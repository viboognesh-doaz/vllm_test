from collections.abc import Callable
from vllm.attention.backends.xformers import _make_alibi_bias
from vllm.attention.ops.chunked_prefill_paged_decode import chunked_prefill_paged_decode
from vllm.attention.ops.prefix_prefill import context_attention_fwd
from vllm.platforms import current_platform
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE
from xformers import ops as xops
from xformers.ops.fmha.attn_bias import BlockDiagonalCausalFromBottomRightMask
import math
import pytest
import random
import time
import torch
NUM_HEADS = [64]
NUM_QUERIES_PER_KV = [1, 8, 64]
HEAD_SIZES = [128, 96, 24]
DTYPES = [torch.float16]
CUDA_DEVICES = [f'cuda:{i}' for i in range(1 if torch.cuda.device_count() == 1 else 2)]
SLIDING_WINDOW = [0, 16, 64, 128, 256, 512, 2048]
KV_CACHE_DTYPES = ['auto', 'fp8', 'fp8_e5m2']
OPS = [chunked_prefill_paged_decode, context_attention_fwd]

@pytest.mark.parametrize('num_heads', NUM_HEADS)
@pytest.mark.parametrize('num_queries_per_kv', NUM_QUERIES_PER_KV)
@pytest.mark.parametrize('head_size', HEAD_SIZES)
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('kv_cache_dtype', KV_CACHE_DTYPES)
@pytest.mark.parametrize('device', CUDA_DEVICES)
@pytest.mark.parametrize('sliding_window', SLIDING_WINDOW)
@pytest.mark.parametrize('op', OPS)
@torch.inference_mode()
def test_contexted_kv_attention(num_heads: int, num_queries_per_kv: int, head_size: int, sliding_window: int, dtype: torch.dtype, kv_cache_dtype: str, device: str, op: Callable) -> None:
    if 'fp8' in kv_cache_dtype and (not current_platform.has_device_capability(89)):
        pytest.skip('Triton limitation: fp8e4nv data type is not supported on CUDA arch < 89')
    current_platform.seed_everything(0)
    torch.set_default_device(device)
    torch.cuda.set_device(device)
    MAX_SEQ_LEN = 1024
    MAX_CTX_LEN = 1024
    BS = 10
    cache_size = 640
    block_size = 32
    max_block_per_request = 64
    query_lens = [random.randint(16, MAX_SEQ_LEN) for _ in range(BS)]
    query_lens[-1] = 1
    ctx_lens = [random.randint(16, MAX_CTX_LEN) for _ in range(BS)]
    seq_lens = [a + b for a, b in zip(query_lens, ctx_lens)]
    num_kv_heads = num_heads // num_queries_per_kv
    num_tokens = sum(query_lens)
    query = torch.empty(num_tokens, num_heads, head_size, dtype=dtype)
    query.uniform_(-0.001, 0.001)
    output = torch.empty(num_tokens, num_heads, head_size, dtype=dtype)
    kv = torch.empty(sum(seq_lens), 2, num_kv_heads, head_size, dtype=dtype)
    kv.uniform_(-0.001, 0.001)
    key, value = kv.unbind(dim=1)
    if kv_cache_dtype == 'auto':
        cache_dtype = dtype
    else:
        cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[kv_cache_dtype]
    k_cache = torch.zeros(cache_size, block_size, num_kv_heads, head_size, dtype=cache_dtype)
    v_cache = torch.zeros(cache_size, block_size, num_kv_heads, head_size, dtype=cache_dtype)
    k = torch.zeros(sum(query_lens), num_kv_heads, head_size, dtype=dtype)
    v = torch.zeros(sum(query_lens), num_kv_heads, head_size, dtype=dtype)
    values = torch.arange(0, cache_size, dtype=torch.long)
    values = values[torch.randperm(cache_size)]
    block_table = values[:BS * max_block_per_request].view(BS, max_block_per_request)
    b_seq_len = torch.tensor(seq_lens, dtype=torch.long)
    b_ctx_len = torch.tensor(ctx_lens, dtype=torch.long)
    b_start_loc = torch.cumsum(torch.tensor([0] + query_lens, dtype=torch.long), dim=0)
    max_input_len = MAX_SEQ_LEN
    b_seq_start_loc = torch.cumsum(torch.tensor([0] + seq_lens[:-1], dtype=torch.long), dim=0)
    for i in range(BS):
        for j in range(query_lens[i]):
            k[b_start_loc[i] + j].copy_(key[b_seq_start_loc[i] + b_ctx_len[i] + j])
            v[b_start_loc[i] + j].copy_(value[b_seq_start_loc[i] + b_ctx_len[i] + j])
        cur_ctx = 0
        block_id = 0
        while cur_ctx < b_ctx_len[i]:
            start_loc = b_seq_start_loc[i] + cur_ctx
            if cur_ctx + block_size > b_ctx_len[i]:
                end_loc = b_seq_start_loc[i] + b_ctx_len[i]
            else:
                end_loc = start_loc + block_size
            start_slot = block_table[i, block_id] * block_size
            end_slot = start_slot + end_loc - start_loc
            k_cache.view(-1, num_kv_heads, head_size)[start_slot:end_slot].copy_(key[start_loc:end_loc])
            v_cache.view(-1, num_kv_heads, head_size)[start_slot:end_slot].copy_(value[start_loc:end_loc])
            cur_ctx += block_size
            block_id += 1
    k_cache = k_cache.view(-1, block_size, num_kv_heads, head_size // 8, 8).permute(0, 2, 3, 1, 4).contiguous()
    v_cache = v_cache.view(-1, block_size, num_kv_heads, head_size).permute(0, 2, 3, 1).contiguous()
    k_scale = v_scale = torch.tensor(1.0, dtype=torch.float32, device=device)
    op(query, k, v, output, kv_cache_dtype, k_cache, v_cache, block_table, b_start_loc, b_seq_len, MAX_CTX_LEN, max_input_len, k_scale, v_scale, sliding_window=sliding_window)
    torch.cuda.synchronize()
    start_time = time.time()
    op(query, k, v, output, kv_cache_dtype, k_cache, v_cache, block_table, b_start_loc, b_seq_len, MAX_CTX_LEN, max_input_len, k_scale, v_scale, sliding_window=sliding_window)
    torch.cuda.synchronize()
    end_time = time.time()
    print(f'triton Time: {(end_time - start_time) * 1000:.2f} ms')
    scale = float(1.0 / head_size ** 0.5)
    attn_op = xops.fmha.cutlass.FwOp()
    if num_kv_heads != num_heads:
        query = query.view(query.shape[0], num_kv_heads, num_queries_per_kv, query.shape[-1])
        key = key[:, :, None, :].expand(key.shape[0], num_kv_heads, num_queries_per_kv, key.shape[-1])
        value = value[:, :, None, :].expand(value.shape[0], num_kv_heads, num_queries_per_kv, value.shape[-1])
    query = query.unsqueeze(0)
    key = key.unsqueeze(0)
    value = value.unsqueeze(0)
    attn_bias = BlockDiagonalCausalFromBottomRightMask.from_seqlens(query_lens, seq_lens)
    if sliding_window > 0:
        attn_bias = attn_bias.make_local_attention_from_bottomright(sliding_window)
    output_ref = xops.memory_efficient_attention_forward(query, key, value, attn_bias=attn_bias, p=0.0, scale=scale, op=attn_op)
    torch.cuda.synchronize()
    start_time = time.time()
    output_ref = xops.memory_efficient_attention_forward(query, key, value, attn_bias=attn_bias, p=0.0, scale=scale, op=attn_op)
    torch.cuda.synchronize()
    end_time = time.time()
    print(f'xformers Time: {(end_time - start_time) * 1000:.2f} ms')
    output_ref = output_ref.reshape(output.shape)
    atol = 0.001 if 'fp8' in kv_cache_dtype else 0.0001
    torch.testing.assert_close(output, output_ref, atol=atol, rtol=0)

@pytest.mark.parametrize('num_heads', NUM_HEADS)
@pytest.mark.parametrize('num_queries_per_kv', NUM_QUERIES_PER_KV)
@pytest.mark.parametrize('head_size', HEAD_SIZES)
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('kv_cache_dtype', KV_CACHE_DTYPES)
@pytest.mark.parametrize('device', CUDA_DEVICES)
@pytest.mark.parametrize('op', OPS)
@torch.inference_mode()
def test_contexted_kv_attention_alibi(num_heads: int, num_queries_per_kv: int, head_size: int, dtype: torch.dtype, kv_cache_dtype: str, device: str, op: Callable) -> None:
    if 'fp8' in kv_cache_dtype and (not current_platform.has_device_capability(89)):
        pytest.skip('Triton limitation: fp8e4nv data type is not supported on CUDA arch < 89')
    current_platform.seed_everything(0)
    torch.set_default_device(device)
    torch.cuda.set_device(device)

    def _get_alibi_slopes(total_num_heads: int) -> torch.Tensor:
        closest_power_of_2 = 2 ** math.floor(math.log2(total_num_heads))
        base = torch.tensor(2 ** (-2 ** (-(math.log2(closest_power_of_2) - 3))), dtype=torch.float32)
        powers = torch.arange(1, 1 + closest_power_of_2, dtype=torch.int32)
        slopes = torch.pow(base, powers)
        if closest_power_of_2 != total_num_heads:
            extra_base = torch.tensor(2 ** (-2 ** (-(math.log2(2 * closest_power_of_2) - 3))), dtype=torch.float32)
            num_remaining_heads = min(closest_power_of_2, total_num_heads - closest_power_of_2)
            extra_powers = torch.arange(start=1, end=1 + 2 * num_remaining_heads, step=2, dtype=torch.int32)
            slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)
        return slopes
    alibi_slopes = _get_alibi_slopes(num_heads).to(device)
    MAX_SEQ_LEN = 1024
    MAX_CTX_LEN = 1024
    BS = 10
    cache_size = 640
    block_size = 32
    max_block_per_request = 64
    query_lens = [random.randint(16, MAX_SEQ_LEN) for _ in range(BS)]
    ctx_lens = [random.randint(16, MAX_CTX_LEN) for _ in range(BS)]
    seq_lens = [a + b for a, b in zip(query_lens, ctx_lens)]
    num_kv_heads = num_heads // num_queries_per_kv
    num_tokens = sum(query_lens)
    query = torch.empty(num_tokens, num_heads, head_size, dtype=dtype)
    query.uniform_(-0.001, 0.001)
    output = torch.empty(num_tokens, num_heads, head_size, dtype=dtype)
    kv = torch.empty(sum(seq_lens), 2, num_kv_heads, head_size, dtype=dtype)
    kv.uniform_(-0.001, 0.001)
    key, value = kv.unbind(dim=1)
    if kv_cache_dtype == 'auto':
        cache_dtype = dtype
    else:
        cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[kv_cache_dtype]
    k_cache = torch.zeros(cache_size, block_size, num_kv_heads, head_size, dtype=cache_dtype)
    v_cache = torch.zeros(cache_size, block_size, num_kv_heads, head_size, dtype=cache_dtype)
    k = torch.zeros(sum(query_lens), num_kv_heads, head_size, dtype=dtype)
    v = torch.zeros(sum(query_lens), num_kv_heads, head_size, dtype=dtype)
    values = torch.arange(0, cache_size, dtype=torch.long)
    values = values[torch.randperm(cache_size)]
    block_table = values[:BS * max_block_per_request].view(BS, max_block_per_request)
    b_seq_len = torch.tensor(seq_lens, dtype=torch.long)
    b_ctx_len = torch.tensor(ctx_lens, dtype=torch.long)
    b_start_loc = torch.cumsum(torch.tensor([0] + query_lens, dtype=torch.long), dim=0)
    max_input_len = MAX_SEQ_LEN
    b_seq_start_loc = torch.cumsum(torch.tensor([0] + seq_lens[:-1], dtype=torch.long), dim=0)
    for i in range(BS):
        for j in range(query_lens[i]):
            k[b_start_loc[i] + j].copy_(key[b_seq_start_loc[i] + b_ctx_len[i] + j])
            v[b_start_loc[i] + j].copy_(value[b_seq_start_loc[i] + b_ctx_len[i] + j])
        cur_ctx = 0
        block_id = 0
        while cur_ctx < b_ctx_len[i]:
            start_loc = b_seq_start_loc[i] + cur_ctx
            if cur_ctx + block_size > b_ctx_len[i]:
                end_loc = b_seq_start_loc[i] + b_ctx_len[i]
            else:
                end_loc = start_loc + block_size
            start_slot = block_table[i, block_id] * block_size
            end_slot = start_slot + end_loc - start_loc
            k_cache.view(-1, num_kv_heads, head_size)[start_slot:end_slot].copy_(key[start_loc:end_loc])
            v_cache.view(-1, num_kv_heads, head_size)[start_slot:end_slot].copy_(value[start_loc:end_loc])
            cur_ctx += block_size
            block_id += 1
    k_cache = k_cache.view(-1, block_size, num_kv_heads, head_size // 8, 8).permute(0, 2, 3, 1, 4).contiguous()
    v_cache = v_cache.view(-1, block_size, num_kv_heads, head_size).permute(0, 2, 3, 1).contiguous()
    k_scale = v_scale = torch.tensor(1.0, dtype=torch.float32, device=device)
    op(query, k, v, output, kv_cache_dtype, k_cache, v_cache, block_table, b_start_loc, b_seq_len, MAX_CTX_LEN, max_input_len, k_scale, v_scale, alibi_slopes=alibi_slopes)
    torch.cuda.synchronize()
    start_time = time.time()
    op(query, k, v, output, kv_cache_dtype, k_cache, v_cache, block_table, b_start_loc, b_seq_len, MAX_CTX_LEN, max_input_len, k_scale, v_scale, alibi_slopes=alibi_slopes)
    torch.cuda.synchronize()
    end_time = time.time()
    print(f'triton Time: {(end_time - start_time) * 1000:.2f} ms')
    scale = float(1.0 / head_size ** 0.5)
    if query.shape[0] != key.shape[0]:
        query_pad = torch.empty(sum(seq_lens), num_heads, head_size, dtype=dtype)
        query_pad.uniform_(-0.001, 0.001)
        seq_start = 0
        query_start = 0
        for i, (query_len, seq_len) in enumerate(zip(query_lens, seq_lens)):
            seq_end = seq_start + seq_len
            query_end = query_start + query_len
            query_pad[seq_start:seq_end, ...] = torch.cat([torch.zeros(seq_len - query_len, num_heads, head_size, dtype=dtype), query[query_start:query_end, ...]], dim=0)
            seq_start += seq_len
            query_start += query_len
        query = query_pad
    if num_kv_heads != num_heads:
        key = key[:, :, None, :].expand(key.shape[0], num_kv_heads, num_queries_per_kv, key.shape[-1])
        value = value[:, :, None, :].expand(value.shape[0], num_kv_heads, num_queries_per_kv, value.shape[-1])
        key = key.reshape(key.shape[0], -1, key.shape[-1])
        value = value.reshape(value.shape[0], -1, value.shape[-1])
    query = query.unsqueeze(0)
    key = key.unsqueeze(0)
    value = value.unsqueeze(0)
    attn_bias = _make_alibi_bias(alibi_slopes, num_kv_heads, dtype, seq_lens)
    output_ref = torch.empty_like(output)
    seq_start = 0
    query_start = 0
    start_time = time.time()
    for i, (query_len, seq_len) in enumerate(zip(query_lens, seq_lens)):
        seq_end = seq_start + seq_len
        query_end = query_start + query_len
        out = xops.memory_efficient_attention_forward(query[:, seq_start:seq_end], key[:, seq_start:seq_end], value[:, seq_start:seq_end], attn_bias=attn_bias[i], p=0.0, scale=scale)
        out = out.view_as(query[:, seq_start:seq_end]).view(seq_len, num_heads, head_size)
        output_ref[query_start:query_end, ...].copy_(out[seq_len - query_len:, ...])
        seq_start += seq_len
        query_start += query_len
    torch.cuda.synchronize()
    end_time = time.time()
    print(f'xformers Time: {(end_time - start_time) * 1000:.2f} ms')
    atol = 0.001 if 'fp8' in kv_cache_dtype else 1e-06
    torch.testing.assert_close(output, output_ref, atol=atol, rtol=0)

@pytest.mark.optional
@pytest.mark.parametrize('num_heads', NUM_HEADS)
@pytest.mark.parametrize('num_queries_per_kv', NUM_QUERIES_PER_KV)
@pytest.mark.parametrize('head_size', HEAD_SIZES)
@pytest.mark.parametrize('dtype', [torch.float32])
@pytest.mark.parametrize('kv_cache_dtype', KV_CACHE_DTYPES)
@pytest.mark.parametrize('device', CUDA_DEVICES)
@pytest.mark.parametrize('sliding_window', SLIDING_WINDOW)
@pytest.mark.parametrize('op', OPS)
@torch.inference_mode()
def test_contexted_kv_attention_f32(num_heads: int, num_queries_per_kv: int, head_size: int, sliding_window: int, dtype: torch.dtype, kv_cache_dtype: str, device: str, op: Callable) -> None:
    test_contexted_kv_attention(num_heads, num_queries_per_kv, head_size, sliding_window, dtype, kv_cache_dtype, device, op)

@pytest.mark.optional
@pytest.mark.parametrize('num_heads', NUM_HEADS)
@pytest.mark.parametrize('num_queries_per_kv', NUM_QUERIES_PER_KV)
@pytest.mark.parametrize('head_size', HEAD_SIZES)
@pytest.mark.parametrize('dtype', [torch.float32])
@pytest.mark.parametrize('kv_cache_dtype', KV_CACHE_DTYPES)
@pytest.mark.parametrize('device', CUDA_DEVICES)
@pytest.mark.parametrize('op', OPS)
@torch.inference_mode()
def test_contexted_kv_attention_alibi_f32(num_heads: int, num_queries_per_kv: int, head_size: int, dtype: torch.dtype, kv_cache_dtype: str, device: str, op: Callable) -> None:
    test_contexted_kv_attention_alibi(num_heads, num_queries_per_kv, head_size, dtype, kv_cache_dtype, device, op)