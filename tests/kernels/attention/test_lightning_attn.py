from vllm.model_executor.layers.lightning_attn import lightning_attention
from vllm.model_executor.layers.lightning_attn import linear_decode_forward_triton
from vllm.platforms import current_platform
import pytest
import torch
NUM_HEADS = [4, 8]
HEAD_SIZES = [64]
BATCH_SIZES = [1, 2]
SEQ_LENGTHS = [16]
DTYPES = [torch.float32]

def reference_lightning_attention(q, k, v, ed, block_size, kv_history):
    """Reference implementation of lightning attention core algorithm
    
    The difference from the main implementation is that this processes 
    each step sequentially, instead of using parallelized triton kernels
    """
    B, H, S, D = q.shape
    E = v.shape[-1]
    dtype = q.dtype
    output = torch.zeros((B, H, S, E), dtype=dtype, device=q.device)
    if kv_history is None:
        kv_cache = torch.zeros((B, H, D, E), dtype=dtype, device=q.device)
    else:
        kv_cache = kv_history.clone()
    if ed.dim() == 1:
        decay = torch.exp(-ed).view(1, -1, 1, 1)
    else:
        decay = torch.exp(-ed)
    for b in range(B):
        for step in range(S):
            q_bs = q[b, :, step]
            k_bs = k[b, :, step]
            v_bs = v[b, :, step]
            for h in range(H):
                kv_outer = torch.outer(k_bs[h], v_bs[h])
                kv_cache[b, h] = decay[0, h, 0, 0] * kv_cache[b, h] + kv_outer
                output[b, h, step] = torch.matmul(q_bs[h], kv_cache[b, h])
    kv_reshaped = kv_cache.unsqueeze(2)
    final_kv_cache = torch.cat([kv_reshaped, kv_reshaped], dim=2)
    return (output, final_kv_cache)

def reference_linear_decode(q, k, v, kv_caches, slope_rate, slot_idx):
    """Reference implementation: linear attention decode function"""
    B, H, _, D = q.shape
    output = torch.zeros(B, H * D, dtype=q.dtype, device=q.device)
    decay = torch.exp(-slope_rate).view(-1, 1, 1)
    for b in range(B):
        slot_id = slot_idx[b].item()
        if slot_id == -1:
            continue
        q_b = q[b, :, 0]
        k_b = k[b, :, 0]
        v_b = v[b, :, 0]
        for h in range(H):
            q_bh = q_b[h]
            k_bh = k_b[h]
            v_bh = v_b[h]
            kv_cache_old = kv_caches[b, h]
            kv_outer = torch.outer(k_bh, v_bh)
            kv_new = kv_outer + decay[h, 0, 0] * kv_cache_old
            out_h = torch.matmul(q_bh, kv_new)
            output[b, h * D:(h + 1) * D] = out_h
            kv_caches[b, h] = kv_new
    return output

@pytest.mark.parametrize('batch_size', BATCH_SIZES)
@pytest.mark.parametrize('num_heads', NUM_HEADS)
@pytest.mark.parametrize('head_size', HEAD_SIZES)
@pytest.mark.parametrize('dtype', DTYPES)
@torch.inference_mode()
def test_linear_decode_forward_triton(batch_size: int, num_heads: int, head_size: int, dtype: torch.dtype):
    torch.set_default_device('cuda')
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    current_platform.seed_everything(42)
    base = 0.01
    q = base * torch.randn(batch_size, num_heads, 1, head_size, dtype=dtype)
    k = base * torch.randn(batch_size, num_heads, 1, head_size, dtype=dtype)
    v = base * torch.randn(batch_size, num_heads, 1, head_size, dtype=dtype)
    kv_caches = base * torch.randn(batch_size, num_heads, head_size, head_size, dtype=dtype, device='cuda')
    kv_caches_copy = kv_caches.clone()
    slope_rate = torch.zeros(num_heads, device='cuda')
    for h in range(num_heads):
        slope_rate[h] = 0.1 * (h + 1)
    slot_idx = torch.arange(batch_size, device='cuda')
    triton_output = linear_decode_forward_triton(q, k, v, kv_caches, slope_rate, slot_idx)
    reference_output = reference_linear_decode(q, k, v, kv_caches_copy, slope_rate, slot_idx)
    torch.testing.assert_close(triton_output, reference_output, rtol=0.1, atol=0.1)
    torch.testing.assert_close(kv_caches, kv_caches_copy, rtol=0.1, atol=0.1)
    assert triton_output.shape == (batch_size, num_heads * head_size)

@pytest.mark.parametrize('num_heads', NUM_HEADS)
@pytest.mark.parametrize('head_size', HEAD_SIZES)
@pytest.mark.parametrize('dtype', DTYPES)
@torch.inference_mode()
def test_linear_decode_forward_triton_with_padding(num_heads: int, head_size: int, dtype: torch.dtype):
    torch.set_default_device('cuda')
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    current_platform.seed_everything(42)
    batch_size = 4
    base = 0.01
    q = base * torch.randn(batch_size, num_heads, 1, head_size, dtype=dtype)
    k = base * torch.randn(batch_size, num_heads, 1, head_size, dtype=dtype)
    v = base * torch.randn(batch_size, num_heads, 1, head_size, dtype=dtype)
    kv_caches = base * torch.randn(batch_size, num_heads, head_size, head_size, dtype=dtype, device='cuda')
    kv_caches_copy = kv_caches.clone()
    slope_rate = torch.zeros(num_heads, device='cuda')
    for h in range(num_heads):
        slope_rate[h] = 0.1 * (h + 1)
    slot_idx = torch.tensor([0, 1, -1, 2], device='cuda')
    triton_output = linear_decode_forward_triton(q, k, v, kv_caches, slope_rate, slot_idx)
    reference_output = reference_linear_decode(q, k, v, kv_caches_copy, slope_rate, slot_idx)
    padding_mask = (slot_idx != -1).unsqueeze(1).expand(-1, num_heads * head_size)
    triton_masked = triton_output[padding_mask]
    reference_masked = reference_output[padding_mask]
    atol, rtol = (0.15, 0.15)
    valid_indices = slot_idx != -1
    for i in range(batch_size):
        if valid_indices[i] > 0:
            torch.testing.assert_close(kv_caches[i], kv_caches_copy[i], rtol=rtol, atol=atol)
    torch.testing.assert_close(triton_masked, reference_masked, rtol=rtol, atol=atol)
    assert triton_output.shape == (batch_size, num_heads * head_size)

@pytest.mark.parametrize('batch_size', BATCH_SIZES)
@pytest.mark.parametrize('num_heads', NUM_HEADS)
@pytest.mark.parametrize('head_size', HEAD_SIZES)
@pytest.mark.parametrize('seq_len', SEQ_LENGTHS)
@pytest.mark.parametrize('dtype', DTYPES)
@torch.inference_mode()
def test_lightning_attention_reference(batch_size: int, num_heads: int, head_size: int, seq_len: int, dtype: torch.dtype):
    torch.set_default_device('cuda')
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    current_platform.seed_everything(42)
    base = 0.01
    q = base * torch.randn(batch_size, num_heads, seq_len, head_size, dtype=dtype)
    k = base * torch.randn(batch_size, num_heads, seq_len, head_size, dtype=dtype)
    v = base * torch.randn(batch_size, num_heads, seq_len, head_size, dtype=dtype)
    ed = torch.zeros(num_heads, device='cuda')
    for h in range(num_heads):
        ed[h] = 0.1 * (h + 1)
    kv_history = base * torch.randn(batch_size, num_heads, head_size, head_size, dtype=dtype, device='cuda')
    kv_history_clone = kv_history.clone()
    ref_output, ref_kv_cache = reference_lightning_attention(q, k, v, ed, 256, kv_history)
    actual_output, actual_kv_cache = lightning_attention(q, k, v, ed, 256, kv_history_clone)
    atol, rtol = (0.15, 0.15)
    torch.testing.assert_close(ref_output, actual_output, rtol=rtol, atol=atol)
    torch.testing.assert_close(ref_kv_cache, actual_kv_cache, rtol=rtol, atol=atol)
    assert ref_output.shape == (batch_size, num_heads, seq_len, head_size)
    assert ref_kv_cache.shape == actual_kv_cache.shape