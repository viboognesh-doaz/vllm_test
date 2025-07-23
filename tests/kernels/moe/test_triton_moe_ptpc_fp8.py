from vllm import _custom_ops as ops
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe import fused_moe
from vllm.platforms import current_platform
import itertools
import pytest
import torch
if current_platform.get_device_capability() < (9, 0):
    pytest.skip('FP8 Triton requires CUDA 9.0 or higher', allow_module_level=True)
vllm_config = VllmConfig()
vllm_config.scheduler_config.max_num_seqs = 128
vllm_config.scheduler_config.max_model_len = 8192

def native_w8a8_per_token_matmul(A, B, As, Bs, output_dtype=torch.float16):
    """Matrix multiplication function that supports per-token input
    quantization and per-column weight quantization"""
    A = A.to(torch.float32)
    B = B.to(torch.float32)
    assert A.shape[-1] == B.shape[-1], 'Dimension mismatch'
    assert B.ndim == 2 and B.is_contiguous(), 'B must be a 2D contiguous tensor'
    M = A.numel() // A.shape[-1]
    B = B.t()
    N, K = B.shape
    origin_C_shape = A.shape[:-1] + (K,)
    A = A.reshape(M, N)
    C = torch.matmul(A, B)
    C = As * C * Bs.view(1, -1)
    return C.reshape(origin_C_shape).to(output_dtype)

def fp8_mask(a, mask):
    dtype = a.dtype
    return a.view(torch.int8)[mask].view(dtype)

def torch_w8a8_per_column_moe(a, w1, w2, w1_s, w2_s, score, topk):
    """This function performs fused moe with per-column int8
    quantization using native torch."""
    B, D = a.shape
    a_q, a_s = ops.scaled_fp8_quant(a, use_per_token_if_dynamic=True)
    a_q = a_q.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
    a_s = a_s.view(B, -1, 1).repeat(1, topk, 1).reshape(-1, 1)
    out = torch.zeros(B * topk, w2.shape[1], dtype=a.dtype, device=a.device)
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)
    topk_weight = topk_weight.view(-1)
    topk_ids = topk_ids.view(-1)
    for i in range(w1.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            inter_out = native_w8a8_per_token_matmul(fp8_mask(a_q, mask), w1[i], fp8_mask(a_s, mask), w1_s[i], output_dtype=a.dtype)
            act_out = SiluAndMul().forward_native(inter_out)
            act_out_q, act_out_s = ops.scaled_fp8_quant(act_out, use_per_token_if_dynamic=True)
            out[mask] = native_w8a8_per_token_matmul(act_out_q, w2[i], act_out_s, w2_s[i], output_dtype=a.dtype)
    return (out.view(B, -1, w2.shape[1]) * topk_weight.view(B, -1, 1).to(out.dtype)).sum(dim=1)

@pytest.fixture(autouse=True, scope='module')
def setup_cuda():
    """Sets the default CUDA device for all tests in this module."""
    torch.set_default_device('cuda')
DTYPES = [torch.half, torch.bfloat16]
M = [1, 33]
N = [128, 1024]
K = [256, 4096]
E = [8]
TOP_KS = [2, 6]
SEEDS = [0]

@pytest.mark.parametrize('M, N, K, E, topk, dtype, seed', itertools.product(M, N, K, E, TOP_KS, DTYPES, SEEDS))
@torch.inference_mode()
def test_w8a8_fp8_fused_moe(M, N, K, E, topk, dtype, seed):
    torch.manual_seed(seed)
    factor_for_scale = 0.01
    finfo = torch.finfo(torch.float8_e4m3fn)
    fp8_max = finfo.max
    fp8_min = finfo.min
    a = torch.randn((M, K), dtype=dtype) / 10
    w1_fp32 = (torch.rand((E, 2 * N, K), dtype=torch.float32) - 0.5) * 2
    w1 = (w1_fp32 * fp8_max).clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)
    w2_fp32 = (torch.rand((E, K, N), dtype=torch.float32) - 0.5) * 2
    w2 = (w2_fp32 * fp8_max).clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)
    w1_s = torch.rand(E, 2 * N, device=w1_fp32.device) * factor_for_scale
    w2_s = torch.rand(E, K, device=w2_fp32.device) * factor_for_scale
    score = torch.randn((M, E), dtype=dtype)
    with set_current_vllm_config(vllm_config):
        ref_out = torch_w8a8_per_column_moe(a, w1, w2, w1_s, w2_s, score, topk)
        out = fused_moe(a, w1, w2, score, topk, renormalize=False, use_fp8_w8a8=True, per_channel_quant=True, w1_scale=w1_s, w2_scale=w2_s, block_shape=None)
    rel_diff = torch.mean(torch.abs(out.to(torch.float32) - ref_out.to(torch.float32))) / torch.mean(torch.abs(ref_out.to(torch.float32)))
    assert rel_diff < 0.05