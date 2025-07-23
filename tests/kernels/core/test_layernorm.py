from tests.kernels.quant_utils import FP8_DTYPE
from tests.kernels.utils import opcheck
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.platforms import current_platform
import pytest
import torch
DTYPES = [torch.half, torch.bfloat16, torch.float]
NUM_TOKENS = [7, 83, 4096]
HIDDEN_SIZES = [8, 768, 769, 770, 771, 5120, 5124, 5125, 5126, 8192, 8199]
ADD_RESIDUAL = [False, True]
SEEDS = [0]
CUDA_DEVICES = [f'cuda:{i}' for i in range(1 if torch.cuda.device_count() == 1 else 2)]

@pytest.mark.parametrize('num_tokens', NUM_TOKENS)
@pytest.mark.parametrize('hidden_size', HIDDEN_SIZES)
@pytest.mark.parametrize('add_residual', ADD_RESIDUAL)
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('seed', SEEDS)
@pytest.mark.parametrize('device', CUDA_DEVICES)
@torch.inference_mode()
def test_rms_norm(num_tokens: int, hidden_size: int, add_residual: bool, dtype: torch.dtype, seed: int, device: str) -> None:
    current_platform.seed_everything(seed)
    torch.set_default_device(device)
    layer = RMSNorm(hidden_size).to(dtype=dtype)
    layer.weight.data.normal_(mean=1.0, std=0.1)
    scale = 1 / (2 * hidden_size)
    x = torch.randn(num_tokens, hidden_size, dtype=dtype)
    x *= scale
    residual = torch.randn_like(x) * scale if add_residual else None
    ref_out = layer.forward_native(x, residual)
    out = layer(x, residual)
    if add_residual:
        torch.testing.assert_close(out[0], ref_out[0], atol=0.01, rtol=0.01)
        torch.testing.assert_close(out[1], ref_out[1], atol=0.01, rtol=0.01)
    else:
        torch.testing.assert_close(out, ref_out, atol=0.01, rtol=0.01)
    if residual is not None:
        opcheck(torch.ops._C.fused_add_rms_norm, (x, residual, layer.weight.data, layer.variance_epsilon))
    else:
        opcheck(torch.ops._C.rms_norm, (out, x, layer.weight.data, layer.variance_epsilon))

@pytest.mark.parametrize('num_tokens', NUM_TOKENS)
@pytest.mark.parametrize('hidden_size', HIDDEN_SIZES)
@pytest.mark.parametrize('add_residual', ADD_RESIDUAL)
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('quant_scale', [1.0, 0.01, 10.0])
@pytest.mark.parametrize('seed', SEEDS)
@pytest.mark.parametrize('device', CUDA_DEVICES)
def test_fused_rms_norm_quant(num_tokens: int, hidden_size: int, add_residual: bool, dtype: torch.dtype, quant_scale: float, seed: int, device: str) -> None:
    current_platform.seed_everything(seed)
    torch.set_default_device(device)
    weight = torch.empty(hidden_size, dtype=dtype).normal_(mean=1.0, std=0.1)
    scale = 1 / (2 * hidden_size)
    x = torch.randn(num_tokens, hidden_size, dtype=dtype)
    x *= scale
    if add_residual:
        residual = torch.randn_like(x) * scale
        residual_fused = residual.clone()
    else:
        residual = residual_fused = None
    out_norm = torch.empty_like(x)
    out_quant = torch.empty_like(x, dtype=FP8_DTYPE)
    out_quant_fused = torch.empty_like(out_quant)
    quant_scale_t = torch.tensor(quant_scale, dtype=torch.float32)
    if add_residual:
        torch.ops._C.fused_add_rms_norm_static_fp8_quant(out_quant_fused, x, residual_fused, weight, quant_scale_t, 1e-06)
        x_unfused = x.clone()
        torch.ops._C.fused_add_rms_norm(x_unfused, residual, weight, 1e-06)
        torch.ops._C.static_scaled_fp8_quant(out_quant, x_unfused, quant_scale_t)
        torch.cuda.synchronize()
        torch.testing.assert_close(residual_fused, residual, atol=0.01, rtol=0.01)
        opcheck(torch.ops._C.fused_add_rms_norm_static_fp8_quant, (out_quant_fused, x, residual_fused, weight, quant_scale_t, 1e-06))
    else:
        torch.ops._C.rms_norm_static_fp8_quant(out_quant_fused, x, weight, quant_scale_t, 1e-06)
        torch.ops._C.rms_norm(out_norm, x, weight, 1e-06)
        torch.ops._C.static_scaled_fp8_quant(out_quant, out_norm, quant_scale_t)
        opcheck(torch.ops._C.rms_norm_static_fp8_quant, (out_quant_fused, x, weight, quant_scale_t, 1e-06))
    torch.testing.assert_close(out_quant_fused.to(dtype=torch.float32), out_quant.to(dtype=torch.float32), atol=0.001, rtol=0.001)