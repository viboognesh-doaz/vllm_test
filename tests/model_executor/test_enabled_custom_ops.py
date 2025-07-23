from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.activation import GeluAndMul, ReLUSquaredActivation, SiluAndMul
from vllm.model_executor.layers.fused_moe.fused_moe import dispatch_topk_func, vllm_topk_softmax
from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import is_rocm_aiter_moe_enabled
from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import rocm_aiter_topk_softmax
from vllm.model_executor.layers.layernorm import RMSNorm, dispatch_cuda_rmsnorm_func, fused_add_rms_norm, rms_norm, rocm_aiter_fused_add_rms_norm, rocm_aiter_rms_norm
from vllm.model_executor.layers.quantization.utils.fp8_utils import cutlass_scaled_mm, dispatch_w8a8_blockscale_func, w8a8_block_fp8_matmul
from vllm.platforms import current_platform
import pytest
import torch

@CustomOp.register('relu3')
class Relu3(ReLUSquaredActivation):
    pass

@pytest.mark.parametrize('env, torch_level, use_inductor, ops_enabled, default_on', [('', 0, False, [True] * 4, True), ('', 1, True, [True] * 4, True), ('', 2, False, [True] * 4, True), ('', 3, True, [False] * 4, False), ('', 4, True, [False] * 4, False), ('', 3, False, [True] * 4, True), ('', 4, False, [True] * 4, True), ('+rms_norm,-silu_and_mul', 0, True, [1, 0, 1, 1], True), ('none,-rms_norm,+relu3', 1, False, [0, 0, 0, 1], False), ('all,-silu_and_mul', 2, True, [1, 0, 1, 1], True), ('-relu3,relu2', 3, False, [1, 1, 1, 0], True), ('none,-relu3,+rms_norm,+silu_and_mul', 4, False, [1, 1, 0, 0], False), ('-rms_norm', 3, False, [0, 1, 1, 1], True), ('-silu_and_mul,+relu3', 3, True, [0, 0, 0, 1], False), ('all,-rms_norm', 4, True, [0, 1, 1, 1], True)])
def test_enabled_ops(env: str, torch_level: int, use_inductor: bool, ops_enabled: list[int], default_on: bool):
    vllm_config = VllmConfig(compilation_config=CompilationConfig(use_inductor=bool(use_inductor), level=torch_level, custom_ops=env.split(',')))
    with set_current_vllm_config(vllm_config):
        assert CustomOp.default_on() == default_on
        ops_enabled = [bool(x) for x in ops_enabled]
        assert RMSNorm(1024).enabled() == ops_enabled[0]
        assert CustomOp.op_registry['rms_norm'].enabled() == ops_enabled[0]
        assert SiluAndMul().enabled() == ops_enabled[1]
        assert CustomOp.op_registry['silu_and_mul'].enabled() == ops_enabled[1]
        assert GeluAndMul().enabled() == ops_enabled[2]
        assert CustomOp.op_registry['gelu_and_mul'].enabled() == ops_enabled[2]
        assert Relu3().enabled() == ops_enabled[3]
        assert CustomOp.op_registry['relu3'].enabled() == ops_enabled[3]

        class SiluAndMul2(SiluAndMul):
            pass
        assert SiluAndMul2().enabled() == SiluAndMul().enabled()

@pytest.mark.parametrize('env', ['all,none', 'all,+rms_norm,all', '+rms_norm,-rms_norm'])
def test_enabled_ops_invalid(env: str):
    with pytest.raises(Exception):
        vllm_config = VllmConfig(compilation_config=CompilationConfig(custom_ops=env.split(',')))
        with set_current_vllm_config(vllm_config):
            RMSNorm(1024).enabled()

@pytest.mark.skipif(not current_platform.is_rocm() or not current_platform.is_fp8_fnuz(), reason='AITER is a feature exclusive for ROCm and FP8_FNUZ')
@pytest.mark.parametrize('use_cutlass', [True, False])
@pytest.mark.parametrize('use_rocm_aiter', ['0', '1'])
@pytest.mark.parametrize('use_rocm_aiter_gemm_w8a8_blockscale', ['0', '1'])
def test_w8a8_blockscale_dispatch(use_cutlass: bool, use_rocm_aiter: str, use_rocm_aiter_gemm_w8a8_blockscale: str, monkeypatch):
    monkeypatch.setenv('VLLM_ROCM_USE_AITER', use_rocm_aiter)
    monkeypatch.setenv('VLLM_ROCM_USE_AITER_LINEAR', use_rocm_aiter_gemm_w8a8_blockscale)
    use_aiter_and_is_supported = bool(int(use_rocm_aiter)) and bool(int(use_rocm_aiter_gemm_w8a8_blockscale))
    block_scale_func = dispatch_w8a8_blockscale_func(use_cutlass, use_aiter_and_is_supported=use_aiter_and_is_supported)
    if use_cutlass:
        assert block_scale_func == cutlass_scaled_mm
    elif current_platform.is_rocm() and int(use_rocm_aiter) and int(use_rocm_aiter_gemm_w8a8_blockscale):
        assert block_scale_func == torch.ops.vllm.rocm_aiter_gemm_w8a8_blockscale
    else:
        assert block_scale_func == w8a8_block_fp8_matmul

@pytest.mark.parametrize('use_rocm_aiter', ['0', '1'])
def test_topk_dispatch(use_rocm_aiter: str, monkeypatch):
    monkeypatch.setenv('VLLM_ROCM_USE_AITER', use_rocm_aiter)
    topk_func = dispatch_topk_func()
    is_rocm_aiter_moe_enabled.cache_clear()
    if current_platform.is_rocm() and int(use_rocm_aiter):
        assert topk_func == rocm_aiter_topk_softmax
    else:
        assert topk_func == vllm_topk_softmax

@pytest.mark.parametrize('add_residual', [True, False])
@pytest.mark.parametrize('use_rocm_aiter', ['0', '1'])
@pytest.mark.parametrize('use_rocm_aiter_norm', ['0', '1'])
@pytest.mark.skipif(not current_platform.is_rocm(), reason='AITER is a feature exclusive for ROCm')
def test_rms_norm_dispatch(add_residual: bool, use_rocm_aiter: str, use_rocm_aiter_norm: str, monkeypatch):
    monkeypatch.setenv('VLLM_ROCM_USE_AITER', use_rocm_aiter)
    monkeypatch.setenv('VLLM_ROCM_USE_AITER_RMSNORM', use_rocm_aiter_norm)
    rms_norm_func = dispatch_cuda_rmsnorm_func(add_residual)
    if not add_residual:
        if current_platform.is_rocm() and int(use_rocm_aiter) and int(use_rocm_aiter_norm):
            assert rms_norm_func == rocm_aiter_rms_norm
        else:
            assert rms_norm_func == rms_norm
    elif current_platform.is_rocm() and int(use_rocm_aiter) and int(use_rocm_aiter_norm):
        assert rms_norm_func == rocm_aiter_fused_add_rms_norm
    else:
        assert rms_norm_func == fused_add_rms_norm