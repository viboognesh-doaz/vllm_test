from tests.quantization.utils import is_quant_method_supported
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.fp8 import Fp8KVCacheMethod, Fp8LinearMethod
from vllm.platforms import current_platform
import pytest
import torch
'Tests whether FP8 computation is enabled correctly.\n\nRun `pytest tests/quantization/test_fp8.py --forked`.\n'
MODELS = ['neuralmagic/Meta-Llama-3-8B-Instruct-FP8-KV', 'nm-testing/Phi-3-mini-128k-instruct-FP8', 'nm-testing/Qwen2-0.5B-Instruct-FP8-SkipQKV']

@pytest.mark.skipif(not is_quant_method_supported('fp8'), reason='FP8 is not supported on this GPU type.')
@pytest.mark.parametrize('model_id', MODELS)
@pytest.mark.parametrize('force_marlin', [False, True])
@pytest.mark.parametrize('use_rocm_aiter', [True, False] if current_platform.is_rocm() else [False])
def test_model_load_and_run(vllm_runner, model_id: str, force_marlin: bool, use_rocm_aiter: bool, monkeypatch) -> None:
    if use_rocm_aiter:
        monkeypatch.setenv('VLLM_ROCM_USE_AITER', '1')
    if force_marlin:
        monkeypatch.setenv('VLLM_TEST_FORCE_FP8_MARLIN', '1')
    with vllm_runner(model_id) as llm:
        outputs = llm.generate_greedy(prompts=['Hello my name is'], max_tokens=10)
        print(outputs[0][1])
KV_CACHE_MODELS = ['neuralmagic/Meta-Llama-3-8B-Instruct-FP8-KV', 'nm-testing/Qwen2-1.5B-Instruct-FP8-K-V']

@pytest.mark.skipif(not is_quant_method_supported('fp8'), reason='FP8 is not supported on this GPU type.')
@pytest.mark.parametrize('model_id', KV_CACHE_MODELS)
@pytest.mark.parametrize('use_rocm_aiter', [True, False] if current_platform.is_rocm() else [False])
def test_kv_cache_model_load_and_run(vllm_runner, model_id: str, use_rocm_aiter: bool, monkeypatch):
    if use_rocm_aiter:
        monkeypatch.setenv('VLLM_ROCM_USE_AITER', '1')
    monkeypatch.setenv('VLLM_USE_V1', '0')
    with vllm_runner(model_id, kv_cache_dtype='fp8') as llm:

        def check_model(model):
            attn = model.model.layers[0].self_attn.attn
            assert isinstance(attn.quant_method, Fp8KVCacheMethod)
            if not current_platform.is_rocm():
                assert 0.0 < attn._k_scale < 1.0
                assert 0.0 < attn._v_scale < 1.0
            else:
                assert 0.0 < attn._k_scale < 1.0 * 2.0
                assert 0.0 < attn._v_scale < 1.0 * 2.0
        llm.apply_model(check_model)
        outputs = llm.generate_greedy(prompts=['Hello my name is'], max_tokens=10)
        print(outputs[0][1])

@pytest.mark.skipif(not is_quant_method_supported('fp8'), reason='FP8 is not supported on this GPU type.')
@pytest.mark.parametrize('kv_cache_dtype', ['auto', 'fp8'])
@pytest.mark.parametrize('force_marlin', [False, True])
@pytest.mark.parametrize('use_rocm_aiter', [True, False] if current_platform.is_rocm() else [False])
def test_load_fp16_model(vllm_runner, kv_cache_dtype: str, force_marlin: bool, use_rocm_aiter: bool, monkeypatch) -> None:
    if use_rocm_aiter:
        monkeypatch.setenv('VLLM_ROCM_USE_AITER', '1')
    monkeypatch.setenv('VLLM_USE_V1', '0')
    if force_marlin:
        monkeypatch.setenv('VLLM_TEST_FORCE_FP8_MARLIN', '1')
    with vllm_runner('facebook/opt-125m', quantization='fp8', kv_cache_dtype=kv_cache_dtype) as llm:

        def check_model(model):
            fc1 = model.model.decoder.layers[0].fc1
            assert isinstance(fc1.quant_method, Fp8LinearMethod)
            if kv_cache_dtype == 'fp8':
                attn = model.model.decoder.layers[0].self_attn.attn
                assert isinstance(attn.quant_method, Fp8KVCacheMethod)
                assert attn._k_scale == 1.0
                assert attn._v_scale == 1.0
            if current_platform.is_cuda():
                if current_platform.supports_fp8() and (not force_marlin):
                    assert fc1.weight.dtype == torch.float8_e4m3fn
                else:
                    assert fc1.weight.dtype == torch.int32
            elif current_platform.is_rocm():
                if current_platform.supports_fp8() and (not force_marlin):
                    assert fc1.weight.dtype == current_platform.fp8_dtype()
                else:
                    pytest.skip('Skip `test_load_fp16_model`. It only runs on ROCm platform with FP8 compute. e.g. MI300X and above.')
            else:
                pytest.skip('Skip `test_load_fp16_model`. It only runs on CUDA and ROCm platform.')
        llm.apply_model(check_model)

@pytest.mark.skipif(not is_quant_method_supported('fp8'), reason='FP8 is not supported on this GPU type.')
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_scaled_fp8_quant(dtype) -> None:

    def quantize_ref(tensor, inv_scale):
        finfo = torch.finfo(torch.float8_e4m3fn)
        scale = inv_scale.reciprocal()
        qweight = (tensor.to(torch.float32) * scale).clamp(min=finfo.min, max=finfo.max)
        qweight = qweight.to(torch.float8_e4m3fn)
        return qweight

    def per_tensor_dequantize(tensor, inv_scale, dtype):
        fake_qweight = tensor.to(dtype)
        dq_weight = fake_qweight * inv_scale
        return dq_weight
    x = (torch.randn(size=(11, 11), device='cuda') * 13).to(dtype)
    ref_y, inv_scale = ops.scaled_fp8_quant(x, None)
    ref_y = per_tensor_dequantize(ref_y, inv_scale, dtype)
    y = quantize_ref(x, inv_scale)
    torch.testing.assert_close(ref_y, per_tensor_dequantize(y, inv_scale, dtype))
    y, _ = ops.scaled_fp8_quant(x, inv_scale)
    torch.testing.assert_close(ref_y, per_tensor_dequantize(y, inv_scale, dtype))
    y, _ = ops.scaled_fp8_quant(x, inv_scale, num_token_padding=17)
    assert y.shape[0] == 17
    torch.testing.assert_close(ref_y, per_tensor_dequantize(torch.narrow(y, 0, 0, x.shape[0]), inv_scale, dtype))