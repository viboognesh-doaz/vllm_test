from vllm.platforms import current_platform
import pytest
'Test model set-up and inference for quantized HF models supported\n on the AutoRound.\n\n Validating the configuration and printing results for manual checking.\n\n Run `pytest tests/quantization/test_auto_round.py`.\n'
MODELS = ['OPEA/Qwen2.5-0.5B-Instruct-int4-sym-inc', 'Intel/Qwen2-0.5B-Instruct-int4-sym-AutoRound']

@pytest.mark.skipif(not current_platform.is_cpu() and (not current_platform.is_xpu()) and (not current_platform.is_cuda()), reason='only supports CPU/XPU/CUDA backend.')
@pytest.mark.parametrize('model', MODELS)
def test_auto_round(vllm_runner, model):
    with vllm_runner(model) as llm:
        output = llm.generate_greedy(['The capital of France is'], max_tokens=8)
    assert output
    print(f'{output[0][1]}')