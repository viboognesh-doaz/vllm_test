from tests.quantization.utils import is_quant_method_supported
import pytest
"Tests RTN quantization startup and generation, \ndoesn't test correctness\n"
MODELS = ['microsoft/Phi-3-mini-4k-instruct']

@pytest.mark.skipif(not is_quant_method_supported('rtn'), reason='RTN is not supported on this GPU type.')
@pytest.mark.parametrize('model', MODELS)
@pytest.mark.parametrize('dtype', ['bfloat16'])
@pytest.mark.parametrize('max_tokens', [10])
def test_model_rtn_startup(hf_runner, vllm_runner, example_prompts, model: str, dtype: str, max_tokens: int) -> None:
    with vllm_runner(model, dtype=dtype, quantization='rtn') as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens)