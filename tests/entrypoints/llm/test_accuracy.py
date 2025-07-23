from vllm.platforms import current_platform
import lm_eval
import pytest
'\nThis file test accuracy of the vLLM server via LMEval.\nIt uses local-completions, which interacts with vLLM\nthrough the OAI API with N concurrent connections.\nThis simulates real work usage of the API and makes\nsure that the zmq frontend mp RPC message passing and\nAsyncLLMEngine are working correctly.\n'
MODEL_NAMES = ['Qwen/Qwen3-1.7B', 'google/gemma-3-1b-it']
FP8_KV_MODEL_NAMES = ['Qwen/Qwen3-1.7B']
NUM_CONCURRENT = 500
TASK = 'gsm8k'
FILTER = 'exact_match,strict-match'
RTOL = 0.03
EXPECTED_VALUES = {'Qwen/Qwen3-1.7B': 0.68, 'google/gemma-3-1b-it': 0.25}

def run_test(model_name, more_args=None):
    """Run the end to end accuracy test."""
    model_args = f'pretrained={model_name},max_model_len=4096'
    if more_args is not None:
        model_args = '{},{}'.format(model_args, more_args)
    results = lm_eval.simple_evaluate(model='vllm', model_args=model_args, tasks='gsm8k', batch_size='auto')
    measured_value = results['results'][TASK][FILTER]
    assert model_name in EXPECTED_VALUES, f'Cannot find the expected value for the model model_name={model_name!r}'
    expected_value = EXPECTED_VALUES[model_name]
    assert measured_value - RTOL < expected_value and measured_value + RTOL > expected_value, f'Expected: {expected_value} |  Measured: {measured_value}'
TPU_TP_TEST_STR = ''

@pytest.mark.skipif(not current_platform.is_cuda() and (not current_platform.is_tpu()), reason='V1 is currently only supported on CUDA and TPU')
@pytest.mark.parametrize('model', MODEL_NAMES)
def test_lm_eval_accuracy_v1_engine(model, monkeypatch: pytest.MonkeyPatch):
    """Run with the V1 Engine."""
    with monkeypatch.context() as m:
        m.setenv('VLLM_USE_V1', '1')
        more_args = None
        if current_platform.is_tpu():
            m.setenv('HF_HUB_DISABLE_XET', '1')
            more_args = 'max_model_len=2048,max_num_seqs=64'
            if TPU_TP_TEST_STR:
                more_args += ',{}'.format(TPU_TP_TEST_STR)
        run_test(model, more_args)

@pytest.mark.skipif(not current_platform.is_cuda() and (not current_platform.is_tpu()), reason='V1 is currently only supported on CUDA and TPU')
@pytest.mark.parametrize('model', FP8_KV_MODEL_NAMES)
def test_lm_eval_accuracy_v1_engine_fp8_kv_cache(model, monkeypatch: pytest.MonkeyPatch):
    """Run with the V1 Engine."""
    with monkeypatch.context() as m:
        m.setenv('VLLM_USE_V1', '1')
        more_args = None
        if current_platform.is_tpu():
            m.setenv('HF_HUB_DISABLE_XET', '1')
            more_args = 'max_model_len=2048,max_num_seqs=128,kv_cache_dtype=fp8'
            if TPU_TP_TEST_STR:
                more_args += ',{}'.format(TPU_TP_TEST_STR)
        run_test(model, more_args)