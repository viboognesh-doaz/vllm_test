from vllm import LLM
from vllm.distributed import cleanup_dist_env_and_memory
import pytest
'Compare the with and without prefix caching.\n\nRun `pytest tests/prefix_caching/test_prefix_caching.py`.\n'
MODEL_LEN_LEN = [('bigcode/starcoder2-3b', 4096, 16384), ('Qwen/Qwen1.5-0.5B-Chat', 32768, 32768), ('TinyLlama/TinyLlama-1.1B-Chat-v1.0', 2048, 2048)]

@pytest.mark.parametrize('model_len_len', MODEL_LEN_LEN)
def test_disable_sliding_window(model_len_len):
    model, sliding_len, full_len = model_len_len
    vllm_disabled_model = LLM(model, disable_sliding_window=True)
    vllm_disabled_model.generate('Hi my name is')
    model_config = vllm_disabled_model.llm_engine.model_config
    assert model_config.max_model_len == sliding_len, ('Max len expected to equal sliding_len of %s, but got %s', sliding_len, model_config.max_model_len)
    del vllm_disabled_model
    cleanup_dist_env_and_memory()
    vllm_enabled_model = LLM(model, enforce_eager=True, disable_sliding_window=False, enable_prefix_caching=False)
    vllm_enabled_model.generate('Hi my name is')
    model_config = vllm_enabled_model.llm_engine.model_config
    assert model_config.max_model_len == full_len, ('Max len expected to equal full_len of %s, but got %s', full_len, model_config.max_model_len)
    del vllm_enabled_model
    cleanup_dist_env_and_memory()