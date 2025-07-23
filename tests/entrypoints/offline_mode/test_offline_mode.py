from vllm import LLM
from vllm.distributed import cleanup_dist_env_and_memory
import importlib
import pytest
import sys
import urllib3
'Tests for HF_HUB_OFFLINE mode'
MODEL_CONFIGS = [{'model': 'facebook/opt-125m', 'enforce_eager': True, 'gpu_memory_utilization': 0.2, 'max_model_len': 64, 'max_num_batched_tokens': 64, 'max_num_seqs': 64, 'tensor_parallel_size': 1}, {'model': 'mistralai/Mistral-7B-Instruct-v0.1', 'enforce_eager': True, 'gpu_memory_utilization': 0.95, 'max_model_len': 64, 'max_num_batched_tokens': 64, 'max_num_seqs': 64, 'tensor_parallel_size': 1, 'tokenizer_mode': 'mistral'}, {'model': 'sentence-transformers/all-MiniLM-L12-v2', 'enforce_eager': True, 'gpu_memory_utilization': 0.2, 'max_model_len': 64, 'max_num_batched_tokens': 64, 'max_num_seqs': 64, 'tensor_parallel_size': 1}]

@pytest.fixture(scope='module')
def cache_models():
    for model_config in MODEL_CONFIGS:
        LLM(**model_config)
        cleanup_dist_env_and_memory()
    yield

@pytest.mark.skip_global_cleanup
@pytest.mark.usefixtures('cache_models')
def test_offline_mode(monkeypatch: pytest.MonkeyPatch):
    with monkeypatch.context() as m:
        try:
            m.setenv('HF_HUB_OFFLINE', '1')
            m.setenv('VLLM_NO_USAGE_STATS', '1')

            def disable_connect(*args, **kwargs):
                raise RuntimeError('No http calls allowed')
            m.setattr(urllib3.connection.HTTPConnection, 'connect', disable_connect)
            m.setattr(urllib3.connection.HTTPSConnection, 'connect', disable_connect)
            _re_import_modules()
            for model_config in MODEL_CONFIGS:
                LLM(**model_config)
        finally:
            _re_import_modules()

def _re_import_modules():
    hf_hub_module_names = [k for k in sys.modules if k.startswith('huggingface_hub')]
    transformers_module_names = [k for k in sys.modules if k.startswith('transformers') and (not k.startswith('transformers_modules'))]
    reload_exception = None
    for module_name in hf_hub_module_names + transformers_module_names:
        try:
            importlib.reload(sys.modules[module_name])
        except Exception as e:
            reload_exception = e
    if reload_exception is not None:
        raise reload_exception