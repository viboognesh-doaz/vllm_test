from vllm import LLM
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
import os
import pytest
import vllm.envs as envs
UNSUPPORTED_MODELS_V1 = ['openai/whisper-large-v3', 'facebook/bart-large-cnn', 'state-spaces/mamba-130m-hf', 'BAAI/bge-m3']
MODEL = 'meta-llama/Llama-3.2-1B-Instruct'

@pytest.mark.parametrize('model', UNSUPPORTED_MODELS_V1)
def test_reject_unsupported_models(monkeypatch, model):
    with monkeypatch.context() as m:
        m.setenv('VLLM_USE_V1', '1')
        args = AsyncEngineArgs(model=model)
        with pytest.raises(NotImplementedError):
            _ = args.create_engine_config()
        m.delenv('VLLM_USE_V1')

def test_reject_bad_config(monkeypatch):
    with monkeypatch.context() as m:
        m.setenv('VLLM_USE_V1', '0')

def test_unsupported_configs(monkeypatch):
    with monkeypatch.context() as m:
        m.setenv('VLLM_USE_V1', '1')
        with pytest.raises(NotImplementedError):
            AsyncEngineArgs(model=MODEL, speculative_config={'model': MODEL}).create_engine_config()
        with pytest.raises(NotImplementedError):
            AsyncEngineArgs(model=MODEL, guided_decoding_backend='lm-format-enforcer', guided_decoding_disable_fallback=True).create_engine_config()
        with pytest.raises(NotImplementedError):
            AsyncEngineArgs(model=MODEL, preemption_mode='swap').create_engine_config()
        with pytest.raises(NotImplementedError):
            AsyncEngineArgs(model=MODEL, disable_async_output_proc=True).create_engine_config()
        with pytest.raises(NotImplementedError):
            AsyncEngineArgs(model=MODEL, num_scheduler_steps=5).create_engine_config()
        with pytest.raises(NotImplementedError):
            AsyncEngineArgs(model=MODEL, scheduler_delay_factor=1.2).create_engine_config()

def test_enable_by_default_fallback(monkeypatch):
    with monkeypatch.context() as m:
        if os.getenv('VLLM_USE_V1', None):
            m.delenv('VLLM_USE_V1')
        _ = AsyncEngineArgs(model=MODEL, enforce_eager=True).create_engine_config()
        assert envs.VLLM_USE_V1
        m.delenv('VLLM_USE_V1')
        _ = AsyncEngineArgs(model=UNSUPPORTED_MODELS_V1[0]).create_engine_config()
        assert not envs.VLLM_USE_V1
        m.delenv('VLLM_USE_V1')

def test_v1_llm_by_default(monkeypatch):
    with monkeypatch.context() as m:
        if os.getenv('VLLM_USE_V1', None):
            m.delenv('VLLM_USE_V1')
        model = LLM(MODEL, enforce_eager=True, enable_lora=True)
        print(model.generate('Hello my name is'))
        assert hasattr(model.llm_engine, 'engine_core')
        m.delenv('VLLM_USE_V1')

def test_v1_attn_backend(monkeypatch):
    with monkeypatch.context() as m:
        if os.getenv('VLLM_USE_V1', None):
            m.delenv('VLLM_USE_V1')
        m.setenv('VLLM_ATTENTION_BACKEND', 'XFORMERS')
        _ = AsyncEngineArgs(model=MODEL).create_engine_config()
        assert not envs.VLLM_USE_V1
        m.delenv('VLLM_USE_V1')
        m.setenv('VLLM_USE_V1', '1')
        with pytest.raises(NotImplementedError):
            AsyncEngineArgs(model=MODEL).create_engine_config()
        m.delenv('VLLM_USE_V1')
        m.setenv('VLLM_ATTENTION_BACKEND', 'FLASHMLA')
        _ = AsyncEngineArgs(model=MODEL).create_engine_config()
        assert envs.VLLM_USE_V1
        m.delenv('VLLM_USE_V1')

def test_reject_using_constructor_directly(monkeypatch):
    with monkeypatch.context() as m:
        if os.getenv('VLLM_USE_V1', None):
            m.delenv('VLLM_USE_V1')
        vllm_config = AsyncEngineArgs(model=MODEL).create_engine_config()
        with pytest.raises(ValueError):
            AsyncLLMEngine(vllm_config, AsyncLLMEngine._get_executor_cls(vllm_config), log_stats=True)
        m.delenv('VLLM_USE_V1')