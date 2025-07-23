from ..conftest import HfRunner, VllmRunner
from ..models.utils import check_outputs_equal
from ..utils import multi_gpu_test
from unittest.mock import Mock
from vllm import LLM, envs
from vllm.envs import VLLM_USE_V1
from vllm.platforms import current_platform
from vllm.v1.engine.llm_engine import LLMEngine as LLMEngineV1
import os
import pytest
import torch
import weakref
'Compare the short outputs of HF and vLLM when using greedy sampling.\n\nRun `pytest tests/basic_correctness/test_basic_correctness.py`.\n'
MODELS = ['google/gemma-2-2b-it', 'meta-llama/Llama-3.2-1B-Instruct']
TARGET_TEST_SUITE = os.environ.get('TARGET_TEST_SUITE', 'L4')

@pytest.fixture(autouse=True)
def v1(run_with_both_engines):
    pass

def test_vllm_gc_ed():
    """Verify vllm instance is GC'ed when it is deleted"""
    llm = LLM('distilbert/distilgpt2')
    weak_llm = weakref.ref(llm)
    del llm
    assert weak_llm() is None

def _fix_prompt_embed_outputs(vllm_outputs: list[tuple[list[int], str]], hf_model: HfRunner, example_prompts: list[str]) -> list[tuple[list[int], str]]:
    fixed_vllm_outputs = []
    for vllm_output, hf_input, prompt in zip(vllm_outputs, hf_model.get_inputs(example_prompts), example_prompts):
        hf_input_ids = hf_input['input_ids'].tolist()[0]
        fixed_vllm_outputs.append((hf_input_ids + vllm_output[0][len(hf_input_ids):], prompt + vllm_output[1]))
    return fixed_vllm_outputs

@pytest.mark.parametrize('model', MODELS)
@pytest.mark.parametrize('backend', ['FLASH_ATTN'])
@pytest.mark.parametrize('max_tokens', [5])
@pytest.mark.parametrize('enforce_eager', [False])
@pytest.mark.parametrize('enable_prompt_embeds', [True, False])
def test_models(monkeypatch: pytest.MonkeyPatch, hf_runner, model: str, backend: str, max_tokens: int, enforce_eager: bool, enable_prompt_embeds: bool) -> None:
    if enable_prompt_embeds and envs.is_set('VLLM_USE_V1') and envs.VLLM_USE_V1:
        pytest.skip('enable_prompt_embeds is not supported in v1.')
    if backend == 'FLASHINFER' and current_platform.is_rocm():
        pytest.skip('Flashinfer does not support ROCm/HIP.')
    if backend in ('XFORMERS', 'FLASHINFER') and model == 'google/gemma-2-2b-it':
        pytest.skip(f'{backend} does not support gemma2 with full context length.')
    with monkeypatch.context() as m:
        m.setenv('VLLM_ATTENTION_BACKEND', backend)
        prompt = 'The following numbers of the sequence ' + ', '.join((str(i) for i in range(1024))) + ' are:'
        example_prompts = [prompt]
        with hf_runner(model) as hf_model:
            hf_outputs = hf_model.generate_greedy(example_prompts, max_tokens)
            if enable_prompt_embeds:
                with torch.no_grad():
                    prompt_embeds = hf_model.get_prompt_embeddings(example_prompts)
        with VllmRunner(model, max_model_len=8192, enforce_eager=enforce_eager, enable_prompt_embeds=enable_prompt_embeds, gpu_memory_utilization=0.7) as vllm_model:
            if enable_prompt_embeds:
                vllm_outputs = vllm_model.generate_greedy(prompt_embeds, max_tokens)
                vllm_outputs = _fix_prompt_embed_outputs(vllm_outputs, hf_model, example_prompts)
            else:
                vllm_outputs = vllm_model.generate_greedy(example_prompts, max_tokens)
        check_outputs_equal(outputs_0_lst=hf_outputs, outputs_1_lst=vllm_outputs, name_0='hf', name_1='vllm')

@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize('model, distributed_executor_backend, attention_backend, test_suite, extra_env', [('distilbert/distilgpt2', 'ray', '', 'L4', {}), ('distilbert/distilgpt2', 'mp', '', 'L4', {}), ('distilbert/distilgpt2', 'ray', '', 'L4', {'VLLM_SLEEP_WHEN_IDLE': '1'}), ('distilbert/distilgpt2', 'mp', '', 'L4', {'VLLM_SLEEP_WHEN_IDLE': '1'}), ('meta-llama/Llama-3.2-1B-Instruct', 'ray', '', 'L4', {}), ('meta-llama/Llama-3.2-1B-Instruct', 'mp', '', 'L4', {}), ('distilbert/distilgpt2', 'ray', '', 'A100', {}), ('distilbert/distilgpt2', 'mp', '', 'A100', {}), ('distilbert/distilgpt2', 'mp', 'FLASHINFER', 'A100', {}), ('meta-llama/Meta-Llama-3-8B', 'ray', 'FLASHINFER', 'A100', {})])
@pytest.mark.parametrize('enable_prompt_embeds', [True, False])
def test_models_distributed(monkeypatch: pytest.MonkeyPatch, hf_runner, vllm_runner, example_prompts, model: str, distributed_executor_backend: str, attention_backend: str, test_suite: str, extra_env: dict[str, str], enable_prompt_embeds: bool) -> None:
    if enable_prompt_embeds and envs.is_set('VLLM_USE_V1') and envs.VLLM_USE_V1:
        pytest.skip('enable_prompt_embeds is not supported in v1.')
    if test_suite != TARGET_TEST_SUITE:
        pytest.skip(f'Skip test for {test_suite}')
    with monkeypatch.context() as monkeypatch_context:
        if model == 'meta-llama/Llama-3.2-1B-Instruct' and distributed_executor_backend == 'ray' and (attention_backend == '') and (test_suite == 'L4'):
            if enable_prompt_embeds:
                pytest.skip('enable_prompt_embeds does not work with ray compiled dag.')
            monkeypatch_context.setenv('VLLM_USE_RAY_SPMD_WORKER', '1')
            monkeypatch_context.setenv('VLLM_USE_RAY_COMPILED_DAG', '1')
        if attention_backend:
            monkeypatch_context.setenv('VLLM_ATTENTION_BACKEND', attention_backend)
        for k, v in extra_env.items():
            monkeypatch_context.setenv(k, v)
        dtype = 'half'
        max_tokens = 5
        with vllm_runner(model, dtype=dtype, tensor_parallel_size=2, distributed_executor_backend=distributed_executor_backend, enable_prompt_embeds=enable_prompt_embeds, gpu_memory_utilization=0.7) as vllm_model:
            if enable_prompt_embeds:
                with hf_runner(model, dtype=dtype) as hf_model:
                    with torch.no_grad():
                        prompt_embeds = hf_model.get_prompt_embeddings(example_prompts)
                    vllm_outputs = vllm_model.generate_greedy(prompt_embeds, max_tokens)
                    vllm_outputs = _fix_prompt_embed_outputs(vllm_outputs, hf_model, example_prompts)
                    hf_outputs = hf_model.generate_greedy(example_prompts, max_tokens)
            else:
                vllm_outputs = vllm_model.generate_greedy(example_prompts, max_tokens)
                with hf_runner(model, dtype=dtype) as hf_model:
                    hf_outputs = hf_model.generate_greedy(example_prompts, max_tokens)
    check_outputs_equal(outputs_0_lst=hf_outputs, outputs_1_lst=vllm_outputs, name_0='hf', name_1='vllm')

def test_failed_model_execution(vllm_runner, monkeypatch) -> None:
    if not VLLM_USE_V1:
        pytest.skip('Skipping V0 test, dump input not supported')
    monkeypatch.setenv('VLLM_ENABLE_V1_MULTIPROCESSING', '0')
    with vllm_runner('facebook/opt-125m', enforce_eager=True) as vllm_model:
        if isinstance(vllm_model.model.llm_engine, LLMEngineV1):
            v1_test_failed_model_execution(vllm_model)

def v1_test_failed_model_execution(vllm_model):
    engine = vllm_model.model.llm_engine
    mocked_execute_model = Mock(side_effect=RuntimeError('Mocked Critical Error'))
    engine.engine_core.engine_core.model_executor.execute_model = mocked_execute_model
    with pytest.raises(RuntimeError) as exc_info:
        prompts = ['Hello, my name is', 'The president of the United States is', 'The capital of France is', 'The future of AI is']
        vllm_model.generate_greedy(prompts, 200, use_tqdm=False)
    assert isinstance(exc_info.value, RuntimeError)
    assert 'Mocked Critical Error' in str(exc_info.value)