from __future__ import annotations
from ..utils import compare_all_settings
from vllm.config import CompilationLevel
from vllm.utils import cuda_device_count_stateless
import dataclasses
import pytest

@dataclasses.dataclass
class TestSetting:
    model: str
    model_args: list[str]
    pp_size: int
    tp_size: int
    attn_backend: str
    method: str
    fullgraph: bool

@pytest.mark.parametrize('test_setting', [TestSetting(model='meta-llama/Llama-3.2-1B-Instruct', model_args=['--max-model-len', '2048'], pp_size=2, tp_size=2, attn_backend='FLASHINFER', method='generate', fullgraph=True), TestSetting(model='TheBloke/TinyLlama-1.1B-Chat-v0.3-GPTQ', model_args=['--quantization', 'gptq', '--max-model-len', '2048'], pp_size=1, tp_size=1, attn_backend='FLASH_ATTN', method='generate', fullgraph=True), TestSetting(model='ibm/PowerMoE-3b', model_args=['--max-model-len', '2048'], pp_size=1, tp_size=2, attn_backend='FLASH_ATTN', method='generate', fullgraph=True), TestSetting(model='BAAI/bge-multilingual-gemma2', model_args=['--task', 'embed', '--dtype', 'bfloat16', '--max-model-len', '2048'], pp_size=1, tp_size=1, attn_backend='FLASH_ATTN', method='encode', fullgraph=True), TestSetting(model='microsoft/Phi-3.5-vision-instruct', model_args=['--trust-remote-code', '--max-model-len', '2048'], pp_size=2, tp_size=1, attn_backend='FLASH_ATTN', method='generate_with_image', fullgraph=False)])
def test_compile_correctness(monkeypatch: pytest.MonkeyPatch, test_setting: TestSetting):
    model = test_setting.model
    model_args = test_setting.model_args
    pp_size = test_setting.pp_size
    tp_size = test_setting.tp_size
    attn_backend = test_setting.attn_backend
    method = test_setting.method
    fullgraph = test_setting.fullgraph
    if cuda_device_count_stateless() != pp_size * tp_size:
        pytest.skip(f'Need exactly {pp_size}*{tp_size} CUDA gpus but got {cuda_device_count_stateless()}')
    with monkeypatch.context() as m:
        m.setenv('VLLM_ATTENTION_BACKEND', attn_backend)
        final_args = ['--enforce-eager', *model_args, '-pp', str(pp_size), '-tp', str(tp_size)]
        all_args: list[list[str]] = []
        all_envs: list[dict[str, str] | None] = []
        for level in [CompilationLevel.NO_COMPILATION, CompilationLevel.PIECEWISE]:
            all_args.append(final_args + [f'-O{level}'])
            all_envs.append({})
        compare_all_settings(model, all_args, all_envs, method=method if method != 'generate' else 'generate_close')
        all_envs.clear()
        all_args.clear()
        for level in [CompilationLevel.NO_COMPILATION, CompilationLevel.DYNAMO_AS_IS, CompilationLevel.DYNAMO_ONCE]:
            all_args.append(final_args + [f'-O{level}'])
            all_envs.append({})
            if level != CompilationLevel.DYNAMO_ONCE and (not fullgraph):
                all_envs[-1]['VLLM_TEST_DYNAMO_FULLGRAPH_CAPTURE'] = '0'
        compare_all_settings(model, all_args * 3, all_envs, method=method)