from transformers import AutoModelForSequenceClassification
from vllm.platforms import current_platform
import pytest
import torch

@pytest.mark.parametrize('model', [pytest.param('jason9693/Qwen2.5-1.5B-apeach', marks=[pytest.mark.core_model, pytest.mark.cpu_model])])
@pytest.mark.parametrize('dtype', ['half'] if current_platform.is_rocm() else ['float'])
def test_models(hf_runner, vllm_runner, example_prompts, model: str, dtype: str, monkeypatch) -> None:
    if current_platform.is_rocm():
        monkeypatch.setenv('VLLM_USE_TRITON_FLASH_ATTN', 'False')
    with vllm_runner(model, max_model_len=512, dtype=dtype) as vllm_model:
        vllm_outputs = vllm_model.classify(example_prompts)
    with hf_runner(model, dtype=dtype, auto_cls=AutoModelForSequenceClassification) as hf_model:
        hf_outputs = hf_model.classify(example_prompts)
    for hf_output, vllm_output in zip(hf_outputs, vllm_outputs):
        hf_output = torch.tensor(hf_output)
        vllm_output = torch.tensor(vllm_output)
        assert torch.allclose(hf_output, vllm_output, 0.001 if dtype == 'float' else 0.01)