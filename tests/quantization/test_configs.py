from dataclasses import dataclass
from vllm.config import ModelConfig
import pytest
'Tests whether Marlin models can be loaded from the autogptq config.\n\nRun `pytest tests/quantization/test_configs.py --forked`.\n'

@dataclass
class ModelPair:
    model_marlin: str
    model_gptq: str
MODEL_ARG_EXPTYPES = [('neuralmagic/TinyLlama-1.1B-Chat-v1.0-marlin', None, 'marlin'), ('neuralmagic/TinyLlama-1.1B-Chat-v1.0-marlin', 'marlin', 'marlin'), ('neuralmagic/TinyLlama-1.1B-Chat-v1.0-marlin', 'gptq', 'marlin'), ('neuralmagic/TinyLlama-1.1B-Chat-v1.0-marlin', 'awq', 'ERROR'), ('TheBloke/Llama-2-7B-Chat-GPTQ', None, 'gptq_marlin'), ('TheBloke/Llama-2-7B-Chat-GPTQ', 'marlin', 'gptq_marlin'), ('TheBloke/Llama-2-7B-Chat-GPTQ', 'gptq', 'gptq'), ('TheBloke/Llama-2-7B-Chat-GPTQ', 'awq', 'ERROR'), ('LnL-AI/TinyLlama-1.1B-Chat-v1.0-GPTQ-Marlin-4bit', None, 'marlin'), ('LnL-AI/TinyLlama-1.1B-Chat-v1.0-GPTQ-Marlin-4bit', 'marlin', 'marlin'), ('LnL-AI/TinyLlama-1.1B-Chat-v1.0-GPTQ-Marlin-4bit', 'gptq', 'marlin'), ('LnL-AI/TinyLlama-1.1B-Chat-v1.0-GPTQ-Marlin-4bit', 'awq', 'ERROR'), ('LnL-AI/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit', None, 'gptq_marlin'), ('LnL-AI/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit', 'marlin', 'gptq_marlin'), ('LnL-AI/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit', 'gptq', 'gptq'), ('LnL-AI/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit', 'awq', 'ERROR'), ('TheBloke/OpenHermes-2.5-Mistral-7B-AWQ', None, 'awq_marlin'), ('TheBloke/OpenHermes-2.5-Mistral-7B-AWQ', 'awq', 'awq'), ('TheBloke/OpenHermes-2.5-Mistral-7B-AWQ', 'marlin', 'awq_marlin'), ('TheBloke/OpenHermes-2.5-Mistral-7B-AWQ', 'gptq', 'ERROR')]

@pytest.mark.parametrize('model_arg_exptype', MODEL_ARG_EXPTYPES)
def test_auto_gptq(model_arg_exptype: tuple[str, None, str]) -> None:
    model_path, quantization_arg, expected_type = model_arg_exptype
    try:
        model_config = ModelConfig(model_path, task='auto', tokenizer=model_path, tokenizer_mode='auto', trust_remote_code=False, seed=0, dtype='float16', revision=None, quantization=quantization_arg)
        found_quantization_type = model_config.quantization
    except ValueError:
        found_quantization_type = 'ERROR'
    assert found_quantization_type == expected_type, f'Expected quant_type == {expected_type} for {model_path}, but found {found_quantization_type} for no --quantization {quantization_arg} case'