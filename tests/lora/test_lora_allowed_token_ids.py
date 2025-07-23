from vllm.config import CacheConfig, DeviceConfig, LoRAConfig, ModelConfig, VllmConfig
from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.tokenizer_group import init_tokenizer_from_configs
from vllm.v1.engine.processor import Processor
import pytest

def test_allowed_token_ids_with_lora_vocab(llama_2_7b_base_huggingface_id, sql_lora_files):
    """
    Test that we properly resolve the range of allowed token ids for lora
    adapters that define additional tokens.
    """
    model_config = ModelConfig(model=llama_2_7b_base_huggingface_id, tokenizer=llama_2_7b_base_huggingface_id, tokenizer_mode='auto')
    vllm_config = VllmConfig(model_config=model_config, cache_config=CacheConfig(), device_config=DeviceConfig(), lora_config=LoRAConfig())
    tokenizer = init_tokenizer_from_configs(model_config=vllm_config.model_config, scheduler_config=vllm_config.scheduler_config, lora_config=vllm_config.lora_config)
    processor = Processor(vllm_config, tokenizer)
    lora_request = LoRARequest('1', 1, str(sql_lora_files))
    request_id = '1'
    prompt = 'a prompt'
    lora_token_ids = [32000, 32001, 32002, 32003]
    processor.process_inputs(request_id, prompt, params=SamplingParams(allowed_token_ids=lora_token_ids), lora_request=lora_request)
    base_token_ids = [1000, 1001, 1002, 1003]
    processor.process_inputs(request_id, prompt, params=SamplingParams(allowed_token_ids=base_token_ids), lora_request=lora_request)
    invalid_token_ids = [35000, 35001, 35002, 35003]
    with pytest.raises(ValueError):
        processor.process_inputs(request_id, prompt, params=SamplingParams(allowed_token_ids=invalid_token_ids), lora_request=lora_request)
    with pytest.raises(ValueError):
        processor.process_inputs(request_id, prompt, params=SamplingParams(allowed_token_ids=lora_token_ids))

def test_allowed_token_ids_with_lora_adapter_no_vocab(qwen25vl_base_huggingface_id, qwen25vl_lora_files):
    """
    Test that we properly resolve the range of allowed token ids for lora
    adapters that do not define additional tokens.
    """
    model_config = ModelConfig(model=qwen25vl_base_huggingface_id, tokenizer=qwen25vl_base_huggingface_id, tokenizer_mode='auto')
    vllm_config = VllmConfig(model_config=model_config, cache_config=CacheConfig(), device_config=DeviceConfig(), lora_config=LoRAConfig())
    tokenizer = init_tokenizer_from_configs(model_config=vllm_config.model_config, scheduler_config=vllm_config.scheduler_config, lora_config=vllm_config.lora_config)
    processor = Processor(vllm_config, tokenizer)
    lora_request = LoRARequest('1', 1, str(qwen25vl_lora_files))
    request_id = '1'
    prompt = 'a prompt'
    base_token_ids = [1000, 1001, 1002, 1003]
    processor.process_inputs(request_id, prompt, params=SamplingParams(allowed_token_ids=base_token_ids), lora_request=lora_request)
    base_token_ids = [1000, 1001, 1002, 1003]
    processor.process_inputs(request_id, prompt, params=SamplingParams(allowed_token_ids=base_token_ids))
    invalid_token_ids = [200000, 200001, 200002, 200003]
    with pytest.raises(ValueError):
        processor.process_inputs(request_id, prompt, params=SamplingParams(allowed_token_ids=invalid_token_ids), lora_request=lora_request)