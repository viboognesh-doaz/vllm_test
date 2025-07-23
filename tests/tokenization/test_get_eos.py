from vllm.transformers_utils.config import try_get_generation_config
from vllm.transformers_utils.tokenizer import get_tokenizer
'\nThis test file includes some cases where it is inappropriate to\nonly get the `eos_token_id` from the tokenizer as defined by\n{meth}`vllm.LLMEngine._get_eos_token_id`.\n'

def test_get_llama3_eos_token():
    model_name = 'meta-llama/Llama-3.2-1B-Instruct'
    tokenizer = get_tokenizer(model_name)
    assert tokenizer.eos_token_id == 128009
    generation_config = try_get_generation_config(model_name, trust_remote_code=False)
    assert generation_config is not None
    assert generation_config.eos_token_id == [128001, 128008, 128009]

def test_get_blip2_eos_token():
    model_name = 'Salesforce/blip2-opt-2.7b'
    tokenizer = get_tokenizer(model_name)
    assert tokenizer.eos_token_id == 2
    generation_config = try_get_generation_config(model_name, trust_remote_code=False)
    assert generation_config is not None
    assert generation_config.eos_token_id == 50118