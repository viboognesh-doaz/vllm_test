from vllm import SamplingParams
import pytest
import transformers
'Test the different finish_reason="stop" situations during generation:\n    1. One of the provided stop strings\n    2. One of the provided stop tokens\n    3. The EOS token\n\nRun `pytest tests/engine/test_stop_reason.py`.\n'
MODEL = 'distilbert/distilgpt2'
STOP_STR = '.'
SEED = 42
MAX_TOKENS = 1024

@pytest.fixture
def vllm_model(vllm_runner):
    with vllm_runner(MODEL) as vllm_model:
        yield vllm_model

def test_stop_reason(vllm_model, example_prompts):
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL)
    stop_token_id = tokenizer.convert_tokens_to_ids(STOP_STR)
    llm = vllm_model.model
    outputs = llm.generate(example_prompts, sampling_params=SamplingParams(ignore_eos=True, seed=SEED, max_tokens=MAX_TOKENS, stop_token_ids=[stop_token_id]))
    for output in outputs:
        output = output.outputs[0]
        assert output.finish_reason == 'stop'
        assert output.stop_reason == stop_token_id
    outputs = llm.generate(example_prompts, sampling_params=SamplingParams(ignore_eos=True, seed=SEED, max_tokens=MAX_TOKENS, stop='.'))
    for output in outputs:
        output = output.outputs[0]
        assert output.finish_reason == 'stop'
        assert output.stop_reason == STOP_STR
    outputs = llm.generate(example_prompts, sampling_params=SamplingParams(seed=SEED, max_tokens=MAX_TOKENS))
    for output in outputs:
        output = output.outputs[0]
        assert output.finish_reason == 'length' or (output.finish_reason == 'stop' and output.stop_reason is None)