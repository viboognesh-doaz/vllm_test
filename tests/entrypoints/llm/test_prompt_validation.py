from vllm import LLM
import pytest

@pytest.fixture(autouse=True)
def v1(run_with_both_engines):
    pass

def test_empty_prompt():
    llm = LLM(model='openai-community/gpt2', enforce_eager=True)
    with pytest.raises(ValueError, match='decoder prompt cannot be empty'):
        llm.generate([''])

@pytest.mark.skip_v1
def test_out_of_vocab_token():
    llm = LLM(model='openai-community/gpt2', enforce_eager=True)
    with pytest.raises(ValueError, match='out of vocabulary'):
        llm.generate({'prompt_token_ids': [999999]})