from tests.conftest import VllmRunner
from tests.core.utils import create_dummy_prompt
from vllm.engine.llm_engine import LLMEngine
from vllm.platforms import current_platform
from vllm.sequence import SequenceGroup
import pytest
MODEL = 'JackFram/llama-160m'

def add_seq_group_to_engine(engine: LLMEngine, seq_group: SequenceGroup):
    scheduler = engine.scheduler[0]
    scheduler.add_seq_group(seq_group)

@pytest.mark.parametrize('num_scheduler_steps', [1, 8])
@pytest.mark.parametrize('enable_chunked_prefill', [False, True])
@pytest.mark.parametrize('enforce_eager', [False, True])
def test_num_computed_tokens_update(num_scheduler_steps: int, enable_chunked_prefill: bool, enforce_eager: bool):
    is_multi_step = num_scheduler_steps > 1
    is_multi_step_chunked_prefill = is_multi_step and enable_chunked_prefill
    if is_multi_step_chunked_prefill and current_platform.is_rocm():
        pytest.skip('Multi-step with Chunked-Prefill does not support rocm_flash_attn backend')
    runner = VllmRunner(model_name=MODEL, gpu_memory_utilization=0.7, num_scheduler_steps=num_scheduler_steps, enable_chunked_prefill=enable_chunked_prefill, enforce_eager=enforce_eager)
    engine: LLMEngine = runner.model.llm_engine
    num_prompt_steps = num_scheduler_steps if is_multi_step_chunked_prefill else 1
    num_output_tokens_list = [4, 8, 12, 15, 16, 17]
    prompt_len = 10
    for req_idx, num_output_tokens in enumerate(num_output_tokens_list):
        seq, seq_group = create_dummy_prompt(request_id=str(req_idx), prompt_length=prompt_len, min_tokens=num_output_tokens, max_tokens=num_output_tokens)
        add_seq_group_to_engine(engine, seq_group)
        assert seq.data.get_num_computed_tokens() == 0
        for _ in range(num_prompt_steps):
            engine.step()
        if not seq.is_finished():
            prompt_num_computed_tokens = seq.data.get_num_computed_tokens()
            assert prompt_num_computed_tokens == prompt_len + num_prompt_steps - 1
            decode_step_counter = 0
            while not seq.is_finished():
                assert seq.data.get_num_computed_tokens() == prompt_num_computed_tokens + decode_step_counter
                for _ in range(num_scheduler_steps):
                    engine.step()
                    decode_step_counter += 1
        assert seq.data.get_num_computed_tokens() == prompt_len + num_output_tokens - 1