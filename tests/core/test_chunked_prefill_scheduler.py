from .utils import create_dummy_prompt
from unittest.mock import MagicMock
from vllm.config import CacheConfig, SchedulerConfig
from vllm.core.scheduler import Scheduler
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.sampling_params import SamplingParams
from vllm.sequence import Logprob, SequenceGroup
import pytest

def get_sequence_groups(scheduler_output):
    return [s.seq_group for s in scheduler_output.scheduled_seq_groups]

def append_new_token(seq_group: SequenceGroup, token_id: int):
    for seq in seq_group.get_seqs():
        seq.append_token_id(token_id, {token_id: Logprob(token_id)})

def schedule_and_update_computed_tokens(scheduler):
    metas, out, _ = scheduler.schedule()
    for s, meta in zip(out.scheduled_seq_groups, metas):
        s.seq_group.update_num_computed_tokens(meta.token_chunk_size)
    return (metas, out)

def test_simple():
    """Verify basic scheduling works."""
    block_size = 4
    num_seq_group = 4
    max_model_len = 16
    max_num_batched_tokens = 64
    scheduler_config = SchedulerConfig('generate', max_num_batched_tokens, num_seq_group, max_model_len, enable_chunked_prefill=True)
    cache_config = CacheConfig(block_size, 1.0, 1, 'auto')
    cache_config.num_cpu_blocks = 8
    cache_config.num_gpu_blocks = 8
    scheduler = Scheduler(scheduler_config, cache_config, None)
    running: list[SequenceGroup] = []
    for i in range(num_seq_group):
        _, seq_group = create_dummy_prompt(str(i), prompt_length=block_size, block_size=block_size)
        scheduler.add_seq_group(seq_group)
        running.append(seq_group)
    num_tokens = block_size * num_seq_group
    seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
    assert set(get_sequence_groups(out)) == set(running)
    assert out.num_batched_tokens == num_tokens
    assert not out.blocks_to_copy and (not out.blocks_to_swap_in) and (not out.blocks_to_swap_out)
    assert len(seq_group_meta) == num_seq_group
    for s in running:
        append_new_token(s, 1)
    seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
    assert set(get_sequence_groups(out)) == set(running)
    assert out.num_batched_tokens == num_seq_group
    assert not out.blocks_to_copy and (not out.blocks_to_swap_in) and (not out.blocks_to_swap_out)
    assert len(seq_group_meta) == num_seq_group

def test_chunk():
    """Verify prefills are chunked properly."""
    block_size = 4
    max_seqs = 60
    max_model_len = 80
    max_num_batched_tokens = 64
    scheduler_config = SchedulerConfig('generate', max_num_batched_tokens, max_seqs, max_model_len, enable_chunked_prefill=True)
    cache_config = CacheConfig(block_size, 1.0, 1, 'auto')
    cache_config.num_cpu_blocks = 32
    cache_config.num_gpu_blocks = 32
    scheduler = Scheduler(scheduler_config, cache_config, None)
    running: list[SequenceGroup] = []
    for i in range(2):
        _, seq_group = create_dummy_prompt(str(i), prompt_length=60, block_size=block_size)
        scheduler.add_seq_group(seq_group)
        running.append(seq_group)
    seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
    print()
    assert set(get_sequence_groups(out)) == set(running)
    assert seq_group_meta[0].token_chunk_size == 60
    assert seq_group_meta[1].token_chunk_size == 4
    assert out.num_prefill_groups == 2
    assert out.num_batched_tokens == 64
    append_new_token(running[0], 1)
    seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
    assert set(get_sequence_groups(out)) == set(running)
    assert seq_group_meta[0].token_chunk_size == 56
    assert seq_group_meta[1].token_chunk_size == 1
    assert out.num_prefill_groups == 1
    assert out.num_batched_tokens == 57

def test_concurrent_chunking():
    """Verify prefills are chunked properly when 
    --max-num-partial-prefills is > 1"""
    block_size = 4
    max_seqs = 60
    max_model_len = 2000
    max_num_batched_tokens = 64
    scheduler_config = SchedulerConfig('generate', max_num_batched_tokens, max_seqs, max_model_len, enable_chunked_prefill=True, max_num_partial_prefills=2)
    cache_config = CacheConfig(block_size, 1.0, 1, 'auto')
    cache_config.num_cpu_blocks = 32
    cache_config.num_gpu_blocks = 32
    scheduler = Scheduler(scheduler_config, cache_config, None)
    running: list[SequenceGroup] = []
    for i in range(2):
        _, seq_group = create_dummy_prompt(str(i), prompt_length=60, block_size=block_size)
        scheduler.add_seq_group(seq_group)
        running.append(seq_group)
    seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
    assert set(get_sequence_groups(out)) == set(running)
    assert seq_group_meta[0].token_chunk_size == 32
    assert seq_group_meta[1].token_chunk_size == 32
    assert out.num_prefill_groups == 2
    assert out.num_batched_tokens == 64
    seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
    assert set(get_sequence_groups(out)) == set(running)
    assert seq_group_meta[0].token_chunk_size == 28
    assert seq_group_meta[1].token_chunk_size == 28
    assert out.num_prefill_groups == 2
    assert out.num_batched_tokens == 56

def test_concurrent_chunking_large_requests():
    """Verify large prefill requests are run one at a time"""
    block_size = 4
    max_seqs = 60
    max_model_len = 2000
    max_num_batched_tokens = 64
    scheduler_config = SchedulerConfig('generate', max_num_batched_tokens, max_seqs, max_model_len, enable_chunked_prefill=True, max_num_partial_prefills=2)
    cache_config = CacheConfig(block_size, 1.0, 1, 'auto')
    cache_config.num_cpu_blocks = 3200
    cache_config.num_gpu_blocks = 3200
    scheduler = Scheduler(scheduler_config, cache_config, None)
    for i in range(2):
        _, seq_group = create_dummy_prompt(str(i), prompt_length=1200, block_size=block_size)
        scheduler.add_seq_group(seq_group)
    seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
    assert len(get_sequence_groups(out)) == 1
    assert seq_group_meta[0].token_chunk_size == 64
    assert out.num_prefill_groups == 1
    assert out.num_batched_tokens == 64

def test_short_prompts_jump_long_prompts_in_queue():
    """Verify large prefill requests are punted behind smaller ones if 
    another large prefill request is already running"""
    block_size = 4
    max_seqs = 60
    max_model_len = 2000
    max_num_batched_tokens = 64
    scheduler_config = SchedulerConfig('generate', max_num_batched_tokens, max_seqs, max_model_len, enable_chunked_prefill=True, max_num_partial_prefills=2)
    cache_config = CacheConfig(block_size, 1.0, 1, 'auto')
    cache_config.num_cpu_blocks = 3200
    cache_config.num_gpu_blocks = 3200
    scheduler = Scheduler(scheduler_config, cache_config, None)
    long_seqs: list[SequenceGroup] = []
    short_seqs: list[SequenceGroup] = []
    for i in range(2):
        _, seq_group = create_dummy_prompt(str(i), prompt_length=1200, block_size=block_size)
        scheduler.add_seq_group(seq_group)
        long_seqs.append(seq_group)
        assert seq_group.is_prefill()
    for i in range(2):
        _, seq_group = create_dummy_prompt(str(i + 2), prompt_length=40, block_size=block_size)
        scheduler.add_seq_group(seq_group)
        short_seqs.append(seq_group)
        assert seq_group.is_prefill()
    seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
    assert seq_group_meta[0].token_chunk_size == 32
    assert seq_group_meta[1].token_chunk_size == 32
    assert long_seqs[0].is_prefill()
    assert long_seqs[1].is_prefill()
    assert short_seqs[0].is_prefill()
    assert short_seqs[1].is_prefill()
    assert long_seqs[0].first_seq.get_num_computed_tokens() == 32
    assert long_seqs[1].first_seq.get_num_computed_tokens() == 0
    assert short_seqs[0].first_seq.get_num_computed_tokens() == 32
    assert short_seqs[1].first_seq.get_num_computed_tokens() == 0
    assert out.num_prefill_groups == 2
    assert out.num_batched_tokens == 64
    seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
    assert seq_group_meta[0].token_chunk_size == 24
    assert seq_group_meta[1].token_chunk_size == 32
    assert seq_group_meta[2].token_chunk_size == 8
    assert long_seqs[0].is_prefill()
    assert long_seqs[1].is_prefill()
    assert not short_seqs[0].is_prefill()
    assert short_seqs[1].is_prefill()
    assert long_seqs[0].first_seq.get_num_computed_tokens() == 64
    assert long_seqs[1].first_seq.get_num_computed_tokens() == 0
    assert short_seqs[0].first_seq.get_num_computed_tokens() == 40
    assert short_seqs[1].first_seq.get_num_computed_tokens() == 24
    assert out.num_prefill_groups == 3
    assert out.num_batched_tokens == 64
    append_new_token(short_seqs[0], 1)
    seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
    assert seq_group_meta[0].token_chunk_size == 32
    assert seq_group_meta[1].token_chunk_size == 16
    assert seq_group_meta[2].token_chunk_size == 1
    assert out.num_prefill_groups == 2
    assert out.num_batched_tokens == 49
    assert long_seqs[0].is_prefill()
    assert long_seqs[1].is_prefill()
    assert not short_seqs[0].is_prefill()
    assert not short_seqs[1].is_prefill()
    assert long_seqs[0].first_seq.get_num_computed_tokens() == 96
    assert long_seqs[1].first_seq.get_num_computed_tokens() == 0
    assert short_seqs[0].first_seq.get_num_computed_tokens() == 41
    assert short_seqs[1].first_seq.get_num_computed_tokens() == 40
    append_new_token(short_seqs[0], 1)
    append_new_token(short_seqs[1], 1)
    seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
    assert seq_group_meta[0].token_chunk_size == 62
    assert seq_group_meta[1].token_chunk_size == 1
    assert seq_group_meta[2].token_chunk_size == 1
    assert out.num_prefill_groups == 1
    assert out.num_batched_tokens == 64
    assert long_seqs[0].first_seq.get_num_computed_tokens() == 158

def test_complex():
    block_size = 4
    max_seqs = 60
    max_model_len = 80
    max_num_batched_tokens = 64
    scheduler_config = SchedulerConfig('generate', max_num_batched_tokens, max_seqs, max_model_len, enable_chunked_prefill=True)
    cache_config = CacheConfig(block_size, 1.0, 1, 'auto')
    cache_config.num_cpu_blocks = 64
    cache_config.num_gpu_blocks = 64
    scheduler = Scheduler(scheduler_config, cache_config, None)
    running: list[SequenceGroup] = []
    for i in range(2):
        _, seq_group = create_dummy_prompt(str(i), prompt_length=60, block_size=block_size)
        scheduler.add_seq_group(seq_group)
        running.append(seq_group)
        assert seq_group.is_prefill()
    seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
    assert set(get_sequence_groups(out)) == set(running)
    assert seq_group_meta[0].token_chunk_size == 60
    assert seq_group_meta[1].token_chunk_size == 4
    assert not running[0].is_prefill()
    assert running[1].is_prefill()
    assert out.num_prefill_groups == 2
    assert out.num_batched_tokens == 64
    append_new_token(running[0], 1)
    for i in range(2, 4):
        _, seq_group = create_dummy_prompt(str(i), prompt_length=60, block_size=block_size)
        scheduler.add_seq_group(seq_group)
        running.append(seq_group)
    seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
    assert len(get_sequence_groups(out)) == 3
    assert seq_group_meta[0].token_chunk_size == 7
    assert seq_group_meta[1].token_chunk_size == 56
    assert seq_group_meta[2].token_chunk_size == 1
    assert out.num_prefill_groups == 2
    assert out.num_batched_tokens == 64
    append_new_token(running[0], 1)
    assert not running[0].is_prefill()
    append_new_token(running[1], 1)
    assert not running[1].is_prefill()
    assert running[2].is_prefill()

def test_maximal_decoding():
    """Verify decoding requests are prioritized."""
    block_size = 4
    max_seqs = 2
    max_model_len = 8
    max_num_batched_tokens = 2
    scheduler_config = SchedulerConfig('generate', max_num_batched_tokens, max_seqs, max_model_len, enable_chunked_prefill=True)
    cache_config = CacheConfig(block_size, 1.0, 1, 'auto')
    cache_config.num_cpu_blocks = 8
    cache_config.num_gpu_blocks = 8
    scheduler = Scheduler(scheduler_config, cache_config, None)
    running: list[SequenceGroup] = []
    for i in range(2):
        _, seq_group = create_dummy_prompt(str(i), prompt_length=2, block_size=block_size)
        scheduler.add_seq_group(seq_group)
        running.append(seq_group)
        assert seq_group.is_prefill()
    seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
    assert len(get_sequence_groups(out)) == 1
    assert seq_group_meta[0].token_chunk_size == 2
    assert not running[0].is_prefill()
    assert running[1].is_prefill()
    assert out.num_prefill_groups == 1
    assert out.num_batched_tokens == 2
    append_new_token(running[0], 1)
    _, seq_group = create_dummy_prompt('3', prompt_length=2, block_size=block_size)
    scheduler.add_seq_group(seq_group)
    running.append(seq_group)
    assert seq_group.is_prefill()
    seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
    assert len(get_sequence_groups(out)) == 2
    assert seq_group_meta[0].token_chunk_size == 1
    assert seq_group_meta[1].token_chunk_size == 1
    assert not running[0].is_prefill()
    assert running[1].is_prefill()
    assert running[2].is_prefill()
    assert out.num_prefill_groups == 1
    assert out.num_batched_tokens == 2
    append_new_token(running[0], 1)
    seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
    assert len(get_sequence_groups(out)) == 2
    assert seq_group_meta[0].token_chunk_size == 1
    assert seq_group_meta[1].token_chunk_size == 1
    assert not running[0].is_prefill()
    assert not running[1].is_prefill()
    assert out.num_prefill_groups == 1
    assert out.num_batched_tokens == 2
    append_new_token(running[0], 1)
    append_new_token(running[1], 1)
    seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
    assert len(get_sequence_groups(out)) == 2
    assert seq_group_meta[0].token_chunk_size == 1
    assert seq_group_meta[1].token_chunk_size == 1
    assert not running[0].is_prefill()
    assert not running[1].is_prefill()
    assert out.num_prefill_groups == 0
    assert out.num_batched_tokens == 2
    append_new_token(running[0], 1)
    append_new_token(running[1], 1)
    scheduler.abort_seq_group(running[0].request_id)
    seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
    assert len(get_sequence_groups(out)) == 2
    assert seq_group_meta[0].token_chunk_size == 1
    assert seq_group_meta[1].token_chunk_size == 1
    assert not running[1].is_prefill()
    assert running[2].is_prefill()
    assert out.num_prefill_groups == 1
    assert out.num_batched_tokens == 2

def test_prompt_limit():
    """Verify max_num_batched_tokens < max_model_len is possible."""
    block_size = 4
    max_seqs = 32
    max_model_len = 64
    max_num_batched_tokens = 32
    scheduler_config = SchedulerConfig('generate', max_num_batched_tokens, max_seqs, max_model_len, enable_chunked_prefill=True)
    cache_config = CacheConfig(block_size, 1.0, 1, 'auto')
    cache_config.num_cpu_blocks = 16
    cache_config.num_gpu_blocks = 16
    scheduler = Scheduler(scheduler_config, cache_config, None)
    running: list[SequenceGroup] = []
    _, seq_group = create_dummy_prompt('1', prompt_length=48, block_size=block_size)
    scheduler.add_seq_group(seq_group)
    running.append(seq_group)
    assert seq_group.is_prefill()
    seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
    assert len(get_sequence_groups(out)) == 1
    assert seq_group_meta[0].token_chunk_size == 32
    assert running[0].is_prefill()
    assert out.num_prefill_groups == 1
    assert out.num_batched_tokens == 32

def test_prompt_limit_exceed():
    block_size = 4
    max_seqs = 64
    max_model_len = 32
    max_num_batched_tokens = 64
    scheduler_config = SchedulerConfig('generate', max_num_batched_tokens, max_seqs, max_model_len, enable_chunked_prefill=True)
    cache_config = CacheConfig(block_size, 1.0, 1, 'auto')
    cache_config.num_cpu_blocks = 16
    cache_config.num_gpu_blocks = 16
    scheduler = Scheduler(scheduler_config, cache_config, None)
    running: list[SequenceGroup] = []
    _, seq_group = create_dummy_prompt('2', prompt_length=48, block_size=block_size)
    scheduler.add_seq_group(seq_group)
    running.append(seq_group)
    assert seq_group.is_prefill()
    seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
    assert len(out.ignored_seq_groups) == 1
    assert out.ignored_seq_groups[0] == seq_group

def test_chunked_prefill_preempt():
    """Verify preempt works with chunked prefill requests"""
    block_size = 4
    max_seqs = 30
    max_model_len = 200
    max_num_batched_tokens = 30
    scheduler_config = SchedulerConfig('generate', max_num_batched_tokens, max_seqs, max_model_len, enable_chunked_prefill=True)
    cache_config = CacheConfig(block_size, 1.0, 1, 'auto')
    cache_config.num_cpu_blocks = 16
    cache_config.num_gpu_blocks = 16
    scheduler = Scheduler(scheduler_config, cache_config, None)
    _, seq_group = create_dummy_prompt('1', prompt_length=60, block_size=block_size)
    scheduler.add_seq_group(seq_group)
    _, out = schedule_and_update_computed_tokens(scheduler)
    assert len(out.scheduled_seq_groups) == 1
    assert out.num_prefill_groups == 1
    assert seq_group.is_prefill()
    assert out.num_batched_tokens == max_num_batched_tokens
    scheduler.block_manager.can_append_slots = MagicMock()

    def cannot_append_second_group1(seq_group, num_lookahead_slots):
        return seq_group.request_id != '1'
    scheduler.block_manager.can_append_slots.side_effect = cannot_append_second_group1
    _, out = schedule_and_update_computed_tokens(scheduler)
    assert len(out.scheduled_seq_groups) == 0
    assert out.num_batched_tokens == 0
    assert out.blocks_to_swap_out == []
    assert out.blocks_to_swap_in == []
    _, out = schedule_and_update_computed_tokens(scheduler)
    assert len(out.scheduled_seq_groups) == 1
    assert out.num_prefill_groups == 1
    assert seq_group.is_prefill()
    assert out.num_batched_tokens == max_num_batched_tokens
    assert seq_group.get_num_uncomputed_tokens() == 30

    def cannot_append_second_group2(seq_group, num_lookahead_slots):
        return True
    scheduler.block_manager.can_append_slots.side_effect = cannot_append_second_group2
    _, out = schedule_and_update_computed_tokens(scheduler)
    assert len(out.scheduled_seq_groups) == 1
    assert out.num_prefill_groups == 1
    assert not seq_group.is_prefill()
    assert out.num_batched_tokens == max_num_batched_tokens

@pytest.mark.parametrize('num_scheduler_steps', [1, 5])
def test_chunked_prefill_spec_prefill(num_scheduler_steps):
    """Verify that the num_lookahead_slots is set appropriately for an all"""
    'prefill batch depending on whether multi-step scheduling is enabled'
    'or not'
    block_size = 4
    max_seqs = 30
    max_model_len = 200
    max_num_batched_tokens = 30
    num_lookahead_slots = 4
    scheduler_config = SchedulerConfig('generate', max_num_batched_tokens, max_seqs, max_model_len, enable_chunked_prefill=True, num_lookahead_slots=num_lookahead_slots, num_scheduler_steps=num_scheduler_steps)
    cache_config = CacheConfig(block_size, 1.0, 1, 'auto')
    cache_config.num_cpu_blocks = 16
    cache_config.num_gpu_blocks = 16
    scheduler = Scheduler(scheduler_config, cache_config, None)
    _, seq_group = create_dummy_prompt('1', prompt_length=30, block_size=block_size)
    scheduler.add_seq_group(seq_group)
    _, out = schedule_and_update_computed_tokens(scheduler)
    assert len(out.scheduled_seq_groups) == 1
    assert out.num_prefill_groups == 1
    assert out.num_batched_tokens == max_num_batched_tokens
    print(out.num_lookahead_slots)
    assert out.num_lookahead_slots == (0 if num_scheduler_steps == 1 else num_lookahead_slots)

def test_chunked_prefill_max_seqs():
    block_size = 4
    max_seqs = 2
    max_model_len = 80
    max_num_batched_tokens = 64
    scheduler_config = SchedulerConfig('generate', max_num_batched_tokens, max_seqs, max_model_len, enable_chunked_prefill=True)
    cache_config = CacheConfig(block_size, 1.0, 1, 'auto')
    cache_config.num_cpu_blocks = 128
    cache_config.num_gpu_blocks = 128
    scheduler = Scheduler(scheduler_config, cache_config, None)
    running: list[SequenceGroup] = []
    _, seq_group = create_dummy_prompt('1', prompt_length=65, block_size=block_size)
    scheduler.add_seq_group(seq_group)
    running.append(seq_group)
    seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
    assert seq_group_meta[0].token_chunk_size == max_num_batched_tokens
    assert len(get_sequence_groups(out)) == 1
    for i in range(4):
        _, seq_group = create_dummy_prompt(str(i), prompt_length=65, block_size=block_size)
        scheduler.add_seq_group(seq_group)
        running.append(seq_group)
    seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
    assert out.num_batched_tokens == max_num_batched_tokens
    assert len(get_sequence_groups(out)) == 2
    assert not running[0].is_prefill()
    assert running[1].is_prefill()
    append_new_token(running[0], 1)
    seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
    assert seq_group_meta[0].token_chunk_size == 2
    assert seq_group_meta[1].token_chunk_size == 1
    assert out.num_batched_tokens == 3
    assert len(get_sequence_groups(out)) == max_seqs
    assert not running[0].is_prefill()
    assert not running[1].is_prefill()

def test_prefix_caching():
    """Verify allocating full blocks when prefix caching is enabled."""
    block_size = 4
    max_seqs = 10
    max_model_len = 80
    max_num_batched_tokens = 64
    scheduler_config = SchedulerConfig('generate', max_num_batched_tokens, max_seqs, max_model_len, enable_chunked_prefill=True)
    cache_config = CacheConfig(block_size, 1.0, 1, 'auto', enable_prefix_caching=True)
    cache_config.num_cpu_blocks = 0
    cache_config.num_gpu_blocks = 32
    scheduler = Scheduler(scheduler_config, cache_config, None)
    running: list[SequenceGroup] = []
    for i in range(2):
        _, seq_group = create_dummy_prompt(str(i), block_size=block_size, prompt_length=50)
        scheduler.add_seq_group(seq_group)
        running.append(seq_group)
    seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
    assert set(get_sequence_groups(out)) == set(running)
    assert seq_group_meta[0].token_chunk_size == 50
    assert seq_group_meta[1].token_chunk_size == 12
    assert out.num_prefill_groups == 2
    assert out.num_batched_tokens == 62

def test_prefix_caching_with_concurrent_partial_prefills():
    """Verify allocating full blocks when prefix caching is enabled with 
    --max-num-partial-prefills > 1."""
    block_size = 4
    max_seqs = 10
    max_model_len = 8000
    max_num_batched_tokens = 60
    scheduler_config = SchedulerConfig('generate', max_num_batched_tokens, max_seqs, max_model_len, enable_chunked_prefill=True, max_num_partial_prefills=2)
    cache_config = CacheConfig(block_size, 1.0, 1, 'auto', enable_prefix_caching=True)
    cache_config.num_cpu_blocks = 0
    cache_config.num_gpu_blocks = 32
    scheduler = Scheduler(scheduler_config, cache_config, None)
    running: list[SequenceGroup] = []
    for i in range(2):
        _, seq_group = create_dummy_prompt(str(i), block_size=block_size, prompt_length=50)
        scheduler.add_seq_group(seq_group)
        running.append(seq_group)
    seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
    assert set(get_sequence_groups(out)) == set(running)
    assert seq_group_meta[0].token_chunk_size == 28
    assert seq_group_meta[1].token_chunk_size == 28
    assert out.num_prefill_groups == 2
    assert out.num_batched_tokens == 56
    seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
    assert set(get_sequence_groups(out)) == set(running)
    assert seq_group_meta[0].token_chunk_size == 22
    assert seq_group_meta[1].token_chunk_size == 22
    assert out.num_prefill_groups == 2
    assert out.num_batched_tokens == 44

@pytest.mark.parametrize('model', ['facebook/opt-125m'])
@pytest.mark.parametrize('max_num_partial_prefills', [2, 4, 8])
def test_chunked_prefill_with_actual_engine(model: str, max_num_partial_prefills: int):
    """Make sure the model can actually sample with concurrent 
    partial prefills
    """
    prompt = 'hello' * 40
    engine_args = EngineArgs(model=model, max_num_partial_prefills=max_num_partial_prefills, max_num_batched_tokens=40, max_num_seqs=8, enable_chunked_prefill=True, gpu_memory_utilization=0.8)
    engine = LLMEngine.from_engine_args(engine_args)
    sampling_params = SamplingParams(temperature=0)
    for req_num in range(max_num_partial_prefills):
        engine.add_request(f'{req_num}', prompt, sampling_params)
    request_outputs = engine.step()
    assert len(request_outputs) == 0
    assert len(engine.scheduler[0].running) == max_num_partial_prefills