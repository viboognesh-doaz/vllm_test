from .utils import assert_scheduler_empty, create_model_runner_output, create_request, create_scheduler, create_vllm_config
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT
from vllm.v1.request import FinishReason, RequestStatus
import copy

def test_basic_lifecycle():
    """Test lifecycle of a remote prefill."""
    vllm_config = create_vllm_config()
    scheduler = create_scheduler(vllm_config)
    BLOCK_SIZE = vllm_config.cache_config.block_size
    NUM_EXTERNAL_FULL_BLOCKS = 2
    NUM_TOKENS = int(BLOCK_SIZE * (NUM_EXTERNAL_FULL_BLOCKS + 0.5))
    START_FREE_BLOCK_QUEUE_SIZE = scheduler.kv_cache_manager.block_pool.free_block_queue.num_free_blocks
    request = create_request(request_id=1, num_tokens=NUM_TOKENS, do_remote_prefill=True)
    scheduler.add_request(request)
    request_id = request.request_id
    scheduler_output = scheduler.schedule()
    assert len(scheduler.running) == 0
    assert len(scheduler_output.scheduled_new_reqs) == 0
    assert scheduler_output.scheduled_cached_reqs.num_reqs == 0
    assert len(scheduler_output.num_scheduled_tokens) == 0
    assert scheduler_output.total_num_scheduled_tokens == 0
    assert len(scheduler.waiting) == 1
    assert request in scheduler.waiting
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert request.num_computed_tokens == 0
    block_pool = scheduler.kv_cache_manager.block_pool
    assert block_pool.free_block_queue.num_free_blocks < START_FREE_BLOCK_QUEUE_SIZE
    assert len(block_pool.cached_block_hash_to_block) == 0
    blocks = scheduler.kv_cache_manager.coordinator.single_type_managers[0].req_to_blocks[request_id]
    for block in blocks:
        assert block._block_hash is None
    model_runner_output = EMPTY_MODEL_RUNNER_OUTPUT
    engine_core_outputs = scheduler.update_from_output(scheduler_output, model_runner_output)
    assert not engine_core_outputs or not engine_core_outputs[0].outputs
    scheduler_output = scheduler.schedule()
    assert len(scheduler.waiting) == 1
    assert len(scheduler.running) == 0
    model_runner_output = copy.deepcopy(EMPTY_MODEL_RUNNER_OUTPUT)
    model_runner_output.finished_recving = [request_id]
    engine_core_outputs = scheduler.update_from_output(scheduler_output, model_runner_output)
    assert len(scheduler.waiting) == 1
    assert request_id in scheduler.finished_recving_kv_req_ids
    scheduler_output = scheduler.schedule()
    assert len(scheduler.running) == 1
    num_hashed_blocks = 0
    blocks = scheduler.kv_cache_manager.coordinator.single_type_managers[0].req_to_blocks[request_id]
    for block in blocks:
        assert block.ref_cnt == 1
        num_hashed_blocks += 1 if block._block_hash is not None else 0
    assert num_hashed_blocks == NUM_EXTERNAL_FULL_BLOCKS
    scheduled_req = scheduler_output.scheduled_new_reqs[0]
    num_scheduled_tokens = scheduler_output.num_scheduled_tokens[request_id]
    num_computed_tokens = scheduled_req.num_computed_tokens
    total_prompt_tokens = len(scheduled_req.prompt_token_ids)
    assert num_scheduled_tokens == total_prompt_tokens - num_computed_tokens
    model_runner_output = create_model_runner_output([request])
    scheduler.update_from_output(scheduler_output, model_runner_output)
    scheduler_output = scheduler.schedule()
    model_runner_output = create_model_runner_output([request], use_eos=True)
    engine_core_outputs = scheduler.update_from_output(scheduler_output, model_runner_output)
    scheduler.schedule()
    outputs = engine_core_outputs[0].outputs
    assert len(outputs) == 1
    output = outputs[0]
    assert output.finish_reason == FinishReason.STOP
    assert_scheduler_empty(scheduler)

def test_interleaved_lifecycle():
    """Test Remote Prefills Work Well With Other Requests."""
    vllm_config = create_vllm_config()
    scheduler = create_scheduler(vllm_config)
    BLOCK_SIZE = vllm_config.cache_config.block_size
    NUM_EXTERNAL_FULL_BLOCKS = 2
    NUM_TOKENS = int(BLOCK_SIZE * (NUM_EXTERNAL_FULL_BLOCKS + 0.5))
    request_remote = create_request(request_id=1, num_tokens=NUM_TOKENS, do_remote_prefill=True)
    request_local_a = create_request(request_id=2, num_tokens=NUM_TOKENS)
    request_local_b = create_request(request_id=3, num_tokens=NUM_TOKENS)
    scheduler.add_request(request_local_a)
    scheduler_output = scheduler.schedule()
    assert len(scheduler.running) == 1
    model_runner_output = create_model_runner_output([request_local_a])
    scheduler.update_from_output(scheduler_output, model_runner_output)
    scheduler.add_request(request_local_b)
    scheduler.add_request(request_remote)
    scheduler_output = scheduler.schedule()
    assert len(scheduler.running) == 2
    assert len(scheduler.waiting) == 1
    assert len(scheduler_output.scheduled_new_reqs) == 1
    assert scheduler_output.scheduled_cached_reqs.num_reqs == 1
    model_runner_output = create_model_runner_output([request_local_a, request_local_b])
    scheduler.update_from_output(scheduler_output, model_runner_output)
    scheduler_output = scheduler.schedule()
    assert len(scheduler.running) == 2
    assert len(scheduler.waiting) == 1
    assert len(scheduler_output.scheduled_new_reqs) == 0
    assert scheduler_output.scheduled_cached_reqs.num_reqs == 2
    model_runner_output = create_model_runner_output(reqs=[request_local_a, request_local_b])
    scheduler.update_from_output(scheduler_output, model_runner_output)
    assert len(scheduler.running) == 2
    assert len(scheduler.waiting) == 1
    assert len(scheduler_output.scheduled_new_reqs) == 0
    assert scheduler_output.scheduled_cached_reqs.num_reqs == 2
    scheduler_output = scheduler.schedule()
    assert len(scheduler.running) == 2
    assert len(scheduler.waiting) == 1
    assert len(scheduler_output.scheduled_new_reqs) == 0
    assert scheduler_output.scheduled_cached_reqs.num_reqs == 2
    model_runner_output = create_model_runner_output([request_local_a, request_local_b], finished_recving=[request_remote.request_id])
    scheduler.update_from_output(scheduler_output, model_runner_output)
    scheduler_output = scheduler.schedule()
    assert len(scheduler.running) == 3
    assert len(scheduler.waiting) == 0
    assert len(scheduler_output.scheduled_new_reqs) == 1
    assert scheduler_output.scheduled_cached_reqs.num_reqs == 2
    model_runner_output = create_model_runner_output([request_local_a, request_local_b, request_remote])
    scheduler.update_from_output(scheduler_output, model_runner_output)
    scheduler_output = scheduler.schedule()
    model_runner_output = create_model_runner_output([request_local_a, request_local_b, request_remote], use_eos=True)
    scheduler.update_from_output(scheduler_output, model_runner_output)
    scheduler.schedule()
    assert_scheduler_empty(scheduler)

def test_no_spurious_prefix_caching():
    """
    With P/D, blocks can be allocated but uncomputed for
    multiple engine steps. This test confirms that we do
    not accidentally have cache hits against uncomputed
    blocks.
    """
    vllm_config = create_vllm_config()
    scheduler = create_scheduler(vllm_config)
    vllm_config = create_vllm_config()
    scheduler = create_scheduler(vllm_config)
    BLOCK_SIZE = vllm_config.cache_config.block_size
    NUM_EXTERNAL_FULL_BLOCKS = 2
    NUM_TOKENS = int(BLOCK_SIZE * (NUM_EXTERNAL_FULL_BLOCKS + 0.5))
    request_remote = create_request(request_id=1, num_tokens=NUM_TOKENS, do_remote_prefill=True, use_all_1s_for_prompt_tokens=True)
    request_local = create_request(request_id=2, num_tokens=NUM_TOKENS, do_remote_prefill=False, use_all_1s_for_prompt_tokens=True)
    scheduler.add_request(request_remote)
    scheduler_output = scheduler.schedule()
    scheduler.update_from_output(scheduler_output, EMPTY_MODEL_RUNNER_OUTPUT)
    assert len(scheduler.waiting) == 1
    scheduler.add_request(request_local)
    scheduler_output = scheduler.schedule()
    assert len(scheduler.running) == 1
    assert len(scheduler.waiting) == 1
    local_blocks = scheduler.kv_cache_manager.coordinator.single_type_managers[0].req_to_blocks[request_local.request_id]
    remote_blocks = scheduler.kv_cache_manager.coordinator.single_type_managers[0].req_to_blocks[request_remote.request_id]
    num_hashed_blocks = 0
    for block in local_blocks:
        assert block.ref_cnt == 1
        num_hashed_blocks += 1 if block._block_hash is not None else 0
    assert num_hashed_blocks > 0
    for block in remote_blocks:
        assert block.ref_cnt == 1
        assert block._block_hash is None

def test_full_block_prompt():
    """Test that we handle a prompt that is the full block size."""
    vllm_config = create_vllm_config()
    scheduler = create_scheduler(vllm_config)
    BLOCK_SIZE = vllm_config.cache_config.block_size
    NUM_EXTERNAL_FULL_BLOCKS = 2
    NUM_TOKENS = int(BLOCK_SIZE * NUM_EXTERNAL_FULL_BLOCKS)
    request = create_request(request_id=1, num_tokens=NUM_TOKENS, do_remote_prefill=True)
    scheduler.add_request(request)
    request_id = request.request_id
    scheduler_output = scheduler.schedule()
    num_blocks = len(scheduler.kv_cache_manager.coordinator.single_type_managers[0].req_to_blocks[request_id])
    assert num_blocks == NUM_EXTERNAL_FULL_BLOCKS
    model_runner_output = EMPTY_MODEL_RUNNER_OUTPUT
    scheduler.update_from_output(scheduler_output, model_runner_output)
    scheduler_output = scheduler.schedule()
    model_runner_output = copy.deepcopy(EMPTY_MODEL_RUNNER_OUTPUT)
    model_runner_output.finished_recving = [request_id]
    scheduler.update_from_output(scheduler_output, model_runner_output)
    assert len(scheduler.waiting) == 1
    assert request_id in scheduler.finished_recving_kv_req_ids
    scheduler_output = scheduler.schedule()
    num_blocks = len(scheduler.kv_cache_manager.coordinator.single_type_managers[0].req_to_blocks[request_id])
    assert num_blocks == NUM_EXTERNAL_FULL_BLOCKS
    assert scheduler_output.scheduled_new_reqs[0].num_computed_tokens == NUM_TOKENS - 1
    assert scheduler_output.num_scheduled_tokens[request_id] == 1
    model_runner_output = create_model_runner_output([request])
    scheduler.update_from_output(scheduler_output, model_runner_output)
    scheduler_output = scheduler.schedule()
    model_runner_output = create_model_runner_output([request], use_eos=True)
    engine_core_outputs = scheduler.update_from_output(scheduler_output, model_runner_output)
    scheduler.schedule()
    outputs = engine_core_outputs[0].outputs
    assert len(outputs) == 1
    output = outputs[0]
    assert output.finish_reason == FinishReason.STOP
    assert_scheduler_empty(scheduler)

def test_cannot_schedule_after_recv():
    """
    Test that we can handle no schedule after recv due to not
    enough remaining KV blocks.
    """
    TOTAL_NUM_BLOCKS = 6
    vllm_config = create_vllm_config()
    scheduler = create_scheduler(vllm_config, num_blocks=TOTAL_NUM_BLOCKS)
    NUM_PROMPT_BLOCKS = 2
    BLOCK_SIZE = vllm_config.cache_config.block_size
    NUM_TOKENS_LOCAL = int(BLOCK_SIZE * NUM_PROMPT_BLOCKS)
    NUM_TOKENS_REMOTE = int(BLOCK_SIZE * (NUM_PROMPT_BLOCKS + 0.5))
    request_normal = create_request(request_id=1, num_tokens=NUM_TOKENS_LOCAL)
    request_remote = create_request(request_id=2, num_tokens=NUM_TOKENS_REMOTE, do_remote_prefill=True)
    scheduler.add_request(request_normal)
    scheduler_output = scheduler.schedule()
    model_runner_output = create_model_runner_output(reqs=[request_normal])
    scheduler.update_from_output(scheduler_output, model_runner_output)
    assert len(scheduler.running) == 1
    assert len(scheduler.waiting) == 0
    scheduler.add_request(request_remote)
    scheduler_output = scheduler.schedule()
    model_runner_output = create_model_runner_output(reqs=[request_normal])
    scheduler.update_from_output(scheduler_output, model_runner_output)
    assert len(scheduler.running) == 1
    assert len(scheduler.waiting) == 1
    scheduler_output = scheduler.schedule()
    model_runner_output = create_model_runner_output(reqs=[request_normal], finished_recving=[request_remote.request_id])
    scheduler.update_from_output(scheduler_output, model_runner_output)
    assert len(scheduler.running) == 1
    assert len(scheduler.waiting) == 1
    scheduler_output = scheduler.schedule()
    model_runner_output = create_model_runner_output(reqs=[request_normal])
    scheduler.update_from_output(scheduler_output, model_runner_output)
    assert len(scheduler.running) == 1
    assert len(scheduler.waiting) == 1
    scheduler_output = scheduler.schedule()
    model_runner_output = create_model_runner_output(reqs=[request_normal], use_eos=True)
    scheduler.update_from_output(scheduler_output, model_runner_output)
    assert len(scheduler.running) == 0
    assert len(scheduler.waiting) == 1
    scheduler_output = scheduler.schedule()
    model_runner_output = create_model_runner_output(reqs=[request_remote])
    assert scheduler_output.scheduled_new_reqs[0].num_computed_tokens == NUM_PROMPT_BLOCKS * BLOCK_SIZE
    scheduler.update_from_output(scheduler_output, model_runner_output)
    assert len(scheduler.running) == 1
    assert len(scheduler.waiting) == 0
    scheduler_output = scheduler.schedule()
    model_runner_output = create_model_runner_output(reqs=[request_remote], use_eos=True)
    scheduler.update_from_output(scheduler_output, model_runner_output)
    _ = scheduler.schedule()
    assert_scheduler_empty(scheduler)