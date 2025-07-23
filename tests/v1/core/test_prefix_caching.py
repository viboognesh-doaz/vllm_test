from typing import Optional
from vllm.distributed.kv_events import AllBlocksCleared, BlockRemoved
from vllm.multimodal.inputs import MultiModalKwargs, PlaceholderRange
from vllm.sampling_params import SamplingParams
from vllm.utils import sha256, sha256_cbor_64bit
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheManager, Request
from vllm.v1.core.kv_cache_utils import BlockHash, BlockHashWithGroupId, KVCacheBlock, hash_block_tokens, init_none_hash
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, SlidingWindowSpec
import copy
import pytest
import torch
'Compare the with and without prefix caching.'

def make_request(request_id, prompt_token_ids, mm_positions=None, mm_hashes=None, prompt_logprobs: Optional[int]=None, cache_salt: Optional[str]=None):
    if mm_positions is None:
        multi_modal_inputs = None
    else:
        multi_modal_inputs = [MultiModalKwargs({})] * len(mm_positions)
    return Request(request_id=request_id, prompt_token_ids=prompt_token_ids, multi_modal_inputs=multi_modal_inputs, multi_modal_hashes=mm_hashes, multi_modal_placeholders=mm_positions, sampling_params=SamplingParams(max_tokens=17, prompt_logprobs=prompt_logprobs), pooling_params=None, eos_token_id=100, lora_request=None, cache_salt=cache_salt)

def make_kv_cache_config(block_size: int, num_blocks: int) -> KVCacheConfig:
    return KVCacheConfig(num_blocks=num_blocks, kv_cache_tensors=[], kv_cache_groups=[KVCacheGroupSpec(['layer'], FullAttentionSpec(block_size, 1, 1, torch.float32, False))])

def make_kv_cache_config_hybrid_model(block_size: int, num_blocks: int) -> KVCacheConfig:
    return KVCacheConfig(num_blocks=num_blocks, kv_cache_tensors=[], kv_cache_groups=[KVCacheGroupSpec(['layer1'], FullAttentionSpec(block_size, 1, 1, torch.float32, False)), KVCacheGroupSpec(['layer2'], SlidingWindowSpec(block_size, 1, 1, torch.float32, False, sliding_window=2 * block_size)), KVCacheGroupSpec(['layer3'], SlidingWindowSpec(block_size, 1, 1, torch.float32, False, sliding_window=2 * block_size))])

@pytest.mark.parametrize('hash_algo', ['sha256', 'sha256_cbor_64bit', 'hash'])
def test_prefill(hash_algo):
    manager = KVCacheManager(make_kv_cache_config(16, 11), max_model_len=8192, enable_caching=True, caching_hash_algo=hash_algo)
    hash_fn = sha256_cbor_64bit if hash_algo == 'sha256_cbor_64bit' else sha256 if hash_algo == 'sha256' else hash
    common_token_ids = [i for i in range(3) for _ in range(16)]
    unique_token_ids = [3] * 7
    all_token_ids = common_token_ids + unique_token_ids
    req0 = make_request('0', all_token_ids)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req0)
    assert len(manager.req_to_block_hashes[req0.request_id]) == 3
    assert not computed_blocks.blocks[0]
    assert num_computed_tokens == 0
    blocks = manager.allocate_slots(req0, 55, len(computed_blocks.blocks[0]) * 16, computed_blocks)
    assert blocks.get_block_ids() == ([1, 2, 3, 4],)
    parent_block_hash = None
    for block_id in (1, 2, 3):
        block_tokens = tuple(all_token_ids[(block_id - 1) * 16:block_id * 16])
        block_hash = hash_block_tokens(hash_fn, parent_block_hash, block_tokens)
        assert manager.block_pool.blocks[block_id].block_hash.block_hash == block_hash
        assert manager.block_pool.blocks[block_id].ref_cnt == 1
        parent_block_hash = block_hash.hash_value
    for block_id in (4,):
        assert manager.block_pool.blocks[block_id].block_hash is None
        assert manager.block_pool.blocks[block_id].ref_cnt == 1
    unique_token_ids = [3] * 5
    req1 = make_request('1', common_token_ids + unique_token_ids)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req1)
    assert len(manager.req_to_block_hashes[req1.request_id]) == 3
    assert computed_blocks.get_block_ids() == ([1, 2, 3],)
    assert num_computed_tokens == 3 * 16
    num_new_tokens = 53 - 3 * 16
    blocks = manager.allocate_slots(req1, num_new_tokens, len(computed_blocks.blocks[0]) * 16, computed_blocks)
    assert blocks.get_block_ids() == ([5],)
    for block in computed_blocks.blocks[0]:
        assert block.ref_cnt == 2
    free_block_queue = manager.block_pool.free_block_queue
    assert free_block_queue.num_free_blocks == 5
    manager.free(req0)
    manager.free(req1)
    assert free_block_queue.num_free_blocks == 10
    assert [b.block_id for b in manager.block_pool.free_block_queue.get_all_free_blocks()] == [6, 7, 8, 9, 10, 4, 5, 3, 2, 1]
    unique_token_ids = [3] * 6
    req2 = make_request('2', common_token_ids + unique_token_ids)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req2)
    assert len(manager.req_to_block_hashes[req2.request_id]) == 3
    assert computed_blocks.get_block_ids() == ([1, 2, 3],)
    assert num_computed_tokens == 3 * 16
    num_new_tokens = 53 - 3 * 16
    blocks = manager.allocate_slots(req2, num_new_tokens, len(computed_blocks.blocks[0]) * 16, computed_blocks)
    assert blocks.get_block_ids() == ([6],)
    assert free_block_queue.num_free_blocks == 6
    assert all([b.ref_cnt == 0 for b in free_block_queue.get_all_free_blocks()])
    assert len([b for b in free_block_queue.get_all_free_blocks()]) == 6
    manager.free(req2)
    req3 = make_request('3', [99] * (16 * 10))
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req3)
    assert not computed_blocks.blocks[0]
    assert num_computed_tokens == 0
    blocks = manager.allocate_slots(req3, 16 * 10, len(computed_blocks.blocks[0]) * 16, computed_blocks)
    assert blocks.get_block_ids() == ([7, 8, 9, 10, 4, 5, 6, 3, 2, 1],)
    assert free_block_queue.num_free_blocks == 0
    assert free_block_queue.fake_free_list_head.next_free_block is free_block_queue.fake_free_list_tail
    assert free_block_queue.fake_free_list_tail.prev_free_block is free_block_queue.fake_free_list_head

def test_prefill_hybrid_model():
    block_size = 16
    manager = KVCacheManager(make_kv_cache_config_hybrid_model(block_size, 21), max_model_len=8192, enable_caching=True)
    hash_fn = hash
    common_token_ids = [i for i in range(3) for _ in range(block_size)]
    unique_token_ids = [3] * 7
    all_token_ids = common_token_ids + unique_token_ids
    req0 = make_request('0', all_token_ids)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req0)
    assert len(manager.req_to_block_hashes[req0.request_id]) == 3
    assert not computed_blocks.blocks[0]
    assert num_computed_tokens == 0
    blocks = manager.allocate_slots(req0, 55, len(computed_blocks.blocks[0]) * 16, computed_blocks)
    assert blocks.get_block_ids() == ([1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12])
    parent_block_hash = None
    for length, block_ids in zip((1, 2, 3), ((1, 5, 9), (2, 6, 10), (3, 7, 11))):
        block_tokens = tuple(all_token_ids[(length - 1) * 16:length * 16])
        block_hash = hash_block_tokens(hash_fn, parent_block_hash, block_tokens)
        for block_id in block_ids:
            assert manager.block_pool.blocks[block_id].block_hash.block_hash == block_hash
            assert manager.block_pool.blocks[block_id].ref_cnt == 1
        parent_block_hash = block_hash.hash_value
    for block_id in (4, 8, 12):
        assert manager.block_pool.blocks[block_id].block_hash is None
        assert manager.block_pool.blocks[block_id].ref_cnt == 1
    unique_token_ids = [3] * 5
    req1 = make_request('1', common_token_ids + unique_token_ids)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req1)
    assert len(manager.req_to_block_hashes[req1.request_id]) == 3
    assert computed_blocks.get_block_ids() == ([1, 2, 3], [0, 6, 7], [0, 10, 11])
    assert num_computed_tokens == 3 * 16
    num_new_tokens = 53 - 3 * 16
    blocks = manager.allocate_slots(req1, num_new_tokens, len(computed_blocks.blocks[0]) * 16, computed_blocks)
    assert blocks.get_block_ids() == ([13], [14], [15])
    for block_per_group in computed_blocks.blocks:
        for block in block_per_group:
            if block != manager.block_pool.null_block:
                assert block.ref_cnt == 2
    block_hashes = manager.req_to_block_hashes[req1.request_id]
    manager.free(req0)
    manager.free(req1)
    cached_block_hash_to_block_bak = copy.copy(manager.block_pool.cached_block_hash_to_block)

    def test_partial_request_hit(request_id: str, hash_to_evict: list[BlockHashWithGroupId], expect_hit_length: int):
        req = make_request(request_id, common_token_ids + unique_token_ids)
        for hash_with_group_id in hash_to_evict:
            manager.block_pool.cached_block_hash_to_block.pop(hash_with_group_id)
        computed_blocks, num_computed_tokens = manager.get_computed_blocks(req)
        assert len(manager.req_to_block_hashes[req.request_id]) == 3
        assert num_computed_tokens == expect_hit_length * block_size
        for block_per_group in computed_blocks.blocks:
            assert len(block_per_group) == num_computed_tokens // block_size
        for hash_with_group_id in hash_to_evict:
            manager.block_pool.cached_block_hash_to_block[hash_with_group_id] = cached_block_hash_to_block_bak[hash_with_group_id]
        manager.free(req)
    test_partial_request_hit('2', [BlockHashWithGroupId(block_hashes[0], 1), BlockHashWithGroupId(block_hashes[0], 2)], 3)
    test_partial_request_hit('3', [BlockHashWithGroupId(block_hashes[0], 0)], 0)
    test_partial_request_hit('4', [BlockHashWithGroupId(block_hashes[2], 0), BlockHashWithGroupId(block_hashes[2], 1), BlockHashWithGroupId(block_hashes[2], 2)], 2)
    test_partial_request_hit('5', [BlockHashWithGroupId(block_hashes[2], 0)], 2)
    test_partial_request_hit('6', [BlockHashWithGroupId(block_hashes[2], 1)], 2)
    test_partial_request_hit('7', [BlockHashWithGroupId(block_hashes[2], 2)], 2)
    test_partial_request_hit('8', [BlockHashWithGroupId(block_hashes[2], 0), BlockHashWithGroupId(block_hashes[0], 1), BlockHashWithGroupId(block_hashes[0], 2)], 0)

def test_prefill_plp():
    """Test prefill with APC and some prompt logprobs (plp) requests.

    1. Schedule plp request and validate APC block allocation
    2. Schedule non-plp request and validate blocks
    3. Schedule plp request; no hit should occur; validate blocks
    """
    manager = KVCacheManager(make_kv_cache_config(16, 11), max_model_len=8192, enable_caching=True)
    hash_fn = hash
    common_token_ids = [i for i in range(3) for _ in range(16)]
    unique_token_ids = [3] * 7
    all_token_ids = common_token_ids + unique_token_ids
    req0 = make_request('0', all_token_ids, prompt_logprobs=5)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req0)
    assert len(manager.req_to_block_hashes[req0.request_id]) == 0
    assert not computed_blocks.blocks[0]
    assert num_computed_tokens == 0
    blocks = manager.allocate_slots(req0, 55, len(computed_blocks.blocks[0]) * 16, computed_blocks)
    assert blocks.get_block_ids() == ([1, 2, 3, 4],)
    req0_block_hashes = [b.block_hash for b in blocks.blocks[0]]
    parent_block_hash = None
    for block_id in (1, 2, 3):
        block_tokens = tuple(all_token_ids[(block_id - 1) * 16:block_id * 16])
        block_hash = hash_block_tokens(hash_fn, parent_block_hash, block_tokens)
        assert manager.block_pool.blocks[block_id].block_hash.block_hash == block_hash
        assert manager.block_pool.blocks[block_id].ref_cnt == 1
        parent_block_hash = block_hash.hash_value
    for block_id in (4,):
        assert manager.block_pool.blocks[block_id].block_hash is None
        assert manager.block_pool.blocks[block_id].ref_cnt == 1
    unique_token_ids = [3] * 5
    req1 = make_request('1', common_token_ids + unique_token_ids)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req1)
    assert len(manager.req_to_block_hashes[req1.request_id]) == 3
    assert computed_blocks.get_block_ids() == ([1, 2, 3],)
    assert num_computed_tokens == 3 * 16
    num_new_tokens = 53 - 3 * 16
    blocks = manager.allocate_slots(req1, num_new_tokens, len(computed_blocks.blocks[0]) * 16, computed_blocks)
    assert blocks.get_block_ids() == ([5],)
    for block in computed_blocks.blocks[0]:
        assert block.ref_cnt == 2
    assert manager.block_pool.free_block_queue.num_free_blocks == 5
    manager.free(req0)
    manager.free(req1)
    assert manager.block_pool.free_block_queue.num_free_blocks == 10
    assert [b.block_id for b in manager.block_pool.free_block_queue.get_all_free_blocks()] == [6, 7, 8, 9, 10, 4, 5, 3, 2, 1]
    unique_token_ids = [3] * 6
    req2 = make_request('2', common_token_ids + unique_token_ids, prompt_logprobs=5)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req2)
    assert len(manager.req_to_block_hashes[req2.request_id]) == 0
    assert not computed_blocks.blocks[0]
    assert num_computed_tokens == 0
    blocks = manager.allocate_slots(req2, 55, len(computed_blocks.blocks[0]) * 16, computed_blocks)
    block_ids = blocks.get_block_ids()
    assert [b.block_hash for b in blocks.blocks[0]] == req0_block_hashes
    assert block_ids != ([1, 2, 3, 4],)
    for block_id in block_ids[0]:
        assert manager.block_pool.blocks[block_id].ref_cnt == 1
    manager.free(req2)

def test_decode():
    manager = KVCacheManager(make_kv_cache_config(16, 11), max_model_len=8192, enable_caching=True)
    common_token_ids = [i for i in range(3) for _ in range(16)]
    unique_token_ids = [3] * 7
    req0 = make_request('0', common_token_ids + unique_token_ids)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req0)
    assert not computed_blocks.blocks[0]
    assert num_computed_tokens == 0
    blocks = manager.allocate_slots(req0, 55, len(computed_blocks.blocks[0]) * 16, computed_blocks)
    assert blocks.get_block_ids() == ([1, 2, 3, 4],)
    req0.num_computed_tokens = 55
    for _ in range(4):
        req0.append_output_token_ids(8)
    new_blocks = manager.allocate_slots(req0, 4, len(computed_blocks.blocks[0]) * 16, computed_blocks)
    assert new_blocks is not None and len(new_blocks.blocks[0]) == 0
    assert manager.coordinator.single_type_managers[0].req_to_blocks[req0.request_id][-1].block_hash is None
    req0.num_computed_tokens = 59
    for _ in range(9 + 10):
        req0.append_output_token_ids(7)
    new_blocks = manager.allocate_slots(req0, 19, len(computed_blocks.blocks[0]) * 16, computed_blocks)
    assert new_blocks is not None and len(new_blocks.blocks[0]) == 1
    assert manager.coordinator.single_type_managers[0].req_to_blocks[req0.request_id][-2].block_hash is not None
    assert manager.coordinator.single_type_managers[0].req_to_blocks[req0.request_id][-1].block_hash is None

def test_evict():
    manager = KVCacheManager(make_kv_cache_config(16, 11), max_model_len=8192, enable_caching=True)
    last_token_id = 5 * 16 + 7
    req0 = make_request('0', list(range(last_token_id)))
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req0)
    assert not computed_blocks.blocks[0]
    assert num_computed_tokens == 0
    blocks = manager.allocate_slots(req0, 5 * 16 + 7, len(computed_blocks.blocks[0]) * 16, computed_blocks)
    assert len(blocks.blocks[0]) == 6
    req1 = make_request('1', list(range(last_token_id, last_token_id + 3 * 16)))
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req1)
    assert not computed_blocks.blocks[0]
    assert num_computed_tokens == 0
    blocks = manager.allocate_slots(req1, 3 * 16, len(computed_blocks.blocks[0]) * 16, computed_blocks)
    assert len(blocks.blocks[0]) == 3
    last_token_id += 3 * 16
    assert manager.block_pool.free_block_queue.num_free_blocks == 1
    manager.free(req0)
    manager.free(req1)
    assert manager.block_pool.free_block_queue.num_free_blocks == 10
    assert [b.block_id for b in manager.block_pool.free_block_queue.get_all_free_blocks()] == [10, 6, 5, 4, 3, 2, 1, 9, 8, 7]
    req2 = make_request('2', list(range(2 * 16 + 3)))
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req2)
    assert computed_blocks.get_block_ids() == ([1, 2],)
    assert num_computed_tokens == 2 * 16
    blocks = manager.allocate_slots(req2, 3, len(computed_blocks.blocks[0]) * 16, computed_blocks)
    assert blocks.get_block_ids() == ([10],)
    assert manager.block_pool.free_block_queue.num_free_blocks == 7

def test_hash_block_correct_reuse():
    """
    This tests when a previously cached block is reused as a new block,
    its hash metadata should be correctly reset.
    """
    block_size = 16
    manager = KVCacheManager(make_kv_cache_config(16, 2), max_model_len=8192, enable_caching=True)
    num_tokens = block_size * 1
    req = make_request('0', list(range(num_tokens)))
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req)
    assert not computed_blocks.blocks[0]
    assert num_computed_tokens == 0
    blocks = manager.allocate_slots(req, num_tokens, len(computed_blocks.blocks[0]) * 16, computed_blocks)
    assert len(blocks.blocks[0]) == 1
    manager.free(req)
    req = make_request('1', list(range(num_tokens - 1)))
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req)
    assert not computed_blocks.blocks[0]
    assert num_computed_tokens == 0
    blocks = manager.allocate_slots(req, num_tokens - 1, len(computed_blocks.blocks[0]) * 16, computed_blocks)
    assert len(blocks.blocks[0]) == 1
    assert manager.block_pool.blocks[blocks.blocks[0][0].block_id].block_hash is None

def test_computed_blocks_not_evicted():
    """
    Test that the computed blocks are not evicted when getting new blocks
    for a request if there are any other free blocks.
    """
    block_size = 16
    manager = KVCacheManager(make_kv_cache_config(block_size, 3), max_model_len=8192, enable_caching=True)
    num_tokens = block_size * 1
    req0 = make_request('0', list(range(num_tokens)))
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req0)
    assert not computed_blocks.blocks[0]
    assert num_computed_tokens == 0
    blocks = manager.allocate_slots(req0, num_tokens, len(computed_blocks.blocks[0]) * 16, computed_blocks)
    assert len(blocks.blocks[0]) == 1
    assert blocks.blocks[0][0].block_id == 1
    req1 = make_request('1', list(range(num_tokens, num_tokens * 2)))
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req1)
    assert not computed_blocks.blocks[0]
    assert num_computed_tokens == 0
    blocks = manager.allocate_slots(req1, num_tokens, len(computed_blocks.blocks[0]) * 16, computed_blocks)
    assert len(blocks.blocks[0]) == 1
    assert blocks.blocks[0][0].block_id == 2
    manager.free(req0)
    manager.free(req1)
    req2 = make_request('2', list(range(num_tokens * 2)))
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req2)
    assert len(computed_blocks.blocks[0]) == 1
    assert computed_blocks.blocks[0][0].block_id == 1
    assert num_computed_tokens == block_size
    blocks = manager.allocate_slots(req2, num_tokens * 2 - num_tokens, len(computed_blocks.blocks[0]) * 16, computed_blocks)
    assert len(blocks.blocks[0]) == 1
    assert blocks.blocks[0][0].block_id == 2

def test_basic_prefix_caching_disabled():
    """
    This tests that the prefix caching is disabled.
    """
    block_size = 4
    manager = KVCacheManager(make_kv_cache_config(block_size, 5), max_model_len=8192, enable_caching=False)
    req1 = make_request('1', list(range(10)))
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req1)
    assert not computed_blocks.blocks[0]
    assert num_computed_tokens == 0
    blocks = manager.allocate_slots(req1, 10, len(computed_blocks.blocks[0]) * 16, computed_blocks)
    assert len(blocks.blocks[0]) == 3
    manager.free(req1)
    req2 = make_request('2', list(range(16)))
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req2)
    assert not computed_blocks.blocks[0]
    assert num_computed_tokens == 0
    blocks = manager.allocate_slots(req2, 16, len(computed_blocks.blocks[0]) * 16, computed_blocks)
    assert len(blocks.blocks[0]) == 4
    req3 = make_request('3', list(range(4)))
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req3)
    assert not computed_blocks.blocks[0]
    assert num_computed_tokens == 0
    blocks = manager.allocate_slots(req3, 4, len(computed_blocks.blocks[0]) * 16, computed_blocks)
    assert not blocks

@pytest.mark.parametrize('hash_fn', [sha256, sha256_cbor_64bit, hash])
def test_cache_blocks(hash_fn):
    """
    This is a unit test that tests the correctness of the _cache_full_blocks
    function of KVCacheManager.
    """
    init_none_hash(hash_fn)
    block_size = 4
    block_pool = BlockPool(num_gpu_blocks=5, enable_caching=True)
    req = make_request('0', list(range(14)))
    blocks = [KVCacheBlock(block_id=i) for i in range(2)]
    block_hashes: list[BlockHash] = []
    block_pool.cache_full_blocks(request=req, blocks=blocks, block_hashes=block_hashes, num_cached_blocks=0, num_full_blocks=2, block_size=block_size, hash_fn=hash_fn, kv_cache_group_id=0)
    assert len(block_pool.cached_block_hash_to_block) == 2
    assert all([block.block_hash is not None for block in blocks])
    blocks += [KVCacheBlock(block_id=2)]
    block_pool.cache_full_blocks(request=req, blocks=blocks, block_hashes=block_hashes, num_cached_blocks=2, num_full_blocks=3, block_size=block_size, hash_fn=hash_fn, kv_cache_group_id=0)
    assert len(block_pool.cached_block_hash_to_block) == 3
    assert blocks[0].block_hash is not None

def test_cache_blocks_multi_group():
    """
    This tests that blocks are cached correctly for different kv cache groups.
    """
    block_size = 4
    block_pool = BlockPool(num_gpu_blocks=10, enable_caching=True)
    req = make_request('0', list(range(14)))
    blocks = [KVCacheBlock(block_id=i) for i in range(2)]
    block_hashes: list[BlockHash] = []
    block_pool.cache_full_blocks(request=req, blocks=blocks, block_hashes=block_hashes, num_cached_blocks=0, num_full_blocks=2, block_size=block_size, hash_fn=hash, kv_cache_group_id=0)
    assert len(block_pool.cached_block_hash_to_block) == 2
    assert len(block_hashes) == 2
    assert all([block.block_hash is not None for block in blocks])
    blocks = [KVCacheBlock(block_id=i) for i in range(3)]
    block_pool.cache_full_blocks(request=req, blocks=blocks, block_hashes=block_hashes, num_cached_blocks=0, num_full_blocks=3, block_size=block_size, hash_fn=hash, kv_cache_group_id=1)
    assert len(block_pool.cached_block_hash_to_block) == 5
    assert len(block_hashes) == 3
    assert all([block.block_hash is not None for block in blocks])
    assert block_pool.get_cached_block(block_hashes[0], kv_cache_group_ids=[0]) is not None
    assert block_pool.get_cached_block(block_hashes[1], kv_cache_group_ids=[0]) is not None
    assert block_pool.get_cached_block(block_hashes[2], kv_cache_group_ids=[0]) is None
    assert block_pool.get_cached_block(block_hashes[0], kv_cache_group_ids=[1]) is not None
    assert block_pool.get_cached_block(block_hashes[1], kv_cache_group_ids=[1]) is not None
    assert block_pool.get_cached_block(block_hashes[2], kv_cache_group_ids=[1]) is not None
    assert block_pool.get_cached_block(block_hashes[0], kv_cache_group_ids=[0, 1]) is not None
    assert block_pool.get_cached_block(block_hashes[1], kv_cache_group_ids=[0, 1]) is not None
    assert block_pool.get_cached_block(block_hashes[2], kv_cache_group_ids=[0, 1]) is None

def test_mm_prefix_caching():
    """
    This tests that the multi-modal prefix caching is correct.
    """
    manager = KVCacheManager(make_kv_cache_config(16, 11), max_model_len=8192, enable_caching=True)
    common_token_ids = list(range(10)) + [-1] * 6
    common_token_ids += [-1] * 4 + list(range(10, 20)) + [-1] * 2
    common_token_ids += [-1] * 16
    common_mm_positions = [PlaceholderRange(offset=11, length=10), PlaceholderRange(offset=30, length=18)]
    common_mm_hashes = ['aaa', 'bbb']
    unique_token_ids = [-1] * 7 + [100] * 4
    all_token_ids = common_token_ids + unique_token_ids
    mm_positions = common_mm_positions + [PlaceholderRange(offset=48, length=7)]
    mm_hashes = common_mm_hashes + ['ccc']
    req0 = make_request('0', all_token_ids, mm_positions=mm_positions, mm_hashes=mm_hashes)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req0)
    assert not computed_blocks.blocks[0]
    assert num_computed_tokens == 0
    block_hashes = manager.req_to_block_hashes[req0.request_id]
    assert len(block_hashes) == 3
    assert block_hashes[0].extra_keys == ('aaa',)
    assert block_hashes[1].extra_keys == ('aaa', 'bbb')
    assert block_hashes[2].extra_keys == ('bbb',)
    blocks = manager.allocate_slots(req0, 59, len(computed_blocks.blocks[0]) * 16, computed_blocks)
    assert blocks.get_block_ids() == ([1, 2, 3, 4],)
    req0.num_computed_tokens = 59
    for _ in range(5):
        req0.append_output_token_ids(8)
    new_blocks = manager.allocate_slots(req0, 5, len(computed_blocks.blocks[0]) * 16, computed_blocks)
    assert new_blocks is not None and len(new_blocks.blocks[0]) == 0
    assert len(block_hashes) == 4
    assert block_hashes[3].extra_keys == ('ccc',)
    unique_token_ids = [-1] * 7 + [200] * 5
    all_token_ids = common_token_ids + unique_token_ids
    mm_positions = common_mm_positions + [PlaceholderRange(offset=48, length=7)]
    mm_hashes = common_mm_hashes + ['ccc']
    req1 = make_request('1', all_token_ids, mm_positions=mm_positions, mm_hashes=mm_hashes)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req1)
    assert len(computed_blocks.blocks[0]) == 3
    assert num_computed_tokens == 3 * 16

def test_cache_key_salting():
    """
    This tests that cache salts are applied during hashing and the cache
    is separated cache as expected.
    """
    block_size = 16
    manager = KVCacheManager(make_kv_cache_config(block_size, 11), max_model_len=8192, enable_caching=True)
    common_token_ids = [i for i in range(3) for _ in range(block_size)]
    token_ids = common_token_ids + [3] * 11
    req0 = make_request('0', token_ids, cache_salt='salt1')
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req0)
    assert not computed_blocks.blocks[0]
    assert num_computed_tokens == 0
    block_hashes = manager.req_to_block_hashes[req0.request_id]
    assert len(block_hashes) == 3
    assert block_hashes[0].extra_keys == ('salt1',)
    assert block_hashes[1].extra_keys is None
    assert block_hashes[2].extra_keys is None
    blocks = manager.allocate_slots(req0, 59, len(computed_blocks.blocks[0]) * 16, computed_blocks)
    assert blocks.get_block_ids() == ([1, 2, 3, 4],)
    req0.num_computed_tokens = 59
    for _ in range(5):
        req0.append_output_token_ids(8)
    new_blocks = manager.allocate_slots(req0, 5, len(computed_blocks.blocks[0]) * 16, computed_blocks)
    assert new_blocks is not None and len(new_blocks.blocks[0]) == 0
    assert len(block_hashes) == 4
    assert block_hashes[3].extra_keys is None
    token_ids = common_token_ids + [4] * 11
    req1 = make_request('1', token_ids, cache_salt='salt1')
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req1)
    assert len(computed_blocks.blocks[0]) == 3
    assert num_computed_tokens == 3 * block_size
    token_ids = common_token_ids + [4] * 11
    req2 = make_request('2', token_ids, cache_salt='salt2')
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req2)
    assert len(computed_blocks.blocks[0]) == 0
    assert num_computed_tokens == 0
    block_hashes = manager.req_to_block_hashes[req2.request_id]
    assert len(block_hashes) == 3
    assert block_hashes[0].extra_keys == ('salt2',)

def test_prefill_not_enough_free_blocks_with_computed_blocks():
    """
    This is a unit test that tests the correctness of the allocate_slots
    when there is not enough free blocks. Specifically, when a request
    has computed blocks but cannot be allocated due to not enough free blocks,
    the computed blocks should not be touched.
    """
    block_size = 16
    manager = KVCacheManager(make_kv_cache_config(block_size, 11), max_model_len=8192, enable_caching=True)
    common_token_ids = [i for i in range(3) for _ in range(16)]
    req0 = make_request('0', common_token_ids)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req0)
    assert not computed_blocks.blocks[0]
    assert num_computed_tokens == 0
    manager.allocate_slots(req0, 48, len(computed_blocks.blocks[0]) * 16, computed_blocks)
    block_part0 = manager.coordinator.single_type_managers[0].req_to_blocks[req0.request_id]
    req1 = make_request('1', common_token_ids * 2)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req1)
    assert computed_blocks.blocks[0] == block_part0
    assert num_computed_tokens == 3 * 16
    manager.allocate_slots(req1, 48, len(computed_blocks.blocks[0]) * 16, computed_blocks)
    block_part1 = manager.coordinator.single_type_managers[0].req_to_blocks[req1.request_id]
    manager.free(req1)
    assert {block.ref_cnt for block in block_part1[:3]} == {1}
    assert {block.ref_cnt for block in block_part1[3:]} == {0}
    req2 = make_request('2', [7] * block_size * 2)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req2)
    assert not computed_blocks.blocks[0]
    assert num_computed_tokens == 0
    manager.allocate_slots(req2, block_size * 2, len(computed_blocks.blocks[0]) * 16, computed_blocks)
    assert manager.block_pool.free_block_queue.num_free_blocks == 5
    req3 = make_request('3', common_token_ids * 3)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req3)
    assert computed_blocks.blocks[0] == block_part1
    assert num_computed_tokens == 6 * 16
    assert manager.allocate_slots(req3, 48, len(computed_blocks.blocks[0]) * 16, computed_blocks) is None
    assert {block.ref_cnt for block in block_part1[:3]} == {1}
    assert {block.ref_cnt for block in block_part1[3:]} == {0}

def test_reset_prefix_cache():
    manager = KVCacheManager(make_kv_cache_config(16, 11), max_model_len=8192, enable_caching=True)
    full_block_token_ids = [i for i in range(3) for _ in range(16)]
    unique_token_ids = [3] * 7
    all_token_ids = full_block_token_ids + unique_token_ids
    req0 = make_request('0', all_token_ids)
    blocks = manager.allocate_slots(req0, 55)
    assert blocks.get_block_ids() == ([1, 2, 3, 4],)
    unique_token_ids = [4] * 7
    all_token_ids = full_block_token_ids + unique_token_ids
    req1 = make_request('1', all_token_ids)
    computed_blocks, _ = manager.get_computed_blocks(req1)
    assert len(manager.req_to_block_hashes[req1.request_id]) == 3
    assert len(computed_blocks.blocks[0]) == 3
    blocks = manager.allocate_slots(req1, 7, len(computed_blocks.blocks[0]) * 16, computed_blocks)
    assert blocks.get_block_ids() == ([5],)
    assert not manager.reset_prefix_cache()
    assert manager.block_pool.cached_block_hash_to_block
    manager.free(req0)
    manager.free(req1)
    assert manager.reset_prefix_cache()
    assert not manager.block_pool.cached_block_hash_to_block
    assert all([blk.block_hash is None for blk in manager.block_pool.blocks])

def test_prefix_cache_stats_disabled():
    """Test that prefix_cache_stats is None when log_stats is False."""
    manager = KVCacheManager(make_kv_cache_config(16, 11), max_model_len=8192, enable_caching=True, log_stats=False)
    assert manager.prefix_cache_stats is None
    req = make_request('0', list(range(16)))
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req)
    assert not computed_blocks.blocks[0]
    assert num_computed_tokens == 0
    manager.allocate_slots(req, 16, len(computed_blocks.blocks[0]) * 16, computed_blocks)
    manager.reset_prefix_cache()
    assert manager.prefix_cache_stats is None

@pytest.mark.parametrize('blocks_to_cache', [2, 3, 10])
def test_kv_cache_events(blocks_to_cache: int):
    block_size = 16
    num_blocks = blocks_to_cache + 1
    manager = KVCacheManager(make_kv_cache_config(block_size, num_blocks), max_model_len=8192, enable_caching=True, enable_kv_cache_events=True)
    num_tokens = block_size * blocks_to_cache
    req0 = make_request('0', list(range(num_tokens)))
    _ = manager.allocate_slots(req0, num_tokens)
    events = manager.take_events()
    block = events[-1]
    assert len(block.block_hashes) == blocks_to_cache == len(manager.block_pool.cached_block_hash_to_block)
    assert len(block.token_ids) == block.block_size * len(block.block_hashes)
    assert len(manager.block_pool.kv_event_queue) == 0
    stored_block_hash = block.block_hashes
    manager.free(req0)
    req1 = make_request('1', list(range(num_tokens)))
    _ = manager.allocate_slots(req1, num_tokens)
    events = manager.take_events()
    for blocks in events[:-1]:
        assert blocks.block_hashes[0] in stored_block_hash
    assert len(events) == blocks_to_cache + 1
    assert isinstance(events[-2], BlockRemoved)
    assert len(events[-1].block_hashes) == blocks_to_cache == len(manager.block_pool.cached_block_hash_to_block)
    manager.free(req1)
    manager.reset_prefix_cache()
    events = manager.take_events()
    assert isinstance(events[-1], AllBlocksCleared)
    assert len(manager.block_pool.cached_block_hash_to_block) == 0

def test_eagle_enabled_removes_last_block():
    """Verify Eagle does NOT remove blocks when request 
    length is divisible by block size."""
    block_size = 16
    manager = KVCacheManager(make_kv_cache_config(block_size, num_blocks=10), max_model_len=8192, enable_caching=True, use_eagle=True)
    token_ids = [0] * (3 * block_size)
    req = make_request('divisible_request', token_ids)
    computed_blocks, _ = manager.get_computed_blocks(req)
    manager.allocate_slots(req, len(token_ids), len(computed_blocks.blocks[0]) * 16, computed_blocks)
    manager.free(req)
    req_eagle = make_request('eagle_divisible', token_ids)
    computed_blocks, num_tokens = manager.get_computed_blocks(req_eagle)
    assert len(computed_blocks.blocks[0]) == 1
    assert num_tokens == 1 * block_size

def test_eagle_with_partial_blocks():
    """Test Eagle behavior with requests containing partial blocks."""
    block_size = 16
    manager = KVCacheManager(make_kv_cache_config(block_size, num_blocks=10), max_model_len=8192, enable_caching=True, use_eagle=True)
    token_ids = [0] * (2 * block_size + 5)
    req = make_request('partial_block_test', token_ids)
    computed_blocks, _ = manager.get_computed_blocks(req)
    manager.allocate_slots(req, len(token_ids), len(computed_blocks.blocks[0]) * 16, computed_blocks)
    manager.free(req)
    req_eagle = make_request('partial_eagle', token_ids)
    computed_blocks, num_tokens = manager.get_computed_blocks(req_eagle)
    assert len(computed_blocks.blocks[0]) == 1
    assert num_tokens == 1 * block_size

def test_eagle_with_sliding_window():
    """Test Eagle behavior with sliding window."""
    block_size = 16
    sliding_window_spec = SlidingWindowSpec(block_size=block_size, num_kv_heads=1, head_size=1, dtype=torch.float32, sliding_window=block_size, use_mla=False)
    manager = KVCacheManager(KVCacheConfig(num_blocks=10, kv_cache_tensors=[], kv_cache_groups=[KVCacheGroupSpec(['layer'], sliding_window_spec)]), max_model_len=8192, enable_caching=True, use_eagle=True)
    token_ids = [0] * (2 * block_size + 5)
    req = make_request('partial_block_test', token_ids)
    computed_blocks, _ = manager.get_computed_blocks(req)
    manager.allocate_slots(req, len(token_ids), len(computed_blocks.blocks[0]) * 16, computed_blocks)
    block_hash_first_block = manager.req_to_block_hashes[req.request_id][0]
    assert block_hash_first_block is not None
    manager.free(req)
    req_eagle = make_request('partial_eagle', token_ids)
    computed_blocks, num_tokens = manager.get_computed_blocks(req_eagle)
    assert len(computed_blocks.blocks[0]) == 1
    assert num_tokens == 1 * block_size
    assert manager.block_pool.get_cached_block(block_hash_first_block, kv_cache_group_ids=[0]) is not None
    manager.block_pool.cached_block_hash_to_block.pop(BlockHashWithGroupId(block_hash_first_block, 0))
    req_after_evict = make_request('partial_eagle_after_evict', token_ids)
    computed_blocks, num_tokens = manager.get_computed_blocks(req_after_evict)
    assert len(computed_blocks.blocks[0]) == 0
    assert num_tokens == 0