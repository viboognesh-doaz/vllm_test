from tests.core.utils import create_dummy_lora_sequence, create_dummy_sequence
from typing import Optional
from unittest.mock import MagicMock
from vllm.core.block.cpu_gpu_block_allocator import CpuGpuBlockAllocator
from vllm.core.block.interfaces import Block, BlockAllocator
from vllm.core.block.prefix_caching_block import ComputedBlocksTracker, PrefixCachingBlock, PrefixCachingBlockAllocator
from vllm.sequence import Logprob
from vllm.utils import Device
import math
import pytest
import random

class TestPrefixCachingBlock:

    @staticmethod
    @pytest.mark.parametrize('seed', list(range(10)))
    @pytest.mark.parametrize('block_size', [1, 16])
    @pytest.mark.parametrize('is_curr_block_full', [True, False])
    def test_first_block_has_correct_content_hash(seed: int, block_size: int, is_curr_block_full: bool):
        """Verify a block which is first in the sequence has the correct hash.
        """
        random.seed(seed)
        num_to_fill = block_size if is_curr_block_full else random.randint(0, block_size - 1)
        token_ids = list(range(num_to_fill))
        mock_allocator = MagicMock(spec=PrefixCachingBlockAllocator)
        block_with_prev = PrefixCachingBlock(prev_block=None, token_ids=token_ids, block_size=block_size, allocator=mock_allocator)
        if is_curr_block_full:
            assert block_with_prev.content_hash == PrefixCachingBlock.hash_block_tokens(is_first_block=True, prev_block_hash=None, cur_block_token_ids=token_ids)
        else:
            assert block_with_prev.content_hash is None

    @staticmethod
    @pytest.mark.parametrize('seed', list(range(10)))
    @pytest.mark.parametrize('block_size', [1, 16])
    @pytest.mark.parametrize('is_curr_block_full', [True, False])
    @pytest.mark.parametrize('prev_block_has_hash', [True, False])
    def test_nth_block_has_correct_content_hash(seed: int, block_size: int, is_curr_block_full: bool, prev_block_has_hash: bool):
        """Verify a block which is not first in the sequence has the correct
        hash.
        """
        random.seed(seed)
        previous_block = MagicMock(spec=PrefixCachingBlock)
        prev_block_hash = random.randint(0, 1000)
        previous_block.content_hash = prev_block_hash if prev_block_has_hash else hash('None')
        num_to_fill = block_size if is_curr_block_full else random.randint(0, block_size - 1)
        token_ids = list(range(num_to_fill))
        mock_allocator = MagicMock(spec=PrefixCachingBlockAllocator)
        block_with_prev = PrefixCachingBlock(prev_block=previous_block, token_ids=token_ids, block_size=block_size, allocator=mock_allocator)
        if is_curr_block_full and prev_block_has_hash:
            assert block_with_prev.content_hash == PrefixCachingBlock.hash_block_tokens(is_first_block=False, prev_block_hash=prev_block_hash, cur_block_token_ids=token_ids)
        else:
            assert block_with_prev.content_hash is None

    @staticmethod
    @pytest.mark.parametrize('block_size', [1, 2, 16])
    @pytest.mark.parametrize('num_tokens', list(range(3)))
    @pytest.mark.parametrize('num_empty_trailing_blocks', [0, 1, 10])
    def test_blocks_have_correct_hash_in_chain(block_size: int, num_tokens: int, num_empty_trailing_blocks: int):
        """Create two chains of logical blocks with the same contents.
        Assert the hashes are equal.
        """
        random.seed(0)
        token_ids = [random.randint(0, 50000) for _ in range(num_tokens)]
        first_chain, second_chain = (TestPrefixCachingBlock.create_chain(block_size=block_size, token_ids=token_ids, num_empty_trailing_blocks=num_empty_trailing_blocks) for _ in range(2))
        for first_chain_block, second_chain_block in zip(first_chain, second_chain):
            assert first_chain_block.content_hash == second_chain_block.content_hash
        if not first_chain or not second_chain:
            assert first_chain == second_chain
            assert num_tokens == 0

    @staticmethod
    def create_chain(block_size: int, token_ids: list[int], num_empty_trailing_blocks=0) -> list[PrefixCachingBlock]:
        """Helper method which creates a chain of blocks.
        """
        blocks: list[PrefixCachingBlock] = []
        num_blocks = math.ceil(len(token_ids) / block_size) + num_empty_trailing_blocks
        if num_blocks == 0:
            return []
        allocator = MagicMock(spec=PrefixCachingBlockAllocator)
        prev_block = None
        for block_number in range(0, num_blocks):
            prev_block = PrefixCachingBlock(prev_block=prev_block, token_ids=[], block_size=block_size, allocator=allocator)
            tokens_to_append = token_ids[block_number * block_size:(block_number + 1) * block_size]
            if tokens_to_append:
                prev_block.append_token_ids(tokens_to_append)
            blocks.append(prev_block)
        return blocks

class TestPrefixCachingBlockAllocator:

    @staticmethod
    def create_allocate_lambda(allocate_type: str, allocator: BlockAllocator, prev_block: Optional[Block], token_ids: list[int]):
        if allocate_type == 'immutable':
            allocate_block = lambda: allocator.allocate_immutable_block(prev_block=prev_block, token_ids=token_ids)
        elif allocate_type == 'mutable':
            allocate_block = lambda: allocator.allocate_mutable_block(prev_block=prev_block)
        else:
            raise ValueError()
        return allocate_block

    @staticmethod
    @pytest.mark.parametrize('num_blocks', [1, 1024])
    @pytest.mark.parametrize('block_size', [1, 16])
    def test_allocate_mutable_ooms(num_blocks: int, block_size: int):
        allocator = PrefixCachingBlockAllocator(num_blocks=num_blocks, block_size=block_size)
        allocate_block = TestPrefixCachingBlockAllocator.create_allocate_lambda(allocate_type='mutable', allocator=allocator, prev_block=None, token_ids=list(range(block_size)))
        [allocate_block() for _ in range(num_blocks)]
        with pytest.raises(BlockAllocator.NoFreeBlocksError):
            allocate_block()

    @staticmethod
    @pytest.mark.parametrize('num_blocks', [1, 1024])
    @pytest.mark.parametrize('block_size', [1, 16])
    def test_allocate_immutable_does_not_oom_single_hash(num_blocks: int, block_size: int):
        allocator = PrefixCachingBlockAllocator(num_blocks=num_blocks, block_size=block_size)
        allocate_block = TestPrefixCachingBlockAllocator.create_allocate_lambda(allocate_type='immutable', allocator=allocator, prev_block=None, token_ids=list(range(block_size)))
        blocks = [allocate_block() for _ in range(num_blocks)]
        non_oom_block = allocate_block()
        for block in blocks:
            assert block.block_id == non_oom_block.block_id

    @staticmethod
    @pytest.mark.parametrize('num_blocks', [1, 1024])
    @pytest.mark.parametrize('block_size', [1, 16])
    def test_allocate_immutable_ooms_many_hash(num_blocks: int, block_size: int):
        """Consume all blocks using many different hashes/block content.

        Do this by creating a sequence that is very long.
        Expect next block to OOM.
        """
        allocator = PrefixCachingBlockAllocator(num_blocks=num_blocks, block_size=block_size)
        token_ids = list(range(num_blocks * block_size))
        chain = TestPrefixCachingBlockAllocator.create_immutable_chain(block_size=block_size, token_ids=token_ids, allocator=allocator)
        with pytest.raises(BlockAllocator.NoFreeBlocksError):
            allocator.allocate_immutable_block(prev_block=chain[-1], token_ids=list(range(block_size)))
        with pytest.raises(BlockAllocator.NoFreeBlocksError):
            allocator.allocate_mutable_block(prev_block=chain[-1])
        second_chain = TestPrefixCachingBlockAllocator.create_immutable_chain(block_size=block_size, token_ids=token_ids, allocator=allocator)
        assert chain and second_chain
        for first_chain_block, second_chain_block in zip(chain, second_chain):
            assert first_chain_block.block_id == second_chain_block.block_id

    @staticmethod
    @pytest.mark.parametrize('num_blocks', [1, 1024])
    @pytest.mark.parametrize('block_size', [1, 16])
    def test_free_prevents_oom(num_blocks: int, block_size: int):
        allocator = PrefixCachingBlockAllocator(num_blocks=num_blocks, block_size=block_size)
        token_ids = list(range(num_blocks * block_size))
        chain = TestPrefixCachingBlockAllocator.create_immutable_chain(block_size=block_size, token_ids=token_ids, allocator=allocator)
        with pytest.raises(BlockAllocator.NoFreeBlocksError):
            allocator.allocate_mutable_block(prev_block=None)
        block_to_free = chain[-1]
        for i in range(100):
            block_id = block_to_free.block_id
            allocator.free(block_to_free)
            assert block_to_free.block_id is None, i
            new_block = allocator.allocate_mutable_block(prev_block=None)
            assert new_block.block_id == block_id, i
            with pytest.raises(BlockAllocator.NoFreeBlocksError):
                allocator.allocate_mutable_block(prev_block=None)
            block_to_free = new_block

    @staticmethod
    @pytest.mark.parametrize('num_blocks', [1024])
    @pytest.mark.parametrize('block_size', [16])
    @pytest.mark.parametrize('seed', list(range(20)))
    def test_get_num_free_blocks(num_blocks: int, block_size: int, seed: int):
        random.seed(seed)
        allocator = PrefixCachingBlockAllocator(num_blocks=num_blocks, block_size=block_size)
        num_blocks_to_consume = random.randint(1, num_blocks - 1)
        token_ids = list(range(num_blocks_to_consume * block_size))
        chain = TestPrefixCachingBlockAllocator.create_immutable_chain(block_size=block_size, token_ids=token_ids, allocator=allocator)
        for i, block in enumerate(chain):
            assert allocator.get_num_free_blocks() == num_blocks - num_blocks_to_consume + i
            allocator.free(block)

    @staticmethod
    @pytest.mark.parametrize('num_blocks', [4])
    @pytest.mark.parametrize('block_size', [8])
    def test_prefix_caching_block_get_num_full_blocks_touched(num_blocks, block_size):
        """ Verify the allocator can correctly return the number of
        blocks touched, when there are cached prefixes.
        """
        allocator_src = PrefixCachingBlockAllocator(num_blocks=num_blocks, block_size=block_size)
        allocator_dst = PrefixCachingBlockAllocator(num_blocks=num_blocks, block_size=block_size)
        token_ids = list(range((num_blocks - 1) * block_size))
        cached_blocks = TestPrefixCachingBlockAllocator.create_immutable_chain(block_size=block_size, token_ids=token_ids, allocator=allocator_dst)
        blocks_to_swap_in = TestPrefixCachingBlockAllocator.create_immutable_chain(block_size=block_size, token_ids=token_ids, allocator=allocator_src)
        assert allocator_dst.get_num_full_blocks_touched(blocks_to_swap_in) == 0
        allocator_dst.free(cached_blocks[0])
        assert allocator_dst.get_num_full_blocks_touched(blocks_to_swap_in) == 1
        non_full_block = allocator_src.allocate_mutable_block(blocks_to_swap_in[-1])
        non_full_block.append_token_ids([0])
        blocks_to_swap_in.append(non_full_block)
        assert allocator_dst.get_num_full_blocks_touched(blocks_to_swap_in) == 1
        non_full_block.append_token_ids([0] * (block_size - 1))
        assert allocator_dst.get_num_full_blocks_touched(blocks_to_swap_in) == 2

    @staticmethod
    @pytest.mark.parametrize('num_blocks', [1024])
    @pytest.mark.parametrize('block_size', [16])
    @pytest.mark.parametrize('seed', list(range(20)))
    def test_get_num_free_blocks_shared(num_blocks: int, block_size: int, seed: int):
        """Verify sharing occurs by allocating two sequences that share prefixes
        and incrementally freeing blocks.
        """
        random.seed(seed)
        allocator = PrefixCachingBlockAllocator(num_blocks=num_blocks, block_size=block_size)
        num_blocks_to_consume = random.randint(1, num_blocks - 1)
        token_ids = list(range(num_blocks_to_consume * block_size))
        first_chain = TestPrefixCachingBlockAllocator.create_immutable_chain(block_size=block_size, token_ids=token_ids, allocator=allocator)
        second_chain = TestPrefixCachingBlockAllocator.create_immutable_chain(block_size=block_size, token_ids=token_ids, allocator=allocator)
        for i, block in enumerate(first_chain):
            assert allocator.get_num_free_blocks() == num_blocks - num_blocks_to_consume
            allocator.free(block)
        for i, block in enumerate(second_chain):
            assert allocator.get_num_free_blocks() == num_blocks - num_blocks_to_consume + i
            allocator.free(block)

    @staticmethod
    @pytest.mark.parametrize('num_blocks', [1024])
    @pytest.mark.parametrize('block_size', [16])
    @pytest.mark.parametrize('seed', list(range(20)))
    def test_get_common_computed_block_ids(num_blocks: int, block_size: int, seed: int):
        """Verify get_common_computed_block_ids could get correct result
        by create two immutable chain sharing prefix at specified pos,
        and compare whether we also could get right result
        from get_common_computed_block_ids.
        """
        random.seed(seed)
        allocator = PrefixCachingBlockAllocator(num_blocks=num_blocks * 2, block_size=block_size)
        num_blocks_to_consume = random.randint(1, num_blocks - 1)
        token_ids = list(range(num_blocks_to_consume * block_size))
        first_chain = TestPrefixCachingBlockAllocator.create_immutable_chain(block_size=block_size, token_ids=token_ids, allocator=allocator)
        zero_point = random.randint(1, len(token_ids) - 1)
        zero_point_blocks = zero_point // block_size
        token_ids[zero_point:] = [-1] * (len(token_ids) - zero_point)
        second_chain = TestPrefixCachingBlockAllocator.create_immutable_chain(block_size=block_size, token_ids=token_ids, allocator=allocator)
        first_computed_ids = [first_chain[i].block_id for i in range(num_blocks_to_consume)]
        second_computed_ids = [second_chain[i].block_id for i in range(num_blocks_to_consume)]
        res = allocator.get_common_computed_block_ids([first_computed_ids, second_computed_ids])
        assert len(res) == zero_point_blocks

    @staticmethod
    @pytest.mark.parametrize('num_blocks', [3])
    @pytest.mark.parametrize('block_size', [16])
    @pytest.mark.parametrize('seed', list(range(10)))
    def test_alloc_promotion(num_blocks: int, block_size: int, seed: int):
        random.seed(seed)
        allocator = PrefixCachingBlockAllocator(num_blocks=num_blocks, block_size=block_size)
        token_ids = list(range(block_size))
        block = allocator.allocate_immutable_block(prev_block=None, token_ids=token_ids)
        assert allocator._refcounter.get(block.block_id) == 1
        m = allocator.allocate_mutable_block(prev_block=None)
        block_id = m.block_id
        for i in range(block_size):
            m.append_token_ids([i])
        assert m.block_id == block.block_id
        assert block_id in allocator._hashless_allocator._free_block_indices
        assert allocator._refcounter.get(block.block_id) == 2

    @staticmethod
    @pytest.mark.parametrize('num_blocks', [3])
    @pytest.mark.parametrize('block_size', [16])
    @pytest.mark.parametrize('seed', list(range(10)))
    def test_eviction_alloc_mixed(num_blocks: int, block_size: int, seed: int):
        random.seed(seed)
        all_blocks_list = [i for i in range(num_blocks)]
        zero_ref = {i: 0 for i in range(num_blocks)}
        one_ref = {i: 1 for i in range(num_blocks)}
        allocator = PrefixCachingBlockAllocator(num_blocks=num_blocks, block_size=block_size)
        token_ids = list(range(num_blocks * block_size))
        assert list(allocator._hashless_allocator._free_block_indices) == all_blocks_list
        assert len(allocator._block_tracker.keys()) == num_blocks
        for block_id in range(num_blocks):
            assert not allocator._block_tracker[block_id].active
        assert len(allocator._cached_blocks.values()) == 0
        assert len(allocator.evictor.free_table.keys()) == 0
        assert allocator._refcounter._refcounts == zero_ref
        new_block = []
        for i in range(num_blocks):
            block = allocator.allocate_immutable_block(prev_block=None, token_ids=token_ids[block_size * i:block_size * (i + 1)])
            new_block.append(block)
        assert len(allocator._hashless_allocator._free_block_indices) == 0
        assert len(allocator._block_tracker.keys()) == num_blocks
        for block_id in range(num_blocks):
            assert allocator._block_tracker[block_id].active
        assert len(allocator._cached_blocks.values()) == num_blocks
        assert len(allocator.evictor.free_table.keys()) == 0
        assert allocator._refcounter._refcounts == one_ref
        for block in new_block:
            allocator.free(block)
        assert len(allocator._block_tracker.keys()) == num_blocks
        for block_id in range(num_blocks):
            assert not allocator._block_tracker[block_id].active
        assert len(allocator._hashless_allocator._free_block_indices) == 0
        assert list(allocator._cached_blocks.values()) == all_blocks_list
        assert list(allocator.evictor.free_table.keys()) == all_blocks_list
        assert allocator._refcounter._refcounts == zero_ref
        mutable = allocator.allocate_mutable_block(prev_block=None)
        assert mutable.block_id == 0
        assert mutable.content_hash is None
        assert allocator._block_tracker[0].active
        assert allocator._refcounter.get(0) == 1
        assert 0 not in allocator._cached_blocks
        assert 0 not in allocator.evictor
        allocator.free(mutable)
        assert not allocator._block_tracker[0].active
        assert allocator._refcounter._refcounts == zero_ref
        assert 0 not in allocator._cached_blocks
        assert 0 not in allocator.evictor
        assert 0 in allocator._hashless_allocator._free_block_indices
        block = allocator.allocate_immutable_block(prev_block=None, token_ids=token_ids[:block_size])
        assert block.block_id == 0
        assert len(allocator._hashless_allocator._free_block_indices) == 0
        assert allocator._block_tracker[0].active
        assert 0 in allocator._cached_blocks.values()
        assert allocator._refcounter.get(0) == 1
        assert 0 not in allocator.evictor
        mutable = allocator.allocate_mutable_block(prev_block=None)
        assert len(allocator._hashless_allocator._free_block_indices) == 0
        assert mutable.block_id not in allocator.evictor.free_table
        assert allocator._refcounter.get(mutable.block_id) == 1

    @staticmethod
    @pytest.mark.parametrize('num_blocks', [1024])
    @pytest.mark.parametrize('block_size', [16])
    @pytest.mark.parametrize('seed', list(range(20)))
    def test_eviction_order(num_blocks: int, block_size: int, seed: int):
        """This test case simulate the two chain created and free in order,
        and together they would exhaust the initial freed blocks.

        So the next block created after those two chain shall use the block
        from the first chain as that block has long access time.
        While first chain has two blocks, it shall pick up the last one, as
        it has larger token number.
        """
        random.seed(seed)
        allocator = PrefixCachingBlockAllocator(num_blocks=num_blocks, block_size=block_size)
        num_blocks_to_consume = num_blocks + 1
        token_ids = list(range(num_blocks_to_consume * block_size))
        num_blocks_in_first_chain = 2
        num_tokens_in_first_chain = block_size * num_blocks_in_first_chain
        first_chain = TestPrefixCachingBlockAllocator.create_immutable_chain(block_size=block_size, token_ids=token_ids[:num_tokens_in_first_chain], allocator=allocator)
        assert allocator.get_num_free_blocks() == num_blocks - num_blocks_in_first_chain
        blocks_ids = [block.block_id for block in first_chain]
        allocator.mark_blocks_as_accessed(blocks_ids, 1)
        second_chain = TestPrefixCachingBlockAllocator.create_immutable_chain(block_size=block_size, token_ids=token_ids[num_tokens_in_first_chain:-block_size], allocator=allocator)
        assert allocator.get_num_free_blocks() == 0
        assert len(first_chain) == num_blocks_in_first_chain
        last_block_id = first_chain[-1].block_id
        for i, block in enumerate(first_chain):
            allocator.free(block)
        blocks_ids = [block.block_id for block in second_chain]
        allocator.mark_blocks_as_accessed(blocks_ids, 2)
        for i, block in enumerate(second_chain):
            allocator.free(block)
        new_block = TestPrefixCachingBlockAllocator.create_immutable_chain(block_size=block_size, token_ids=token_ids[-block_size:], allocator=allocator)
        assert new_block[0].block_id == last_block_id

    @staticmethod
    def test_metric():
        block_size = 16
        allocator = PrefixCachingBlockAllocator(num_blocks=4, block_size=block_size)
        assert allocator.get_prefix_cache_hit_rate() == 0.0
        token_ids = list(range(block_size))
        allocator.allocate_immutable_block(prev_block=None, token_ids=token_ids)
        assert allocator.get_prefix_cache_hit_rate() == 0.0
        allocator.allocate_immutable_block(prev_block=None, token_ids=token_ids)
        assert allocator.get_prefix_cache_hit_rate() == 0.5
        for _ in range(2, 1005):
            allocator.allocate_immutable_block(prev_block=None, token_ids=token_ids)
        assert allocator.get_prefix_cache_hit_rate() > 0.99

    @staticmethod
    def test_touch_block():
        block_size = 16
        common_blocks = 4
        allocator = PrefixCachingBlockAllocator(num_blocks=8, block_size=block_size)
        common_token_ids = list(range(block_size * common_blocks))
        for _ in range(3):
            blocks = TestPrefixCachingBlockAllocator.create_immutable_chain(block_size=block_size, token_ids=common_token_ids, allocator=allocator)
            block_hashes = [block.content_hash for block in blocks]
            computed_block_ids = allocator.find_cached_blocks_prefix(block_hashes)
            assert len(computed_block_ids) == 0
        allocator.mark_blocks_as_computed([])
        computed_block_ids = allocator.find_cached_blocks_prefix(block_hashes=block_hashes)
        assert len(computed_block_ids) == common_blocks

    @staticmethod
    def test_find_cached_blocks_prefix():
        """
        This test verifies the behavior of find_cached_blocks_prefix.
        """
        block_size = 4
        num_blocks = 8
        total_test_blocks = 12
        allocator = PrefixCachingBlockAllocator(num_blocks=num_blocks, block_size=block_size)
        token_ids = list(range(total_test_blocks * block_size))
        block_tokens_seq1 = token_ids[:num_blocks * block_size]
        blocks_seq1 = TestPrefixCachingBlockAllocator.create_immutable_chain(block_size=block_size, token_ids=block_tokens_seq1, allocator=allocator)
        block_hashes_seq1 = [block.content_hash for block in blocks_seq1]
        allocator.mark_blocks_as_computed([])
        cached_blocks_seq1 = allocator.find_cached_blocks_prefix(block_hashes=block_hashes_seq1)
        assert len(cached_blocks_seq1) == num_blocks
        for block in blocks_seq1:
            allocator.free(block)
        cached_blocks = allocator.find_cached_blocks_prefix(block_hashes=block_hashes_seq1)
        assert len(cached_blocks) == num_blocks
        block_tokens_seq2 = token_ids[num_blocks * block_size:]
        blocks_seq2 = TestPrefixCachingBlockAllocator.create_immutable_chain(block_size=block_size, token_ids=block_tokens_seq2, allocator=allocator)
        block_hashes_seq2 = [block.content_hash for block in blocks_seq2]
        allocator.mark_blocks_as_computed([])
        cached_blocks = allocator.find_cached_blocks_prefix(block_hashes=block_hashes_seq2)
        assert len(cached_blocks) == len(blocks_seq2)
        num_evicted_blocks = len(blocks_seq2)
        cached_blocks = allocator.find_cached_blocks_prefix(block_hashes=block_hashes_seq1)
        assert len(cached_blocks) == len(blocks_seq1) - num_evicted_blocks

    @staticmethod
    @pytest.mark.parametrize('num_blocks', [10])
    @pytest.mark.parametrize('block_size', [16])
    def test_reset_prefix_cache(num_blocks: int, block_size: int):
        """This test case simulates the case of resetting the prefix cache."""
        allocator = PrefixCachingBlockAllocator(num_blocks=num_blocks, block_size=block_size)
        token_ids = list(range(3 * block_size))
        first_chain = TestPrefixCachingBlockAllocator.create_immutable_chain(block_size=block_size, token_ids=token_ids, allocator=allocator)
        second_chain = TestPrefixCachingBlockAllocator.create_immutable_chain(block_size=block_size, token_ids=token_ids, allocator=allocator)
        for block in first_chain:
            allocator.free(block)
        assert not allocator.reset_prefix_cache()
        assert allocator.get_prefix_cache_hit_rate() > 0.0
        for block in second_chain:
            allocator.free(block)
        assert allocator.reset_prefix_cache()
        assert allocator.get_prefix_cache_hit_rate() == 0.0

    @staticmethod
    def create_immutable_chain(block_size: int, token_ids: list[int], allocator: PrefixCachingBlockAllocator, extra_hash: Optional[int]=None) -> list[PrefixCachingBlock]:
        """Helper method which creates a chain of blocks.
        """
        blocks: list[Block] = []
        num_blocks = math.ceil(len(token_ids) / block_size)
        if num_blocks == 0:
            return []
        prev_block = None
        for block_number in range(0, num_blocks):
            block_token_ids = token_ids[block_number * block_size:(block_number + 1) * block_size]
            prev_block = allocator.allocate_immutable_block(prev_block=prev_block, token_ids=block_token_ids, extra_hash=extra_hash)
            blocks.append(prev_block)
        return blocks

class TestComputedBlocksTracker:

    @staticmethod
    def _get_mock_allocator():
        return MagicMock(spec=PrefixCachingBlockAllocator)

    @staticmethod
    def test_get_num_cached_tokens():
        """
        Test it correctly computes the number of cached tokens for a given
        sequence:

        - The cache token count is derived from the number of cached blocks.
        - The cache token count is updated when the allocator is updated.
        - When a sequence is removed, the cache token count should be updated
        accordingly.

        # TODO(rickyx): This behaviour for prefill sequence is a hack until
        we fix the computed blocks tracking.
        - The cache token count for prefill sequence doesn't change while
        the sequence is in continuous prefill (chunked prefill).
        """
        block_size = 4
        mock_allocator = TestComputedBlocksTracker._get_mock_allocator()
        tracker = ComputedBlocksTracker(allocator=mock_allocator, block_size=block_size, enable_caching=True)
        tokens = [0, 1, 2, 3, 4, 5]
        seq1 = create_dummy_sequence(request_id=0, token_ids=tokens, block_size=block_size)
        mock_allocator.find_cached_blocks_prefix.return_value = []
        assert tracker.get_num_cached_tokens(seq1) == 0
        mock_allocator.find_cached_blocks_prefix.return_value = [None]
        assert tracker.get_num_cached_tokens(seq1) == 0
        seq1.data.update_num_computed_tokens(len(tokens))
        assert not seq1.is_prefill()
        assert tracker.get_num_cached_tokens(seq1) == 4
        num_new_tokens = 3
        for i in range(num_new_tokens):
            seq1.append_token_id(i, {i: Logprob(logprob=0.0)})
        assert tracker.get_num_cached_tokens(seq1) == 4
        mock_allocator.find_cached_blocks_prefix.return_value = [None] * 2
        assert tracker.get_num_cached_tokens(seq1) == 8
        tracker.remove_seq(seq1.seq_id)
        seq1 = create_dummy_sequence(request_id=0, token_ids=tokens, block_size=block_size)
        mock_allocator.find_cached_blocks_prefix.return_value = []
        assert tracker.get_num_cached_tokens(seq1) == 0

    @staticmethod
    def test_correct_block_hash():
        """
        Test that the block hash is correctly computed for a sequence (should
        match the underlying block allocator's block hash). So the number of
        cached tokens is correctly retrieved.
        """
        block_size = 4
        allocator = CpuGpuBlockAllocator.create(allocator_type='prefix_caching', num_gpu_blocks=16, num_cpu_blocks=16, block_size=block_size)
        gpu_allocator = allocator._allocators[Device.GPU]
        tracker = ComputedBlocksTracker(allocator=allocator, block_size=block_size, enable_caching=True)
        tokens = list(range(block_size * 4))
        seq = create_dummy_sequence(request_id=0, token_ids=tokens, block_size=block_size)
        _ = TestPrefixCachingBlockAllocator.create_immutable_chain(block_size=block_size, token_ids=tokens, allocator=gpu_allocator)
        allocator.mark_blocks_as_computed([])
        assert tracker.get_num_cached_tokens(seq) == len(tokens)

    @staticmethod
    def test_correct_extra_hash():
        """
        Test that the block hash is correctly computed based on the extra hash,
        ensuring it matches the allocator's block hash, specifically for the
        LoRA case, and that the correct number of cached tokens is retrieved.
        """
        block_size = 4
        allocator = CpuGpuBlockAllocator.create(allocator_type='prefix_caching', num_gpu_blocks=16, num_cpu_blocks=16, block_size=block_size)
        gpu_allocator = allocator._allocators[Device.GPU]
        tracker = ComputedBlocksTracker(allocator=allocator, block_size=block_size, enable_caching=True)
        tokens = list(range(block_size * 4))
        lora_seq = create_dummy_lora_sequence(request_id=0, token_ids=tokens, block_size=block_size, lora_int_id=1)
        _ = TestPrefixCachingBlockAllocator.create_immutable_chain(block_size=block_size, token_ids=tokens, allocator=gpu_allocator, extra_hash=lora_seq.extra_hash())
        allocator.mark_blocks_as_computed([])
        seq = create_dummy_sequence(request_id=1, token_ids=tokens, block_size=block_size)
        different_lora_seq = create_dummy_lora_sequence(request_id=2, token_ids=tokens, block_size=block_size, lora_int_id=2)
        assert tracker.get_num_cached_tokens(seq) == 0
        assert tracker.get_num_cached_tokens(different_lora_seq) == 0
        assert tracker.get_num_cached_tokens(lora_seq) == len(tokens)