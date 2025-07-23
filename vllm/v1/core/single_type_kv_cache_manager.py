from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Callable
from vllm.utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHash, KVCacheBlock
from vllm.v1.kv_cache_interface import ChunkedLocalAttentionSpec, FullAttentionSpec, KVCacheSpec, MambaSpec, SlidingWindowSpec
from vllm.v1.request import Request

class SingleTypeKVCacheManager(ABC):
    """
    An abstract base class for a manager that handle the kv cache management 
    logic of one specific type of attention layer.
    """

    def __init__(self, kv_cache_spec: KVCacheSpec, block_pool: BlockPool, kv_cache_group_id: int, caching_hash_fn: Callable) -> None:
        """
        Initializes the SpecializedManager.
        Args:
            kv_cache_spec: The kv_cache_spec for this manager.
            block_pool: The block pool.
            kv_cache_group_id: The id of the kv cache group of this manager.
            caching_hash_fn: The caching hash function.
        """
        self.block_size = kv_cache_spec.block_size
        self.kv_cache_spec = kv_cache_spec
        self.block_pool = block_pool
        self.req_to_blocks: defaultdict[str, list[KVCacheBlock]] = defaultdict(list)
        self.num_cached_block: dict[str, int] = {}
        self.caching_hash_fn = caching_hash_fn
        self.kv_cache_group_id = kv_cache_group_id
        self._null_block = block_pool.null_block

    def get_num_blocks_to_allocate(self, request_id: str, num_tokens: int, new_computed_blocks: list[KVCacheBlock]) -> int:
        """
        Get the number of blocks needed to be allocated for the request.

        Args:
            request_id: The request ID.
            num_tokens: The total number of tokens that need a slot (including 
                tokens that are already allocated).
            new_computed_blocks: The new computed blocks just hitting the
                prefix caching.

        Returns:
            The number of blocks.
        """
        num_required_blocks = cdiv(num_tokens, self.block_size)
        num_new_blocks = num_required_blocks - len(new_computed_blocks) - len(self.req_to_blocks[request_id])
        num_evictable_computed_blocks = sum((blk.ref_cnt == 0 and (not blk.is_null) for blk in new_computed_blocks))
        return num_new_blocks + num_evictable_computed_blocks

    def save_new_computed_blocks(self, request_id: str, new_computed_blocks: list[KVCacheBlock]) -> None:
        """
        Add the new computed blocks to the request.

        Args:
            request_id: The request ID.
            new_computed_blocks: The new computed blocks just hitting the
                prefix cache.
        """
        if request_id not in self.num_cached_block:
            req_blocks = self.req_to_blocks[request_id]
            assert len(req_blocks) == 0
            req_blocks.extend(new_computed_blocks)
            self.num_cached_block[request_id] = len(new_computed_blocks)
        else:
            assert len(new_computed_blocks) == 0

    def allocate_new_blocks(self, request_id: str, num_tokens: int) -> list[KVCacheBlock]:
        """
        Allocate new blocks for the request to give it at least `num_tokens` 
        token slots.

        Args:
            request_id: The request ID.
            num_tokens: The total number of tokens that need a slot (including 
                tokens that are already allocated).

        Returns:
            The new allocated blocks.
        """
        req_blocks = self.req_to_blocks[request_id]
        num_required_blocks = cdiv(num_tokens, self.block_size)
        num_new_blocks = num_required_blocks - len(req_blocks)
        if num_new_blocks <= 0:
            return []
        else:
            new_blocks = self.block_pool.get_new_blocks(num_new_blocks)
            req_blocks.extend(new_blocks)
            return new_blocks

    def cache_blocks(self, request: Request, block_hashes: list[BlockHash], num_tokens: int) -> None:
        """
        Cache the blocks for the request.

        Args:
            request: The request.
            block_hashes: The block hashes of the request.
            num_tokens: The total number of tokens that need to be cached 
                (including tokens that are already cached).
        """
        num_cached_blocks = self.num_cached_block[request.request_id]
        num_full_blocks = num_tokens // self.block_size
        self.block_pool.cache_full_blocks(request=request, blocks=self.req_to_blocks[request.request_id], block_hashes=block_hashes, num_cached_blocks=num_cached_blocks, num_full_blocks=num_full_blocks, block_size=self.block_size, kv_cache_group_id=self.kv_cache_group_id, hash_fn=self.caching_hash_fn)
        self.num_cached_block[request.request_id] = num_full_blocks

    def free(self, request_id: str) -> None:
        """
        Free the blocks for the request.

        Args:
            request_id: The request ID.
        """
        req_blocks = self.req_to_blocks.pop(request_id, [])
        ordered_blocks = reversed(req_blocks)
        self.block_pool.free_blocks(ordered_blocks)
        self.num_cached_block.pop(request_id, None)

    @abstractmethod
    def get_num_common_prefix_blocks(self, request_id: str, num_running_requests: int) -> int:
        """
        Get the number of common prefix blocks for a request.

        Args:
            request_id: The request ID.
            block_hashes: The block hashes of the request.

        Returns:
            The number of common prefix blocks.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def find_longest_cache_hit(cls, block_hashes: list[BlockHash], max_length: int, kv_cache_group_ids: list[int], block_pool: BlockPool, kv_cache_spec: KVCacheSpec, use_eagle: bool) -> tuple[list[KVCacheBlock], ...]:
        """
        Get the longest cache hit prefix of the blocks that is not longer than 
        `max_length`. The prefix should be a common prefix hit for all the 
        kv cache groups in `kv_cache_group_ids`. If no cache hit is found, 
        return an empty list. 
        If eagle is enabled, drop the last matched block to force recompute the 
        last block to get the required hidden states for eagle drafting head. 
        Need to be customized for each attention type.

        Args:
            block_hashes: The block hashes of the request.
            max_length: The maximum length of the cache hit prefix.
            kv_cache_group_ids: The ids of the kv cache groups.
            block_pool: The block pool.
            kv_cache_spec: The kv cache spec.
            use_eagle: Whether to use eagle.

        Returns:
            A list of cached blocks with skipped blocks replaced by null block
            for each kv cache group in `kv_cache_group_ids`.
            Return a list of length `len(kv_cache_group_ids)`, where the i-th
            element is a list of cached blocks for the i-th kv cache group
            in `kv_cache_group_ids`.
            For example, sliding window manager should return a list like
            ([NULL, NULL, KVCacheBlock(7), KVCacheBlock(8)]) for block size 4
            and sliding window 8 and len(kv_cache_group_ids) = 1.
        """
        raise NotImplementedError

    @abstractmethod
    def remove_skipped_blocks(self, request_id: str, num_computed_tokens: int) -> None:
        """
        Remove the blocks that are no longer needed from `blocks` and free the 
        blocks. The removed blocks should be replaced by null_block.
        Need to be customized for each attention type.

        Args:
            request_id: The request ID.
            num_computed_tokens: The number of tokens that have been computed.
        """
        raise NotImplementedError

class FullAttentionManager(SingleTypeKVCacheManager):

    @classmethod
    def find_longest_cache_hit(cls, block_hashes: list[BlockHash], max_length: int, kv_cache_group_ids: list[int], block_pool: BlockPool, kv_cache_spec: KVCacheSpec, use_eagle: bool) -> tuple[list[KVCacheBlock], ...]:
        assert isinstance(kv_cache_spec, (FullAttentionSpec, ChunkedLocalAttentionSpec)), 'FullAttentionManager can only be used for full attention and chunked local attention groups'
        computed_blocks: tuple[list[KVCacheBlock], ...] = tuple(([] for _ in range(len(kv_cache_group_ids))))
        max_num_blocks = max_length // kv_cache_spec.block_size
        for i, block_hash in zip(range(max_num_blocks), block_hashes):
            if (cached_block := block_pool.get_cached_block(block_hash, kv_cache_group_ids)):
                for computed, cached in zip(computed_blocks, cached_block):
                    computed.append(cached)
            else:
                break
        if use_eagle and computed_blocks[0]:
            for computed in computed_blocks:
                computed.pop()
        return computed_blocks

    def remove_skipped_blocks(self, request_id: str, num_computed_tokens: int) -> None:
        pass

    def get_num_common_prefix_blocks(self, request_id: str, num_running_requests: int) -> int:
        blocks = self.req_to_blocks[request_id]
        num_common_blocks = 0
        for block in blocks:
            if block.ref_cnt == num_running_requests:
                num_common_blocks += 1
            else:
                break
        return num_common_blocks

class SlidingWindowManager(SingleTypeKVCacheManager):

    def __init__(self, kv_cache_spec: SlidingWindowSpec, block_pool: BlockPool, **kwargs) -> None:
        super().__init__(kv_cache_spec, block_pool, **kwargs)
        self.sliding_window = kv_cache_spec.sliding_window
        self._null_block = block_pool.null_block

    @classmethod
    def find_longest_cache_hit(cls, block_hashes: list[BlockHash], max_length: int, kv_cache_group_ids: list[int], block_pool: BlockPool, kv_cache_spec: KVCacheSpec, use_eagle: bool) -> tuple[list[KVCacheBlock], ...]:
        assert isinstance(kv_cache_spec, SlidingWindowSpec), 'SlidingWindowManager can only be used for sliding window groups'
        sliding_window_contiguous_blocks = cdiv(kv_cache_spec.sliding_window - 1, kv_cache_spec.block_size)
        if use_eagle:
            sliding_window_contiguous_blocks += 1
        max_num_blocks = max_length // kv_cache_spec.block_size
        computed_blocks = tuple(([block_pool.null_block] * max_num_blocks for _ in range(len(kv_cache_group_ids))))
        num_contiguous_blocks = 0
        match_found = False
        for i in range(max_num_blocks - 1, -1, -1):
            if (cached_block := block_pool.get_cached_block(block_hashes[i], kv_cache_group_ids)):
                for computed, cached in zip(computed_blocks, cached_block):
                    computed[i] = cached
                num_contiguous_blocks += 1
                if num_contiguous_blocks >= sliding_window_contiguous_blocks:
                    for computed in computed_blocks:
                        del computed[i + num_contiguous_blocks:]
                    match_found = True
                    break
            else:
                num_contiguous_blocks = 0
        if not match_found:
            for computed in computed_blocks:
                del computed[num_contiguous_blocks:]
        if use_eagle and computed_blocks[0]:
            for computed in computed_blocks:
                computed.pop()
        return computed_blocks

    def remove_skipped_blocks(self, request_id: str, num_computed_tokens: int) -> None:
        last_useful_token = num_computed_tokens - self.sliding_window + 1
        last_useful_block = last_useful_token // self.block_size
        blocks = self.req_to_blocks[request_id]
        removed_blocks: list[KVCacheBlock] = []
        for i in range(last_useful_block - 1, -1, -1):
            if blocks[i] == self._null_block:
                break
            removed_blocks.append(blocks[i])
            blocks[i] = self._null_block
        self.block_pool.free_blocks(removed_blocks)

    def get_num_common_prefix_blocks(self, request_id: str, num_running_requests: int) -> int:
        """
        NOTE(Chen): The prefix blocks are null blocks for sliding window layers.
        So it's not correct to count ref_cnt like FullAttentionManager. Return 
        0 here for correctness. Need to support cascade attention + sliding 
        window in the future.
        """
        return 0

class ChunkedLocalAttentionManager(SingleTypeKVCacheManager):

    def __init__(self, kv_cache_spec: ChunkedLocalAttentionSpec, block_pool: BlockPool, **kwargs) -> None:
        super().__init__(kv_cache_spec, block_pool, **kwargs)
        self.attention_chunk_size = kv_cache_spec.attention_chunk_size
        self._null_block = block_pool.null_block

    @classmethod
    def find_longest_cache_hit(cls, block_hashes: list[BlockHash], max_length: int, kv_cache_group_ids: list[int], block_pool: BlockPool, kv_cache_spec: KVCacheSpec, use_eagle: bool) -> tuple[list[KVCacheBlock], ...]:
        """
        For chunked local attention, we need to find the longest cache hit
        prefix of the blocks that is not longer than `max_length`. The prefix
        should be a common prefix hit for all the kv cache groups in
        `kv_cache_group_ids`. If no cache hit is found, return an empty list.
        note we mark as computed if the whole block is outside of the local 
        window, and set the block as null. Examples:

        1. Attention chunk size of 8, block size of 4, max length of 15
        for next token at 15th (zero-indexed), 8th - 14th tokens are in 
        the window(needs lookup), 0th - 7th are not in the window, 
        so they are already marked as computed. We check the complete 
        block3 (8th - 11th tokens), Assume block 3 is hit, we will return 
        [null, null, block 3], otherwise, we return [null, null]

        2. Attention chunk size of 8, block size of 4, max length of 16
        for next token at 16th (zero-indexed), 0th - 15th tokens are not 
        in the window, so they are already marked as computed. 
        we return 4 blocks[null, null, null, null]

        Args:
            block_hashes: The block hashes of the request.
            max_length: The maximum length of the cache hit prefix.
            kv_cache_group_ids: The ids of the kv cache groups.
            block_pool: The block pool.
            kv_cache_spec: The kv cache spec.
            use_eagle: Whether to use eagle.

        Returns:
            A list of cached blocks
        """
        assert isinstance(kv_cache_spec, ChunkedLocalAttentionSpec), 'ChunkedLocalAttentionManager can only be used for ' + 'chunked local attention groups'
        assert use_eagle is False, 'Hybrid KV cache is not supported for ' + 'eagle + chunked local attention.'
        max_num_blocks = max_length // kv_cache_spec.block_size
        if max_length > 0:
            local_attention_start_idx = max_length // kv_cache_spec.attention_chunk_size * kv_cache_spec.attention_chunk_size
        else:
            local_attention_start_idx = 0
        local_attention_start_block_idx = local_attention_start_idx // kv_cache_spec.block_size
        computed_blocks: tuple[list[KVCacheBlock], ...] = tuple(([block_pool.null_block] * local_attention_start_block_idx for _ in range(len(kv_cache_group_ids))))
        for i in range(local_attention_start_block_idx, max_num_blocks):
            block_hash = block_hashes[i]
            if (cached_block := block_pool.get_cached_block(block_hash, kv_cache_group_ids)):
                for computed, cached in zip(computed_blocks, cached_block):
                    computed.append(cached)
            else:
                break
        return computed_blocks

    def remove_skipped_blocks(self, request_id: str, num_computed_tokens: int) -> None:
        num_cached_block = self.num_cached_block.get(request_id, 0)
        local_attention_start_idx = num_computed_tokens // self.attention_chunk_size * self.attention_chunk_size
        first_useful_block_idx = local_attention_start_idx // self.block_size
        if num_cached_block > 0:
            first_useful_block_idx = min(first_useful_block_idx, num_cached_block - 1)
        blocks = self.req_to_blocks[request_id]
        removed_blocks: list[KVCacheBlock] = []
        for i in range(first_useful_block_idx - 1, -1, -1):
            if blocks[i] == self._null_block:
                break
            removed_blocks.append(blocks[i])
            blocks[i] = self._null_block
        self.block_pool.free_blocks(removed_blocks)

    def get_num_common_prefix_blocks(self, request_id: str, num_running_requests: int) -> int:
        """
        cascade attention is not supported by chunked local attention.
        """
        return 0

class MambaManager(SingleTypeKVCacheManager):

    @classmethod
    def find_longest_cache_hit(cls, block_hashes: list[BlockHash], max_length: int, kv_cache_group_ids: list[int], block_pool: BlockPool, kv_cache_spec: KVCacheSpec, use_eagle: bool) -> tuple[list[KVCacheBlock], ...]:
        assert isinstance(kv_cache_spec, MambaSpec), 'MambaManager can only be used for mamba groups'
        computed_blocks: tuple[list[KVCacheBlock], ...] = tuple(([] for _ in range(len(kv_cache_group_ids))))
        return computed_blocks

    def remove_skipped_blocks(self, request_id: str, num_computed_tokens: int) -> None:
        pass

    def get_num_common_prefix_blocks(self, request_id: str, num_running_requests: int) -> int:
        return 0

    def allocate_new_blocks(self, request_id: str, num_tokens: int) -> list[KVCacheBlock]:
        new_blocks = super().allocate_new_blocks(request_id, num_tokens)
        assert len(self.req_to_blocks[request_id]) == 1, 'MambaManager should only allocate 1 block for each request.'
        return new_blocks
spec_manager_map: dict[type[KVCacheSpec], type[SingleTypeKVCacheManager]] = {FullAttentionSpec: FullAttentionManager, SlidingWindowSpec: SlidingWindowManager, ChunkedLocalAttentionSpec: ChunkedLocalAttentionManager, MambaSpec: MambaManager}

def get_manager_for_kv_cache_spec(kv_cache_spec: KVCacheSpec, **kwargs) -> SingleTypeKVCacheManager:
    manager_class = spec_manager_map[type(kv_cache_spec)]
    manager = manager_class(kv_cache_spec, **kwargs)
    return manager