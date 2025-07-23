from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, Iterable, List, Optional
from typing import Sequence as GenericSequence
from typing import Set, Tuple, Union
from vllm.config import CacheConfig, LoRAConfig, SchedulerConfig
from vllm.core.interfaces import AllocStatus, BlockSpaceManager
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sequence import Sequence, SequenceData, SequenceGroup, SequenceGroupBase, SequenceGroupMetadata, SequenceGroupMetadataDelta, SequenceStage, SequenceStatus
from vllm.utils import Device, PyObjectCache
import enum
import os
import random
import time
logger = init_logger(__name__)
ENABLE_ARTIFICIAL_PREEMPT = bool(os.getenv('VLLM_TEST_ENABLE_ARTIFICIAL_PREEMPT', False))
ARTIFICIAL_PREEMPTION_PROB = 0.5
ARTIFICIAL_PREEMPTION_MAX_CNT = 500

class PreemptionMode(enum.Enum):
    """Preemption modes.

    1. Swapping: Swap out the blocks of the preempted sequences to CPU memory
    and swap them back in when the sequences are resumed.
    2. Recomputation: Discard the blocks of the preempted sequences and
    recompute them when the sequences are resumed, treating the sequences as
    new prompts.
    """
    SWAP = enum.auto()
    RECOMPUTE = enum.auto()

@dataclass
class SchedulingBudget:
    """The available slots for scheduling.

    TODO(sang): Right now, the budget is request_id-aware meaning it can ignore
    budget update from the same request_id. It is because in normal scheduling
    path, we update RUNNING num_seqs ahead of time, meaning it could be
    updated more than once when scheduling RUNNING requests. Since this won't
    happen if we only have chunked prefill scheduling, we can remove this
    feature from the API when chunked prefill is enabled by default.
    """
    token_budget: int
    max_num_seqs: int
    _request_ids_num_batched_tokens: Set[str] = field(default_factory=set)
    _request_ids_num_curr_seqs: Set[str] = field(default_factory=set)
    _num_cached_tokens: int = 0
    _num_batched_tokens: int = 0
    _num_curr_seqs: int = 0

    def can_schedule(self, *, num_new_tokens: int, num_new_seqs: int):
        assert num_new_tokens >= 0
        assert num_new_seqs != 0
        return self.num_batched_tokens + num_new_tokens <= self.token_budget and self.num_curr_seqs + num_new_seqs <= self.max_num_seqs

    def remaining_token_budget(self):
        return self.token_budget - self.num_batched_tokens

    def add_num_batched_tokens(self, req_id: str, num_batched_tokens: int, num_cached_tokens: int=0):
        if req_id in self._request_ids_num_batched_tokens:
            return
        assert num_cached_tokens >= 0
        assert num_batched_tokens >= 0
        self._request_ids_num_batched_tokens.add(req_id)
        self._num_batched_tokens += num_batched_tokens
        self._num_cached_tokens += num_cached_tokens

    def subtract_num_batched_tokens(self, req_id: str, num_batched_tokens: int):
        if req_id in self._request_ids_num_batched_tokens:
            self._request_ids_num_batched_tokens.remove(req_id)
            self._num_batched_tokens -= num_batched_tokens

    def add_num_seqs(self, req_id: str, num_curr_seqs: int):
        if req_id in self._request_ids_num_curr_seqs:
            return
        self._request_ids_num_curr_seqs.add(req_id)
        self._num_curr_seqs += num_curr_seqs

    def subtract_num_seqs(self, req_id: str, num_curr_seqs: int):
        if req_id in self._request_ids_num_curr_seqs:
            self._request_ids_num_curr_seqs.remove(req_id)
            self._num_curr_seqs -= num_curr_seqs

    @property
    def num_batched_tokens(self):
        return self._num_batched_tokens

    @property
    def num_curr_seqs(self):
        return self._num_curr_seqs

    @property
    def num_cached_tokens(self):
        return self._num_cached_tokens

@dataclass
class ScheduledSequenceGroup:
    seq_group: SequenceGroup
    token_chunk_size: int

@dataclass
class SchedulerOutputs:
    """The scheduling decision made from a scheduler."""
    scheduled_seq_groups: GenericSequence[ScheduledSequenceGroup]
    num_prefill_groups: int
    num_batched_tokens: int
    blocks_to_swap_in: List[Tuple[int, int]]
    blocks_to_swap_out: List[Tuple[int, int]]
    blocks_to_copy: List[Tuple[int, int]]
    ignored_seq_groups: List[SequenceGroup]
    num_lookahead_slots: int
    running_queue_size: int
    preempted: int

    def __post_init__(self):
        assert not (self.blocks_to_swap_in and self.blocks_to_swap_out)
        self.num_loras: int = len(self.lora_requests)
        if self.num_loras > 0:
            self._sort_by_lora_ids()
        self.num_prompt_adapters: int = len(self.prompt_adapter_requests)

    def is_empty(self) -> bool:
        return not self.scheduled_seq_groups and (not self.blocks_to_swap_in) and (not self.blocks_to_swap_out) and (not self.blocks_to_copy)

    def _sort_by_lora_ids(self):
        assert 0 <= self.num_prefill_groups <= len(self.scheduled_seq_groups)

        def key_fn(group: ScheduledSequenceGroup):
            key = (group.seq_group.lora_int_id, group.seq_group.request_id)
            if 0 < self.num_prefill_groups < len(self.scheduled_seq_groups):
                return (not group.seq_group.is_prefill(), *key)
            return key
        self.scheduled_seq_groups = sorted(self.scheduled_seq_groups, key=key_fn)

    @property
    def lora_requests(self) -> Set[LoRARequest]:
        return {g.seq_group.lora_request for g in self.scheduled_seq_groups if g.seq_group.lora_request is not None}

    @property
    def prompt_adapter_requests(self) -> Set[PromptAdapterRequest]:
        return {g.seq_group.prompt_adapter_request for g in self.scheduled_seq_groups if g.seq_group.prompt_adapter_request is not None}

@dataclass
class SchedulerRunningOutputs:
    """The requests that are scheduled from a running queue.

    Could contain prefill (prefill that's chunked) or decodes. If there's not
    enough memory, it can be preempted (for recompute) or swapped out.
    """
    decode_seq_groups: List[ScheduledSequenceGroup]
    prefill_seq_groups: List[ScheduledSequenceGroup]
    preempted: List[SequenceGroup]
    swapped_out: List[SequenceGroup]
    blocks_to_swap_out: List[Tuple[int, int]]
    blocks_to_copy: List[Tuple[int, int]]
    num_lookahead_slots: int
    decode_seq_groups_list: List[SequenceGroup]
    prefill_seq_groups_list: List[SequenceGroup]

    @classmethod
    def create_empty(cls) -> 'SchedulerRunningOutputs':
        return SchedulerRunningOutputs(decode_seq_groups=[], prefill_seq_groups=[], preempted=[], swapped_out=[], blocks_to_swap_out=[], blocks_to_copy=[], num_lookahead_slots=0, decode_seq_groups_list=[], prefill_seq_groups_list=[])

@dataclass
class SchedulerSwappedInOutputs:
    """The requests that are scheduled from a swap queue.

    Could contain prefill (prefill that's chunked) or decodes.
    """
    decode_seq_groups: List[ScheduledSequenceGroup]
    prefill_seq_groups: List[ScheduledSequenceGroup]
    blocks_to_swap_in: List[Tuple[int, int]]
    blocks_to_copy: List[Tuple[int, int]]
    num_lookahead_slots: int
    infeasible_seq_groups: List[SequenceGroup]

    @classmethod
    def create_empty(cls) -> 'SchedulerSwappedInOutputs':
        return SchedulerSwappedInOutputs(decode_seq_groups=[], prefill_seq_groups=[], blocks_to_swap_in=[], blocks_to_copy=[], num_lookahead_slots=0, infeasible_seq_groups=[])

@dataclass
class SchedulerPrefillOutputs:
    """The requests that are scheduled from a waiting queue.

    Could contain a fresh prefill requests or preempted requests that need
    to be recomputed from scratch.
    """
    seq_groups: List[ScheduledSequenceGroup]
    ignored_seq_groups: List[SequenceGroup]
    num_lookahead_slots: int

    @classmethod
    def create_empty(cls) -> 'SchedulerPrefillOutputs':
        return SchedulerPrefillOutputs(seq_groups=[], ignored_seq_groups=[], num_lookahead_slots=0)

def seq_group_metadata_builder():
    return SequenceGroupMetadata(request_id='', is_prompt=False, seq_data={}, sampling_params=None, block_tables={})

def scheduler_running_outputs_builder():
    return SchedulerRunningOutputs(decode_seq_groups=[], prefill_seq_groups=[], preempted=[], swapped_out=[], blocks_to_swap_out=[], blocks_to_copy=[], num_lookahead_slots=0, prefill_seq_groups_list=[], decode_seq_groups_list=[])

def scheduled_seq_group_builder():
    return ScheduledSequenceGroup(SequenceGroup.__new__(SequenceGroup), token_chunk_size=0)

@dataclass
class PartialPrefillMetadata:
    """Holds information about the partial prefills that are currently running
    during a single iteration of the Scheduler.
    When chunked prefill is enabled, we allow a certain number of seqs to be
    partially prefilled during each iteration. Having multiple partial prefills
    in flight allows us to minimize TTFT and avoid decode starvation in cases
    where a single sequence group with a very large prompt blocks the queue for
    too many iterations.
    The number of long prefill requests is limited so that smaller
    requests may jump the queue in front of them and get to the decode
    phase faster.
    """
    schedulable_prefills: int
    long_prefills: int
    scheduler_config: SchedulerConfig

    def can_schedule(self, seq_group: SequenceGroup) -> bool:
        """When concurrent partial prefills are enabled,
        we limit the number of long requests and only accept
        shorter requests from the queue while running them
        concurrently"""
        return not (seq_group.first_seq.get_num_new_tokens() > self.scheduler_config.long_prefill_token_threshold and self.long_prefills >= self.scheduler_config.max_long_partial_prefills and (self.scheduler_config.max_num_partial_prefills > 1))

    def maybe_increment_partial_prefills(self, seq_group: SequenceGroup) -> None:
        if seq_group.first_seq.get_num_new_tokens() > self.scheduler_config.long_prefill_token_threshold:
            self.long_prefills += 1

    @classmethod
    def from_queues(cls, running: Deque[SequenceGroup], waiting: Deque[SequenceGroup], scheduler_config: SchedulerConfig) -> 'PartialPrefillMetadata':
        """Create a PartialPrefillMetadata object from the current state of
        the scheduler's queues.
        This accounts for the currently running prefill requests, and peeks into
        the waiting queue to see if there are more prefills to potentially be
        scheduled during this iteration."""
        prefills = 0
        long_prefills = 0
        waiting_long_prefills = 0
        for sg in running:
            if sg.first_seq.data.stage == SequenceStage.PREFILL:
                prefills += 1
                if sg.first_seq.get_num_new_tokens() > scheduler_config.long_prefill_token_threshold:
                    long_prefills += 1
        for sg in waiting:
            if prefills >= scheduler_config.max_num_partial_prefills:
                break
            if sg.first_seq.get_num_new_tokens() > scheduler_config.long_prefill_token_threshold:
                if long_prefills + waiting_long_prefills >= scheduler_config.max_long_partial_prefills:
                    continue
                waiting_long_prefills += 1
            prefills += 1
        return PartialPrefillMetadata(schedulable_prefills=min(prefills, scheduler_config.max_num_partial_prefills), long_prefills=long_prefills, scheduler_config=scheduler_config)

class Scheduler:

    def __init__(self, scheduler_config: SchedulerConfig, cache_config: CacheConfig, lora_config: Optional[LoRAConfig], pipeline_parallel_size: int=1, output_proc_callback: Optional[Callable]=None) -> None:
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config
        self.lora_config = lora_config
        version = 'selfattn'
        if self.scheduler_config.runner_type == 'pooling' or self.cache_config.is_attention_free:
            version = 'placeholder'
        BlockSpaceManagerImpl = BlockSpaceManager.get_block_space_manager_class(version)
        num_gpu_blocks = cache_config.num_gpu_blocks
        if num_gpu_blocks:
            num_gpu_blocks //= pipeline_parallel_size
        num_cpu_blocks = cache_config.num_cpu_blocks
        if num_cpu_blocks:
            num_cpu_blocks //= pipeline_parallel_size
        self.block_manager = BlockSpaceManagerImpl(block_size=self.cache_config.block_size, num_gpu_blocks=num_gpu_blocks, num_cpu_blocks=num_cpu_blocks, sliding_window=self.cache_config.sliding_window, enable_caching=self.cache_config.enable_prefix_caching)
        self.waiting: Deque[SequenceGroup] = deque()
        self.running: Deque[SequenceGroup] = deque()
        self.swapped: Deque[SequenceGroup] = deque()
        self._finished_requests_ids: List[str] = list()
        self.prev_time = 0.0
        self.prev_prompt = False
        self.last_prompt_latency = 0.0
        self.user_specified_preemption_mode = scheduler_config.preemption_mode
        self.enable_artificial_preemption = ENABLE_ARTIFICIAL_PREEMPT
        self.artificial_preempt_cnt = ARTIFICIAL_PREEMPTION_MAX_CNT if self.enable_artificial_preemption else 0
        self.num_cumulative_preemption: int = 0
        self._seq_group_metadata_cache: List[PyObjectCache] = []
        self._scheduler_running_outputs_cache: List[PyObjectCache] = []
        self._scheduled_seq_group_cache: List[PyObjectCache] = []
        self.output_proc_callback = output_proc_callback
        self.use_async_output_proc = self.output_proc_callback is not None
        self.num_cache_iters = 2 if self.use_async_output_proc else 1
        self.cache_id = 0
        for i in range(self.num_cache_iters):
            self._seq_group_metadata_cache.append(PyObjectCache(seq_group_metadata_builder))
            self._scheduler_running_outputs_cache.append(PyObjectCache(scheduler_running_outputs_builder))
            self._scheduled_seq_group_cache.append(PyObjectCache(scheduled_seq_group_builder))
        self._async_stopped: List[SequenceGroup] = []
        self.partial_prefill_budget_lookup_list = [0] * (self.scheduler_config.max_num_partial_prefills + 1)
        self.partial_prefill_budget_lookup_list[0] = scheduler_config.max_num_batched_tokens
        for i in range(1, self.scheduler_config.max_num_partial_prefills + 1):
            self.partial_prefill_budget_lookup_list[i] = scheduler_config.max_num_batched_tokens // i

    @property
    def next_cache_id(self):
        return (self.cache_id + 1) % self.num_cache_iters

    @property
    def lora_enabled(self) -> bool:
        return bool(self.lora_config)

    @property
    def num_decoding_tokens_per_seq(self) -> int:
        """The number of new tokens."""
        return 1

    def add_seq_group(self, seq_group: SequenceGroup) -> None:
        self.waiting.append(seq_group)

    def _add_seq_group_to_running(self, seq_group: SequenceGroup) -> None:
        self.running.append(seq_group)

    def _add_seq_group_to_swapped(self, seq_group: SequenceGroup) -> None:
        self.swapped.append(seq_group)

    def abort_seq_group(self, request_id: Union[str, Iterable[str]], seq_id_to_seq_group: Optional[Dict[str, SequenceGroupBase]]=None) -> None:
        """Aborts a sequence group with the given ID.

        Check if the sequence group with the given ID
            is present in any of the state queue.
        If present, remove the sequence group from the state queue.
            Also, if any of the sequences in the sequence group is not finished,
                free the sequence with status `FINISHED_ABORTED`.
        Otherwise, do nothing.

        Args:
            request_id: The ID(s) of the sequence group to abort.
            seq_id_to_seq_group: helper for groups with n>1
        """
        if isinstance(request_id, str):
            request_id = (request_id,)
        request_ids = set(request_id)
        seq_id_to_seq_group = seq_id_to_seq_group or {}
        for state_queue in [self.waiting, self.running, self.swapped]:
            aborted_groups: List[SequenceGroup] = []
            for seq_group in state_queue:
                if seq_group.request_id in seq_id_to_seq_group:
                    real_request_id = seq_id_to_seq_group[seq_group.request_id].group_id
                else:
                    real_request_id = seq_group.request_id
                if real_request_id in request_ids:
                    aborted_groups.append(seq_group)
            for aborted_group in aborted_groups:
                state_queue.remove(aborted_group)
                self._finished_requests_ids.append(aborted_group.request_id)
                for seq in aborted_group.get_seqs():
                    if seq.is_finished():
                        continue
                    seq.status = SequenceStatus.FINISHED_ABORTED
                    self.free_seq(seq)
                if aborted_group.request_id in seq_id_to_seq_group:
                    del seq_id_to_seq_group[aborted_group.request_id]
                self._free_seq_group_cross_attn_blocks(aborted_group)

    def _free_seq_group_cross_attn_blocks(self, seq_group: SequenceGroup) -> None:
        """
        Free a sequence group from a cross-attention block table.
        Has no effect on decoder-only models.
        """
        if seq_group.is_encoder_decoder():
            self.block_manager.free_cross(seq_group)

    def has_unfinished_seqs(self) -> bool:
        return len(self.waiting) != 0 or len(self.running) != 0 or len(self.swapped) != 0

    def get_prefix_cache_hit_rate(self, device: Device) -> float:
        return self.block_manager.get_prefix_cache_hit_rate(device)

    def reset_prefix_cache(self, device: Optional[Device]=None) -> bool:
        return self.block_manager.reset_prefix_cache(device)

    def get_num_unfinished_seq_groups(self) -> int:
        return len(self.waiting) + len(self.running) + len(self.swapped)

    def get_and_reset_finished_requests_ids(self) -> List[str]:
        """Flushes the list of request ids of previously finished seq_groups."""
        finished_requests_ids = self._finished_requests_ids
        self._finished_requests_ids = list()
        return finished_requests_ids

    def _schedule_running(self, budget: SchedulingBudget, curr_loras: Optional[Set[int]], enable_chunking: bool=False, partial_prefill_metadata: Optional[PartialPrefillMetadata]=None) -> SchedulerRunningOutputs:
        """Schedule sequence groups that are running.

        Running queue should include decode and chunked prefill requests.

        Args:
            budget: The scheduling budget. The argument is in-place updated
                when any decodes are preempted.
            curr_loras: Currently batched lora request ids. The argument is
                in-place updated when any decodes are preempted.
            enable_chunking: If True, seq group can be chunked and only a
                chunked number of tokens are scheduled  if
                `budget.num_batched_tokens` has not enough capacity to schedule
                all tokens.
            partial_prefill_metadata: information about the partial prefills
            that are currently running

        Returns:
            SchedulerRunningOutputs.
        """
        ret: SchedulerRunningOutputs = self._scheduler_running_outputs_cache[self.cache_id].get_object()
        ret.blocks_to_swap_out.clear()
        ret.blocks_to_copy.clear()
        ret.decode_seq_groups.clear()
        ret.prefill_seq_groups.clear()
        ret.preempted.clear()
        ret.swapped_out.clear()
        ret.num_lookahead_slots = self._get_num_lookahead_slots(is_prefill=False, enable_chunking=enable_chunking)
        ret.decode_seq_groups_list.clear()
        ret.prefill_seq_groups_list.clear()
        blocks_to_swap_out: List[Tuple[int, int]] = ret.blocks_to_swap_out
        blocks_to_copy: List[Tuple[int, int]] = ret.blocks_to_copy
        decode_seq_groups: List[ScheduledSequenceGroup] = ret.decode_seq_groups
        prefill_seq_groups: List[ScheduledSequenceGroup] = ret.prefill_seq_groups
        preempted: List[SequenceGroup] = ret.preempted
        swapped_out: List[SequenceGroup] = ret.swapped_out
        running_queue = self.running
        assert len(self._async_stopped) == 0
        while running_queue:
            seq_group = running_queue[0]
            num_uncached_new_tokens, _ = self._get_num_new_uncached_and_cached_tokens(seq_group, SequenceStatus.RUNNING, enable_chunking, budget, partial_prefill_metadata)
            num_running_tokens = num_uncached_new_tokens
            if num_running_tokens == 0:
                break
            running_queue.popleft()
            if self.use_async_output_proc and seq_group.seqs[0].get_len() > self.scheduler_config.max_model_len:
                self._async_stopped.append(seq_group)
                continue
            while not self._can_append_slots(seq_group, enable_chunking):
                budget.subtract_num_batched_tokens(seq_group.request_id, num_running_tokens)
                num_running_seqs = seq_group.get_max_num_running_seqs()
                budget.subtract_num_seqs(seq_group.request_id, num_running_seqs)
                if curr_loras is not None and seq_group.lora_int_id > 0 and (seq_group.lora_int_id in curr_loras):
                    curr_loras.remove(seq_group.lora_int_id)
                cont_loop = True
                if running_queue:
                    victim_seq_group = running_queue.pop()
                else:
                    victim_seq_group = seq_group
                    cont_loop = False
                do_preempt = True
                if self.use_async_output_proc:
                    assert self.output_proc_callback is not None
                    self.output_proc_callback(request_id=victim_seq_group.request_id)
                    if victim_seq_group.is_finished():
                        self._free_finished_seq_group(victim_seq_group)
                        do_preempt = False
                if do_preempt:
                    preempted_mode = self._preempt(victim_seq_group, blocks_to_swap_out)
                    if preempted_mode == PreemptionMode.RECOMPUTE:
                        preempted.append(victim_seq_group)
                    else:
                        swapped_out.append(victim_seq_group)
                if not cont_loop:
                    break
            else:
                self._append_slots(seq_group, blocks_to_copy, enable_chunking)
                is_prefill = seq_group.is_prefill()
                scheduled_seq_group: ScheduledSequenceGroup = self._scheduled_seq_group_cache[self.cache_id].get_object()
                scheduled_seq_group.seq_group = seq_group
                if is_prefill:
                    scheduled_seq_group.token_chunk_size = num_running_tokens
                    prefill_seq_groups.append(scheduled_seq_group)
                    ret.prefill_seq_groups_list.append(seq_group)
                else:
                    scheduled_seq_group.token_chunk_size = 1
                    decode_seq_groups.append(scheduled_seq_group)
                    ret.decode_seq_groups_list.append(seq_group)
                budget.add_num_batched_tokens(seq_group.request_id, num_running_tokens)
                if enable_chunking:
                    num_running_seqs = seq_group.get_max_num_running_seqs()
                    budget.add_num_seqs(seq_group.request_id, num_running_seqs)
                if curr_loras is not None and seq_group.lora_int_id > 0:
                    curr_loras.add(seq_group.lora_int_id)
        self._scheduler_running_outputs_cache[self.next_cache_id].reset()
        self._scheduled_seq_group_cache[self.next_cache_id].reset()
        return ret

    def _schedule_swapped(self, budget: SchedulingBudget, curr_loras: Optional[Set[int]], enable_chunking: bool=False) -> SchedulerSwappedInOutputs:
        """Schedule sequence groups that are swapped out.

        It schedules swapped requests as long as it fits `budget` and
        curr_loras <= max_lora from the scheduling config. The input arguments
        `budget` and `curr_loras` are updated based on scheduled seq_groups.

        Args:
            budget: The scheduling budget. The argument is in-place updated
                when any requests are swapped in.
            curr_loras: Currently batched lora request ids. The argument is
                in-place updated when any requests are swapped in.
            enable_chunking: If True, seq group can be chunked and only a
                chunked number of tokens are scheduled  if
                `budget.num_batched_tokens` has not enough capacity to schedule
                all tokens.

        Returns:
            SchedulerSwappedInOutputs.
        """
        blocks_to_swap_in: List[Tuple[int, int]] = []
        blocks_to_copy: List[Tuple[int, int]] = []
        decode_seq_groups: List[ScheduledSequenceGroup] = []
        prefill_seq_groups: List[ScheduledSequenceGroup] = []
        infeasible_seq_groups: List[SequenceGroup] = []
        swapped_queue = self.swapped
        leftover_swapped: Deque[SequenceGroup] = deque()
        while swapped_queue:
            seq_group = swapped_queue[0]
            is_prefill = seq_group.is_prefill()
            alloc_status = self.block_manager.can_swap_in(seq_group, self._get_num_lookahead_slots(is_prefill, enable_chunking))
            if alloc_status == AllocStatus.LATER:
                break
            elif alloc_status == AllocStatus.NEVER:
                logger.warning("Failing the request %s because there's not enough kv cache blocks to run the entire sequence.", seq_group.request_id)
                for seq in seq_group.get_seqs():
                    seq.status = SequenceStatus.FINISHED_IGNORED
                infeasible_seq_groups.append(seq_group)
                swapped_queue.popleft()
                continue
            lora_int_id = 0
            if self.lora_enabled:
                lora_int_id = seq_group.lora_int_id
                assert curr_loras is not None
                assert self.lora_config is not None
                if lora_int_id > 0 and lora_int_id not in curr_loras and (len(curr_loras) >= self.lora_config.max_loras):
                    leftover_swapped.appendleft(seq_group)
                    swapped_queue.popleft()
                    continue
            num_new_seqs = seq_group.get_max_num_running_seqs()
            num_new_tokens_uncached, num_new_tokens_cached = self._get_num_new_uncached_and_cached_tokens(seq_group, SequenceStatus.SWAPPED, enable_chunking, budget)
            if num_new_tokens_uncached == 0 or not budget.can_schedule(num_new_tokens=num_new_tokens_uncached, num_new_seqs=num_new_seqs):
                self.remove_seq_from_computed_blocks_tracker(seq_group, SequenceStatus.SWAPPED)
                break
            if lora_int_id > 0 and curr_loras is not None:
                curr_loras.add(lora_int_id)
            swapped_queue.popleft()
            self._swap_in(seq_group, blocks_to_swap_in)
            self._append_slots(seq_group, blocks_to_copy, enable_chunking)
            if is_prefill:
                prefill_seq_groups.append(ScheduledSequenceGroup(seq_group, token_chunk_size=num_new_tokens_uncached + num_new_tokens_cached))
            else:
                decode_seq_groups.append(ScheduledSequenceGroup(seq_group, token_chunk_size=1))
            budget.add_num_batched_tokens(seq_group.request_id, num_batched_tokens=num_new_tokens_uncached, num_cached_tokens=num_new_tokens_cached)
            budget.add_num_seqs(seq_group.request_id, num_new_seqs)
        swapped_queue.extendleft(leftover_swapped)
        return SchedulerSwappedInOutputs(decode_seq_groups=decode_seq_groups, prefill_seq_groups=prefill_seq_groups, blocks_to_swap_in=blocks_to_swap_in, blocks_to_copy=blocks_to_copy, num_lookahead_slots=self._get_num_lookahead_slots(is_prefill=False, enable_chunking=enable_chunking), infeasible_seq_groups=infeasible_seq_groups)

    def _get_prompt_limit(self, seq_group: SequenceGroup) -> int:
        if self.scheduler_config.chunked_prefill_enabled and (not self.scheduler_config.is_multi_step):
            prompt_limit = self.scheduler_config.max_model_len
        else:
            prompt_limit = min(self.scheduler_config.max_model_len, self.scheduler_config.max_num_batched_tokens)
        if seq_group.lora_request and seq_group.lora_request.long_lora_max_len:
            assert prompt_limit <= seq_group.lora_request.long_lora_max_len
            return seq_group.lora_request.long_lora_max_len
        else:
            return prompt_limit

    def _get_priority(self, seq_group: SequenceGroup) -> Tuple[Optional[int], float]:
        """Get the priority of the sequence group.
        Highest preference to user-defined priority, followed by arrival time.
        Args:
            seq_group: The sequence group input.
        Returns:
            The priority of the sequence group.
        """
        return (seq_group.priority, seq_group.arrival_time)

    def _schedule_priority_preemption(self, budget: SchedulingBudget) -> int:
        """Sorts waiting and running queue. Also, force preempt requests
        from the running queue if their priority is lower.
        Priority-based preemption is used with the priority policy.
        Args:
            budget: The scheduling budget. The argument is in-place updated
                when any requests are scheduled.
        Returns:
            A count of priority-based preemptions.
        """
        waiting_queue = self.waiting
        running_queue = deque(sorted(self.running, key=self._get_priority))
        blocks_to_swap_out: List[Tuple[int, int]] = []
        force_preemption_count = 0
        if waiting_queue:
            seq_group = waiting_queue.popleft()
            num_new_seqs = seq_group.get_max_num_running_seqs()
            num_new_tokens_uncached, _ = self._get_num_new_uncached_and_cached_tokens(seq_group, SequenceStatus.WAITING, False, budget)
            while running_queue and self._get_priority(running_queue[-1]) > self._get_priority(seq_group):
                can_allocate = self.block_manager.can_allocate(seq_group)
                if num_new_tokens_uncached > 0 and can_allocate == AllocStatus.OK and budget.can_schedule(num_new_tokens=num_new_tokens_uncached, num_new_seqs=num_new_seqs):
                    break
                vseq_group = running_queue.pop()
                num_running_tokens_uncached, _ = self._get_num_new_uncached_and_cached_tokens(vseq_group, SequenceStatus.RUNNING, False, budget)
                budget.subtract_num_batched_tokens(vseq_group.request_id, num_running_tokens_uncached)
                num_running_seqs = vseq_group.get_max_num_running_seqs()
                budget.subtract_num_seqs(vseq_group.request_id, num_running_seqs)
                self._preempt(vseq_group, blocks_to_swap_out)
                waiting_queue.appendleft(vseq_group)
                force_preemption_count += 1
            waiting_queue.appendleft(seq_group)
            self.remove_seq_from_computed_blocks_tracker(seq_group, SequenceStatus.WAITING)
        waiting_queue = deque(sorted(waiting_queue, key=self._get_priority))
        self.waiting = waiting_queue
        self.running = running_queue
        return force_preemption_count

    def _schedule_prefills(self, budget: SchedulingBudget, curr_loras: Optional[Set[int]], enable_chunking: bool=False, partial_prefill_metadata: Optional[PartialPrefillMetadata]=None) -> SchedulerPrefillOutputs:
        """Schedule sequence groups that are in prefill stage.

        Note that the current scheduler treats PREEMPTED_FOR_RECOMPUTE
        as a new prefill (that starts from beginning -> most recently generated
        tokens).

        It schedules waiting requests as long as it fits `budget` and
        curr_loras <= max_lora from the scheduling config. The input arguments
        `budget` and `curr_loras` are updated based on scheduled seq_groups.

        Args:
            budget: The scheduling budget. The argument is in-place updated
                when any requests are scheduled.
            curr_loras: Currently batched lora request ids. The argument is
                in-place updated when any requests are scheduled.
            enable_chunking: If True, seq group can be chunked and only a
                chunked number of tokens are scheduled  if
                `budget.num_batched_tokens` has not enough capacity to schedule
                all tokens.
            partial_prefill_metadata: information about the partial prefills
                that are currently running

        Returns:
            SchedulerPrefillOutputs.
        """
        if budget.remaining_token_budget() == 0:
            return SchedulerPrefillOutputs(seq_groups=[], ignored_seq_groups=[], num_lookahead_slots=self._get_num_lookahead_slots(is_prefill=True, enable_chunking=enable_chunking))
        ignored_seq_groups: List[SequenceGroup] = []
        seq_groups: List[ScheduledSequenceGroup] = []
        using_prompt_embeds: bool = False
        waiting_queue = self.waiting
        leftover_waiting_sequences: Deque[SequenceGroup] = deque()
        while self._passed_delay(time.time()) and waiting_queue:
            seq_group = waiting_queue[0]
            waiting_seqs = seq_group.get_seqs(status=SequenceStatus.WAITING)
            assert len(waiting_seqs) == 1, 'Waiting sequence group should have only one prompt sequence.'
            if partial_prefill_metadata is not None and (not partial_prefill_metadata.can_schedule(seq_group)):
                leftover_waiting_sequences.appendleft(seq_group)
                waiting_queue.popleft()
                continue
            num_new_tokens_uncached, num_new_tokens_cached = self._get_num_new_uncached_and_cached_tokens(seq_group, SequenceStatus.WAITING, enable_chunking, budget, partial_prefill_metadata=partial_prefill_metadata)
            num_new_tokens = num_new_tokens_uncached + num_new_tokens_cached
            if not enable_chunking:
                num_prompt_tokens = waiting_seqs[0].get_len()
                assert num_new_tokens == num_prompt_tokens
            prompt_limit = self._get_prompt_limit(seq_group)
            if num_new_tokens > prompt_limit:
                logger.warning('Input prompt (%d tokens) is too long and exceeds limit of %d', num_new_tokens, prompt_limit)
                for seq in waiting_seqs:
                    seq.status = SequenceStatus.FINISHED_IGNORED
                self.remove_seq_from_computed_blocks_tracker(seq_group, SequenceStatus.FINISHED_IGNORED)
                ignored_seq_groups.append(seq_group)
                waiting_queue.popleft()
                continue
            num_lookahead_slots: int = 0
            if self.scheduler_config.is_multi_step and enable_chunking:
                num_lookahead_slots = self._get_num_lookahead_slots(True, enable_chunking)
            can_allocate = self.block_manager.can_allocate(seq_group, num_lookahead_slots=num_lookahead_slots)
            if can_allocate == AllocStatus.LATER:
                self.remove_seq_from_computed_blocks_tracker(seq_group, SequenceStatus.WAITING)
                break
            elif can_allocate == AllocStatus.NEVER:
                logger.warning('Input prompt (%d tokens) + lookahead slots (%d) is too long and exceeds the capacity of block_manager', num_new_tokens, num_lookahead_slots)
                for seq in waiting_seqs:
                    seq.status = SequenceStatus.FINISHED_IGNORED
                self.remove_seq_from_computed_blocks_tracker(seq_group, SequenceStatus.FINISHED_IGNORED)
                ignored_seq_groups.append(seq_group)
                waiting_queue.popleft()
                continue
            if len(seq_groups) == 0:
                using_prompt_embeds = seq_group.uses_prompt_embeds()
            if using_prompt_embeds != seq_group.uses_prompt_embeds():
                self.remove_seq_from_computed_blocks_tracker(seq_group, SequenceStatus.WAITING)
                leftover_waiting_sequences.appendleft(seq_group)
                waiting_queue.popleft()
                continue
            lora_int_id = 0
            if self.lora_enabled:
                lora_int_id = seq_group.lora_int_id
                assert curr_loras is not None
                assert self.lora_config is not None
                if self.lora_enabled and lora_int_id > 0 and (lora_int_id not in curr_loras) and (len(curr_loras) >= self.lora_config.max_loras):
                    self.remove_seq_from_computed_blocks_tracker(seq_group, SequenceStatus.WAITING)
                    leftover_waiting_sequences.appendleft(seq_group)
                    waiting_queue.popleft()
                    continue
            if budget.num_batched_tokens >= self.scheduler_config.max_num_batched_tokens:
                self.remove_seq_from_computed_blocks_tracker(seq_group, SequenceStatus.WAITING)
                break
            num_new_seqs = seq_group.get_max_num_running_seqs()
            if num_new_tokens_uncached == 0 or not budget.can_schedule(num_new_tokens=num_new_tokens_uncached, num_new_seqs=num_new_seqs):
                self.remove_seq_from_computed_blocks_tracker(seq_group, SequenceStatus.WAITING)
                break
            if curr_loras is not None and lora_int_id > 0:
                curr_loras.add(lora_int_id)
            waiting_queue.popleft()
            self._allocate_and_set_running(seq_group)
            if partial_prefill_metadata is not None:
                partial_prefill_metadata.maybe_increment_partial_prefills(seq_group)
            if enable_chunking and self.scheduler_config.is_multi_step:
                blocks_to_copy: List[Tuple[int, int]] = []
                self._append_slots(seq_group, blocks_to_copy, enable_chunking)
                assert not blocks_to_copy
            else:
                seq_group.init_multi_step_from_lookahead_slots(num_lookahead_slots, num_scheduler_steps=self.scheduler_config.num_scheduler_steps, is_multi_step=self.scheduler_config.is_multi_step, enable_chunking=enable_chunking)
            seq_groups.append(ScheduledSequenceGroup(seq_group=seq_group, token_chunk_size=num_new_tokens))
            budget.add_num_batched_tokens(seq_group.request_id, num_batched_tokens=num_new_tokens_uncached, num_cached_tokens=num_new_tokens_cached)
            budget.add_num_seqs(seq_group.request_id, num_new_seqs)
        waiting_queue.extendleft(leftover_waiting_sequences)
        if len(seq_groups) > 0:
            self.prev_prompt = True
        return SchedulerPrefillOutputs(seq_groups=seq_groups, ignored_seq_groups=ignored_seq_groups, num_lookahead_slots=self._get_num_lookahead_slots(is_prefill=True, enable_chunking=enable_chunking))

    def _schedule_default(self) -> SchedulerOutputs:
        """Schedule queued requests.

        The current policy is designed to optimize the throughput. First,
        it batches as many prefill requests as possible. And it schedules
        decodes. If there's a pressure on GPU memory, decode requests can
        be swapped or preempted.
        """
        budget = SchedulingBudget(token_budget=self.scheduler_config.max_num_batched_tokens, max_num_seqs=self.scheduler_config.max_num_seqs)
        for seq_group in self.running:
            budget.add_num_seqs(seq_group.request_id, seq_group.get_max_num_running_seqs())
        curr_loras = set((seq_group.lora_int_id for seq_group in self.running if seq_group.lora_int_id > 0)) if self.lora_enabled else None
        prefills = SchedulerPrefillOutputs.create_empty()
        running_scheduled = SchedulerRunningOutputs.create_empty()
        swapped_in = SchedulerSwappedInOutputs.create_empty()
        if not self.swapped:
            prefills = self._schedule_prefills(budget, curr_loras, enable_chunking=False)
        if len(prefills.seq_groups) == 0 and self.scheduler_config.policy == 'priority':
            self._schedule_priority_preemption(budget)
        if len(prefills.seq_groups) == 0:
            running_scheduled = self._schedule_running(budget, curr_loras, enable_chunking=False)
            if len(running_scheduled.preempted) + len(running_scheduled.swapped_out) == 0:
                swapped_in = self._schedule_swapped(budget, curr_loras)
        assert budget.num_batched_tokens <= self.scheduler_config.max_num_batched_tokens
        assert budget.num_curr_seqs <= self.scheduler_config.max_num_seqs
        self.waiting.extendleft(running_scheduled.preempted)
        if len(prefills.seq_groups) > 0:
            self.running.extend([s.seq_group for s in prefills.seq_groups])
        self.running.extend(running_scheduled.decode_seq_groups_list)
        if len(swapped_in.decode_seq_groups) > 0:
            self.running.extend([s.seq_group for s in swapped_in.decode_seq_groups])
        self.swapped.extend(running_scheduled.swapped_out)
        preempted = len(running_scheduled.preempted) + len(running_scheduled.swapped_out)
        assert len(running_scheduled.prefill_seq_groups) == 0
        assert len(swapped_in.prefill_seq_groups) == 0
        num_prefill_groups = len(prefills.seq_groups)
        ignored_seq_groups_for_embeds = list[SequenceGroup]()
        if num_prefill_groups > 0:
            scheduled_seq_groups = prefills.seq_groups
            scheduled_seq_groups.extend(running_scheduled.decode_seq_groups)
            ignored_seq_groups_for_embeds.clear()
        else:
            scheduled_seq_groups = running_scheduled.decode_seq_groups
            if len(scheduled_seq_groups) > 0:
                using_prompt_embeds = scheduled_seq_groups[0].seq_group.uses_prompt_embeds()
                ignored_seq_groups_for_embeds.clear()
                indices_ignored = list[int]()
                for i, schedule_seq_group in enumerate(scheduled_seq_groups):
                    if using_prompt_embeds != schedule_seq_group.seq_group.uses_prompt_embeds():
                        ignored_seq_groups_for_embeds.append(schedule_seq_group.seq_group)
                        indices_ignored.append(i)
                if len(ignored_seq_groups_for_embeds) > 0:
                    scheduled_seq_groups = [group for i, group in enumerate(scheduled_seq_groups) if i not in indices_ignored]
            else:
                ignored_seq_groups_for_embeds.clear()
        scheduled_seq_groups.extend(swapped_in.decode_seq_groups)
        blocks_to_copy = running_scheduled.blocks_to_copy
        blocks_to_copy.extend(swapped_in.blocks_to_copy)
        ignored_seq_groups = prefills.ignored_seq_groups
        ignored_seq_groups.extend(ignored_seq_groups_for_embeds)
        ignored_seq_groups.extend(swapped_in.infeasible_seq_groups)
        return SchedulerOutputs(scheduled_seq_groups=scheduled_seq_groups, num_prefill_groups=num_prefill_groups, num_batched_tokens=budget.num_batched_tokens + budget.num_cached_tokens, blocks_to_swap_in=swapped_in.blocks_to_swap_in, blocks_to_swap_out=running_scheduled.blocks_to_swap_out, blocks_to_copy=blocks_to_copy, ignored_seq_groups=ignored_seq_groups, num_lookahead_slots=running_scheduled.num_lookahead_slots, running_queue_size=len(self.running), preempted=preempted)

    def _schedule_chunked_prefill(self) -> SchedulerOutputs:
        """Schedule queued requests.

        Chunked prefill allows to chunk prefill requests, batch them together
        with decode requests. This policy 1. schedule as many decoding requests
        as possible. 2. schedule chunked prefill requests that are not
        finished. 3. schedule swapped request. 4. schedule new prefill
        requests.

        The policy can sustain the high GPU utilization because it can put
        prefill and decodes requests to the same batch, while it improves
        inter token latency because decodes requests don't need to be blocked
        by prefill requests.
        """
        budget = SchedulingBudget(token_budget=self.scheduler_config.max_num_batched_tokens, max_num_seqs=self.scheduler_config.max_num_seqs)
        curr_loras: Set[int] = set()
        prefills = SchedulerPrefillOutputs.create_empty()
        swapped_in = SchedulerSwappedInOutputs.create_empty()
        partial_prefill_metadata = PartialPrefillMetadata.from_queues(running=self.running, waiting=self.waiting, scheduler_config=self.scheduler_config)
        running_scheduled = self._schedule_running(budget, curr_loras, enable_chunking=True, partial_prefill_metadata=partial_prefill_metadata)
        if len(running_scheduled.preempted) + len(running_scheduled.swapped_out) == 0:
            swapped_in = self._schedule_swapped(budget, curr_loras)
        prefills = self._schedule_prefills(budget, curr_loras, enable_chunking=True, partial_prefill_metadata=partial_prefill_metadata)
        assert budget.num_batched_tokens <= self.scheduler_config.max_num_batched_tokens
        assert budget.num_curr_seqs <= self.scheduler_config.max_num_seqs
        self.waiting.extendleft(running_scheduled.preempted)
        self.running.extend([s.seq_group for s in swapped_in.decode_seq_groups])
        self.running.extend([s.seq_group for s in swapped_in.prefill_seq_groups])
        self.running.extend([s.seq_group for s in running_scheduled.decode_seq_groups])
        self.running.extend(self._order_finishing_prefills_first(running_scheduled.prefill_seq_groups))
        self.running.extend([s.seq_group for s in prefills.seq_groups])
        self.swapped.extend(running_scheduled.swapped_out)
        scheduled_seq_groups = prefills.seq_groups + running_scheduled.prefill_seq_groups + swapped_in.prefill_seq_groups + running_scheduled.decode_seq_groups + swapped_in.decode_seq_groups
        num_prefill_groups = len(prefills.seq_groups) + len(swapped_in.prefill_seq_groups) + len(running_scheduled.prefill_seq_groups)
        all_prefills = len(scheduled_seq_groups) == num_prefill_groups
        num_lookahead_slots = 0 if all_prefills and (not self.scheduler_config.is_multi_step) else running_scheduled.num_lookahead_slots
        return SchedulerOutputs(scheduled_seq_groups=scheduled_seq_groups, num_prefill_groups=num_prefill_groups, num_batched_tokens=budget.num_batched_tokens + budget.num_cached_tokens, blocks_to_swap_in=swapped_in.blocks_to_swap_in, blocks_to_swap_out=running_scheduled.blocks_to_swap_out, blocks_to_copy=running_scheduled.blocks_to_copy + swapped_in.blocks_to_copy, ignored_seq_groups=prefills.ignored_seq_groups + swapped_in.infeasible_seq_groups, num_lookahead_slots=num_lookahead_slots, running_queue_size=len(self.running), preempted=len(running_scheduled.preempted) + len(running_scheduled.swapped_out))

    def _order_finishing_prefills_first(self, scheduled_prefill_seqs: List[ScheduledSequenceGroup]) -> List[SequenceGroup]:
        """Returns a list of prefilling SequenceGroups where sequences that are
        scheduled to finish prefilling are listed first"""
        finishing = [s.seq_group for s in scheduled_prefill_seqs if s.seq_group.get_num_uncomputed_tokens() == s.token_chunk_size]
        not_finishing = [s.seq_group for s in scheduled_prefill_seqs if s.seq_group.get_num_uncomputed_tokens() != s.token_chunk_size]
        return finishing + not_finishing

    def _schedule(self) -> SchedulerOutputs:
        """Schedule queued requests."""
        if self.scheduler_config.chunked_prefill_enabled:
            return self._schedule_chunked_prefill()
        else:
            return self._schedule_default()

    def _can_append_slots(self, seq_group: SequenceGroup, enable_chunking: bool) -> bool:
        """Determine whether or not we have enough space in the KV cache to
        continue generation of the sequence group.
        """
        if self.enable_artificial_preemption and random.uniform(0, 1) < ARTIFICIAL_PREEMPTION_PROB and (self.artificial_preempt_cnt > 0):
            self.artificial_preempt_cnt -= 1
            return False
        is_prefill = seq_group.is_prefill()
        num_lookahead_slots = self._get_num_lookahead_slots(is_prefill, enable_chunking)
        if is_prefill and num_lookahead_slots > 0:
            assert self.scheduler_config.is_multi_step and enable_chunking
        return self.block_manager.can_append_slots(seq_group=seq_group, num_lookahead_slots=num_lookahead_slots)

    def _allow_async_output_proc(self, seq_group: SequenceGroup) -> bool:
        no_single_seq = seq_group.sampling_params is None or seq_group.sampling_params.n == 1
        return no_single_seq

    def schedule(self) -> Tuple[List[SequenceGroupMetadata], SchedulerOutputs, bool]:
        scheduler_start_time = time.perf_counter()
        scheduler_outputs: SchedulerOutputs = self._schedule()
        now = time.time()
        if not self.cache_config.enable_prefix_caching:
            common_computed_block_nums = []
        allow_async_output_proc: bool = self.use_async_output_proc
        seq_group_metadata_list: List[SequenceGroupMetadata] = []
        for i, scheduled_seq_group in enumerate(scheduler_outputs.scheduled_seq_groups):
            seq_group = scheduled_seq_group.seq_group
            token_chunk_size = scheduled_seq_group.token_chunk_size
            seq_group.maybe_set_first_scheduled_time(now)
            seq_group_metadata = self._seq_group_metadata_cache[self.cache_id].get_object()
            seq_group_metadata.seq_data.clear()
            seq_group_metadata.block_tables.clear()
            seq_data: Dict[int, SequenceData] = {}
            block_tables: Dict[int, List[int]] = {}
            if seq_group.is_encoder_decoder():
                encoder_seq = seq_group.get_encoder_seq()
                assert encoder_seq is not None
                encoder_seq_data = encoder_seq.data
                cross_block_table = self.block_manager.get_cross_block_table(seq_group)
            else:
                encoder_seq_data = None
                cross_block_table = None
            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                seq_id = seq.seq_id
                seq_data[seq_id] = seq.data
                block_tables[seq_id] = self.block_manager.get_block_table(seq)
                self.block_manager.access_all_blocks_in_seq(seq, now)
            if self.cache_config.enable_prefix_caching:
                common_computed_block_nums = self.block_manager.get_common_computed_block_ids(seq_group.get_seqs(status=SequenceStatus.RUNNING))
            do_sample = True
            is_prompt = seq_group.is_prefill()
            is_first_prefill = False
            if is_prompt:
                seqs = seq_group.get_seqs()
                assert len(seqs) == 1
                num_computed_tokens = seqs[0].data.get_num_computed_tokens()
                is_first_prefill = num_computed_tokens == 0
                if token_chunk_size + num_computed_tokens < seqs[0].data.get_len():
                    do_sample = False
            if is_first_prefill or not self.scheduler_config.send_delta_data:
                seq_group_metadata = SequenceGroupMetadata(request_id=seq_group.request_id, is_prompt=is_prompt, seq_data=seq_data, sampling_params=seq_group.sampling_params, block_tables=block_tables, do_sample=do_sample, pooling_params=seq_group.pooling_params, token_chunk_size=token_chunk_size, lora_request=seq_group.lora_request, computed_block_nums=common_computed_block_nums, encoder_seq_data=encoder_seq_data, cross_block_table=cross_block_table, state=seq_group.state, token_type_ids=seq_group.token_type_ids, multi_modal_data=seq_group.multi_modal_data if scheduler_outputs.num_prefill_groups > 0 else None, multi_modal_placeholders=seq_group.multi_modal_placeholders if scheduler_outputs.num_prefill_groups > 0 else None, prompt_adapter_request=seq_group.prompt_adapter_request)
            else:
                seq_data_delta = {}
                for id, data in seq_data.items():
                    seq_data_delta[id] = data.get_delta_and_reset()
                seq_group_metadata = SequenceGroupMetadataDelta(seq_data_delta, seq_group.request_id, block_tables, is_prompt, do_sample=do_sample, token_chunk_size=token_chunk_size, computed_block_nums=common_computed_block_nums)
            seq_group_metadata_list.append(seq_group_metadata)
            if allow_async_output_proc:
                allow_async_output_proc = self._allow_async_output_proc(seq_group)
        for scheduled_seq_group in scheduler_outputs.scheduled_seq_groups:
            self.block_manager.mark_blocks_as_computed(scheduled_seq_group.seq_group, scheduled_seq_group.token_chunk_size)
        self._seq_group_metadata_cache[self.next_cache_id].reset()
        scheduler_time = time.perf_counter() - scheduler_start_time
        for seq_group in self.running:
            if seq_group is not None and seq_group.metrics is not None:
                if seq_group.metrics.scheduler_time is not None:
                    seq_group.metrics.scheduler_time += scheduler_time
                else:
                    seq_group.metrics.scheduler_time = scheduler_time
        self.cache_id = self.next_cache_id
        return (seq_group_metadata_list, scheduler_outputs, allow_async_output_proc)

    def fork_seq(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        self.block_manager.fork(parent_seq, child_seq)

    def free_seq(self, seq: Sequence) -> None:
        """Free a sequence from a block table."""
        self.block_manager.free(seq)

    def remove_seq_from_computed_blocks_tracker(self, seq_group: SequenceGroup, status: Optional[SequenceStatus]) -> None:
        seqs = seq_group.get_seqs(status=status)
        for seq in seqs:
            self._remove_seq_from_computed_blocks_tracker(seq)

    def _remove_seq_from_computed_blocks_tracker(self, seq: Sequence) -> None:
        """
        Free a sequence computed blocks tracker _seq_id_to_blocks_hashes
        and _seq_id_to_num_tokens_computed.
        """
        self.block_manager.remove_seq_from_computed_blocks_tracker(seq)

    def _free_finished_seqs(self, seq_group: SequenceGroup) -> None:
        """Free finished seqs in a sequence group."""
        for seq in seq_group.get_seqs():
            if seq.is_finished():
                self.free_seq(seq)

    def _free_finished_seq_group(self, seq_group: SequenceGroup) -> None:
        if seq_group.is_finished():
            self._free_seq_group_cross_attn_blocks(seq_group)
            self._finished_requests_ids.append(seq_group.request_id)
        self._free_finished_seqs(seq_group)

    def free_finished_seq_groups(self) -> None:
        remaining: Deque[SequenceGroup] = deque()
        for seq_group in self.running:
            self._free_finished_seq_group(seq_group)
            if not seq_group.is_finished():
                remaining.append(seq_group)
        self.running = remaining
        if self._async_stopped:
            for seq_group in self._async_stopped:
                self._free_seq_group_cross_attn_blocks(seq_group)
                self._finished_requests_ids.append(seq_group.request_id)
                self._free_finished_seqs(seq_group)
            self._async_stopped.clear()

    def _allocate_and_set_running(self, seq_group: SequenceGroup) -> None:
        self.block_manager.allocate(seq_group)
        for seq in seq_group.get_seqs(status=SequenceStatus.WAITING):
            seq.status = SequenceStatus.RUNNING

    def _append_slots(self, seq_group: SequenceGroup, blocks_to_copy: List[Tuple[int, int]], enable_chunking: bool=False) -> None:
        """Appends new slots to the sequences in the given sequence group.

        Args:
            seq_group (SequenceGroup): The sequence group containing the
                sequences to append slots to.
            blocks_to_copy (List[Tuple[int, int]]): A list of tuple of two
                ints, the first int is the source block index, and the second
                int is the destination block index. This list is updated with
                the new source and destination block indices for the appended
                slots.
            enable_chunking (bool): True if chunked prefill is enabled.
        """
        is_prefill: bool = seq_group.is_prefill()
        num_lookahead_slots: int = self._get_num_lookahead_slots(is_prefill, enable_chunking)
        seq_group.init_multi_step_from_lookahead_slots(num_lookahead_slots, num_scheduler_steps=self.scheduler_config.num_scheduler_steps, is_multi_step=self.scheduler_config.is_multi_step, enable_chunking=enable_chunking)
        seq_status: Optional[SequenceStatus] = SequenceStatus.RUNNING
        if self.scheduler_config.is_multi_step and enable_chunking:
            seq_status = None
        for seq in seq_group.get_seqs(status=seq_status):
            cows = self.block_manager.append_slots(seq, num_lookahead_slots)
            if len(cows) > 0:
                blocks_to_copy.extend(cows)

    def _preempt(self, seq_group: SequenceGroup, blocks_to_swap_out: List[Tuple[int, int]]) -> PreemptionMode:
        if self.user_specified_preemption_mode is None:
            if seq_group.get_max_num_running_seqs() == 1:
                preemption_mode = PreemptionMode.RECOMPUTE
            else:
                preemption_mode = PreemptionMode.SWAP
        elif self.user_specified_preemption_mode == 'swap':
            preemption_mode = PreemptionMode.SWAP
        else:
            preemption_mode = PreemptionMode.RECOMPUTE
        if self.num_cumulative_preemption % 50 == 0:
            logger.warning('Sequence group %s is preempted by %s mode because there is not enough KV cache space. This can affect the end-to-end performance. Increase gpu_memory_utilization or tensor_parallel_size to provide more KV cache memory. total_num_cumulative_preemption=%d', seq_group.request_id, preemption_mode, self.num_cumulative_preemption + 1)
        self.num_cumulative_preemption += 1
        if preemption_mode == PreemptionMode.RECOMPUTE:
            self._preempt_by_recompute(seq_group)
        elif preemption_mode == PreemptionMode.SWAP:
            self._preempt_by_swap(seq_group, blocks_to_swap_out)
        else:
            raise AssertionError('Invalid preemption mode.')
        return preemption_mode

    def _preempt_by_recompute(self, seq_group: SequenceGroup) -> None:
        seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
        assert len(seqs) == 1
        for seq in seqs:
            seq.status = SequenceStatus.WAITING
            self.free_seq(seq)
            seq.reset_state_for_recompute()
        self._free_seq_group_cross_attn_blocks(seq_group)

    def _preempt_by_swap(self, seq_group: SequenceGroup, blocks_to_swap_out: List[Tuple[int, int]]) -> None:
        self._swap_out(seq_group, blocks_to_swap_out)

    def _swap_in(self, seq_group: SequenceGroup, blocks_to_swap_in: List[Tuple[int, int]]) -> None:
        mapping = self.block_manager.swap_in(seq_group)
        blocks_to_swap_in.extend(mapping)
        for seq in seq_group.get_seqs(status=SequenceStatus.SWAPPED):
            seq.status = SequenceStatus.RUNNING

    def _swap_out(self, seq_group: SequenceGroup, blocks_to_swap_out: List[Tuple[int, int]]) -> None:
        if not self.block_manager.can_swap_out(seq_group):
            raise RuntimeError('Aborted due to the lack of CPU swap space. Please increase the swap space to avoid this error.')
        mapping = self.block_manager.swap_out(seq_group)
        blocks_to_swap_out.extend(mapping)
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            seq.status = SequenceStatus.SWAPPED

    def _passed_delay(self, now: float) -> bool:
        if self.prev_prompt:
            self.last_prompt_latency = now - self.prev_time
        self.prev_time, self.prev_prompt = (now, False)
        if self.scheduler_config.delay_factor > 0 and self.waiting:
            earliest_arrival_time = min([e.metrics.arrival_time for e in self.waiting])
            passed_delay = now - earliest_arrival_time > self.scheduler_config.delay_factor * self.last_prompt_latency or not self.running
        else:
            passed_delay = True
        return passed_delay

    def _get_num_lookahead_slots(self, is_prefill: bool, enable_chunking: bool) -> int:
        """The number of slots to allocate per sequence per step, beyond known
        token ids. Speculative decoding uses these slots to store KV activations
        of tokens which may or may not be accepted.

        Speculative decoding does not yet support prefill, so we do not perform
        lookahead allocation for prefill.

        When chunking is enabled with multi-step, we allocate lookahead slots
        for the prefills for when the prefills turn into decodes in the first
        step.
        """
        if is_prefill:
            if self.scheduler_config.is_multi_step and enable_chunking:
                return self.scheduler_config.num_lookahead_slots + 1
            else:
                return 0
        return self.scheduler_config.num_lookahead_slots

    def _get_num_new_uncached_and_cached_tokens(self, seq_group: SequenceGroup, status: SequenceStatus, enable_chunking: bool, budget: SchedulingBudget, partial_prefill_metadata: Optional[PartialPrefillMetadata]=None) -> Tuple[int, int]:
        """
        Returns the number of new uncached and cached tokens to schedule for a
        given sequence group that's in a given `status`.

        The API could chunk the number of tokens to compute based on `budget`
        if `enable_chunking` is True. If a sequence group has multiple
        sequences (e.g., running beam search), it means it is in decoding
        phase, so chunking doesn't happen.

        Returns (0, 0) if the new token cannot be computed due to token budget.

        The cached tokens's blocks are already computed, and the attention
        backend will reuse the cached blocks rather than recomputing them. So
        the scheduler could schedule these cached tokens "for free".

        Args:
            seq_group: The sequence group to get the number of new tokens to
                schedule.
            status: The status of the sequences to get the number of new tokens
                to schedule.
            enable_chunking: Whether to chunk the number of tokens to compute.
            budget: The budget to chunk the number of tokens to compute.
            partial_prefill_metadata: information about the partial prefills
                that are currently running


        Returns:
            A tuple of two ints. The first int is the number of new uncached
            tokens to schedule. The second int is the number of cached tokens.
            If no more new tokens can be scheduled, returns (0, 0).
        """
        num_cached_new_tokens = 0
        num_uncached_new_tokens = 0
        seqs = seq_group.get_seqs(status=status)
        for seq in seqs:
            if not seq.is_prefill():
                num_uncached_new_tokens += 1
                continue
            num_computed_tokens_seq = seq.get_num_computed_tokens()
            all_num_new_tokens_seq = seq.get_len() - num_computed_tokens_seq
            if not self.cache_config.enable_prefix_caching:
                num_uncached_new_tokens += all_num_new_tokens_seq
                continue
            num_cached_tokens_seq = self.block_manager.get_num_cached_tokens(seq)
            if num_cached_tokens_seq < num_computed_tokens_seq:
                assert seq.is_prefill() and seq.status == SequenceStatus.RUNNING and self.scheduler_config.chunked_prefill_enabled, f"Number of cached tokens should not be less than the number of computed tokens for a sequence that's still in prefill. But there are {num_cached_tokens_seq} cached tokens and {num_computed_tokens_seq} computed tokens for sequence {seq.seq_id}."
            num_cached_new_tokens_seq = max(0, num_cached_tokens_seq - num_computed_tokens_seq)
            num_uncached_new_tokens_seq = all_num_new_tokens_seq - num_cached_new_tokens_seq
            num_uncached_new_tokens += num_uncached_new_tokens_seq
            num_cached_new_tokens += num_cached_new_tokens_seq
        if num_uncached_new_tokens == 0 and num_cached_new_tokens > 0:
            num_uncached_new_tokens = 1
            num_cached_new_tokens -= 1
        if enable_chunking and len(seqs) == 1:
            num_uncached_new_tokens = self._chunk_new_tokens_to_schedule(self.scheduler_config, self.cache_config, budget, self._get_prompt_limit(seq_group), num_uncached_new_tokens, self.partial_prefill_budget_lookup_list, partial_prefill_metadata)
        return (num_uncached_new_tokens, num_cached_new_tokens)

    @staticmethod
    def _chunk_new_tokens_to_schedule(scheduler_config: SchedulerConfig, cache_config: CacheConfig, budget: SchedulingBudget, prompt_limit: int, num_new_tokens: int, partial_prefill_budget_lookup_list: List[int], partial_prefill_metadata: Optional[PartialPrefillMetadata]=None) -> int:
        """
        Chunks the number of new tokens to schedule based on the budget when
        chunked prefill is enabled.

        Args:
            scheduler_config: The scheduler config.
            cache_config: The cache config.
            budget: The budget to chunk the number of tokens to compute.
            prompt_limit: The maximum number of tokens allowed in a prompt.
            num_new_tokens: The number of new tokens to schedule.

        Returns:
            The number of new tokens to schedule after chunking.
        """
        remaining_token_budget = budget.remaining_token_budget()
        if scheduler_config.is_multi_step:
            if num_new_tokens > prompt_limit:
                return num_new_tokens
            return 0 if num_new_tokens > remaining_token_budget else num_new_tokens
        prefill_slot_budget = remaining_token_budget if partial_prefill_metadata is None else partial_prefill_budget_lookup_list[partial_prefill_metadata.schedulable_prefills]
        if cache_config.enable_prefix_caching:
            block_size = cache_config.block_size
            remaining_token_budget = min(remaining_token_budget, prefill_slot_budget) // block_size * block_size
        num_new_tokens = min(num_new_tokens, remaining_token_budget, prefill_slot_budget)
        return num_new_tokens