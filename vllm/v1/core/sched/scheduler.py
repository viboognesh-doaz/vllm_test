from __future__ import annotations
from collections import defaultdict
from collections.abc import Iterable
from typing import Any, Optional, Union
from vllm.config import VllmConfig
from vllm.distributed.kv_events import EventPublisherFactory, KVEventBatch
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorBase_V1, KVConnectorRole
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.v1.core.encoder_cache_manager import EncoderCacheManager, compute_encoder_budget
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.interface import SchedulerInterface
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.core.sched.utils import check_stop
from vllm.v1.engine import EngineCoreEventType, EngineCoreOutput, EngineCoreOutputs
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.spec_decode.metrics import SpecDecodingStats
from vllm.v1.structured_output import StructuredOutputManager
import itertools
import time
logger = init_logger(__name__)

class Scheduler(SchedulerInterface):

    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig, structured_output_manager: StructuredOutputManager, mm_registry: MultiModalRegistry=MULTIMODAL_REGISTRY, include_finished_set: bool=False, log_stats: bool=False) -> None:
        self.vllm_config = vllm_config
        self.scheduler_config = vllm_config.scheduler_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.kv_cache_config = kv_cache_config
        self.kv_events_config = vllm_config.kv_events_config
        self.parallel_config = vllm_config.parallel_config
        self.log_stats = log_stats
        self.structured_output_manager = structured_output_manager
        self.finished_req_ids_dict: Optional[dict[int, set[str]]] = defaultdict(set) if include_finished_set else None
        self.max_num_running_reqs = self.scheduler_config.max_num_seqs
        self.max_num_scheduled_tokens = self.scheduler_config.max_num_batched_tokens
        self.max_model_len = self.scheduler_config.max_model_len
        self.enable_kv_cache_events = self.kv_events_config is not None and self.kv_events_config.enable_kv_cache_events
        self.connector = None
        if self.vllm_config.kv_transfer_config is not None:
            assert len(self.kv_cache_config.kv_cache_groups) == 1, 'Multiple KV cache groups are not currently supported with KV connectors'
            self.connector = KVConnectorFactory.create_connector_v1(config=self.vllm_config, role=KVConnectorRole.SCHEDULER)
        self.kv_event_publisher = EventPublisherFactory.create(self.kv_events_config, self.parallel_config.data_parallel_rank)
        num_gpu_blocks = self.cache_config.num_gpu_blocks
        assert num_gpu_blocks is not None and num_gpu_blocks > 0
        self.block_size = self.cache_config.block_size
        self.requests: dict[str, Request] = {}
        if self.scheduler_config.policy == 'priority':
            self.policy = SchedulingPolicy.PRIORITY
        elif self.scheduler_config.policy == 'fcfs':
            self.policy = SchedulingPolicy.FCFS
        else:
            raise ValueError(f'Unknown scheduling policy: {self.scheduler_config.policy}')
        self.waiting = create_request_queue(self.policy)
        self.running: list[Request] = []
        self.finished_req_ids: set[str] = set()
        self.finished_recving_kv_req_ids: set[str] = set()
        encoder_compute_budget, encoder_cache_size = compute_encoder_budget(model_config=vllm_config.model_config, scheduler_config=vllm_config.scheduler_config, mm_registry=mm_registry)
        self.max_num_encoder_input_tokens = encoder_compute_budget
        self.encoder_cache_manager = EncoderCacheManager(cache_size=encoder_cache_size)
        speculative_config = vllm_config.speculative_config
        self.use_eagle = False
        self.num_spec_tokens = self.num_lookahead_tokens = 0
        if speculative_config:
            self.num_spec_tokens = speculative_config.num_speculative_tokens
            if speculative_config.use_eagle():
                self.use_eagle = True
                self.num_lookahead_tokens = self.num_spec_tokens
        self.kv_cache_manager = KVCacheManager(kv_cache_config=kv_cache_config, max_model_len=self.max_model_len, enable_caching=self.cache_config.enable_prefix_caching, caching_hash_algo=self.cache_config.prefix_caching_hash_algo, use_eagle=self.use_eagle, log_stats=self.log_stats, enable_kv_cache_events=self.enable_kv_cache_events)
        self.use_pp = self.parallel_config.pipeline_parallel_size > 1

    def schedule(self) -> SchedulerOutput:
        scheduled_new_reqs: list[Request] = []
        scheduled_resumed_reqs: list[Request] = []
        scheduled_running_reqs: list[Request] = []
        preempted_reqs: list[Request] = []
        structured_output_request_ids: dict[str, int] = {}
        req_to_new_block_ids: dict[str, tuple[list[int], ...]] = {}
        num_scheduled_tokens: dict[str, int] = {}
        token_budget = self.max_num_scheduled_tokens
        scheduled_encoder_inputs: dict[str, list[int]] = {}
        encoder_budget = self.max_num_encoder_input_tokens
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}
        scheduled_timestamp = time.monotonic()
        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]
            num_new_tokens = request.num_tokens_with_spec + request.num_output_placeholders - request.num_computed_tokens
            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold
            num_new_tokens = min(num_new_tokens, token_budget)
            num_new_tokens = min(num_new_tokens, self.max_model_len - 1 - request.num_computed_tokens)
            encoder_inputs_to_schedule = None
            new_encoder_budget = encoder_budget
            if request.has_encoder_inputs:
                encoder_inputs_to_schedule, num_new_tokens, new_encoder_budget = self._try_schedule_encoder_inputs(request, request.num_computed_tokens, num_new_tokens, encoder_budget)
            if num_new_tokens == 0:
                req_index += 1
                continue
            while True:
                new_blocks = self.kv_cache_manager.allocate_slots(request, num_new_tokens, num_lookahead_tokens=self.num_lookahead_tokens)
                if new_blocks is None:
                    if self.policy == SchedulingPolicy.PRIORITY:
                        preempted_req = max(self.running, key=lambda r: (r.priority, r.arrival_time))
                        self.running.remove(preempted_req)
                    else:
                        preempted_req = self.running.pop()
                    self.kv_cache_manager.free(preempted_req)
                    preempted_req.status = RequestStatus.PREEMPTED
                    preempted_req.num_computed_tokens = 0
                    if self.log_stats:
                        preempted_req.record_event(EngineCoreEventType.PREEMPTED, scheduled_timestamp)
                    self.waiting.prepend_request(preempted_req)
                    preempted_reqs.append(preempted_req)
                    if preempted_req == request:
                        can_schedule = False
                        break
                else:
                    can_schedule = True
                    break
            if not can_schedule:
                break
            assert new_blocks is not None
            scheduled_running_reqs.append(request)
            if request.use_structured_output:
                structured_output_request_ids[request.request_id] = req_index
            req_to_new_block_ids[request.request_id] = new_blocks.get_block_ids()
            num_scheduled_tokens[request.request_id] = num_new_tokens
            token_budget -= num_new_tokens
            req_index += 1
            if request.spec_token_ids:
                num_scheduled_spec_tokens = num_new_tokens + request.num_computed_tokens - request.num_tokens
                if num_scheduled_spec_tokens > 0:
                    del request.spec_token_ids[num_scheduled_spec_tokens:]
                    scheduled_spec_decode_tokens[request.request_id] = request.spec_token_ids
            if encoder_inputs_to_schedule:
                scheduled_encoder_inputs[request.request_id] = encoder_inputs_to_schedule
                for i in encoder_inputs_to_schedule:
                    self.encoder_cache_manager.allocate(request, i)
                encoder_budget = new_encoder_budget
        scheduled_loras: set[int] = set()
        if self.lora_config:
            scheduled_loras = set((req.lora_request.lora_int_id for req in scheduled_running_reqs if req.lora_request and req.lora_request.lora_int_id > 0))
            assert len(scheduled_loras) <= self.lora_config.max_loras
        skipped_waiting_requests = create_request_queue(self.policy)
        if not preempted_reqs:
            while self.waiting and token_budget > 0:
                if len(self.running) == self.max_num_running_reqs:
                    break
                request = self.waiting.peek_request()
                if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                    is_ready = self._update_waiting_for_remote_kv(request)
                    if is_ready:
                        request.status = RequestStatus.WAITING
                    else:
                        logger.debug('%s is still in WAITING_FOR_REMOTE_KVS state.', request.request_id)
                        self.waiting.pop_request()
                        skipped_waiting_requests.prepend_request(request)
                        continue
                if request.status == RequestStatus.WAITING_FOR_FSM:
                    structured_output_req = request.structured_output_request
                    if structured_output_req and structured_output_req.grammar:
                        request.status = RequestStatus.WAITING
                    else:
                        self.waiting.pop_request()
                        skipped_waiting_requests.prepend_request(request)
                        continue
                if self.lora_config and request.lora_request and (len(scheduled_loras) == self.lora_config.max_loras and request.lora_request.lora_int_id not in scheduled_loras):
                    self.waiting.pop_request()
                    skipped_waiting_requests.prepend_request(request)
                    continue
                num_external_computed_tokens = 0
                load_kv_async = False
                if request.num_computed_tokens == 0:
                    new_computed_blocks, num_new_local_computed_tokens = self.kv_cache_manager.get_computed_blocks(request)
                    if self.connector is not None:
                        num_external_computed_tokens, load_kv_async = self.connector.get_num_new_matched_tokens(request, num_new_local_computed_tokens)
                    num_computed_tokens = num_new_local_computed_tokens + num_external_computed_tokens
                else:
                    new_computed_blocks = self.kv_cache_manager.create_empty_block_list()
                    num_new_local_computed_tokens = 0
                    num_computed_tokens = request.num_computed_tokens
                encoder_inputs_to_schedule = None
                new_encoder_budget = encoder_budget
                if load_kv_async:
                    assert num_external_computed_tokens > 0
                    num_new_tokens = 0
                else:
                    num_new_tokens = request.num_tokens - num_computed_tokens
                    if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
                        num_new_tokens = self.scheduler_config.long_prefill_token_threshold
                    if not self.scheduler_config.chunked_prefill_enabled and num_new_tokens > token_budget:
                        self.waiting.pop_request()
                        skipped_waiting_requests.prepend_request(request)
                        continue
                    num_new_tokens = min(num_new_tokens, token_budget)
                    assert num_new_tokens > 0
                    if request.has_encoder_inputs:
                        encoder_inputs_to_schedule, num_new_tokens, new_encoder_budget = self._try_schedule_encoder_inputs(request, num_computed_tokens, num_new_tokens, encoder_budget)
                        if num_new_tokens == 0:
                            break
                new_blocks = self.kv_cache_manager.allocate_slots(request, num_new_tokens + num_external_computed_tokens, num_new_local_computed_tokens, new_computed_blocks, num_lookahead_tokens=self.num_lookahead_tokens, delay_cache_blocks=load_kv_async)
                if new_blocks is None:
                    break
                if self.connector is not None:
                    self.connector.update_state_after_alloc(request, new_computed_blocks + new_blocks, num_external_computed_tokens)
                request = self.waiting.pop_request()
                if load_kv_async:
                    skipped_waiting_requests.prepend_request(request)
                    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
                    continue
                if request.use_structured_output:
                    structured_output_request_ids[request.request_id] = req_index
                req_index += 1
                self.running.append(request)
                if self.log_stats:
                    request.record_event(EngineCoreEventType.SCHEDULED, scheduled_timestamp)
                if request.status == RequestStatus.WAITING:
                    scheduled_new_reqs.append(request)
                elif request.status == RequestStatus.PREEMPTED:
                    scheduled_resumed_reqs.append(request)
                else:
                    raise RuntimeError(f'Invalid request status: {request.status}')
                if self.lora_config and request.lora_request:
                    scheduled_loras.add(request.lora_request.lora_int_id)
                req_to_new_block_ids[request.request_id] = self.kv_cache_manager.get_block_ids(request.request_id)
                num_scheduled_tokens[request.request_id] = num_new_tokens
                token_budget -= num_new_tokens
                request.status = RequestStatus.RUNNING
                request.num_computed_tokens = num_computed_tokens
                if request.num_cached_tokens < 0:
                    request.num_cached_tokens = num_computed_tokens
                if encoder_inputs_to_schedule:
                    scheduled_encoder_inputs[request.request_id] = encoder_inputs_to_schedule
                    for i in encoder_inputs_to_schedule:
                        self.encoder_cache_manager.allocate(request, i)
                    encoder_budget = new_encoder_budget
        if skipped_waiting_requests:
            self.waiting.prepend_requests(skipped_waiting_requests)
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens
        assert token_budget >= 0
        assert len(self.running) <= self.max_num_running_reqs
        assert len(scheduled_new_reqs) + len(scheduled_resumed_reqs) + len(scheduled_running_reqs) <= len(self.running)
        num_common_prefix_blocks = [0] * len(self.kv_cache_config.kv_cache_groups)
        if self.running:
            any_request = self.running[0]
            num_common_prefix_blocks = self.kv_cache_manager.get_num_common_prefix_blocks(any_request, len(self.running))
        grammar_bitmask = self.structured_output_manager.grammar_bitmask(self.requests, structured_output_request_ids, scheduled_spec_decode_tokens)
        new_reqs_data = [NewRequestData.from_request(req, req_to_new_block_ids[req.request_id]) for req in scheduled_new_reqs]
        cached_reqs_data = self._make_cached_request_data(scheduled_running_reqs, scheduled_resumed_reqs, num_scheduled_tokens, scheduled_spec_decode_tokens, req_to_new_block_ids)
        scheduler_output = SchedulerOutput(scheduled_new_reqs=new_reqs_data, scheduled_cached_reqs=cached_reqs_data, num_scheduled_tokens=num_scheduled_tokens, total_num_scheduled_tokens=total_num_scheduled_tokens, scheduled_spec_decode_tokens=scheduled_spec_decode_tokens, scheduled_encoder_inputs=scheduled_encoder_inputs, num_common_prefix_blocks=num_common_prefix_blocks, finished_req_ids=self.finished_req_ids, free_encoder_input_ids=self.encoder_cache_manager.get_freed_ids(), structured_output_request_ids=structured_output_request_ids, grammar_bitmask=grammar_bitmask)
        if self.connector is not None:
            meta = self.connector.build_connector_meta(scheduler_output)
            scheduler_output.kv_connector_metadata = meta
        events = self.kv_cache_manager.take_events()
        if events:
            batch = KVEventBatch(ts=time.time(), events=events)
            self.kv_event_publisher.publish(batch)
        self._update_after_schedule(scheduler_output)
        return scheduler_output

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        for req_id, num_scheduled_token in num_scheduled_tokens.items():
            request = self.requests[req_id]
            request.num_computed_tokens += num_scheduled_token
            if request.has_encoder_inputs:
                self._free_encoder_inputs(request)
        self.finished_req_ids = set()

    def _make_cached_request_data(self, running_reqs: list[Request], resumed_reqs: list[Request], num_scheduled_tokens: dict[str, int], spec_decode_tokens: dict[str, list[int]], req_to_new_block_ids: dict[str, tuple[list[int], ...]]) -> CachedRequestData:
        req_ids: list[str] = []
        new_token_ids: list[list[int]] = []
        new_block_ids: list[tuple[list[int], ...]] = []
        num_computed_tokens: list[int] = []
        use_connector = self.connector is not None
        for req in itertools.chain(running_reqs, resumed_reqs):
            req_id = req.request_id
            req_ids.append(req_id)
            num_tokens = num_scheduled_tokens[req_id] - len(spec_decode_tokens.get(req_id, ()))
            if self.use_pp:
                token_ids = req.all_token_ids[req.num_computed_tokens:req.num_computed_tokens + num_tokens]
                new_token_ids.append(token_ids)
            elif use_connector:
                new_token_ids.append([])
            new_block_ids.append(req_to_new_block_ids[req_id])
            num_computed_tokens.append(req.num_computed_tokens)
        resumed_from_preemption = [False] * len(running_reqs)
        resumed_from_preemption += [True] * len(resumed_reqs)
        return CachedRequestData(req_ids=req_ids, resumed_from_preemption=resumed_from_preemption, new_token_ids=new_token_ids, new_block_ids=new_block_ids, num_computed_tokens=num_computed_tokens)

    def _try_schedule_encoder_inputs(self, request: Request, num_computed_tokens: int, num_new_tokens: int, encoder_budget: int) -> tuple[list[int], int, int]:
        """
        Determine which encoder inputs need to be scheduled in the current step,
        and update `num_new_tokens` and encoder token budget accordingly.

        An encoder input will be scheduled if:
        - Its output tokens overlap with the range of tokens being computed
        in this step, i.e.,
        [num_computed_tokens, num_computed_tokens + num_new_tokens).
        - It is not already computed and stored in the encoder cache.
        - There is sufficient encoder token budget to process it.
        - The encoder cache has space to store it.

        If an encoder input cannot be scheduled due to cache or budget
        limitations, the method adjusts `num_new_tokens` to schedule only the
        decoder tokens up to just before the unschedulable encoder input.

        Note that num_computed_tokens includes both locally cached
        blocks and externally cached blocks (via KVConnector).
        """
        if num_new_tokens == 0 or not request.has_encoder_inputs:
            return ([], num_new_tokens, encoder_budget)
        encoder_inputs_to_schedule: list[int] = []
        mm_positions = request.mm_positions
        assert mm_positions is not None
        assert len(mm_positions) > 0
        for i, pos_info in enumerate(mm_positions):
            start_pos = pos_info.offset
            num_encoder_tokens = pos_info.length
            if start_pos >= num_computed_tokens + num_new_tokens:
                break
            if start_pos + num_encoder_tokens <= num_computed_tokens:
                continue
            if self.encoder_cache_manager.has_cache(request, i):
                continue
            if self.scheduler_config.disable_chunked_mm_input and num_computed_tokens < start_pos and (num_computed_tokens + num_new_tokens < start_pos + num_encoder_tokens):
                num_new_tokens = start_pos - num_computed_tokens
                break
            if not self.encoder_cache_manager.can_allocate(request, i) or num_encoder_tokens > encoder_budget:
                if num_computed_tokens < start_pos:
                    num_new_tokens = start_pos - num_computed_tokens
                else:
                    num_new_tokens = 0
                break
            encoder_budget -= num_encoder_tokens
            encoder_inputs_to_schedule.append(i)
        return (encoder_inputs_to_schedule, num_new_tokens, encoder_budget)

    def update_from_output(self, scheduler_output: SchedulerOutput, model_runner_output: ModelRunnerOutput) -> dict[int, EngineCoreOutputs]:
        sampled_token_ids = model_runner_output.sampled_token_ids
        spec_token_ids = model_runner_output.spec_token_ids
        logprobs = model_runner_output.logprobs
        prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        pooler_outputs = model_runner_output.pooler_output
        num_nans_in_logits = model_runner_output.num_nans_in_logits
        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
        spec_decoding_stats: Optional[SpecDecodingStats] = None
        stopped_running_reqs: set[Request] = set()
        stopped_preempted_reqs: set[Request] = set()
        for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
            assert num_tokens_scheduled > 0
            request = self.requests.get(req_id)
            if request is None:
                continue
            req_index = model_runner_output.req_id_to_index[req_id]
            generated_token_ids = sampled_token_ids[req_index] if sampled_token_ids else []
            scheduled_spec_token_ids = scheduler_output.scheduled_spec_decode_tokens.get(req_id)
            if scheduled_spec_token_ids:
                num_tokens_rejected = len(scheduled_spec_token_ids) + 1 - len(generated_token_ids)
                request.num_computed_tokens -= num_tokens_rejected
                spec_decoding_stats = self.make_spec_decoding_stats(spec_decoding_stats, num_draft_tokens=len(scheduled_spec_token_ids), num_accepted_tokens=len(generated_token_ids) - 1)
            stopped = False
            new_logprobs = None
            new_token_ids = generated_token_ids
            kv_transfer_params = None
            status_before_stop = request.status
            if new_token_ids:
                new_token_ids, stopped = self._update_request_with_output(request, new_token_ids)
            pooler_output = None
            if pooler_outputs:
                pooler_output = pooler_outputs[req_index]
                stopped = check_stop(request, self.max_model_len, pooler_output)
            if stopped:
                kv_transfer_params = self._free_request(request)
                if status_before_stop == RequestStatus.RUNNING:
                    stopped_running_reqs.add(request)
                else:
                    stopped_preempted_reqs.add(request)
            if request.sampling_params is not None and request.sampling_params.logprobs is not None and logprobs:
                new_logprobs = logprobs.slice(req_index, req_index + 1)
            if new_token_ids and self.structured_output_manager.should_advance(request):
                request.structured_output_request.grammar.accept_tokens(req_id, new_token_ids)
            if num_nans_in_logits is not None and req_id in num_nans_in_logits:
                request.num_nans_in_logits = num_nans_in_logits[req_id]
            if spec_token_ids is not None:
                if self.structured_output_manager.should_advance(request):
                    metadata = request.structured_output_request
                    request.spec_token_ids = metadata.grammar.validate_tokens(spec_token_ids[req_index])
                else:
                    request.spec_token_ids = spec_token_ids[req_index]
            prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
            if new_token_ids or pooler_output is not None or kv_transfer_params:
                outputs[request.client_index].append(EngineCoreOutput(request_id=req_id, new_token_ids=new_token_ids, finish_reason=request.get_finished_reason(), new_logprobs=new_logprobs, new_prompt_logprobs_tensors=prompt_logprobs_tensors, pooling_output=pooler_output, stop_reason=request.stop_reason, events=request.take_events(), kv_transfer_params=kv_transfer_params, num_cached_tokens=request.num_cached_tokens))
            else:
                assert not prompt_logprobs_tensors
        if stopped_running_reqs:
            self.running = [req for req in self.running if req not in stopped_running_reqs]
        if stopped_preempted_reqs:
            self.waiting.remove_requests(stopped_preempted_reqs)
        self._update_from_kv_xfer_finished(model_runner_output)
        engine_core_outputs = {client_index: EngineCoreOutputs(outputs=outs) for client_index, outs in outputs.items()}
        finished_req_ids = self.finished_req_ids_dict
        if finished_req_ids:
            for client_index, finished_set in finished_req_ids.items():
                if (eco := engine_core_outputs.get(client_index)) is not None:
                    eco.finished_requests = finished_set
                else:
                    engine_core_outputs[client_index] = EngineCoreOutputs(finished_requests=finished_set)
            finished_req_ids.clear()
        if engine_core_outputs:
            next(iter(engine_core_outputs.values())).scheduler_stats = self.make_stats(spec_decoding_stats)
        return engine_core_outputs

    def _update_request_with_output(self, request: Request, new_token_ids: list[int]) -> tuple[list[int], bool]:
        stopped = False
        for num_new, output_token_id in enumerate(new_token_ids, 1):
            request.append_output_token_ids(output_token_id)
            stopped = check_stop(request, self.max_model_len)
            if stopped:
                del new_token_ids[num_new:]
                break
        return (new_token_ids, stopped)

    def _free_encoder_inputs(self, request: Request) -> None:
        cached_encoder_input_ids = self.encoder_cache_manager.get_cached_input_ids(request)
        if not cached_encoder_input_ids:
            return
        for input_id in list(cached_encoder_input_ids):
            mm_positions = request.mm_positions[input_id]
            start_pos = mm_positions.offset
            num_tokens = mm_positions.length
            if start_pos + num_tokens <= request.num_computed_tokens:
                self.encoder_cache_manager.free_encoder_input(request, input_id)

    def get_request_counts(self) -> tuple[int, int]:
        """Returns (num_running_reqs, num_waiting_reqs)."""
        return (len(self.running), len(self.waiting))

    def add_request(self, request: Request) -> None:
        self.waiting.add_request(request)
        self.requests[request.request_id] = request
        if self.log_stats:
            request.record_event(EngineCoreEventType.QUEUED)

    def finish_requests(self, request_ids: Union[str, Iterable[str]], finished_status: RequestStatus) -> None:
        """Handles the finish signal from outside the scheduler.

        For example, the API server can abort a request when the client
        disconnects.
        """
        assert RequestStatus.is_finished(finished_status)
        if isinstance(request_ids, str):
            request_ids = (request_ids,)
        else:
            request_ids = set(request_ids)
        running_requests_to_remove = []
        waiting_requests_to_remove = []
        valid_requests = []
        for req_id in request_ids:
            request = self.requests.get(req_id)
            if request is None:
                continue
            valid_requests.append(request)
            if request.status == RequestStatus.RUNNING:
                running_requests_to_remove.append(request)
            else:
                waiting_requests_to_remove.append(request)
        for request in running_requests_to_remove:
            self.running.remove(request)
        if waiting_requests_to_remove:
            self.waiting.remove_requests(waiting_requests_to_remove)
        for request in valid_requests:
            request.status = finished_status
            self._free_request(request)

    def _free_request(self, request: Request) -> Optional[dict[str, Any]]:
        assert request.is_finished()
        delay_free_blocks, kv_xfer_params = self._connector_finished(request)
        self.encoder_cache_manager.free(request)
        request_id = request.request_id
        self.finished_req_ids.add(request_id)
        if self.finished_req_ids_dict is not None:
            self.finished_req_ids_dict[request.client_index].add(request_id)
        if not delay_free_blocks:
            self._free_blocks(request)
        return kv_xfer_params

    def _free_blocks(self, request: Request):
        assert request.is_finished()
        self.kv_cache_manager.free(request)
        self.kv_cache_manager.free_block_hashes(request)
        del self.requests[request.request_id]

    def get_num_unfinished_requests(self) -> int:
        return len(self.waiting) + len(self.running)

    def has_finished_requests(self) -> bool:
        return len(self.finished_req_ids) > 0

    def reset_prefix_cache(self) -> bool:
        return self.kv_cache_manager.reset_prefix_cache()

    def make_stats(self, spec_decoding_stats: Optional[SpecDecodingStats]=None) -> Optional[SchedulerStats]:
        if not self.log_stats:
            return None
        prefix_cache_stats = self.kv_cache_manager.make_prefix_cache_stats()
        assert prefix_cache_stats is not None
        return SchedulerStats(num_running_reqs=len(self.running), num_waiting_reqs=len(self.waiting), kv_cache_usage=self.kv_cache_manager.usage, prefix_cache_stats=prefix_cache_stats, spec_decoding_stats=spec_decoding_stats, num_corrupted_reqs=sum((req.is_output_corrupted for req in self.running)))

    def make_spec_decoding_stats(self, spec_decoding_stats: Optional[SpecDecodingStats], num_draft_tokens: int, num_accepted_tokens: int) -> Optional[SpecDecodingStats]:
        if not self.log_stats:
            return None
        if spec_decoding_stats is None:
            spec_decoding_stats = SpecDecodingStats.new(self.num_spec_tokens)
        spec_decoding_stats.observe_draft(num_draft_tokens=num_draft_tokens, num_accepted_tokens=num_accepted_tokens)
        return spec_decoding_stats

    def shutdown(self) -> None:
        if self.kv_event_publisher:
            self.kv_event_publisher.shutdown()

    def get_kv_connector(self) -> Optional[KVConnectorBase_V1]:
        return self.connector

    def _connector_finished(self, request: Request) -> tuple[bool, Optional[dict[str, Any]]]:
        """
        Invoke the KV connector request_finished() method if applicable.

        Returns optional kv transfer parameters to be included with the
        request outputs.
        """
        if self.connector is None:
            return (False, None)
        block_ids, = self.kv_cache_manager.get_block_ids(request.request_id)
        return self.connector.request_finished(request, block_ids)

    def _update_waiting_for_remote_kv(self, request: Request) -> bool:
        """
        KV Connector: check if the request_id is finished_recving.

        The finished_recving_kv_req_ids list is populated
        on the previous steps()'s update_from_output based
        on the worker side connector.

        When the kv transfer is ready, we cache the blocks
        and the request state will be moved back to WAITING from
        WAITING_FOR_REMOTE_KV.
        """
        assert self.connector is not None
        if request.request_id not in self.finished_recving_kv_req_ids:
            return False
        block_ids, = self.kv_cache_manager.get_block_ids(request.request_id)
        num_computed_tokens = len(block_ids) * self.block_size
        num_computed_tokens = min(num_computed_tokens, request.num_tokens)
        if num_computed_tokens == request.num_tokens:
            num_computed_tokens -= 1
        self.kv_cache_manager.cache_blocks(request, num_computed_tokens)
        request.num_computed_tokens = num_computed_tokens
        self.finished_recving_kv_req_ids.remove(request.request_id)
        return True

    def _update_from_kv_xfer_finished(self, model_runner_output: ModelRunnerOutput):
        """
        KV Connector: update the scheduler state based on the output.

        The Worker side connectors add finished_recving and
        finished_sending reqs to the output.
        * if finished_sending: free the blocks
        # if finished_recving: add to state so we can
            scheduler the request during the next step.
        """
        for req_id in model_runner_output.finished_recving or ():
            logger.debug('Finished recving KV transfer for request %s', req_id)
            self.finished_recving_kv_req_ids.add(req_id)
        for req_id in model_runner_output.finished_sending or ():
            logger.debug('Finished sending KV transfer for request %s', req_id)
            self._free_blocks(self.requests[req_id])