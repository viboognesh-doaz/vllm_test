from ..sample.logits_processor import LogitsProcessorManager
from .utils import bind_kv_cache, gather_mm_placeholders, initialize_kv_cache_for_kv_sharing, sanity_check_mm_encoder_outputs, scatter_mm_placeholders
from contextlib import contextmanager
from tqdm import tqdm
from typing import TYPE_CHECKING, Any, Optional, Union, cast, get_args
from vllm.attention import AttentionType, get_attn_backend
from vllm.attention.backends.abstract import AttentionBackend
from vllm.attention.layer import Attention
from vllm.compilation.counter import compilation_counter
from vllm.config import CompilationLevel, VllmConfig, get_layers_from_vllm_config, update_config
from vllm.distributed.eplb.eplb_state import EplbState
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorBase_V1
from vllm.distributed.parallel_state import get_ep_group
from vllm.distributed.parallel_state import get_pp_group, get_tp_group, graph_capture, is_global_first_rank, prepare_communication_buffer_for_model
from vllm.forward_context import DPMetadata, get_forward_context, set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaBase
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding
from vllm.model_executor.model_loader import TensorizerLoader, get_model_loader
from vllm.model_executor.model_loader.tensorizer import TensorizerConfig
from vllm.model_executor.models.interfaces import is_mixture_of_experts
from vllm.model_executor.models.interfaces_base import VllmModelForPooling, is_pooling_model
from vllm.model_executor.models.utils import set_cpu_offload_max_bytes
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalKwargs, PlaceholderRange
from vllm.multimodal.utils import group_mm_inputs_by_modality
from vllm.pooling_params import PoolingParams, PoolingTask
from vllm.sampling_params import SamplingType
from vllm.sequence import IntermediateTensors
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE, DeviceMemoryProfiler, GiB_bytes, LazyLoader, check_use_alibi, get_dtype_size, is_pin_memory_available, round_up
from vllm.v1.attention.backends.mamba_attn import Mamba2AttentionBackend
from vllm.v1.attention.backends.utils import AttentionMetadataBuilder, CommonAttentionMetadata, make_local_attention_virtual_batches
from vllm.v1.core.encoder_cache_manager import compute_encoder_budget
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import AttentionSpec, ChunkedLocalAttentionSpec, FullAttentionSpec, KVCacheConfig, KVCacheSpec, MambaSpec, SlidingWindowSpec
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, LogprobsTensors, ModelRunnerOutput
from vllm.v1.pool.metadata import PoolingMetadata
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import RejectionSampler
from vllm.v1.sample.sampler import Sampler
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.spec_decode.medusa import MedusaProposer
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.spec_decode.ngram_proposer import NgramProposer
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin
import copy
import functools
import gc
import numpy as np
import time
import torch
import torch.distributed
import torch.nn as nn
import vllm.envs as envs
import xgrammar as xgr
import xgrammar.kernels.apply_token_bitmask_inplace_torch_compile as xgr_torch_compile
if TYPE_CHECKING:
else:
    xgr = LazyLoader('xgr', globals(), 'xgrammar')
    xgr_torch_compile = LazyLoader('xgr_torch_compile', globals(), 'xgrammar.kernels.apply_token_bitmask_inplace_torch_compile')
logger = init_logger(__name__)

class GPUModelRunner(LoRAModelRunnerMixin):

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.compilation_config = vllm_config.compilation_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.speculative_config = vllm_config.speculative_config
        self.prompt_adapter_config = vllm_config.prompt_adapter_config
        self.observability_config = vllm_config.observability_config
        set_cpu_offload_max_bytes(int(self.cache_config.cpu_offload_gb * 1024 ** 3))
        model_config = self.model_config
        cache_config = self.cache_config
        scheduler_config = self.scheduler_config
        parallel_config = self.parallel_config
        self.device = device
        self.pin_memory = is_pin_memory_available()
        self.dtype = self.model_config.dtype
        if cache_config.cache_dtype == 'auto':
            self.kv_cache_dtype = self.dtype
        else:
            self.kv_cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]
        self.is_multimodal_model = model_config.is_multimodal_model
        self.is_pooling_model = model_config.pooler_config is not None
        self.max_model_len = model_config.max_model_len
        self.max_num_tokens = scheduler_config.max_num_batched_tokens
        self.max_num_reqs = scheduler_config.max_num_seqs
        self.num_query_heads = model_config.get_num_attention_heads(parallel_config)
        self.hidden_size = model_config.get_hidden_size()
        self.attention_chunk_size = model_config.attention_chunk_size
        self.cascade_attn_enabled = not self.model_config.disable_cascade_attn
        self.mm_registry = MULTIMODAL_REGISTRY
        self.uses_mrope = model_config.uses_mrope
        encoder_compute_budget, encoder_cache_size = compute_encoder_budget(model_config=model_config, scheduler_config=scheduler_config, mm_registry=self.mm_registry)
        self.max_num_encoder_input_tokens = encoder_compute_budget
        self.encoder_cache_size = encoder_cache_size
        self.sampler = Sampler()
        self.eplb_state: Optional[EplbState] = None
        '\n        State of the expert parallelism load balancer.\n\n        Will be lazily initialized when the model is loaded.\n        '
        self.kv_caches: list[torch.Tensor] = []
        self.attn_metadata_builders: list[AttentionMetadataBuilder] = []
        self.attn_backends: list[type[AttentionBackend]] = []
        self.encoder_cache: dict[str, dict[int, torch.Tensor]] = {}
        self.use_aux_hidden_state_outputs = False
        if self.speculative_config and get_pp_group().is_last_rank:
            if self.speculative_config.method == 'ngram':
                self.drafter = NgramProposer(self.vllm_config)
            elif self.speculative_config.use_eagle():
                self.drafter = EagleProposer(self.vllm_config, self.device, self)
                if self.speculative_config.method == 'eagle3':
                    self.use_aux_hidden_state_outputs = True
            elif self.speculative_config.method == 'medusa':
                self.drafter = MedusaProposer(vllm_config=self.vllm_config, device=self.device)
            else:
                raise ValueError(f'Unknown speculative decoding method: {self.speculative_config.method}')
            self.rejection_sampler = RejectionSampler()
        self.requests: dict[str, CachedRequestState] = {}
        self.input_batch = InputBatch(max_num_reqs=self.max_num_reqs, max_model_len=self.max_model_len, max_num_batched_tokens=self.max_num_tokens, device=self.device, pin_memory=self.pin_memory, vocab_size=self.model_config.get_vocab_size(), block_sizes=[self.cache_config.block_size], is_spec_decode=bool(self.vllm_config.speculative_config))
        self.use_cuda_graph = self.vllm_config.compilation_config.level == CompilationLevel.PIECEWISE and self.vllm_config.compilation_config.use_cudagraph and (not self.model_config.enforce_eager)
        self.cudagraph_batch_sizes = list(reversed(self.compilation_config.cudagraph_capture_sizes))
        self.full_cuda_graph = self.compilation_config.full_cuda_graph
        self._init_device_properties()
        self.input_ids = torch.zeros(self.max_num_tokens, dtype=torch.int32, device=self.device)
        self.positions = torch.zeros(self.max_num_tokens, dtype=torch.int64, device=self.device)
        self.query_start_loc = torch.zeros(self.max_num_reqs + 1, dtype=torch.int32, device=self.device)
        self.seq_lens = torch.zeros(self.max_num_reqs, dtype=torch.int32, device=self.device)
        self.slot_mapping = torch.zeros(self.max_num_tokens, dtype=torch.int64, device=self.device)
        self.intermediate_tensors: Optional[IntermediateTensors] = None
        if self.uses_mrope:
            self.mrope_positions = torch.zeros((3, self.max_num_tokens + 1), dtype=torch.int64, device=self.device)
            self.mrope_positions_cpu = torch.zeros((3, self.max_num_tokens + 1), dtype=torch.int64, device='cpu', pin_memory=self.pin_memory)
            self.mrope_positions_np = self.mrope_positions_cpu.numpy()
        self.use_alibi = check_use_alibi(model_config)
        self.inputs_embeds = torch.zeros((self.max_num_tokens, self.hidden_size), dtype=self.dtype, device=self.device)
        self.arange_np = np.arange(max(self.max_num_reqs + 1, self.max_model_len, self.max_num_tokens), dtype=np.int64)
        self.input_ids_cpu = torch.zeros(self.max_num_tokens, dtype=torch.int32, device='cpu', pin_memory=self.pin_memory)
        self.positions_cpu = torch.zeros(self.max_num_tokens, dtype=torch.int64, device='cpu', pin_memory=self.pin_memory)
        self.positions_np = self.positions_cpu.numpy()
        self.query_start_loc_cpu = torch.zeros(self.max_num_reqs + 1, dtype=torch.int32, device='cpu', pin_memory=self.pin_memory)
        self.query_start_loc_np = self.query_start_loc_cpu.numpy()
        self.seq_lens_cpu = torch.zeros(self.max_num_reqs, dtype=torch.int32, device='cpu', pin_memory=self.pin_memory)
        self.seq_lens_np = self.seq_lens_cpu.numpy()
        self.shared_kv_cache_layers: dict[str, str] = {}

    def _may_reorder_batch(self, scheduler_output: 'SchedulerOutput') -> None:
        """
        Update the order of requests in the batch based on the attention
        backend's needs. For example, some attention backends (namely MLA) may
        want to separate requests based on if the attention computation will be
        compute-bound or memory-bound.

        Args:
            scheduler_output: The scheduler output.
        """
        self.attn_metadata_builders[0].reorder_batch(self.input_batch, scheduler_output)
        for i in range(1, len(self.kv_cache_config.kv_cache_groups)):
            batch_reordered = self.attn_metadata_builders[i].reorder_batch(self.input_batch, scheduler_output)
            assert not batch_reordered

    def _init_device_properties(self) -> None:
        """Initialize attributes from torch.cuda.get_device_properties
        """
        self.device_properties = torch.cuda.get_device_properties(self.device)
        self.num_sms = self.device_properties.multi_processor_count

    def _sync_device(self) -> None:
        torch.cuda.synchronize()

    def _update_states(self, scheduler_output: 'SchedulerOutput') -> None:
        """Update the cached states and the persistent batch with the scheduler
        output.

        The updated states are used by the `_prepare_inputs` function to create
        the input GPU tensors for the model.

        The SamplingMetadata is updated and copied to the GPU if there is a
        new/resumed/paused/finished request in the batch.
        """
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)
            self.encoder_cache.pop(req_id, None)
        for req_id in scheduler_output.finished_req_ids:
            self.input_batch.remove_request(req_id)
        for req_id, input_id in scheduler_output.free_encoder_input_ids:
            encoder_outputs = self.encoder_cache.get(req_id)
            if encoder_outputs is not None:
                encoder_outputs.pop(input_id, None)
                if not encoder_outputs:
                    self.encoder_cache.pop(req_id, None)
        scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()
        cached_req_ids = self.input_batch.req_id_to_index.keys()
        unscheduled_req_ids = cached_req_ids - scheduled_req_ids
        for req_id in unscheduled_req_ids:
            self.input_batch.remove_request(req_id)
        req_ids_to_add: list[str] = []
        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_id = new_req_data.req_id
            sampling_params = new_req_data.sampling_params
            pooling_params = new_req_data.pooling_params
            if sampling_params and sampling_params.sampling_type == SamplingType.RANDOM_SEED:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(sampling_params.seed)
            else:
                generator = None
            if pooling_params:
                assert pooling_params.task is not None, 'You did not set `task` in the API'
                model = cast(VllmModelForPooling, self.model)
                to_update = model.pooler.get_pooling_updates(pooling_params.task)
                assert to_update is not None, f'pooling_params.task={pooling_params.task!r} is not supported by the model'
                to_update.apply(pooling_params)
            self.requests[req_id] = CachedRequestState(req_id=req_id, prompt_token_ids=new_req_data.prompt_token_ids, mm_inputs=new_req_data.mm_inputs, mm_positions=new_req_data.mm_positions, sampling_params=sampling_params, pooling_params=pooling_params, generator=generator, block_ids=new_req_data.block_ids, num_computed_tokens=new_req_data.num_computed_tokens, output_token_ids=[], lora_request=new_req_data.lora_request)
            if self.uses_mrope:
                image_grid_thw = []
                video_grid_thw = []
                second_per_grid_ts = []
                audio_feature_lengths = []
                use_audio_in_video = False
                for mm_input in self.requests[req_id].mm_inputs:
                    if mm_input.get('image_grid_thw') is not None:
                        image_grid_thw.extend(mm_input['image_grid_thw'].tolist())
                    if mm_input.get('video_grid_thw') is not None:
                        video_grid_thw.extend(mm_input['video_grid_thw'].tolist())
                    if mm_input.get('second_per_grid_ts') is not None:
                        second_per_grid_ts.extend(mm_input['second_per_grid_ts'])
                    if mm_input.get('audio_feature_lengths') is not None:
                        audio_feature_lengths.extend(mm_input['audio_feature_lengths'])
                    if mm_input.get('use_audio_in_video') is True:
                        use_audio_in_video = True
                hf_config = self.model_config.hf_config
                self.requests[req_id].mrope_positions, self.requests[req_id].mrope_position_delta = MRotaryEmbedding.get_input_positions_tensor(self.requests[req_id].prompt_token_ids, hf_config=hf_config, image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw, second_per_grid_ts=second_per_grid_ts, audio_feature_lengths=audio_feature_lengths, use_audio_in_video=use_audio_in_video)
            req_ids_to_add.append(req_id)
        is_last_rank = get_pp_group().is_last_rank
        req_data = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(req_data.req_ids):
            req_state = self.requests[req_id]
            num_computed_tokens = req_data.num_computed_tokens[i]
            new_block_ids = req_data.new_block_ids[i]
            resumed_from_preemption = req_data.resumed_from_preemption[i]
            req_state.num_computed_tokens = num_computed_tokens
            if not is_last_rank:
                new_token_ids = req_data.new_token_ids[i]
                num_new_tokens = num_computed_tokens + len(new_token_ids) - req_state.num_tokens
                if num_new_tokens == 1:
                    req_state.output_token_ids.append(new_token_ids[-1])
                elif num_new_tokens > 0:
                    req_state.output_token_ids.extend(new_token_ids[-num_new_tokens:])
            if not resumed_from_preemption:
                for block_ids, new_ids in zip(req_state.block_ids, new_block_ids):
                    block_ids.extend(new_ids)
            else:
                req_state.block_ids = new_block_ids
            req_index = self.input_batch.req_id_to_index.get(req_id)
            if req_index is None:
                req_ids_to_add.append(req_id)
                continue
            self.input_batch.num_computed_tokens_cpu[req_index] = num_computed_tokens
            self.input_batch.block_table.append_row(new_block_ids, req_index)
            if not is_last_rank:
                start_token_index = num_computed_tokens
                end_token_index = num_computed_tokens + len(new_token_ids)
                self.input_batch.token_ids_cpu[req_index, start_token_index:end_token_index] = new_token_ids
                self.input_batch.num_tokens_no_spec[req_index] = end_token_index
                self.input_batch.num_tokens[req_index] = end_token_index
            spec_token_ids = scheduler_output.scheduled_spec_decode_tokens.get(req_id, ())
            if spec_token_ids:
                num_spec_tokens = len(spec_token_ids)
                start_index = self.input_batch.num_tokens_no_spec[req_index]
                end_token_index = start_index + num_spec_tokens
                self.input_batch.token_ids_cpu[req_index, start_index:end_token_index] = spec_token_ids
                self.input_batch.num_tokens[req_index] += num_spec_tokens
        for req_id in req_ids_to_add:
            req_state = self.requests[req_id]
            self.input_batch.add_request(req_state)
        self.input_batch.condense()
        self._may_reorder_batch(scheduler_output)
        self.input_batch.refresh_metadata()

    def _get_cumsum_and_arange(self, num_tokens: np.ndarray, cumsum_dtype: Optional[np.dtype]=None) -> tuple[np.ndarray, np.ndarray]:
        """Get the cumulative sum and batched arange of the given array.
        # E.g., [2, 5, 3] -> ([2, 7, 10], [0, 1, 0, 1, 2, 3, 4, 0, 1, 2])
        # Equivalent to but faster than:
        # np.concatenate([np.arange(n) for n in num_tokens])
        """
        cu_num_tokens = np.cumsum(num_tokens, dtype=cumsum_dtype)
        total_num_tokens = cu_num_tokens[-1]
        cumsums_offsets = np.repeat(cu_num_tokens - num_tokens, num_tokens)
        arange = self.arange_np[:total_num_tokens] - cumsums_offsets
        return (cu_num_tokens, arange)

    def _prepare_inputs(self, scheduler_output: 'SchedulerOutput') -> tuple[dict[str, Any], bool, torch.Tensor, Optional[SpecDecodeMetadata], np.ndarray, Optional[CommonAttentionMetadata]]:
        """
        :return: tuple[
            attn_metadata: layer-to-attention_metadata mapping,
            attention_cuda_graphs: whether attention can run in cudagraph
            logits_indices, spec_decode_metadata
        ]
        """
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0
        self.input_batch.block_table.commit_block_table(num_reqs)
        req_ids = self.input_batch.req_ids
        tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
        num_scheduled_tokens = np.array(tokens, dtype=np.int32)
        max_num_scheduled_tokens = max(tokens)
        req_indices = np.repeat(self.arange_np[:num_reqs], num_scheduled_tokens)
        cu_num_tokens, arange = self._get_cumsum_and_arange(num_scheduled_tokens)
        positions_np = self.positions_np[:total_num_scheduled_tokens]
        np.add(self.input_batch.num_computed_tokens_cpu[req_indices], arange, out=positions_np)
        if self.uses_mrope:
            self._calc_mrope_positions(scheduler_output)
        token_indices = positions_np + req_indices * self.input_batch.token_ids_cpu.shape[1]
        torch.index_select(self.input_batch.token_ids_cpu_tensor.flatten(), 0, torch.from_numpy(token_indices), out=self.input_ids_cpu[:total_num_scheduled_tokens])
        self.input_batch.block_table.compute_slot_mapping(req_indices, positions_np)
        self.input_batch.block_table.commit_slot_mapping(total_num_scheduled_tokens)
        self.query_start_loc_np[0] = 0
        self.query_start_loc_np[1:num_reqs + 1] = cu_num_tokens
        self.seq_lens_np[:num_reqs] = self.input_batch.num_computed_tokens_cpu[:num_reqs] + num_scheduled_tokens
        self.input_ids[:total_num_scheduled_tokens].copy_(self.input_ids_cpu[:total_num_scheduled_tokens], non_blocking=True)
        if self.uses_mrope:
            self.mrope_positions[:, :total_num_scheduled_tokens].copy_(self.mrope_positions_cpu[:, :total_num_scheduled_tokens], non_blocking=True)
        else:
            self.positions[:total_num_scheduled_tokens].copy_(self.positions_cpu[:total_num_scheduled_tokens], non_blocking=True)
        self.query_start_loc[:num_reqs + 1].copy_(self.query_start_loc_cpu[:num_reqs + 1], non_blocking=True)
        self.seq_lens[:num_reqs].copy_(self.seq_lens_cpu[:num_reqs], non_blocking=True)
        self.seq_lens[num_reqs:].fill_(0)
        self.query_start_loc[num_reqs + 1:].fill_(self.query_start_loc_cpu[num_reqs].item())
        query_start_loc = self.query_start_loc[:num_reqs + 1]
        spec_decode_common_attn_metadata = None
        attn_metadata: dict[str, Any] = {}
        for kv_cache_group_id, kv_cache_group_spec in enumerate(self.kv_cache_config.kv_cache_groups):
            blk_table = self.input_batch.block_table[kv_cache_group_id]
            blk_table_tensor = blk_table.get_device_tensor()[:num_reqs]
            slot_mapping = blk_table.slot_mapping[:total_num_scheduled_tokens]
            blk_table.slot_mapping[total_num_scheduled_tokens:].fill_(-1)
            common_attn_metadata = CommonAttentionMetadata(query_start_loc=self.query_start_loc[:num_reqs + 1], query_start_loc_cpu=self.query_start_loc_cpu[:num_reqs + 1], seq_lens=self.seq_lens[:num_reqs], seq_lens_cpu=self.seq_lens_cpu[:num_reqs], num_computed_tokens_cpu=self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs], num_reqs=num_reqs, num_actual_tokens=total_num_scheduled_tokens, max_query_len=max_num_scheduled_tokens, block_table_tensor=blk_table_tensor, slot_mapping=slot_mapping)
            if self.speculative_config and spec_decode_common_attn_metadata is None:
                spec_decode_common_attn_metadata = common_attn_metadata
            if isinstance(kv_cache_group_spec.kv_cache_spec, ChunkedLocalAttentionSpec):
                common_attn_metadata = make_local_attention_virtual_batches(kv_cache_group_spec.kv_cache_spec.attention_chunk_size, common_attn_metadata, self.cache_config.block_size)
            common_prefix_len = 0
            builder = self.attn_metadata_builders[kv_cache_group_id]
            if self.cascade_attn_enabled:
                common_prefix_len = self._compute_cascade_attn_prefix_len(num_scheduled_tokens, scheduler_output.num_common_prefix_blocks[kv_cache_group_id], kv_cache_group_spec.kv_cache_spec, builder)
            attn_metadata_i = builder.build(common_prefix_len=common_prefix_len, common_attn_metadata=common_attn_metadata)
            for layer_name in kv_cache_group_spec.layer_names:
                attn_metadata[layer_name] = attn_metadata_i
        attention_cuda_graphs = all((b.can_run_in_cudagraph(common_attn_metadata) for b in self.attn_metadata_builders))
        use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
        if not use_spec_decode:
            logits_indices = query_start_loc[1:] - 1
            spec_decode_metadata = None
        else:
            num_draft_tokens = np.zeros(num_reqs, dtype=np.int32)
            for req_id, draft_token_ids in scheduler_output.scheduled_spec_decode_tokens.items():
                req_idx = self.input_batch.req_id_to_index[req_id]
                num_draft_tokens[req_idx] = len(draft_token_ids)
            spec_decode_metadata = self._calc_spec_decode_metadata(num_draft_tokens, cu_num_tokens)
            logits_indices = spec_decode_metadata.logits_indices
        if self.lora_config:
            self.set_active_loras(self.input_batch, num_scheduled_tokens)
        return (attn_metadata, attention_cuda_graphs, logits_indices, spec_decode_metadata, num_scheduled_tokens, spec_decode_common_attn_metadata)

    def _compute_cascade_attn_prefix_len(self, num_scheduled_tokens: np.ndarray, num_common_prefix_blocks: int, kv_cache_spec: KVCacheSpec, attn_metadata_builder: AttentionMetadataBuilder) -> int:
        """Compute the length of the common prefix for cascade attention.

        NOTE(woosuk): The common prefix length returned by this function
        represents the length used specifically for cascade attention, not the
        actual number of tokens shared between requests. When cascade attention
        is disabled (use_cascade=False), this function returns 0 even if
        requests share common tokens. Additionally, the common prefix length is
        truncated to a multiple of the block size and may be further truncated
        due to implementation details explained below.

        Args:
            num_scheduled_tokens: Number of tokens scheduled per request.
            num_common_prefix_blocks: Number of shared KV cache blocks.

        Returns:
            int: Length of common prefix in tokens.
        """
        common_prefix_len = num_common_prefix_blocks * kv_cache_spec.block_size
        if common_prefix_len == 0:
            return 0
        num_reqs = len(num_scheduled_tokens)
        common_prefix_len = min(common_prefix_len, self.input_batch.num_computed_tokens_cpu[:num_reqs].min())
        common_prefix_len = common_prefix_len // kv_cache_spec.block_size * kv_cache_spec.block_size
        use_sliding_window = isinstance(kv_cache_spec, SlidingWindowSpec) or (isinstance(kv_cache_spec, FullAttentionSpec) and kv_cache_spec.sliding_window is not None)
        use_local_attention = isinstance(kv_cache_spec, ChunkedLocalAttentionSpec) or (isinstance(kv_cache_spec, FullAttentionSpec) and kv_cache_spec.attention_chunk_size is not None)
        assert isinstance(kv_cache_spec, AttentionSpec)
        use_cascade = attn_metadata_builder.use_cascade_attention(common_prefix_len=common_prefix_len, query_lens=num_scheduled_tokens, num_query_heads=self.num_query_heads, num_kv_heads=kv_cache_spec.num_kv_heads, use_alibi=self.use_alibi, use_sliding_window=use_sliding_window, use_local_attention=use_local_attention, num_sms=self.num_sms)
        return common_prefix_len if use_cascade else 0

    def _calc_mrope_positions(self, scheduler_output: 'SchedulerOutput'):
        mrope_pos_ptr = 0
        for index, req_id in enumerate(self.input_batch.req_ids):
            req = self.requests[req_id]
            assert req.mrope_positions is not None
            num_computed_tokens = self.input_batch.num_computed_tokens_cpu[index]
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
            num_prompt_tokens = len(req.prompt_token_ids)
            if num_computed_tokens + num_scheduled_tokens > num_prompt_tokens:
                prompt_part_len = max(0, num_prompt_tokens - num_computed_tokens)
                completion_part_len = max(0, num_scheduled_tokens - prompt_part_len)
            else:
                prompt_part_len = num_scheduled_tokens
                completion_part_len = 0
            assert num_scheduled_tokens == prompt_part_len + completion_part_len
            if prompt_part_len > 0:
                dst_start = mrope_pos_ptr
                dst_end = mrope_pos_ptr + prompt_part_len
                src_start = num_computed_tokens
                src_end = num_computed_tokens + prompt_part_len
                self.mrope_positions_cpu[:, dst_start:dst_end] = req.mrope_positions[:, src_start:src_end]
                mrope_pos_ptr += prompt_part_len
            if completion_part_len > 0:
                dst_start = mrope_pos_ptr
                dst_end = mrope_pos_ptr + completion_part_len
                MRotaryEmbedding.get_next_input_positions_tensor(out=self.mrope_positions_np, out_offset=dst_start, mrope_position_delta=req.mrope_position_delta, context_len=num_computed_tokens + prompt_part_len, num_new_tokens=completion_part_len)
                mrope_pos_ptr += completion_part_len

    def _calc_spec_decode_metadata(self, num_draft_tokens: np.ndarray, cu_num_scheduled_tokens: np.ndarray) -> SpecDecodeMetadata:
        num_sampled_tokens = num_draft_tokens + 1
        cu_num_sampled_tokens, arange = self._get_cumsum_and_arange(num_sampled_tokens, cumsum_dtype=np.int32)
        logits_indices = np.repeat(cu_num_scheduled_tokens - num_sampled_tokens, num_sampled_tokens)
        logits_indices += arange
        bonus_logits_indices = cu_num_sampled_tokens - 1
        cu_num_draft_tokens, arange = self._get_cumsum_and_arange(num_draft_tokens, cumsum_dtype=np.int32)
        target_logits_indices = np.repeat(cu_num_sampled_tokens - num_sampled_tokens, num_draft_tokens)
        target_logits_indices += arange
        cu_num_draft_tokens = torch.from_numpy(cu_num_draft_tokens).to(self.device, non_blocking=True)
        logits_indices = torch.from_numpy(logits_indices).to(self.device, non_blocking=True)
        target_logits_indices = torch.from_numpy(target_logits_indices).to(self.device, non_blocking=True)
        bonus_logits_indices = torch.from_numpy(bonus_logits_indices).to(self.device, non_blocking=True)
        draft_token_ids = self.input_ids[logits_indices]
        draft_token_ids = draft_token_ids[target_logits_indices + 1]
        metadata = SpecDecodeMetadata(draft_token_ids=draft_token_ids, num_draft_tokens=num_draft_tokens.tolist(), cu_num_draft_tokens=cu_num_draft_tokens, target_logits_indices=target_logits_indices, bonus_logits_indices=bonus_logits_indices, logits_indices=logits_indices)
        return metadata

    def _execute_mm_encoder(self, scheduler_output: 'SchedulerOutput'):
        scheduled_encoder_inputs = scheduler_output.scheduled_encoder_inputs
        if not scheduled_encoder_inputs:
            return
        mm_inputs = list[MultiModalKwargs]()
        req_ids_pos = list[tuple[str, int, PlaceholderRange]]()
        for req_id, encoder_input_ids in scheduled_encoder_inputs.items():
            req_state = self.requests[req_id]
            for mm_input_id in encoder_input_ids:
                mm_inputs.append(req_state.mm_inputs[mm_input_id])
                req_ids_pos.append((req_id, mm_input_id, req_state.mm_positions[mm_input_id]))
        grouped_mm_inputs_list = group_mm_inputs_by_modality(mm_inputs)
        encoder_outputs = []
        for grouped_mm_inputs in grouped_mm_inputs_list:
            batched_mm_inputs = MultiModalKwargs.batch(grouped_mm_inputs, pin_memory=self.pin_memory)
            batched_mm_inputs = MultiModalKwargs.as_kwargs(batched_mm_inputs, device=self.device)
            curr_group_outputs = self.model.get_multimodal_embeddings(**batched_mm_inputs)
            sanity_check_mm_encoder_outputs(curr_group_outputs, expected_num_items=len(grouped_mm_inputs))
            for output in curr_group_outputs:
                encoder_outputs.append(output)
        for (req_id, input_id, pos_info), output in zip(req_ids_pos, encoder_outputs):
            if req_id not in self.encoder_cache:
                self.encoder_cache[req_id] = {}
            self.encoder_cache[req_id][input_id] = scatter_mm_placeholders(output, is_embed=pos_info.is_embed)

    def _gather_mm_embeddings(self, scheduler_output: 'SchedulerOutput') -> list[torch.Tensor]:
        mm_embeds: list[torch.Tensor] = []
        for req_id in self.input_batch.req_ids:
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
            req_state = self.requests[req_id]
            num_computed_tokens = req_state.num_computed_tokens
            mm_positions = req_state.mm_positions
            for i, pos_info in enumerate(mm_positions):
                start_pos = pos_info.offset
                num_encoder_tokens = pos_info.length
                if start_pos >= num_computed_tokens + num_scheduled_tokens:
                    break
                if start_pos + num_encoder_tokens <= num_computed_tokens:
                    continue
                start_idx = max(num_computed_tokens - start_pos, 0)
                end_idx = min(num_computed_tokens - start_pos + num_scheduled_tokens, num_encoder_tokens)
                assert start_idx < end_idx
                assert req_id in self.encoder_cache
                assert i in self.encoder_cache[req_id]
                encoder_output = self.encoder_cache[req_id][i]
                if (is_embed := pos_info.is_embed) is not None:
                    is_embed = is_embed[start_idx:end_idx]
                mm_embeds_item = gather_mm_placeholders(encoder_output[start_idx:end_idx], is_embed=is_embed)
                mm_embeds.append(mm_embeds_item)
        return mm_embeds

    def get_model(self) -> nn.Module:
        return self.model

    def get_supported_pooling_tasks(self) -> list[PoolingTask]:
        model = self.get_model()
        if not is_pooling_model(model):
            return []
        return [task for task in get_args(PoolingTask) if model.pooler.get_pooling_updates(task)]

    def apply_grammar_bitmask(self, scheduler_output: 'SchedulerOutput', logits: torch.Tensor):
        grammar_bitmask = scheduler_output.grammar_bitmask
        if grammar_bitmask is None:
            return
        struct_out_req_batch_indices: dict[str, int] = {}
        cumulative_offset = 0
        seq = sorted(self.input_batch.req_id_to_index.items(), key=lambda x: x[1])
        for req_id, batch_index in seq:
            logit_index = batch_index + cumulative_offset
            cumulative_offset += len(scheduler_output.scheduled_spec_decode_tokens.get(req_id, []))
            if req_id in scheduler_output.structured_output_request_ids:
                struct_out_req_batch_indices[req_id] = logit_index
        out_indices = []
        sorted_bitmask = np.zeros_like(grammar_bitmask, shape=(logits.shape[0], grammar_bitmask.shape[1]))
        cumulative_index = 0
        seq = sorted(scheduler_output.structured_output_request_ids.items(), key=lambda x: x[1])
        for req_id, _ in seq:
            logit_index = struct_out_req_batch_indices[req_id]
            num_spec_tokens = len(scheduler_output.scheduled_spec_decode_tokens.get(req_id, []))
            for i in range(1 + num_spec_tokens):
                sorted_bitmask[logit_index + i] = grammar_bitmask[cumulative_index + i]
                out_indices.append(logit_index + i)
            cumulative_index += 1 + num_spec_tokens
        grammar_bitmask = sorted_bitmask
        grammar_bitmask = torch.from_numpy(grammar_bitmask)
        xgr_torch_compile.apply_token_bitmask_inplace_torch_compile(logits, grammar_bitmask.to(self.device, non_blocking=True), indices=out_indices)

    def sync_and_slice_intermediate_tensors(self, num_tokens: int, intermediate_tensors: IntermediateTensors, sync_self: bool) -> IntermediateTensors:
        assert self.intermediate_tensors is not None
        tp = self.vllm_config.parallel_config.tensor_parallel_size
        enabled_sp = self.compilation_config.pass_config.enable_sequence_parallelism
        if enabled_sp:
            assert num_tokens % tp == 0
        is_residual_scattered = tp > 1 and enabled_sp and (num_tokens % tp == 0)
        if sync_self:
            assert intermediate_tensors is not None
            for k, v in intermediate_tensors.items():
                is_scattered = 'residual' and is_residual_scattered
                copy_len = num_tokens // tp if is_scattered else num_tokens
                self.intermediate_tensors[k][:copy_len].copy_(v[:copy_len], non_blocking=True)
        return IntermediateTensors({k: v[:num_tokens // tp] if k == 'residual' and is_residual_scattered else v[:num_tokens] for k, v in self.intermediate_tensors.items()})

    def eplb_step(self, is_dummy: bool=False, is_profile: bool=False) -> None:
        """
        Step for the EPLB (Expert Parallelism Load Balancing) state.
        """
        if not self.parallel_config.enable_eplb:
            return
        assert self.eplb_state is not None
        assert is_mixture_of_experts(self.model)
        self.eplb_state.step(self.model, is_dummy, is_profile, log_stats=self.parallel_config.eplb_log_balancedness)

    def get_dp_padding(self, num_tokens: int) -> tuple[int, Optional[torch.Tensor]]:
        dp_size = self.vllm_config.parallel_config.data_parallel_size
        dp_rank = self.vllm_config.parallel_config.data_parallel_rank
        if dp_size == 1 or self.vllm_config.model_config.enforce_eager:
            return (0, None)
        num_tokens_across_dp = DPMetadata.num_tokens_across_dp(num_tokens, dp_size, dp_rank)
        max_tokens_across_dp_cpu = torch.max(num_tokens_across_dp).item()
        num_tokens_after_padding = torch.tensor([max_tokens_across_dp_cpu] * dp_size, device='cpu', dtype=torch.int32)
        return (max_tokens_across_dp_cpu - num_tokens, num_tokens_after_padding)

    def _pool(self, hidden_states: torch.Tensor, num_scheduled_tokens: int, num_scheduled_tokens_np: np.ndarray, finished_sending: Optional[set[str]], finished_recving: Optional[set[str]]) -> ModelRunnerOutput:
        assert self.input_batch.num_reqs == len(self.input_batch.pooling_params), 'Either all or none of the requests in a batch must be pooling request'
        extracted_hidden_states = list(torch.split(hidden_states[:num_scheduled_tokens], num_scheduled_tokens_np.tolist()))
        pooling_metadata = self.input_batch.pooling_metadata
        raw_pooler_output = self.model.pooler(hidden_states=extracted_hidden_states, pooling_metadata=pooling_metadata)
        pooler_output: list[Optional[torch.Tensor]] = []
        seq_lens = self.seq_lens[:self.input_batch.num_reqs]
        for raw_output, seq_len, prompt_len in zip(raw_pooler_output, seq_lens, pooling_metadata.prompt_lens):
            if seq_len == prompt_len:
                pooler_output.append(raw_output.data.cpu())
            else:
                pooler_output.append(None)
        return ModelRunnerOutput(req_ids=self.input_batch.req_ids, req_id_to_index=self.input_batch.req_id_to_index, sampled_token_ids=[], spec_token_ids=None, logprobs=None, prompt_logprobs_dict={}, pooler_output=pooler_output, finished_sending=finished_sending, finished_recving=finished_recving)

    @torch.inference_mode()
    def execute_model(self, scheduler_output: 'SchedulerOutput', intermediate_tensors: Optional[IntermediateTensors]=None) -> Union[ModelRunnerOutput, IntermediateTensors]:
        self._update_states(scheduler_output)
        if not scheduler_output.total_num_scheduled_tokens:
            if not has_kv_transfer_group():
                return EMPTY_MODEL_RUNNER_OUTPUT
            return self.kv_connector_no_forward(scheduler_output)
        attn_metadata, attention_cuda_graphs, logits_indices, spec_decode_metadata, num_scheduled_tokens_np, spec_decode_common_attn_metadata = self._prepare_inputs(scheduler_output)
        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        if self.use_cuda_graph and num_scheduled_tokens <= self.cudagraph_batch_sizes[-1]:
            num_input_tokens = self.vllm_config.pad_for_cudagraph(num_scheduled_tokens)
        else:
            tp_size = self.vllm_config.parallel_config.tensor_parallel_size
            if self.compilation_config.pass_config.enable_sequence_parallelism and tp_size > 1:
                num_input_tokens = round_up(num_scheduled_tokens, tp_size)
            else:
                num_input_tokens = num_scheduled_tokens
        num_pad, num_tokens_across_dp = self.get_dp_padding(num_input_tokens)
        num_input_tokens += num_pad
        if self.is_multimodal_model:
            self._execute_mm_encoder(scheduler_output)
            mm_embeds = self._gather_mm_embeddings(scheduler_output)
        else:
            mm_embeds = []
        if self.is_multimodal_model and get_pp_group().is_first_rank:
            input_ids = self.input_ids[:num_scheduled_tokens]
            inputs_embeds = self.model.get_input_embeddings(input_ids=input_ids, multimodal_embeddings=mm_embeds or None)
            self.inputs_embeds[:num_scheduled_tokens].copy_(inputs_embeds)
            inputs_embeds = self.inputs_embeds[:num_input_tokens]
            input_ids = None
        else:
            input_ids = self.input_ids[:num_input_tokens]
            inputs_embeds = None
        if self.uses_mrope:
            positions = self.mrope_positions[:, :num_input_tokens]
        else:
            positions = self.positions[:num_input_tokens]
        if get_pp_group().is_first_rank:
            intermediate_tensors = None
        else:
            intermediate_tensors = self.sync_and_slice_intermediate_tensors(num_input_tokens, intermediate_tensors, True)
        skip_cuda_graphs = self.full_cuda_graph and (not attention_cuda_graphs)
        with set_forward_context(attn_metadata, self.vllm_config, num_tokens=num_input_tokens, num_tokens_across_dp=num_tokens_across_dp, skip_cuda_graphs=skip_cuda_graphs):
            self.maybe_setup_kv_connector(scheduler_output)
            model_output = self.model(input_ids=input_ids, positions=positions, intermediate_tensors=intermediate_tensors, inputs_embeds=inputs_embeds)
            self.maybe_wait_for_kv_save()
            finished_sending, finished_recving = self.get_finished_kv_transfers(scheduler_output)
        if self.use_aux_hidden_state_outputs:
            hidden_states, aux_hidden_states = model_output
        else:
            hidden_states = model_output
            aux_hidden_states = None
        broadcast_pp_output = self.parallel_config.distributed_executor_backend == 'external_launcher' and len(get_pp_group().ranks) > 0
        if not get_pp_group().is_last_rank:
            if not broadcast_pp_output:
                if finished_sending or finished_recving:
                    hidden_states.finished_sending = finished_sending
                    hidden_states.finished_recving = finished_recving
                return hidden_states
            assert isinstance(hidden_states, IntermediateTensors)
            get_pp_group().send_tensor_dict(hidden_states.tensors, all_gather_group=get_tp_group())
            logits = None
        else:
            if self.input_batch.pooling_params:
                return self._pool(hidden_states, num_scheduled_tokens, num_scheduled_tokens_np, finished_sending, finished_recving)
            sample_hidden_states = hidden_states[logits_indices]
            logits = self.model.compute_logits(sample_hidden_states, None)
        if broadcast_pp_output:
            model_output_broadcast_data = {'logits': logits.contiguous()} if logits is not None else {}
            model_output_broadcast_data = get_pp_group().broadcast_tensor_dict(model_output_broadcast_data, src=len(get_pp_group().ranks) - 1)
            assert model_output_broadcast_data is not None
            logits = model_output_broadcast_data['logits']
        if scheduler_output.grammar_bitmask is not None:
            self.apply_grammar_bitmask(scheduler_output, logits)
        sampling_metadata = self.input_batch.sampling_metadata
        if spec_decode_metadata is None:
            sampler_output = self.sampler(logits=logits, sampling_metadata=sampling_metadata)
        else:
            assert logits is not None
            bonus_logits = logits[spec_decode_metadata.bonus_logits_indices]
            sampler_output = self.sampler(logits=bonus_logits, sampling_metadata=sampling_metadata)
            bonus_token_ids = sampler_output.sampled_token_ids
            target_logits = logits[spec_decode_metadata.target_logits_indices]
            output_token_ids = self.rejection_sampler(spec_decode_metadata, None, target_logits, bonus_token_ids, sampling_metadata)
            sampler_output.sampled_token_ids = output_token_ids
        num_nans_in_logits = {}
        if envs.VLLM_COMPUTE_NANS_IN_LOGITS:
            num_nans_in_logits = self._get_nans_in_logits(logits)
        discard_sampled_tokens_req_indices = []
        for i, req_id in enumerate(self.input_batch.req_ids):
            req_state = self.requests[req_id]
            seq_len = req_state.num_computed_tokens + scheduler_output.num_scheduled_tokens[req_id]
            if seq_len < req_state.num_tokens:
                generator = self.input_batch.generators.get(i)
                if generator is not None:
                    generator.set_offset(generator.get_offset() - 4)
                discard_sampled_tokens_req_indices.append(i)
        logprobs_tensors = sampler_output.logprobs_tensors
        logprobs_lists = logprobs_tensors.tolists() if logprobs_tensors is not None else None
        prompt_logprobs_dict = self._get_prompt_logprobs_dict(hidden_states[:num_scheduled_tokens], scheduler_output)
        sampled_token_ids = sampler_output.sampled_token_ids
        max_gen_len = sampled_token_ids.shape[-1]
        if max_gen_len == 1:
            valid_sampled_token_ids = sampled_token_ids.tolist()
        else:
            valid_sampled_token_ids = self.rejection_sampler.parse_output(sampled_token_ids, self.input_batch.vocab_size)
        for i in discard_sampled_tokens_req_indices:
            valid_sampled_token_ids[i].clear()
        for req_idx, sampled_ids in enumerate(valid_sampled_token_ids):
            if not sampled_ids:
                continue
            start_idx = self.input_batch.num_tokens_no_spec[req_idx]
            end_idx = start_idx + len(sampled_ids)
            assert end_idx <= self.max_model_len, f'Sampled token IDs exceed the max model length. Total number of tokens: {end_idx} > max_model_len: {self.max_model_len}'
            self.input_batch.token_ids_cpu[req_idx, start_idx:end_idx] = sampled_ids
            self.input_batch.num_tokens_no_spec[req_idx] = end_idx
            self.input_batch.num_tokens[req_idx] = end_idx
            req_id = self.input_batch.req_ids[req_idx]
            req_state = self.requests[req_id]
            req_state.output_token_ids.extend(sampled_ids)
        if not self.speculative_config:
            spec_token_ids = None
        else:
            assert spec_decode_common_attn_metadata is not None
            spec_token_ids = self.propose_draft_token_ids(scheduler_output, valid_sampled_token_ids, sampling_metadata, hidden_states, sample_hidden_states, aux_hidden_states, spec_decode_metadata, spec_decode_common_attn_metadata)
        self.eplb_step()
        return ModelRunnerOutput(req_ids=self.input_batch.req_ids, req_id_to_index=self.input_batch.req_id_to_index, sampled_token_ids=valid_sampled_token_ids, spec_token_ids=spec_token_ids, logprobs=logprobs_lists, prompt_logprobs_dict=prompt_logprobs_dict, pooler_output=[], finished_sending=finished_sending, finished_recving=finished_recving, num_nans_in_logits=num_nans_in_logits)

    def propose_draft_token_ids(self, scheduler_output: 'SchedulerOutput', sampled_token_ids: list[list[int]], sampling_metadata: SamplingMetadata, hidden_states: torch.Tensor, sample_hidden_states: torch.Tensor, aux_hidden_states: Optional[torch.Tensor], spec_decode_metadata: Optional[SpecDecodeMetadata], common_attn_metadata: CommonAttentionMetadata) -> list[list[int]]:
        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        if self.speculative_config.method == 'ngram':
            assert isinstance(self.drafter, NgramProposer)
            spec_token_ids = self.propose_ngram_draft_token_ids(sampled_token_ids)
        elif self.speculative_config.method == 'medusa':
            assert isinstance(self.drafter, MedusaProposer)
            if sample_hidden_states.shape[0] == len(sampled_token_ids):
                hidden_states = sample_hidden_states
            else:
                indices = []
                offset = 0
                for num_draft, tokens in zip(spec_decode_metadata.num_draft_tokens, sampled_token_ids):
                    indices.append(offset + len(tokens) - 1)
                    offset += num_draft + 1
                indices = torch.tensor(indices, device=self.device)
                hidden_states = sample_hidden_states[indices]
            spec_token_ids = self.drafter.propose(target_hidden_states=hidden_states, sampling_metadata=sampling_metadata)
        elif self.speculative_config.use_eagle():
            assert isinstance(self.drafter, EagleProposer)
            next_token_ids: list[int] = []
            for i, token_ids in enumerate(sampled_token_ids):
                if token_ids:
                    next_token_id = token_ids[-1]
                else:
                    req_id = self.input_batch.req_ids[i]
                    req_state = self.requests[req_id]
                    seq_len = req_state.num_computed_tokens + scheduler_output.num_scheduled_tokens[req_id]
                    next_token_id = req_state.get_token_id(seq_len)
                next_token_ids.append(next_token_id)
            next_token_ids = torch.tensor(next_token_ids, dtype=torch.int32, device=self.device)
            if spec_decode_metadata is None:
                target_token_ids = self.input_ids[:num_scheduled_tokens]
                target_positions = self.positions[:num_scheduled_tokens]
                if self.use_aux_hidden_state_outputs:
                    target_hidden_states = torch.cat([h[:num_scheduled_tokens] for h in aux_hidden_states], dim=-1)
                else:
                    target_hidden_states = hidden_states[:num_scheduled_tokens]
            else:
                num_draft_tokens = spec_decode_metadata.num_draft_tokens
                num_rejected_tokens = [n + 1 - len(sampled_token_ids[i]) if n > 0 else 0 for i, n in enumerate(num_draft_tokens)]
                num_rejected_tokens_cpu = torch.tensor(num_rejected_tokens, dtype=torch.int32)
                common_attn_metadata, token_indices = self.drafter.prepare_inputs(common_attn_metadata, num_rejected_tokens_cpu)
                target_token_ids = self.input_ids[token_indices]
                target_positions = self.positions[token_indices]
                if self.use_aux_hidden_state_outputs:
                    target_hidden_states = torch.cat([h[token_indices] for h in aux_hidden_states], dim=-1)
                else:
                    target_hidden_states = hidden_states[token_indices]
            draft_token_ids = self.drafter.propose(target_token_ids=target_token_ids, target_positions=target_positions, target_hidden_states=target_hidden_states, next_token_ids=next_token_ids, sampling_metadata=sampling_metadata, common_attn_metadata=common_attn_metadata)
            spec_token_ids = draft_token_ids.tolist()
        return spec_token_ids

    @staticmethod
    def maybe_setup_kv_connector(scheduler_output: 'SchedulerOutput'):
        if has_kv_transfer_group():
            kv_connector = get_kv_transfer_group()
            assert isinstance(kv_connector, KVConnectorBase_V1)
            assert scheduler_output.kv_connector_metadata is not None
            kv_connector.bind_connector_metadata(scheduler_output.kv_connector_metadata)
            kv_connector.start_load_kv(get_forward_context())

    @staticmethod
    def maybe_wait_for_kv_save() -> None:
        if has_kv_transfer_group():
            get_kv_transfer_group().wait_for_save()

    @staticmethod
    def get_finished_kv_transfers(scheduler_output: 'SchedulerOutput') -> tuple[Optional[set[str]], Optional[set[str]]]:
        if has_kv_transfer_group():
            return get_kv_transfer_group().get_finished(scheduler_output.finished_req_ids)
        return (None, None)

    def kv_connector_no_forward(self, scheduler_output: 'SchedulerOutput') -> ModelRunnerOutput:
        with set_forward_context(None, self.vllm_config):
            self.maybe_setup_kv_connector(scheduler_output)
            finished_sending, finished_recving = self.get_finished_kv_transfers(scheduler_output)
        if not finished_sending and (not finished_recving):
            return EMPTY_MODEL_RUNNER_OUTPUT
        output = copy.copy(EMPTY_MODEL_RUNNER_OUTPUT)
        output.finished_sending = finished_sending
        output.finished_recving = finished_recving
        return output

    def propose_ngram_draft_token_ids(self, sampled_token_ids: list[list[int]]) -> list[list[int]]:
        draft_token_ids: list[list[int]] = []
        for i, sampled_ids in enumerate(sampled_token_ids):
            num_sampled_ids = len(sampled_ids)
            if not num_sampled_ids:
                draft_token_ids.append([])
                continue
            req_id = self.input_batch.req_ids[i]
            if req_id in self.input_batch.spec_decode_unsupported_reqs:
                draft_token_ids.append([])
                continue
            num_tokens = self.input_batch.num_tokens_no_spec[i]
            if num_tokens >= self.max_model_len:
                draft_token_ids.append([])
                continue
            drafter_output = self.drafter.propose(self.input_batch.token_ids_cpu[i, :num_tokens])
            if drafter_output is None or len(drafter_output) == 0:
                draft_token_ids.append([])
            else:
                draft_token_ids.append(drafter_output.tolist())
        return draft_token_ids

    def update_config(self, overrides: dict[str, Any]) -> None:
        allowed_config_names = {'load_config', 'model_config'}
        for config_name, config_overrides in overrides.items():
            assert config_name in allowed_config_names, f'Config `{config_name}` not supported. Allowed configs: {allowed_config_names}'
            config = getattr(self, config_name)
            new_config = update_config(config, config_overrides)
            setattr(self, config_name, new_config)

    def load_model(self, eep_scale_up: bool=False) -> None:
        """
        Args:
            eep_scale_up: the model loading is for elastic EP scale up.
        """
        logger.info('Starting to load model %s...', self.model_config.model)
        if eep_scale_up:
            num_local_physical_experts = torch.empty(1, dtype=torch.int32, device='cpu')
            torch.distributed.broadcast(num_local_physical_experts, group=get_ep_group().cpu_group, group_src=0)
            num_local_physical_experts = int(num_local_physical_experts.item())
            new_ep_size = get_ep_group().world_size
            global_expert_load, old_global_expert_indices = EplbState.recv_state()
            num_logical_experts = global_expert_load.shape[1]
            self.parallel_config.num_redundant_experts = num_local_physical_experts * new_ep_size - num_logical_experts
            assert old_global_expert_indices.shape[1] % num_local_physical_experts == 0
            old_ep_size = old_global_expert_indices.shape[1] // num_local_physical_experts
            rank_mapping = {old_ep_rank: old_ep_rank for old_ep_rank in range(old_ep_size)}
        else:
            global_expert_load = None
            old_global_expert_indices = None
            rank_mapping = None
        with DeviceMemoryProfiler() as m:
            time_before_load = time.perf_counter()
            model_loader = get_model_loader(self.load_config)
            if not hasattr(self, 'model'):
                logger.info('Loading model from scratch...')
                self.model = model_loader.load_model(vllm_config=self.vllm_config, model_config=self.model_config)
            else:
                logger.info('Model was already initialized. Loading weights inplace...')
                model_loader.load_weights(self.model, model_config=self.model_config)
            if self.lora_config:
                self.model = self.load_lora_model(self.model, self.model_config, self.scheduler_config, self.lora_config, self.device)
            if hasattr(self, 'drafter'):
                logger.info('Loading drafter model...')
                self.drafter.load_model(self.model)
            if self.use_aux_hidden_state_outputs:
                self.model.set_aux_hidden_state_layers(self.model.get_eagle3_aux_hidden_state_layers())
            time_after_load = time.perf_counter()
        self.model_memory_usage = m.consumed_memory
        logger.info('Model loading took %.4f GiB and %.6f seconds', self.model_memory_usage / GiB_bytes, time_after_load - time_before_load)
        prepare_communication_buffer_for_model(self.model)
        if is_mixture_of_experts(self.model) and self.parallel_config.enable_eplb:
            logger.info('EPLB is enabled for model %s.', self.model_config.model)
            self.eplb_state = EplbState.build(self.model, self.device, self.parallel_config, global_expert_load, old_global_expert_indices, rank_mapping)

    def save_tensorized_model(self, tensorizer_config: 'TensorizerConfig') -> None:
        TensorizerLoader.save_model(self.model, tensorizer_config=tensorizer_config, model_config=self.model_config)

    def _get_prompt_logprobs_dict(self, hidden_states: torch.Tensor, scheduler_output: 'SchedulerOutput') -> dict[str, Optional[LogprobsTensors]]:
        num_prompt_logprobs_dict = self.input_batch.num_prompt_logprobs
        if not num_prompt_logprobs_dict:
            return {}
        in_progress_dict = self.input_batch.in_progress_prompt_logprobs_cpu
        prompt_logprobs_dict: dict[str, Optional[LogprobsTensors]] = {}
        completed_prefill_reqs = []
        for req_id, num_prompt_logprobs in num_prompt_logprobs_dict.items():
            num_tokens = scheduler_output.num_scheduled_tokens[req_id]
            request = self.requests[req_id]
            num_prompt_tokens = len(request.prompt_token_ids)
            prompt_token_ids = torch.tensor(request.prompt_token_ids).to(self.device, non_blocking=True)
            logprobs_tensors = in_progress_dict.get(req_id)
            if not logprobs_tensors:
                logprobs_tensors = LogprobsTensors.empty_cpu(num_prompt_tokens - 1, num_prompt_logprobs + 1)
                in_progress_dict[req_id] = logprobs_tensors
            start_idx = request.num_computed_tokens
            start_tok = start_idx + 1
            num_remaining_tokens = num_prompt_tokens - start_tok
            if num_tokens <= num_remaining_tokens:
                num_logits = num_tokens
            else:
                num_logits = num_remaining_tokens
                completed_prefill_reqs.append(req_id)
                prompt_logprobs_dict[req_id] = logprobs_tensors
            if num_logits <= 0:
                continue
            req_idx = self.input_batch.req_id_to_index[req_id]
            offset = self.query_start_loc_np[req_idx].item()
            prompt_hidden_states = hidden_states[offset:offset + num_logits]
            logits = self.model.compute_logits(prompt_hidden_states, None)
            tgt_token_ids = prompt_token_ids[start_tok:start_tok + num_logits]
            logprobs = self.sampler.compute_logprobs(logits)
            token_ids, logprobs, ranks = self.sampler.gather_logprobs(logprobs, num_prompt_logprobs, tgt_token_ids)
            chunk_slice = slice(start_idx, start_idx + num_logits)
            logprobs_tensors.logprob_token_ids[chunk_slice].copy_(token_ids, non_blocking=True)
            logprobs_tensors.logprobs[chunk_slice].copy_(logprobs, non_blocking=True)
            logprobs_tensors.selected_token_ranks[chunk_slice].copy_(ranks, non_blocking=True)
        for req_id in completed_prefill_reqs:
            del num_prompt_logprobs_dict[req_id]
            del in_progress_dict[req_id]
        if prompt_logprobs_dict:
            self._sync_device()
        return prompt_logprobs_dict

    def _get_nans_in_logits(self, logits: Optional[torch.Tensor]) -> dict[str, int]:
        try:
            if logits is None:
                return {req_id: 0 for req_id in self.input_batch.req_ids}
            num_nans_in_logits = {}
            num_nans_for_index = logits.isnan().sum(dim=-1).cpu().numpy()
            for req_id in self.input_batch.req_ids:
                req_index = self.input_batch.req_id_to_index[req_id]
                num_nans_in_logits[req_id] = int(num_nans_for_index[req_index]) if num_nans_for_index is not None and req_index < logits.shape[0] else 0
            return num_nans_in_logits
        except IndexError:
            return {}

    @contextmanager
    def maybe_randomize_inputs(self, input_ids: torch.Tensor):
        """
        Randomize input_ids if VLLM_RANDOMIZE_DP_DUMMY_INPUTS is set.
        This is to help balance expert-selection
         - during profile_run
         - during DP rank dummy run 
        """
        dp_size = self.vllm_config.parallel_config.data_parallel_size
        randomize_inputs = envs.VLLM_RANDOMIZE_DP_DUMMY_INPUTS and dp_size > 1
        if not randomize_inputs:
            yield
        else:

            @functools.cache
            def rand_input_ids() -> torch.Tensor:
                return torch.randint_like(self.input_ids, low=0, high=self.model_config.get_vocab_size(), dtype=input_ids.dtype)
            logger.debug('Randomizing dummy data for DP Rank')
            input_ids.copy_(rand_input_ids()[:input_ids.size(0)], non_blocking=True)
            yield
            input_ids.fill_(0)

    @torch.inference_mode()
    def _dummy_run(self, num_tokens: int, capture_attn_cudagraph: bool=False, skip_eplb: bool=False, is_profile: bool=False) -> tuple[torch.Tensor, torch.Tensor]:
        num_pad, num_tokens_across_dp = self.get_dp_padding(num_tokens)
        num_tokens += num_pad
        assert num_tokens <= self.scheduler_config.max_num_batched_tokens
        max_num_reqs = self.scheduler_config.max_num_seqs
        num_reqs = min(num_tokens, max_num_reqs)
        min_tokens_per_req = num_tokens // num_reqs
        num_scheduled_tokens_list = [min_tokens_per_req] * num_reqs
        num_scheduled_tokens_list[-1] += num_tokens % num_reqs
        assert sum(num_scheduled_tokens_list) == num_tokens
        assert len(num_scheduled_tokens_list) == num_reqs
        num_scheduled_tokens = np.array(num_scheduled_tokens_list, dtype=np.int32)
        attn_metadata: Optional[dict[str, Any]] = None
        if capture_attn_cudagraph:
            attn_metadata = {}
            self.seq_lens_np[:num_reqs] = self.max_model_len
            self.seq_lens_np[num_reqs:] = 0
            self.seq_lens[:num_reqs].copy_(self.seq_lens_cpu[:num_reqs], non_blocking=True)
            for kv_cache_group_id, kv_cache_group_spec in enumerate(self.kv_cache_config.kv_cache_groups):
                common_attn_metadata = CommonAttentionMetadata(query_start_loc=self.query_start_loc[:num_reqs + 1], query_start_loc_cpu=self.query_start_loc_cpu[:num_reqs + 1], seq_lens=self.seq_lens[:num_reqs], seq_lens_cpu=self.seq_lens_cpu[:num_reqs], num_computed_tokens_cpu=self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs], num_reqs=num_reqs, num_actual_tokens=num_tokens, max_query_len=num_tokens, block_table_tensor=self.input_batch.block_table[kv_cache_group_id].get_device_tensor()[:num_reqs], slot_mapping=self.input_batch.block_table[kv_cache_group_id].slot_mapping[:num_tokens])
                attn_metadata_i = self.attn_metadata_builders[kv_cache_group_id].build_for_cudagraph_capture(common_attn_metadata)
                for layer_name in kv_cache_group_spec.layer_names:
                    attn_metadata[layer_name] = attn_metadata_i
        with self.maybe_dummy_run_with_lora(self.lora_config, num_scheduled_tokens):
            model = self.model
            if self.is_multimodal_model:
                input_ids = None
                inputs_embeds = self.inputs_embeds[:num_tokens]
            else:
                input_ids = self.input_ids[:num_tokens]
                inputs_embeds = None
            if self.uses_mrope:
                positions = self.mrope_positions[:, :num_tokens]
            else:
                positions = self.positions[:num_tokens]
            if get_pp_group().is_first_rank:
                intermediate_tensors = None
            else:
                if self.intermediate_tensors is None:
                    self.intermediate_tensors = self.model.make_empty_intermediate_tensors(batch_size=self.max_num_tokens, dtype=self.model_config.dtype, device=self.device)
                intermediate_tensors = self.sync_and_slice_intermediate_tensors(num_tokens, None, False)
            with self.maybe_randomize_inputs(input_ids), set_forward_context(attn_metadata, self.vllm_config, num_tokens=num_tokens, num_tokens_across_dp=num_tokens_across_dp):
                outputs = model(input_ids=input_ids, positions=positions, intermediate_tensors=intermediate_tensors, inputs_embeds=inputs_embeds)
            if self.use_aux_hidden_state_outputs:
                hidden_states, _ = outputs
            else:
                hidden_states = outputs
            if self.speculative_config and self.speculative_config.use_eagle():
                assert isinstance(self.drafter, EagleProposer)
                self.drafter.dummy_run(num_tokens)
        if not skip_eplb:
            self.eplb_step(is_dummy=True, is_profile=is_profile)
        logit_indices = np.cumsum(num_scheduled_tokens) - 1
        return (hidden_states, hidden_states[logit_indices])

    @torch.inference_mode()
    def _dummy_sampler_run(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = torch.rand_like(hidden_states)
        logits = self.model.compute_logits(hidden_states, None)
        num_reqs = logits.size(0)
        dummy_tensors = lambda v: torch.full((num_reqs,), v, device=self.device)
        dummy_metadata = SamplingMetadata(temperature=dummy_tensors(0.5), all_greedy=False, all_random=False, top_p=dummy_tensors(0.9), top_k=dummy_tensors(logits.size(1) - 1), generators={}, max_num_logprobs=None, no_penalties=True, prompt_token_ids=None, frequency_penalties=dummy_tensors(0.1), presence_penalties=dummy_tensors(0.1), repetition_penalties=dummy_tensors(0.1), output_token_ids=[[] for _ in range(num_reqs)], allowed_token_ids_mask=None, bad_words_token_ids={}, logitsprocs=LogitsProcessorManager())
        try:
            sampler_output = self.sampler(logits=logits, sampling_metadata=dummy_metadata)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                raise RuntimeError(f'CUDA out of memory occurred when warming up sampler with {num_reqs} dummy requests. Please try lowering `max_num_seqs` or `gpu_memory_utilization` when initializing the engine.') from e
            else:
                raise e
        if self.speculative_config:
            draft_token_ids = [[0] for _ in range(num_reqs)]
            dummy_spec_decode_metadata = SpecDecodeMetadata.make_dummy(draft_token_ids, self.device)
            num_tokens = sum((len(ids) for ids in draft_token_ids))
            draft_probs = None
            target_logits = torch.randn(num_tokens, logits.shape[-1], device=self.device, dtype=logits.dtype)
            bonus_token_ids = torch.zeros(num_reqs, device=self.device, dtype=torch.int32)
            self.rejection_sampler(dummy_spec_decode_metadata, draft_probs, target_logits, bonus_token_ids, dummy_metadata)
        return sampler_output

    @torch.inference_mode()
    def _dummy_pooler_run(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        max_num_reqs = self.scheduler_config.max_num_seqs
        num_reqs = min(num_tokens, max_num_reqs)
        min_tokens_per_req = num_tokens // num_reqs
        num_scheduled_tokens_list = [min_tokens_per_req] * num_reqs
        num_scheduled_tokens_list[-1] += num_tokens % num_reqs
        assert sum(num_scheduled_tokens_list) == num_tokens
        assert len(num_scheduled_tokens_list) == num_reqs
        hidden_states_list = list(torch.split(hidden_states, num_scheduled_tokens_list))
        req_num_tokens = num_tokens // num_reqs
        model = cast(VllmModelForPooling, self.model)
        dummy_task = self.get_supported_pooling_tasks()[0]
        dummy_pooling_params = PoolingParams(task=dummy_task)
        to_update = model.pooler.get_pooling_updates(dummy_task)
        assert to_update is not None
        to_update.apply(dummy_pooling_params)
        dummy_metadata = PoolingMetadata(prompt_lens=torch.tensor([h.shape[0] for h in hidden_states_list], device=self.device), prompt_token_ids=torch.zeros((num_reqs, req_num_tokens), dtype=torch.int32, device=self.device), pooling_params=[dummy_pooling_params] * num_reqs)
        try:
            pooler_output = model.pooler(hidden_states=hidden_states_list, pooling_metadata=dummy_metadata)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                raise RuntimeError(f'CUDA out of memory occurred when warming up pooler with {num_reqs} dummy requests. Please try lowering `max_num_seqs` or `gpu_memory_utilization` when initializing the engine.') from e
            else:
                raise e
        return pooler_output

    def profile_run(self) -> None:
        if self.is_multimodal_model and self.max_num_encoder_input_tokens > 0 and (self.encoder_cache_size > 0):
            max_tokens_by_modality_dict = self.mm_registry.get_max_tokens_per_item_by_nonzero_modality(self.model_config)
            dummy_data_modality, max_tokens_per_mm_item = max(max_tokens_by_modality_dict.items(), key=lambda item: item[1])
            encoder_budget = min(self.max_num_encoder_input_tokens, self.encoder_cache_size)
            max_num_mm_items_encoder_budget = encoder_budget // max_tokens_per_mm_item
            max_mm_items_per_req = self.mm_registry.get_mm_limits_per_prompt(self.model_config)[dummy_data_modality]
            max_num_mm_items_decoder_budget = self.max_num_reqs * max_mm_items_per_req
            max_num_mm_items = max(1, min(max_num_mm_items_encoder_budget, max_num_mm_items_decoder_budget))
            logger.info('Encoder cache will be initialized with a budget of %s tokens, and profiled with %s %s items of the maximum feature size.', encoder_budget, max_num_mm_items, dummy_data_modality)
            dummy_mm_kwargs = self.mm_registry.get_decoder_dummy_data(model_config=self.model_config, seq_len=max_tokens_per_mm_item, mm_counts={dummy_data_modality: 1}).multi_modal_data
            batched_dummy_mm_inputs = MultiModalKwargs.batch([dummy_mm_kwargs] * max_num_mm_items, pin_memory=self.pin_memory)
            batched_dummy_mm_inputs = MultiModalKwargs.as_kwargs(batched_dummy_mm_inputs, device=self.device)
            dummy_encoder_outputs = self.model.get_multimodal_embeddings(**batched_dummy_mm_inputs)
            sanity_check_mm_encoder_outputs(dummy_encoder_outputs, expected_num_items=max_num_mm_items)
            self.encoder_cache['tmp'] = dict(enumerate(dummy_encoder_outputs))
        hidden_states, last_hidden_states = self._dummy_run(self.max_num_tokens, is_profile=True)
        if get_pp_group().is_last_rank:
            if self.is_pooling_model:
                output = self._dummy_pooler_run(hidden_states)
            else:
                output = self._dummy_sampler_run(last_hidden_states)
        else:
            output = None
        self._sync_device()
        del hidden_states, output
        self.encoder_cache.clear()
        gc.collect()

    def capture_model(self) -> None:
        if not self.use_cuda_graph:
            logger.warning('Skipping CUDA graph capture. To turn on CUDA graph capture, set -O %s and ensure `use_cudagraph` was not manually set to False', CompilationLevel.PIECEWISE)
            return
        compilation_counter.num_gpu_runner_capture_triggers += 1
        start_time = time.perf_counter()
        start_free_gpu_memory = torch.cuda.mem_get_info()[0]
        with graph_capture(device=self.device):
            full_cg = self.full_cuda_graph
            compilation_cases = reversed(self.cudagraph_batch_sizes)
            if is_global_first_rank():
                compilation_cases = tqdm(list(compilation_cases), disable=not self.load_config.use_tqdm_on_load, desc='Capturing CUDA graph shapes')
            for num_tokens in compilation_cases:
                for _ in range(self.compilation_config.cudagraph_num_of_warmups):
                    self._dummy_run(num_tokens, capture_attn_cudagraph=full_cg, skip_eplb=True)
                self._dummy_run(num_tokens, capture_attn_cudagraph=full_cg, skip_eplb=True)
        end_time = time.perf_counter()
        end_free_gpu_memory = torch.cuda.mem_get_info()[0]
        elapsed_time = end_time - start_time
        cuda_graph_size = start_free_gpu_memory - end_free_gpu_memory
        logger.info('Graph capturing finished in %.0f secs, took %.2f GiB', elapsed_time, cuda_graph_size / (1 << 30))

    def initialize_attn_backend(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize the attention backends and attention metadata builders.
        """
        assert len(self.attn_backends) == 0 and len(self.attn_metadata_builders) == 0, 'Attention backends are already initialized'
        for i, kv_cache_group_spec in enumerate(kv_cache_config.kv_cache_groups):
            kv_cache_spec = kv_cache_group_spec.kv_cache_spec
            if isinstance(kv_cache_spec, AttentionSpec):
                attn_backend_i = get_attn_backend(kv_cache_spec.head_size, self.dtype, kv_cache_spec.dtype, kv_cache_spec.block_size, self.model_config.is_attention_free, use_mla=kv_cache_spec.use_mla)
                if attn_backend_i is None:
                    error_msg = f'Error with get_attn_backend: kv_cache_spec.head_size={kv_cache_spec.head_size!r}, self.dtype={self.dtype!r}, kv_cache_spec.dtype={kv_cache_spec.dtype!r}, kv_cache_spec.block_size={kv_cache_spec.block_size!r}, self.model_config.is_attention_free={self.model_config.is_attention_free!r}, kv_cache_spec.use_mla={kv_cache_spec.use_mla!r}'
                    logger.error(error_msg)
                    raise NotImplementedError('Non-Attention backend is not supported by V1 GPUModelRunner.')
            elif isinstance(kv_cache_spec, MambaSpec):
                attn_backend_i = Mamba2AttentionBackend
            else:
                raise ValueError(f'Unknown KV cache spec type: {type(kv_cache_spec)}')
            attn_metadata_builder_i = attn_backend_i.get_builder_cls()(kv_cache_spec, self.vllm_config, self.device)
            if self.full_cuda_graph and (not attn_metadata_builder_i.full_cudagraph_supported):
                raise ValueError(f'Full CUDAGraph not supported for {attn_backend_i.__name__}. Turn off CompilationConfig.full_cuda_graph or use a different attention backend.')
            self.attn_backends.append(attn_backend_i)
            self.attn_metadata_builders.append(attn_metadata_builder_i)

    def may_reinitialize_input_batch(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Re-initialize the input batch if the block sizes are different from
        `[self.cache_config.block_size]`. This usually happens when there
        are multiple KV cache groups.

        Args:
            kv_cache_config: The KV cache configuration.
        """
        block_sizes = [kv_cache_group.kv_cache_spec.block_size for kv_cache_group in kv_cache_config.kv_cache_groups]
        if block_sizes != [self.cache_config.block_size]:
            assert self.cache_config.cpu_offload_gb == 0, 'Cannot re-initialize the input batch when CPU weight offloading is enabled. See https://github.com/vllm-project/vllm/pull/18298 for more details.'
            self.input_batch = InputBatch(max_num_reqs=self.max_num_reqs, max_model_len=self.max_model_len, max_num_batched_tokens=self.max_num_tokens, device=self.device, pin_memory=self.pin_memory, vocab_size=self.model_config.get_vocab_size(), block_sizes=block_sizes, is_spec_decode=bool(self.vllm_config.speculative_config))

    def _allocate_kv_cache_tensors(self, kv_cache_config: KVCacheConfig) -> dict[str, torch.Tensor]:
        """
        Initializes the KV cache buffer with the correct size. The buffer needs
        to be reshaped to the desired shape before being used by the models.

        Args:
            kv_cache_config: The KV cache config
        Returns:
            dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer for KV cache.
         """
        kv_cache_raw_tensors: dict[str, torch.Tensor] = {}
        for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
            tensor = torch.zeros(kv_cache_tensor.size, dtype=torch.int8, device=self.device)
            for layer_name in kv_cache_tensor.shared_by:
                kv_cache_raw_tensors[layer_name] = tensor
        layer_names = set()
        for group in kv_cache_config.kv_cache_groups:
            layer_names.update(group.layer_names)
        assert layer_names == set(kv_cache_raw_tensors.keys()), 'Some layers are not correctly initialized'
        return kv_cache_raw_tensors

    def _reshape_kv_cache_tensors(self, kv_cache_config: KVCacheConfig, kv_cache_raw_tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Reshape the KV cache tensors to the desired shape and dtype.

        Args:
            kv_cache_config: The KV cache config
            kv_cache_raw_tensors: The KV cache buffer of each layer, with
            correct size but uninitialized shape.
        Returns:
            Dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer for KV cache.
        """
        kv_caches: dict[str, torch.Tensor] = {}
        has_attn, has_mamba = (False, False)
        for i, kv_cache_group_spec in enumerate(kv_cache_config.kv_cache_groups):
            kv_cache_spec = kv_cache_group_spec.kv_cache_spec
            for layer_name in kv_cache_group_spec.layer_names:
                raw_tensor = kv_cache_raw_tensors[layer_name]
                assert raw_tensor.numel() % kv_cache_spec.page_size_bytes == 0
                num_blocks = raw_tensor.numel() // kv_cache_spec.page_size_bytes
                if isinstance(kv_cache_spec, AttentionSpec):
                    has_attn = True
                    kv_cache_shape = self.attn_backends[i].get_kv_cache_shape(num_blocks, kv_cache_spec.block_size, kv_cache_spec.num_kv_heads, kv_cache_spec.head_size)
                    dtype = kv_cache_spec.dtype
                    try:
                        kv_cache_stride_order = self.attn_backends[i].get_kv_cache_stride_order()
                        assert len(kv_cache_stride_order) == len(kv_cache_shape)
                    except (AttributeError, NotImplementedError):
                        kv_cache_stride_order = tuple(range(len(kv_cache_shape)))
                    kv_cache_shape = tuple((kv_cache_shape[i] for i in kv_cache_stride_order))
                    inv_order = [kv_cache_stride_order.index(i) for i in range(len(kv_cache_stride_order))]
                    kv_caches[layer_name] = kv_cache_raw_tensors[layer_name].view(dtype).view(kv_cache_shape).permute(*inv_order)
                elif isinstance(kv_cache_spec, MambaSpec):
                    has_mamba = True
                    raw_tensor = kv_cache_raw_tensors[layer_name]
                    dtype = kv_cache_spec.dtype
                    num_element_per_page = kv_cache_spec.page_size_bytes // get_dtype_size(dtype)
                    state_tensors = []
                    storage_offset = 0
                    for shape in kv_cache_spec.shapes:
                        target_shape = (num_blocks, *shape)
                        stride = torch.empty(target_shape).stride()
                        target_stride = (num_element_per_page, *stride[1:])
                        tensor = torch.as_strided(raw_tensor.view(dtype), size=target_shape, stride=target_stride, storage_offset=storage_offset)
                        state_tensors.append(tensor)
                        storage_offset += stride[0]
                    kv_caches[layer_name] = state_tensors
                else:
                    raise NotImplementedError
        if has_attn and has_mamba:
            self._verify_hybrid_attention_mamba_layout(kv_cache_config, kv_cache_raw_tensors)
        return kv_caches

    def _verify_hybrid_attention_mamba_layout(self, kv_cache_config: KVCacheConfig, kv_cache_raw_tensors: dict[str, torch.Tensor]) -> None:
        """
        Verify that the KV cache memory layout is compatible for
        models with both attention and mamba KV cache groups.

        Args:
            kv_cache_config: The KV cache config
            kv_cache_raw_tensors: The KV cache buffer of each layer.
        """
        for i, kv_cache_group_spec in enumerate(kv_cache_config.kv_cache_groups):
            kv_cache_spec = kv_cache_group_spec.kv_cache_spec
            for layer_name in kv_cache_group_spec.layer_names:
                raw_tensor = kv_cache_raw_tensors[layer_name]
                num_blocks = raw_tensor.numel() // kv_cache_spec.page_size_bytes
                if isinstance(kv_cache_spec, AttentionSpec):
                    kv_cache_shape = self.attn_backends[i].get_kv_cache_shape(num_blocks, kv_cache_spec.block_size, kv_cache_spec.num_kv_heads, kv_cache_spec.head_size)
                    if kv_cache_shape[0] != num_blocks or kv_cache_shape[1] != 2:
                        raise ValueError('Hybrid models in V1 require an attention backend with kv_cache_shape=(num_blocks, 2, ...). Please try setting VLLM_ATTENTION_BACKEND=FLASHINFER')

    def initialize_kv_cache_tensors(self, kv_cache_config: KVCacheConfig) -> dict[str, torch.Tensor]:
        """
        Initialize the memory buffer for KV cache.

        Args:
            kv_cache_config: The KV cache config
        Returns:
            Dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer for KV cache.
        """
        kv_cache_raw_tensors = self._allocate_kv_cache_tensors(kv_cache_config)
        kv_caches = self._reshape_kv_cache_tensors(kv_cache_config, kv_cache_raw_tensors)
        if self.shared_kv_cache_layers:
            initialize_kv_cache_for_kv_sharing(self.shared_kv_cache_layers, kv_cache_config.kv_cache_groups, kv_caches)
        bind_kv_cache(kv_caches, self.compilation_config.static_forward_context, self.kv_caches)
        return kv_caches

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize KV cache based on `kv_cache_config`.
        Args:
            kv_cache_config: Configuration for the KV cache, including the KV
            cache size of each layer
        """
        self.kv_cache_config = kv_cache_config
        self.may_reinitialize_input_batch(kv_cache_config)
        self.initialize_attn_backend(kv_cache_config)
        kv_caches = self.initialize_kv_cache_tensors(kv_cache_config)
        if self.speculative_config and self.speculative_config.use_eagle():
            assert isinstance(self.drafter, EagleProposer)
            self.drafter.validate_same_kv_cache_group(kv_cache_config)
        if has_kv_transfer_group():
            get_kv_transfer_group().register_kv_caches(kv_caches)

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """
        Generates the KVCacheSpec by parsing the kv cache format from each
        Attention module in the static forward context.
        Returns:
            KVCacheSpec: A dictionary mapping layer names to their KV cache
            format. Layers that do not need KV cache are not included.
        """
        block_size = self.vllm_config.cache_config.block_size
        use_mla = self.vllm_config.model_config.use_mla
        kv_cache_spec: dict[str, KVCacheSpec] = {}
        attn_layers = get_layers_from_vllm_config(self.vllm_config, Attention)
        for layer_name, attn_module in attn_layers.items():
            if (kv_tgt_layer := attn_module.kv_sharing_target_layer_name) is not None:
                self.shared_kv_cache_layers[layer_name] = kv_tgt_layer
                continue
            if attn_module.attn_type == AttentionType.DECODER:
                use_local_attention = self.attention_chunk_size is not None and getattr(attn_module.impl, 'use_irope', False)
                if attn_module.sliding_window is not None:
                    kv_cache_spec[layer_name] = SlidingWindowSpec(block_size=block_size, num_kv_heads=attn_module.num_kv_heads, head_size=attn_module.head_size, dtype=self.kv_cache_dtype, sliding_window=attn_module.sliding_window, use_mla=use_mla)
                    assert not use_local_attention, ('attention module can not be with ', 'both local attention and sliding window')
                elif use_local_attention:
                    kv_cache_spec[layer_name] = ChunkedLocalAttentionSpec(block_size=block_size, num_kv_heads=attn_module.num_kv_heads, head_size=attn_module.head_size, dtype=self.kv_cache_dtype, attention_chunk_size=self.attention_chunk_size, use_mla=use_mla)
                else:
                    kv_cache_spec[layer_name] = FullAttentionSpec(block_size=block_size, num_kv_heads=attn_module.num_kv_heads, head_size=attn_module.head_size, dtype=self.kv_cache_dtype, use_mla=use_mla)
            elif attn_module.attn_type in (AttentionType.ENCODER, AttentionType.ENCODER_ONLY):
                continue
            elif attn_module.attn_type == AttentionType.ENCODER_DECODER:
                raise NotImplementedError
            else:
                raise ValueError(f'Unknown attention type: {attn_module.attn_type}')
        mamba_layers = get_layers_from_vllm_config(self.vllm_config, MambaBase)
        if len(mamba_layers) > 0:
            if self.vllm_config.speculative_config is not None:
                raise NotImplementedError('Mamba with speculative decoding is not supported yet.')
            if self.vllm_config.cache_config.enable_prefix_caching:
                raise NotImplementedError('Prefix caching is not supported for Mamba yet.')
            max_model_len = self.vllm_config.model_config.max_model_len
            page_size_padded = self.vllm_config.cache_config.mamba_page_size_padded
            for layer_name, mamba_module in mamba_layers.items():
                kv_cache_spec[layer_name] = MambaSpec(shapes=mamba_module.get_state_shape(), dtype=self.kv_cache_dtype, block_size=max_model_len, page_size_padded=page_size_padded)
        return kv_cache_spec