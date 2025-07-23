from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Generic, Optional, TypeVar
from vllm.attention.backends.abstract import AttentionImpl
from vllm.attention.layer import Attention
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.distributed.kv_transfer.kv_connector.utils import get_kv_connector_cache_layout
from vllm.logger import init_logger
from vllm.utils import cdiv
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.worker.gpu_input_batch import InputBatch
import abc
import functools
import numpy as np
import torch
import vllm.envs as envs
if TYPE_CHECKING:
logger = init_logger(__name__)
_KV_CACHE_LAYOUT_OVERRIDE = None

@dataclass
class CommonAttentionMetadata:
    """
    Per-batch attention metadata, shared across layers and backends.
    AttentionMetadataBuilder instances use it to construct per-layer metadata.
    
    For many of the tensors we keep both GPU and CPU versions.
    """
    query_start_loc: torch.Tensor
    query_start_loc_cpu: torch.Tensor
    '(batch_size + 1,), the start location of each request in query Tensor'
    seq_lens: torch.Tensor
    seq_lens_cpu: torch.Tensor
    '(batch_size,), the length of each request including both computed tokens\n    and newly scheduled tokens'
    num_computed_tokens_cpu: torch.Tensor
    '(batch_size,), the number of computed tokens for each request'
    num_reqs: int
    'Number of requests'
    num_actual_tokens: int
    'Total number of tokens in batch'
    max_query_len: int
    'Longest query in batch'
    block_table_tensor: torch.Tensor
    slot_mapping: torch.Tensor
M = TypeVar('M')

class AttentionMetadataBuilder(abc.ABC, Generic[M]):
    full_cudagraph_supported: ClassVar[bool] = False

    @abstractmethod
    def __init__(self, kv_cache_spec: AttentionSpec, vllm_config: VllmConfig, device: torch.device):
        self.kv_cache_spec = kv_cache_spec

    @abstractmethod
    def build(self, common_prefix_len: int, common_attn_metadata: CommonAttentionMetadata, fast_build: bool=False) -> M:
        """
        Central method that builds attention metadata.
        Some builders (MLA) require reorder_batch to be called prior to build.
        
        Args:
            common_prefix_len: The length of the common prefix of the batch.
            common_attn_metadata: The common attention metadata.
            fast_build: The meta-data will prioritize speed of building over
                then speed at execution. Can be used for spec-decode where the
                result of a build call may only be used for few layers/iters.
        """
        raise NotImplementedError

    def can_run_in_cudagraph(self, common_attn_metadata: CommonAttentionMetadata) -> bool:
        """
        Can this batch (with given metadata) use CUDA Graphs for attention.
        """
        return False

    def build_for_cudagraph_capture(self, common_attn_metadata: CommonAttentionMetadata) -> M:
        """
        Build attention metadata for CUDA graph capture. Uses build by default.
        Subclasses that override this method should call self.build or
        super().build_for_cudagraph_capture.
        """
        return self.build(common_prefix_len=0, common_attn_metadata=common_attn_metadata)

    def use_cascade_attention(self, common_prefix_len: int, query_lens: np.ndarray, num_query_heads: int, num_kv_heads: int, use_alibi: bool, use_sliding_window: bool, use_local_attention: bool, num_sms: int) -> bool:
        return False

    def reorder_batch(self, input_batch: 'InputBatch', scheduler_output: 'SchedulerOutput') -> bool:
        """
        This method can reorder the batch if desired by the backend.
        :return: Has the batch been reordered (default False).
        """
        return False

@functools.lru_cache
def get_kv_cache_layout():
    global _KV_CACHE_LAYOUT_OVERRIDE
    cache_layout = envs.VLLM_KV_CACHE_LAYOUT
    if cache_layout is None:
        cache_layout = get_kv_connector_cache_layout()
    else:
        logger.info_once('`VLLM_KV_CACHE_LAYOUT` environment variable detected. Setting KV cache layout to %s.', cache_layout)
    if _KV_CACHE_LAYOUT_OVERRIDE is not None:
        cache_layout = _KV_CACHE_LAYOUT_OVERRIDE
    return cache_layout

def set_kv_cache_layout(cache_layout: str):
    global _KV_CACHE_LAYOUT_OVERRIDE
    _KV_CACHE_LAYOUT_OVERRIDE = cache_layout

@dataclass
class PerLayerParameters:
    """
    Currently, FlashInfer backend only support models in which all layers share
    the same values for the following hyperparameters.
    """
    window_left: int
    logits_soft_cap: Optional[float]
    sm_scale: float

def get_per_layer_parameters(vllm_config: VllmConfig, cls_: type['AttentionImpl']) -> dict[str, PerLayerParameters]:
    """
    Scan all attention layers and determine some hyperparameters
    to use during `plan`.
    """
    layers = get_layers_from_vllm_config(vllm_config, Attention)
    per_layer_params: dict[str, PerLayerParameters] = {}
    for key, layer in layers.items():
        impl = layer.impl
        assert isinstance(impl, cls_)
        window_size = getattr(impl, 'sliding_window', None)
        window_left = window_size[0] if window_size is not None else -1
        logits_soft_cap = getattr(impl, 'logits_soft_cap', None)
        sm_scale = impl.scale
        per_layer_params[key] = PerLayerParameters(window_left, logits_soft_cap, sm_scale)
    return per_layer_params

def infer_global_hyperparameters(per_layer_params: dict[str, PerLayerParameters]) -> PerLayerParameters:
    """
    Currently, FlashInfer backend only support models in which all layers share
    the same values for the following hyperparameters:
    - `window_left`
    - `logits_soft_cap`
    - `sm_scale`

    So this function asserts that all layers share the same values for these
    hyperparameters and returns the global values.
    """
    assert len(per_layer_params) > 0, 'No attention layers found in the model.'
    param_sets = list(per_layer_params.values())
    global_params = param_sets[0]
    for params in param_sets:
        assert params == global_params, 'FlashInfer backend currently only supports models in which all layers share the same values for the following hyperparameters: `window_left`, `logits_soft_cap`, `sm_scale`.'
    return global_params

def make_local_attention_virtual_batches(attn_chunk_size: int, common_attn_metadata: CommonAttentionMetadata, block_size: int=0) -> CommonAttentionMetadata:
    query_start_loc_np = common_attn_metadata.query_start_loc_cpu.numpy()
    seq_lens_np = common_attn_metadata.seq_lens_cpu.numpy()
    block_table = common_attn_metadata.block_table_tensor
    device = common_attn_metadata.query_start_loc.device
    q_seqlens = query_start_loc_np[1:] - query_start_loc_np[:-1]
    actual_batch_size = seq_lens_np.shape[0]
    q_tokens_in_first_block = np.minimum(attn_chunk_size - (seq_lens_np - q_seqlens) % attn_chunk_size, q_seqlens).astype(np.int32)
    tokens_in_last_block = attn_chunk_size + seq_lens_np % -attn_chunk_size
    local_blocks = 1 + cdiv(q_seqlens - q_tokens_in_first_block, attn_chunk_size)
    cu_num_blocks = np.cumsum(local_blocks)
    virtual_batches = cu_num_blocks[-1]
    block_offsets = np.repeat(cu_num_blocks - local_blocks, local_blocks)
    arange = np.arange(virtual_batches, dtype=np.int32) - block_offsets
    rarange = np.repeat(local_blocks, local_blocks) - arange - 1
    seqlens_q_local = np.repeat(q_seqlens - q_tokens_in_first_block, local_blocks)
    seqlens_q_local[arange == 0] = q_tokens_in_first_block
    seqlens_q_local[arange > 0] = np.minimum(seqlens_q_local - attn_chunk_size * (arange - 1), attn_chunk_size)[arange > 0]
    cu_seqlens_q_local = np.pad(np.cumsum(seqlens_q_local), (1, 0)).astype(np.int32)
    seqlens_k_local = np.full(cu_num_blocks[-1], attn_chunk_size, dtype=np.int32)
    seqlens_k_local[cu_num_blocks - 1] = tokens_in_last_block
    num_computed_tokens_local = seqlens_k_local - seqlens_q_local
    k_seqstarts_absolute = np.repeat(seq_lens_np, local_blocks) - (rarange * attn_chunk_size + np.repeat(tokens_in_last_block, local_blocks))
    block_starts = k_seqstarts_absolute // block_size
    assert attn_chunk_size % block_size == 0, f'attn_chunk_size {attn_chunk_size} is not divisible by block_size {block_size}'
    pages_per_local_batch = attn_chunk_size // block_size
    block_indices = np.broadcast_to(np.arange(pages_per_local_batch, dtype=np.int32), (virtual_batches, pages_per_local_batch)) + np.expand_dims(block_starts, axis=1)
    block_indices = block_indices.flatten().clip(max=block_table.shape[1] - 1)
    batch_indices = np.repeat(np.arange(actual_batch_size, dtype=np.int32), local_blocks * pages_per_local_batch)
    block_table_local = block_table[batch_indices, block_indices].view(virtual_batches, -1)
    query_start_loc_cpu = torch.from_numpy(cu_seqlens_q_local)
    seq_lens_cpu = torch.from_numpy(seqlens_k_local)
    return CommonAttentionMetadata(query_start_loc_cpu=query_start_loc_cpu, query_start_loc=query_start_loc_cpu.to(device=device, non_blocking=True), seq_lens_cpu=seq_lens_cpu, seq_lens=seq_lens_cpu.to(device=device, non_blocking=True), num_computed_tokens_cpu=torch.from_numpy(num_computed_tokens_local), num_reqs=len(seq_lens_cpu), num_actual_tokens=common_attn_metadata.num_actual_tokens, max_query_len=seqlens_q_local.max(), block_table_tensor=block_table_local, slot_mapping=common_attn_metadata.slot_mapping)

def split_decodes_and_prefills(common_attn_metadata: CommonAttentionMetadata, decode_threshold: int=1) -> tuple[int, int, int, int]:
    """
    Assuming a reordered batch, finds the boundary between prefill and decode
    requests.

    Args:
        common_attn_metadata: CommonAttentionMetadata object containing the
            batch metadata.
        decode_threshold: The maximum query length to be considered a decode.

    Returns:
        num_decodes: The number of decode requests.
        num_prefills: The number of prefill requests.
        num_decode_tokens: The number of tokens in the decode requests.
        num_prefill_tokens: The number of tokens in the prefill requests.
    """
    max_query_len = common_attn_metadata.max_query_len
    num_reqs = common_attn_metadata.num_reqs
    num_tokens = common_attn_metadata.num_actual_tokens
    query_start_loc = common_attn_metadata.query_start_loc_cpu
    if max_query_len <= decode_threshold:
        return (num_reqs, 0, num_tokens, 0)
    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    is_prefill = query_lens > decode_threshold
    if not torch.any(is_prefill):
        return (num_reqs, 0, num_tokens, 0)
    first_prefill = is_prefill.int().argmax(dim=-1).item()
    assert torch.all(query_lens[first_prefill:] > decode_threshold)
    assert torch.all(query_lens[:first_prefill] <= decode_threshold)
    num_decodes = first_prefill
    num_prefills = num_reqs - num_decodes
    num_decode_tokens = query_start_loc[first_prefill].item()
    num_prefill_tokens = num_tokens - num_decode_tokens
    return (num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens)

def reorder_batch_to_split_decodes_and_prefills(input_batch: 'InputBatch', scheduler_output: 'SchedulerOutput', decode_threshold: int=1) -> bool:
    """
    Reorders the batch to split into prefill and decode requests; places all
    requests with <= decode_threshold tokens at the front of the batch.
    
    Returns:
        True if the batch was modified, False otherwise.
    """
    decodes = []
    prefills = []
    num_decode_tokens = 0
    num_prefill_tokens = 0
    for i, req_id in enumerate(input_batch.req_ids):
        num_tokens = scheduler_output.num_scheduled_tokens[req_id]
        if num_tokens <= decode_threshold:
            decodes.append(i)
            num_decode_tokens += num_tokens
        else:
            prefills.append(i)
            num_prefill_tokens += num_tokens
    num_decodes = len(decodes)
    num_prefills = len(prefills)
    modified_batch = False
    for i in range(1, min(num_decodes, num_prefills) + 1):
        decode_idx = decodes[num_decodes - i]
        if decode_idx < num_decodes:
            break
        input_batch.swap_states(prefills[i - 1], decode_idx)
        modified_batch = True
    return modified_batch