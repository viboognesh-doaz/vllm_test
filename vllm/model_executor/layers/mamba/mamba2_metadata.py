from dataclasses import dataclass
from typing import Optional, Union
from vllm.attention.backends.abstract import AttentionMetadata
from vllm.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.attention.backends.placeholder_attn import PlaceholderAttentionMetadata
from vllm.attention.backends.rocm_flash_attn import ROCmFlashAttentionMetadata
from vllm.attention.backends.utils import PAD_SLOT_ID
from vllm.attention.backends.xformers import XFormersMetadata
from vllm.platforms import current_platform
from vllm.v1.attention.backends.mamba_attn import Mamba2AttentionMetadata, _query_start_loc_to_chunk_indices_offsets
import numpy as np
import torch

@dataclass
class Mamba2Metadata:
    has_initial_states: torch.Tensor
    prep_initial_states: bool
    chunk_size: int
    seq_idx: torch.Tensor
    chunk_indices: torch.Tensor
    chunk_offsets: torch.Tensor
    '\n    With continuous batching layout of `x` in vLLM, to enable a Triton program\n    to handle a request in parallel, two supporting tensors are used\n    (batch_ptr, token_chunk_offset_ptr)\n    BLOCK_M = the # tokens to be handled by a Triton program\n              (can be customized for different hardware)\n\n    nums_dict:\n       tracks the data associated with a given value of BLOCK_M\n       BLOCK_M = #tokens handled by a Triton program\n    cu_seqlen: total tokens per batch\n           (used as flag to update other data at each new input)\n    batch_ptr: tracks batch-id handled by the Triton program\n    token_chunk_offset_ptr: tracks token group_idx handled by the Triton program\n           (Triton implementation of causal_conv1d handles parallelism in 3-axes\n           - feature-axis\n           - batch-axis\n           - sequence-axis)\n    '
    nums_dict: Optional[dict] = None
    cu_seqlen: Optional[int] = None
    batch_ptr: Optional[torch.tensor] = None
    token_chunk_offset_ptr: Optional[torch.tensor] = None

def get_platform_metadata_classes() -> tuple[type[AttentionMetadata], ...]:
    """Returns the appropriate metadata classes for the current platform."""
    if current_platform.is_rocm():
        return (ROCmFlashAttentionMetadata, PlaceholderAttentionMetadata)
    elif current_platform.is_cuda():
        return (FlashAttentionMetadata, XFormersMetadata, PlaceholderAttentionMetadata)
    raise ValueError(f'Unsupported platform for Mamba2: {current_platform.device_type}')

def prepare_mamba2_metadata(chunk_size: int, attn_metadata: AttentionMetadata, mamba2_metadata=None) -> Mamba2Metadata:
    num_prefills = attn_metadata.num_prefills
    num_prefill_tokens = attn_metadata.num_prefill_tokens
    seq_idx = None
    chunk_indices, chunk_offsets = (None, None)
    has_initial_states = None
    prep_initial_states = False
    if num_prefills > 0:
        attn_metadata_instances = get_platform_metadata_classes()
        if isinstance(attn_metadata, attn_metadata_instances) and attn_metadata.context_lens_tensor is not None:
            has_initial_states = attn_metadata.context_lens_tensor > 0
            prep_initial_states = torch.any(has_initial_states[:num_prefills]).item()
        query_start_loc = attn_metadata.query_start_loc[:num_prefills + 1]
        seq_idx = torch.repeat_interleave(torch.arange(num_prefills, dtype=torch.int32, device=query_start_loc.device), query_start_loc.diff(), output_size=num_prefill_tokens)
        seq_idx.unsqueeze_(0)
        if prep_initial_states:
            chunk_indices, chunk_offsets = _query_start_loc_to_chunk_indices_offsets(query_start_loc, chunk_size, num_prefill_tokens)
    if mamba2_metadata is not None:
        mamba2_metadata.has_initial_states = has_initial_states
        mamba2_metadata.prep_initial_states = prep_initial_states
        mamba2_metadata.chunk_size = chunk_size
        mamba2_metadata.seq_idx = seq_idx
        mamba2_metadata.chunk_indices = chunk_indices
        mamba2_metadata.chunk_offsets = chunk_offsets
        mamba2_metadata.cu_seqlen = None
        return mamba2_metadata
    return Mamba2Metadata(has_initial_states=has_initial_states, prep_initial_states=prep_initial_states, chunk_size=chunk_size, seq_idx=seq_idx, chunk_indices=chunk_indices, chunk_offsets=chunk_offsets)

def update_metadata(x: torch.Tensor, query_start_loc: torch.Tensor, mamba2_metadata: Union[Mamba2Metadata, Mamba2AttentionMetadata]):
    """
    this is triggered upon handling a new input at the first layer
    """
    dim, cu_seqlen = x.shape
    mamba2_metadata.cu_seqlen = cu_seqlen
    seqlens = np.diff(query_start_loc.to('cpu'))
    nums_dict = {}
    for BLOCK_M in [8]:
        nums = -(-seqlens // BLOCK_M)
        nums_dict[BLOCK_M] = {}
        nums_dict[BLOCK_M]['nums'] = nums
        nums_dict[BLOCK_M]['tot'] = nums.sum().item()
        mlist = torch.from_numpy(np.repeat(np.arange(len(nums)), nums))
        nums_dict[BLOCK_M]['mlist'] = mlist
        mlist_len = len(nums_dict[BLOCK_M]['mlist'])
        nums_dict[BLOCK_M]['mlist_len'] = mlist_len
        MAX_NUM_PROGRAMS = max(1024, mlist_len) * 2
        offsetlist = []
        for idx, num in enumerate(nums):
            offsetlist.extend(range(num))
        offsetlist = torch.tensor(offsetlist, dtype=torch.int32)
        nums_dict[BLOCK_M]['offsetlist'] = offsetlist
        if mamba2_metadata.batch_ptr is None:
            mamba2_metadata.batch_ptr = torch.full((MAX_NUM_PROGRAMS,), PAD_SLOT_ID, dtype=torch.int32, device='cuda')
            mamba2_metadata.token_chunk_offset_ptr = torch.full((MAX_NUM_PROGRAMS,), PAD_SLOT_ID, dtype=torch.int32, device='cuda')
        elif mamba2_metadata.batch_ptr.nelement() < MAX_NUM_PROGRAMS:
            mamba2_metadata.batch_ptr.resize_(MAX_NUM_PROGRAMS).fill_(PAD_SLOT_ID)
            mamba2_metadata.token_chunk_offset_ptr.resize_(MAX_NUM_PROGRAMS).fill_(PAD_SLOT_ID)
        mamba2_metadata.batch_ptr[0:mlist_len].copy_(mlist)
        mamba2_metadata.token_chunk_offset_ptr[0:mlist_len].copy_(offsetlist)
        nums_dict[BLOCK_M]['batch_ptr'] = mamba2_metadata.batch_ptr
        nums_dict[BLOCK_M]['token_chunk_offset_ptr'] = mamba2_metadata.token_chunk_offset_ptr
    mamba2_metadata.nums_dict = nums_dict
    return mamba2_metadata