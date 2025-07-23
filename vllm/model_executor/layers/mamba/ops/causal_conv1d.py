from typing import Optional, Union
from vllm.attention.backends.utils import PAD_SLOT_ID
from vllm.triton_utils import tl, triton
import numpy as np
import torch

@triton.jit()
def _causal_conv1d_fwd_kernel(x_ptr, w_ptr, bias_ptr, initial_states_ptr, cache_indices_ptr, has_initial_states_ptr, query_start_loc_ptr, batch_ptr, token_chunk_offset_ptr, o_ptr, batch: tl.int32, dim: tl.constexpr, seqlen: tl.int32, num_cache_lines: tl.constexpr, stride_x_seq: tl.constexpr, stride_x_dim: tl.constexpr, stride_x_token: tl.constexpr, stride_w_dim: tl.constexpr, stride_w_width: tl.constexpr, stride_istate_seq: tl.constexpr, stride_istate_dim: tl.constexpr, stride_istate_token: tl.constexpr, stride_o_seq: tl.constexpr, stride_o_dim: tl.constexpr, stride_o_token: tl.constexpr, pad_slot_id: tl.constexpr, HAS_BIAS: tl.constexpr, KERNEL_WIDTH: tl.constexpr, SILU_ACTIVATION: tl.constexpr, HAS_INITIAL_STATES: tl.constexpr, HAS_CACHE: tl.constexpr, IS_CONTINUOUS_BATCHING: tl.constexpr, USE_PAD_SLOT: tl.constexpr, NP2_STATELEN: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    conv_states_ptr = initial_states_ptr
    conv_state_indices_ptr = cache_indices_ptr
    stride_conv_state_seq = stride_istate_seq
    stride_conv_state_dim = stride_istate_dim
    stride_conv_state_tok = stride_istate_token
    state_len = KERNEL_WIDTH - 1
    idx_seq = tl.load(batch_ptr + tl.program_id(0))
    chunk_offset = tl.load(token_chunk_offset_ptr + tl.program_id(0))
    idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    if idx_seq == pad_slot_id:
        return
    sequence_start_index = tl.load(query_start_loc_ptr + idx_seq)
    sequence_end_index = tl.load(query_start_loc_ptr + idx_seq + 1)
    seqlen = sequence_end_index - sequence_start_index
    token_offset = BLOCK_M * chunk_offset
    segment_len = min(BLOCK_M, seqlen - token_offset)
    x_base = x_ptr + sequence_start_index * stride_x_token + idx_feats * stride_x_dim
    if IS_CONTINUOUS_BATCHING:
        conv_state_batch_coord = tl.load(conv_state_indices_ptr + idx_seq).to(tl.int64)
    else:
        conv_state_batch_coord = idx_seq
    if USE_PAD_SLOT:
        if conv_state_batch_coord == pad_slot_id:
            return
    conv_states_base = conv_states_ptr + conv_state_batch_coord * stride_conv_state_seq + idx_feats * stride_conv_state_dim
    w_base = w_ptr + idx_feats * stride_w_dim
    if chunk_offset == 0:
        load_init_state = False
        if HAS_INITIAL_STATES:
            load_init_state = tl.load(has_initial_states_ptr + idx_seq).to(tl.int1)
        if load_init_state:
            prior_tokens = conv_states_base + (state_len - 1) * stride_conv_state_tok
            mask_w = idx_feats < dim
            if KERNEL_WIDTH == 2:
                conv_states_ptrs = prior_tokens
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
            if KERNEL_WIDTH == 3:
                conv_states_ptrs = prior_tokens
                col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 1 * stride_conv_state_tok
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
            if KERNEL_WIDTH == 4:
                conv_states_ptrs = prior_tokens
                col2 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 1 * stride_conv_state_tok
                col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 2 * stride_conv_state_tok
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
            if KERNEL_WIDTH == 5:
                conv_states_ptrs = prior_tokens
                col3 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 1 * stride_conv_state_tok
                col2 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 2 * stride_conv_state_tok
                col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 3 * stride_conv_state_tok
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
        else:
            if KERNEL_WIDTH >= 2:
                col0 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            if KERNEL_WIDTH >= 3:
                col1 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            if KERNEL_WIDTH >= 4:
                col2 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            if KERNEL_WIDTH >= 5:
                col3 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
        if state_len <= seqlen:
            idx_tokens_last = seqlen - state_len + tl.arange(0, NP2_STATELEN)
            x_ptrs = x_ptr + ((sequence_start_index + idx_tokens_last) * stride_x_token)[:, None] + (idx_feats * stride_x_dim)[None, :]
            mask_x = (idx_tokens_last >= 0)[:, None] & (idx_tokens_last < seqlen)[:, None] & (idx_feats < dim)[None, :]
            loaded_x = tl.load(x_ptrs, mask_x, 0.0)
            new_conv_state = tl.load(x_ptrs, mask_x, 0.0)
            idx_tokens_conv = tl.arange(0, NP2_STATELEN)
            conv_states_ptrs_target = conv_states_base[None, :] + (idx_tokens_conv * stride_conv_state_tok)[:, None]
            mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[None, :]
            tl.debug_barrier()
            tl.store(conv_states_ptrs_target, new_conv_state, mask)
        elif load_init_state:
            idx_tokens_conv = tl.arange(0, NP2_STATELEN)
            conv_states_ptrs_source = conv_states_ptr + conv_state_batch_coord * stride_conv_state_seq + (idx_feats * stride_conv_state_dim)[None, :] + ((idx_tokens_conv + seqlen) * stride_conv_state_tok)[:, None]
            mask = (conv_state_batch_coord < num_cache_lines) & (idx_tokens_conv + seqlen < state_len)[:, None] & (idx_feats < dim)[None, :]
            conv_state = tl.load(conv_states_ptrs_source, mask, other=0.0)
            VAL = state_len - seqlen
            x_ptrs = x_base[None, :] + ((idx_tokens_conv - VAL) * stride_x_token)[:, None]
            mask_x = (idx_tokens_conv - VAL >= 0)[:, None] & (idx_tokens_conv - VAL < seqlen)[:, None] & (idx_feats < dim)[None, :]
            loaded_x = tl.load(x_ptrs, mask_x, 0.0)
            tl.debug_barrier()
            new_conv_state = tl.where(mask, conv_state, loaded_x)
            conv_states_ptrs_target = conv_states_base + (idx_tokens_conv * stride_conv_state_tok)[:, None]
            mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[None, :]
            tl.store(conv_states_ptrs_target, new_conv_state, mask)
        else:
            idx_tokens_conv = tl.arange(0, NP2_STATELEN)
            VAL = state_len - seqlen
            x_ptrs = x_base[None, :] + ((idx_tokens_conv - VAL) * stride_x_token)[:, None]
            mask_x = (idx_tokens_conv - VAL >= 0)[:, None] & (idx_tokens_conv - VAL < seqlen)[:, None] & (idx_feats < dim)[None, :]
            new_conv_state = tl.load(x_ptrs, mask_x, 0.0)
            conv_states_ptrs_target = conv_states_base + (idx_tokens_conv * stride_conv_state_tok)[:, None]
            mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[None, :]
            tl.store(conv_states_ptrs_target, new_conv_state, mask)
    else:
        load_init_state = True
        prior_tokens = x_base + (token_offset - 1) * stride_x_token
        mask_w = idx_feats < dim
        if KERNEL_WIDTH == 2:
            conv_states_ptrs = prior_tokens
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier='.ca')
        if KERNEL_WIDTH == 3:
            conv_states_ptrs = prior_tokens
            col1 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier='.ca')
            conv_states_ptrs = prior_tokens - 1 * stride_x_token
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier='.ca')
        if KERNEL_WIDTH == 4:
            conv_states_ptrs = prior_tokens
            col2 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier='.ca')
            conv_states_ptrs = prior_tokens - 1 * stride_x_token
            col1 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier='.ca')
            conv_states_ptrs = prior_tokens - 2 * stride_x_token
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier='.ca')
        if KERNEL_WIDTH == 5:
            conv_states_ptrs = prior_tokens
            col3 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier='.ca')
            conv_states_ptrs = prior_tokens - 1 * stride_x_token
            col2 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier='.ca')
            conv_states_ptrs = prior_tokens - 2 * stride_x_token
            col1 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier='.ca')
            conv_states_ptrs = prior_tokens - 3 * stride_x_token
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier='.ca')
    if HAS_BIAS:
        bias = bias_ptr + idx_feats
        mask_bias = idx_feats < dim
        acc_preload = tl.load(bias, mask=mask_bias, other=0.0).to(tl.float32)
    else:
        acc_preload = tl.zeros((BLOCK_N,), dtype=tl.float32)
    x_base_1d = x_base + token_offset * stride_x_token
    mask_w = idx_feats < dim
    if KERNEL_WIDTH >= 2:
        w_ptrs = w_base + 0 * stride_w_width
        w_col0 = tl.load(w_ptrs, mask_w, other=0.0)
        w_ptrs = w_base + 1 * stride_w_width
        w_col1 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 3:
        w_ptrs = w_base + 2 * stride_w_width
        w_col2 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 4:
        w_ptrs = w_base + 3 * stride_w_width
        w_col3 = tl.load(w_ptrs, mask_w, other=0.0)
    mask_x_1d = idx_feats < dim
    for idx_token in range(segment_len):
        acc = acc_preload
        matrix_w = w_col0
        matrix_x = col0
        for j in tl.static_range(KERNEL_WIDTH):
            if KERNEL_WIDTH == 2:
                if j == 1:
                    matrix_w = w_col1
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 3:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 4:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = col2
                elif j == 3:
                    matrix_w = w_col3
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            acc += matrix_x * matrix_w
        if KERNEL_WIDTH == 2:
            col0 = matrix_x
        elif KERNEL_WIDTH == 3:
            col0 = col1
            col1 = matrix_x
        elif KERNEL_WIDTH == 4:
            col0 = col1
            col1 = col2
            col2 = matrix_x
        if SILU_ACTIVATION:
            acc = acc / (1 + tl.exp(-acc))
        mask_1d = (idx_token < segment_len) & (idx_feats < dim)
        o_ptrs = o_ptr + (sequence_start_index + token_offset + idx_token) * stride_o_token + idx_feats * stride_o_dim
        tl.store(o_ptrs, acc, mask=mask_1d)

def causal_conv1d_fn(x: torch.Tensor, weight: torch.Tensor, bias: Union[torch.Tensor, None], conv_states: torch.Tensor, query_start_loc: torch.Tensor, cache_indices: Optional[torch.Tensor]=None, has_initial_state: Optional[torch.Tensor]=None, activation: Optional[str]='silu', pad_slot_id: int=PAD_SLOT_ID, metadata=None, validate_data=False):
    """support varlen + continuous batching when x is 2D tensor

    x: (dim,cu_seq_len)
        cu_seq_len = total tokens of all seqs in that batch
        sequences are concatenated from left to right for varlen
    weight: (dim, width)
    conv_states: (...,dim,width - 1) itype
        updated inplace if provided
        [it use `cache_indices` to get the index to the cache of conv_state for that sequence

        conv_state[cache_indices[i]] for seq-i - to be used as initial_state when has_initial_state[i] = True
             and after that conv_state[cache_indices[i]] need to be shift-left and updated with values from 'x'
        ]
    query_start_loc: (batch + 1) int32
        The cumulative sequence lengths of the sequences in
        the batch, used to index into sequence. prepended by 0.
        if
        x = [5, 1, 1, 1] <- continuous batching (batch=4)
        then
        query_start_loc = [0, 5, 6, 7, 8] <- the starting index of the next sequence; while the last value is
           the ending index of the last sequence
        [length(query_start_loc)-1 == batch]
        for example: query_start_loc = torch.Tensor([0,10,16,17]),
        x.shape=(dim,17)
    cache_indices: (batch)  int32
        indicates the corresponding state index,
        like so: conv_state = conv_states[cache_indices[batch_id]]
    has_initial_state: (batch) bool
        indicates whether should the kernel take the current state as initial
        state for the calculations
        [single boolean for each sequence in the batch: True or False]
    bias: (dim,)
    activation: either None or "silu" or "swish" or True
    pad_slot_id: int
        if cache_indices is passed, lets the kernel identify padded
        entries that will not be processed,
        for example: cache_indices = [pad_slot_id, 1, 20, pad_slot_id]
        in this case, the kernel will not process entries at
        indices 0 and 3

    out: same shape as `x`
    """
    if isinstance(activation, bool) and activation:
        activation = 'silu'
    args = None
    out = torch.empty_like(x)
    if metadata is not None:
        cu_seqlen = metadata.cu_seqlen
        nums_dict = metadata.nums_dict
        args = nums_dict
        batch_ptr = metadata.batch_ptr
        token_chunk_offset_ptr = metadata.token_chunk_offset_ptr
    else:
        seqlens = np.diff(query_start_loc.to('cpu'))
        args = seqlens
        MAX_NUM_PROGRAMS = 1024
        batch_ptr = torch.full((MAX_NUM_PROGRAMS,), PAD_SLOT_ID, dtype=torch.int32, device=x.device)
        token_chunk_offset_ptr = torch.full((MAX_NUM_PROGRAMS,), PAD_SLOT_ID, dtype=torch.int32, device=x.device)
    is_channel_last = (x.stride(0) == 1) & (x.stride(1) > 1)
    dim, cu_seqlen = x.shape
    _, width = weight.shape
    state_len = width - 1
    np2_statelen = triton.next_power_of_2(state_len)
    padded_batch = query_start_loc.size(0) - 1
    stride_x_seq = 0
    stride_x_dim = x.stride(0)
    stride_x_token = x.stride(1)
    stride_w_dim = weight.stride(0)
    stride_w_width = weight.stride(1)
    stride_istate_seq = 0
    stride_istate_dim = 0
    stride_istate_token = 0
    num_cache_lines = 0
    if conv_states is not None:
        num_cache_lines = conv_states.size(0)
        assert (num_cache_lines, dim, width - 1) == conv_states.shape
        stride_istate_seq = conv_states.stride(0)
        stride_istate_dim = conv_states.stride(1)
        stride_istate_token = conv_states.stride(2)
        assert stride_istate_dim == 1
    if out.dim() == 2:
        stride_o_seq = 0
        stride_o_dim = out.stride(0)
        stride_o_token = out.stride(1)
    else:
        stride_o_seq = out.stride(0)
        stride_o_dim = out.stride(1)
        stride_o_token = out.stride(2)
    if validate_data:
        assert x.dim() == 2
        assert query_start_loc is not None
        assert query_start_loc.dim() == 1
        assert x.stride(0) == 1 or x.stride(1) == 1
        if bias is not None:
            assert bias.dim() == 1
            assert dim == bias.size(0)
        if cache_indices is not None:
            assert cache_indices.dim() == 1
            assert padded_batch == cache_indices.size(0)
        if has_initial_state is not None:
            assert has_initial_state.size() == (padded_batch,)
            assert conv_states is not None, 'ERROR: `has_initial_state` is used, which needs also `conv_states`'
        assert weight.stride(1) == 1
        assert (dim, width) == weight.shape
        assert is_channel_last, 'Need to run in channel-last layout'
    if metadata is None:

        def num_program(META, seqlens):
            tot = 0
            mlist = []
            offsetlist = []
            nums = -(-seqlens // META['BLOCK_M'])
            tot = nums.sum().item()
            mlist = np.repeat(np.arange(len(nums)), nums)
            for idx, num in enumerate(nums):
                offsetlist.extend(range(num))
            if META['batch_ptr'].nelement() < len(mlist):
                newlen = len(mlist) + 1
                META['batch_ptr'].resize_(newlen).fill_(PAD_SLOT_ID)
                META['token_chunk_offset_ptr'].resize_(newlen).fill_(PAD_SLOT_ID)
            if META['batch_ptr'].nelement() >= len(mlist):
                META['batch_ptr'][0:len(mlist)].copy_(torch.from_numpy(np.array(mlist)))
                META['token_chunk_offset_ptr'][0:len(mlist)].copy_(torch.from_numpy(np.array(offsetlist)))
            META['batch_ptr'] = META['batch_ptr'].to(META['x_ptr'].device)
            META['token_chunk_offset_ptr'] = META['token_chunk_offset_ptr'].to(META['x_ptr'].device)
            return tot
    else:

        def num_program(META, nums_dict):
            tot = nums_dict[META['BLOCK_M']]['tot']
            mlist = nums_dict[META['BLOCK_M']]['mlist']
            mlist_len = nums_dict[META['BLOCK_M']]['mlist_len']
            offsetlist = nums_dict[META['BLOCK_M']]['offsetlist']
            if nums_dict[META['BLOCK_M']]['batch_ptr'] is not None:
                META['batch_ptr'] = nums_dict[META['BLOCK_M']]['batch_ptr']
                META['token_chunk_offset_ptr'] = nums_dict[META['BLOCK_M']]['token_chunk_offset_ptr']
            else:
                if META['batch_ptr'].nelement() < mlist_len:
                    newlen = mlist_len + 1
                    META['batch_ptr'].resize_(newlen).fill_(PAD_SLOT_ID)
                    META['token_chunk_offset_ptr'].resize_(newlen).fill_(PAD_SLOT_ID)
                if META['batch_ptr'].nelement() >= mlist_len:
                    META['batch_ptr'][0:mlist_len].copy_(mlist)
                    META['token_chunk_offset_ptr'][0:mlist_len].copy_(offsetlist)
            return tot

    def grid(META):
        return (num_program(META, args), triton.cdiv(dim, META['BLOCK_N']))
    if batch_ptr.device != x.device:
        batch_ptr = batch_ptr.to(x.device)
        token_chunk_offset_ptr = token_chunk_offset_ptr.to(x.device)
    _causal_conv1d_fwd_kernel[grid](x, weight, bias, conv_states, cache_indices, has_initial_state, query_start_loc, batch_ptr, token_chunk_offset_ptr, out, padded_batch, dim, cu_seqlen, num_cache_lines, stride_x_seq, stride_x_dim, stride_x_token, stride_w_dim, stride_w_width, stride_istate_seq, stride_istate_dim, stride_istate_token, stride_o_seq, stride_o_dim, stride_o_token, pad_slot_id, HAS_BIAS=bias is not None, KERNEL_WIDTH=width, SILU_ACTIVATION=activation in ['silu', 'swish'], HAS_INITIAL_STATES=has_initial_state is not None, HAS_CACHE=conv_states is not None, IS_CONTINUOUS_BATCHING=cache_indices is not None, USE_PAD_SLOT=pad_slot_id is not None, NP2_STATELEN=np2_statelen, BLOCK_M=8, BLOCK_N=256, num_stages=2)
    return out

@triton.jit()
def _causal_conv1d_update_kernel(x_ptr, w_ptr, bias_ptr, conv_state_ptr, cache_seqlens_ptr, conv_state_indices_ptr, o_ptr, batch: int, dim: tl.constexpr, seqlen: tl.constexpr, state_len: tl.constexpr, num_cache_lines: tl.constexpr, stride_x_seq: tl.constexpr, stride_x_dim: tl.constexpr, stride_x_token: tl.constexpr, stride_w_dim: tl.constexpr, stride_w_width: tl.constexpr, stride_conv_state_seq: tl.constexpr, stride_conv_state_dim: tl.constexpr, stride_conv_state_tok: tl.constexpr, stride_o_seq: tl.constexpr, stride_o_dim: tl.constexpr, stride_o_token: tl.constexpr, pad_slot_id: tl.constexpr, HAS_BIAS: tl.constexpr, KERNEL_WIDTH: tl.constexpr, SILU_ACTIVATION: tl.constexpr, IS_CONTINUOUS_BATCHING: tl.constexpr, NP2_STATELEN: tl.constexpr, USE_PAD_SLOT: tl.constexpr, BLOCK_N: tl.constexpr):
    idx_seq = tl.program_id(0)
    if idx_seq >= batch:
        return
    idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    if IS_CONTINUOUS_BATCHING:
        conv_state_batch_coord = tl.load(conv_state_indices_ptr + idx_seq).to(tl.int64)
    else:
        conv_state_batch_coord = idx_seq
    if USE_PAD_SLOT:
        if conv_state_batch_coord == pad_slot_id:
            return
    conv_states_base = conv_state_ptr + conv_state_batch_coord * stride_conv_state_seq + idx_feats * stride_conv_state_dim
    mask_w = idx_feats < dim
    prior_tokens = conv_states_base
    if KERNEL_WIDTH >= 2:
        conv_states_ptrs = prior_tokens
        col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
    if KERNEL_WIDTH >= 3:
        conv_states_ptrs = prior_tokens + 1 * stride_conv_state_tok
        col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
    if KERNEL_WIDTH >= 4:
        conv_states_ptrs = prior_tokens + 2 * stride_conv_state_tok
        col2 = tl.load(conv_states_ptrs, mask_w, 0.0)
    if KERNEL_WIDTH == 5:
        conv_states_ptrs = prior_tokens + 3 * stride_conv_state_tok
        col3 = tl.load(conv_states_ptrs, mask_w, 0.0)
    idx_tokens = tl.arange(0, NP2_STATELEN)
    conv_state_ptrs_source = conv_state_ptr + conv_state_batch_coord * stride_conv_state_seq + (idx_feats * stride_conv_state_dim)[None, :] + ((idx_tokens + seqlen) * stride_conv_state_tok)[:, None]
    mask = (conv_state_batch_coord < num_cache_lines) & (idx_tokens + seqlen < state_len)[:, None] & (idx_feats < dim)[None, :]
    conv_state = tl.load(conv_state_ptrs_source, mask, other=0.0)
    VAL = state_len - seqlen
    x_base = x_ptr + idx_seq * stride_x_seq + idx_feats * stride_x_dim
    x_ptrs = x_base[None, :] + ((idx_tokens - VAL) * stride_x_token)[:, None]
    mask_x = (idx_tokens - VAL >= 0)[:, None] & (idx_tokens - VAL < seqlen)[:, None] & (idx_feats < dim)[None, :]
    loaded_x = tl.load(x_ptrs, mask_x, 0.0)
    tl.debug_barrier()
    new_conv_state = tl.where(mask, conv_state, loaded_x)
    conv_state_base = conv_state_ptr + conv_state_batch_coord * stride_conv_state_seq + idx_feats * stride_conv_state_dim
    conv_state_ptrs_target = conv_state_base + (idx_tokens * stride_conv_state_tok)[:, None]
    mask = (idx_tokens < state_len)[:, None] & (idx_feats < dim)[None, :]
    tl.store(conv_state_ptrs_target, new_conv_state, mask)
    if HAS_BIAS:
        bias = bias_ptr + idx_feats
        mask_bias = idx_feats < dim
        acc_preload = tl.load(bias, mask=mask_bias, other=0.0).to(tl.float32)
    else:
        acc_preload = tl.zeros((BLOCK_N,), dtype=tl.float32)
    w_base = w_ptr + idx_feats * stride_w_dim
    mask_w = idx_feats < dim
    if KERNEL_WIDTH >= 2:
        w_ptrs = w_base + 0 * stride_w_width
        w_col0 = tl.load(w_ptrs, mask_w, other=0.0)
        w_ptrs = w_base + 1 * stride_w_width
        w_col1 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 3:
        w_ptrs = w_base + 2 * stride_w_width
        w_col2 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 4:
        w_ptrs = w_base + 3 * stride_w_width
        w_col3 = tl.load(w_ptrs, mask_w, other=0.0)
    x_base_1d = x_base
    mask_x_1d = idx_feats < dim
    for idx_token in tl.static_range(seqlen):
        acc = acc_preload
        matrix_w = w_col0
        matrix_x = col0
        for j in tl.static_range(KERNEL_WIDTH):
            if KERNEL_WIDTH == 2:
                if j == 1:
                    matrix_w = w_col1
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 3:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 4:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = col2
                elif j == 3:
                    matrix_w = w_col3
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            acc += matrix_x * matrix_w
        if KERNEL_WIDTH == 2:
            col0 = matrix_x
        elif KERNEL_WIDTH == 3:
            col0 = col1
            col1 = matrix_x
        elif KERNEL_WIDTH == 4:
            col0 = col1
            col1 = col2
            col2 = matrix_x
        if SILU_ACTIVATION:
            acc = acc / (1 + tl.exp(-acc))
        mask_1d = (idx_token < seqlen) & (idx_feats < dim)
        o_ptrs = o_ptr + idx_seq * stride_o_seq + idx_token * stride_o_token + idx_feats * stride_o_dim
        tl.store(o_ptrs, acc, mask=mask_1d)

def causal_conv1d_update(x: torch.Tensor, conv_state: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]=None, activation: Union[bool, str, None]=None, cache_seqlens: Optional[torch.Tensor]=None, conv_state_indices: Optional[torch.Tensor]=None, pad_slot_id: int=PAD_SLOT_ID, metadata=None, validate_data=False):
    """
    x: (batch, dim) or (batch, dim, seqlen)
        [shape=2: single token prediction]
        [shape=3: single or multiple tokens prediction]
    conv_state: (..., dim, state_len), where state_len >= width - 1
    weight: (dim, width)
    bias: (dim,)
    cache_seqlens: (batch,), dtype int32.
        If not None, the conv_state is treated as a circular buffer.
        The conv_state will be updated by copying x to the conv_state
        starting at the index
        @cache_seqlens % state_len.
    conv_state_indices: (batch,), dtype int32
        If not None, the conv_state is a larger tensor along the batch dim,
        and we are selecting the batch coords specified by conv_state_indices.
        Useful for a continuous batching scenario.
    pad_slot_id: int
            if cache_indices is passed, lets the kernel identify padded
            entries that will not be processed,
            for example: cache_indices = [pad_slot_id, 1 ,20 ,pad_slot_id]
            in this case, the kernel will not process entries at
            indices 0 and 3
    out: (batch, dim) or (batch, dim, seqlen)
    """
    if validate_data:
        assert cache_seqlens is None
        assert pad_slot_id is not None
        assert x.stride(1) == 1
    if isinstance(activation, bool):
        activation = 'silu' if activation is True else None
    elif activation is not None:
        assert activation in ['silu', 'swish']
    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)
    batch, dim, seqlen = x.shape
    _, width = weight.shape
    num_cache_lines, _, state_len = conv_state.size()
    if validate_data:
        assert dim == weight.size(0)
        assert conv_state.stride(-2) == 1, f'ERROR: expect contiguous along feat-dim of conv_state (currently stride={conv_state.stride()})'
        assert state_len >= width - 1
        assert dim == conv_state.size(1)
        if conv_state_indices is None:
            assert conv_state.size(0) >= batch
        else:
            assert (batch,) == conv_state_indices.shape
        assert num_cache_lines >= batch
        assert weight.stride(1) == 1
        assert cache_seqlens is None
    out = x
    stride_w_dim, stride_w_width = weight.stride()
    stride_x_seq, stride_x_dim, stride_x_token = x.stride()
    stride_o_seq, stride_o_dim, stride_o_token = out.stride()
    stride_istate_seq, stride_istate_dim, stride_istate_token = conv_state.stride()
    state_len = width - 1
    np2_statelen = triton.next_power_of_2(state_len)

    def grid(META):
        return (batch, triton.cdiv(dim, META['BLOCK_N']))
    _causal_conv1d_update_kernel[grid](x, weight, bias, conv_state, cache_seqlens, conv_state_indices, out, batch, dim, seqlen, state_len, num_cache_lines, stride_x_seq, stride_x_dim, stride_x_token, stride_w_dim, stride_w_width, stride_istate_seq, stride_istate_dim, stride_istate_token, stride_o_seq, stride_o_dim, stride_o_token, pad_slot_id, HAS_BIAS=bias is not None, KERNEL_WIDTH=width, SILU_ACTIVATION=activation in ['silu', 'swish'], IS_CONTINUOUS_BATCHING=conv_state_indices is not None, NP2_STATELEN=np2_statelen, USE_PAD_SLOT=pad_slot_id is not None, BLOCK_N=256)
    if unsqueeze:
        out = out.squeeze(-1)
    return out