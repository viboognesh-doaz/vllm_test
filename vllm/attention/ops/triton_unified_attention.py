from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
import torch
logger = init_logger(__name__)

@triton.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y

@triton.jit
def apply_softcap(S, x):
    Sdiv = S / x
    p1 = tl.exp(Sdiv)
    p2 = tl.exp(-Sdiv)
    return x * (p1 - p2) / (p1 + p2)

@triton.jit
def find_seq_idx(query_start_len_ptr, target_idx, num_seqs, BLOCK_Q: tl.constexpr, use_q_block_mode: tl.constexpr):
    left: tl.int32 = 0
    right = num_seqs
    while left < right:
        mid = (left + right) // 2
        val = tl.load(query_start_len_ptr + mid)
        mid_val = val // BLOCK_Q + mid if use_q_block_mode else val
        if mid_val <= target_idx:
            left = mid + 1
        else:
            right = mid
    return left - 1

@triton.jit
def kernel_unified_attention_2d(output_ptr, query_ptr, key_cache_ptr, value_cache_ptr, block_tables_ptr, seq_lens_ptr, alibi_slopes_ptr, scale, k_scale, v_scale, softcap, num_query_heads: tl.constexpr, num_queries_per_kv: tl.constexpr, block_table_stride: tl.int64, query_stride_0: tl.int64, query_stride_1: tl.int64, output_stride_0: tl.int64, output_stride_1: tl.int64, BLOCK_SIZE: tl.constexpr, HEAD_SIZE: tl.constexpr, HEAD_SIZE_PADDED: tl.constexpr, USE_ALIBI_SLOPES: tl.constexpr, USE_SOFTCAP: tl.constexpr, SLIDING_WINDOW: tl.constexpr, stride_k_cache_0: tl.int64, stride_k_cache_1: tl.int64, stride_k_cache_2: tl.int64, stride_k_cache_3: tl.constexpr, stride_v_cache_0: tl.int64, stride_v_cache_1: tl.int64, stride_v_cache_2: tl.int64, stride_v_cache_3: tl.constexpr, query_start_len_ptr, BLOCK_Q: tl.constexpr, num_seqs: tl.int32, BLOCK_M: tl.constexpr):
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    seq_idx = find_seq_idx(query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True)
    q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx
    q_block_local_idx = q_block_global_idx - q_block_start_idx
    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)
    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index
    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv
    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    query_offset = query_offset_0[:, None] * query_stride_0 + query_offset_1[:, None] * query_stride_1 + offs_d[None, :]
    dim_mask = tl.where(offs_d < HEAD_SIZE, 1, 0).to(tl.int1)
    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)
    Q = tl.load(query_ptr + query_offset, mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None], other=0.0)
    block_table_offset = seq_idx * block_table_stride
    M = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)
    seq_len = tl.load(seq_lens_ptr + seq_idx)
    context_len = seq_len - cur_batch_query_len
    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(alibi_slopes_ptr + query_offset_1, mask=query_mask_1, other=0.0)
    max_seq_prefix_len = context_len + q_block_local_idx * BLOCK_Q + (BLOCK_M - 1) // num_queries_per_kv + 1
    max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)
    num_blocks = cdiv_fn(max_seq_prefix_len, BLOCK_SIZE)
    for j in range(0, num_blocks):
        physical_block_idx = tl.load(block_tables_ptr + block_table_offset + j)
        offs_n = tl.arange(0, BLOCK_SIZE)
        v_offset = physical_block_idx * stride_v_cache_0 + kv_head_idx * stride_v_cache_2 + offs_d[None, :] * stride_v_cache_3 + offs_n[:, None] * stride_v_cache_1
        k_offset = physical_block_idx * stride_k_cache_0 + kv_head_idx * stride_k_cache_2 + offs_d[:, None] * stride_k_cache_3 + offs_n[None, :] * stride_k_cache_1
        K_load = tl.load(key_cache_ptr + k_offset, mask=dim_mask[:, None], other=0.0)
        if K_load.dtype.is_fp8():
            if Q.dtype.is_fp8():
                K = K_load
            else:
                K = (K_load.to(tl.float32) * tl.load(k_scale)).to(Q.dtype)
        else:
            K = K_load
        V_load = tl.load(value_cache_ptr + v_offset, mask=dim_mask[None, :], other=0.0)
        if V_load.dtype.is_fp8():
            if Q.dtype.is_fp8():
                V = V_load
            else:
                V = (V_load.to(tl.float32) * tl.load(v_scale)).to(Q.dtype)
        else:
            V = V_load
        seq_offset = j * BLOCK_SIZE + offs_n
        seq_mask = seq_offset[None, :] < context_len + query_pos[:, None] + 1
        S = tl.zeros(shape=(BLOCK_M, BLOCK_SIZE), dtype=tl.float32)
        S += scale * tl.dot(Q, K)
        if USE_SOFTCAP:
            S = apply_softcap(S, softcap)
        S = tl.where(query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, S, float('-inf'))
        if SLIDING_WINDOW > 0:
            S = tl.where(context_len + query_pos[:, None] - seq_offset < SLIDING_WINDOW, S, float('-inf'))
        if USE_ALIBI_SLOPES:
            S += alibi_slope[:, None] * (seq_offset - context_len)
        m_j = tl.maximum(M, tl.max(S, axis=1))
        m_j = tl.where(m_j > float('-inf'), m_j, 0.0)
        P = tl.exp(S - m_j[:, None])
        l_j = tl.sum(P, axis=1)
        alpha = tl.exp(M - m_j)
        acc = acc * alpha[:, None]
        L = L * alpha + l_j
        M = m_j
        acc += tl.dot(P.to(V.dtype), V)
    acc = acc / L[:, None]
    output_offset = query_offset_0[:, None] * output_stride_0 + query_offset_1[:, None] * output_stride_1 + offs_d[None, :]
    tl.store(output_ptr + output_offset, acc, mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None])

@triton.jit
def kernel_unified_attention_3d(segm_output_ptr, segm_max_ptr, segm_expsum_ptr, query_ptr, key_cache_ptr, value_cache_ptr, block_tables_ptr, seq_lens_ptr, alibi_slopes_ptr, scale, k_scale, v_scale, softcap, num_query_heads: tl.constexpr, num_queries_per_kv: tl.constexpr, block_table_stride: tl.int64, query_stride_0: tl.int64, query_stride_1: tl.int64, BLOCK_SIZE: tl.constexpr, HEAD_SIZE: tl.constexpr, HEAD_SIZE_PADDED: tl.constexpr, USE_ALIBI_SLOPES: tl.constexpr, USE_SOFTCAP: tl.constexpr, SLIDING_WINDOW: tl.constexpr, stride_k_cache_0: tl.int64, stride_k_cache_1: tl.int64, stride_k_cache_2: tl.int64, stride_k_cache_3: tl.constexpr, stride_v_cache_0: tl.int64, stride_v_cache_1: tl.int64, stride_v_cache_2: tl.int64, stride_v_cache_3: tl.constexpr, query_start_len_ptr, BLOCK_Q: tl.constexpr, num_seqs: tl.int32, BLOCK_M: tl.constexpr, NUM_SEGMENTS_PER_SEQ: tl.constexpr):
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    segm_idx = tl.program_id(2)
    seq_idx = find_seq_idx(query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True)
    q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx
    q_block_local_idx = q_block_global_idx - q_block_start_idx
    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)
    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index
    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return
    seq_len = tl.load(seq_lens_ptr + seq_idx)
    num_segments = NUM_SEGMENTS_PER_SEQ
    blocks_per_segment = cdiv_fn(seq_len, num_segments * BLOCK_SIZE)
    if segm_idx * blocks_per_segment * BLOCK_SIZE >= seq_len:
        return
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv
    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    query_offset = query_offset_0[:, None] * query_stride_0 + query_offset_1[:, None] * query_stride_1 + offs_d[None, :]
    dim_mask = tl.where(offs_d < HEAD_SIZE, 1, 0).to(tl.int1)
    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)
    Q = tl.load(query_ptr + query_offset, mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None], other=0.0)
    block_table_offset = seq_idx * block_table_stride
    M = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)
    context_len = seq_len - cur_batch_query_len
    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(alibi_slopes_ptr + query_offset_1, mask=query_mask_1, other=0.0)
    num_blocks = cdiv_fn(seq_len, BLOCK_SIZE)
    for j in range(segm_idx * blocks_per_segment, min((segm_idx + 1) * blocks_per_segment, num_blocks)):
        physical_block_idx = tl.load(block_tables_ptr + block_table_offset + j)
        offs_n = tl.arange(0, BLOCK_SIZE)
        v_offset = physical_block_idx * stride_v_cache_0 + kv_head_idx * stride_v_cache_2 + offs_d[None, :] * stride_v_cache_3 + offs_n[:, None] * stride_v_cache_1
        k_offset = physical_block_idx * stride_k_cache_0 + kv_head_idx * stride_k_cache_2 + offs_d[:, None] * stride_k_cache_3 + offs_n[None, :] * stride_k_cache_1
        K_load = tl.load(key_cache_ptr + k_offset, mask=dim_mask[:, None], other=0.0)
        if K_load.dtype.is_fp8():
            if Q.dtype.is_fp8():
                K = K_load
            else:
                K = (K_load.to(tl.float32) * tl.load(k_scale)).to(Q.dtype)
        else:
            K = K_load
        V_load = tl.load(value_cache_ptr + v_offset, mask=dim_mask[None, :], other=0.0)
        if V_load.dtype.is_fp8():
            if Q.dtype.is_fp8():
                V = V_load
            else:
                V = (V_load.to(tl.float32) * tl.load(v_scale)).to(Q.dtype)
        else:
            V = V_load
        seq_offset = j * BLOCK_SIZE + offs_n
        seq_mask = seq_offset[None, :] < context_len + query_pos[:, None] + 1
        S = tl.zeros(shape=(BLOCK_M, BLOCK_SIZE), dtype=tl.float32)
        S += scale * tl.dot(Q, K)
        if USE_SOFTCAP:
            S = apply_softcap(S, softcap)
        S = tl.where(query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, S, float('-inf'))
        if SLIDING_WINDOW > 0:
            S = tl.where(context_len + query_pos[:, None] - seq_offset < SLIDING_WINDOW, S, float('-inf'))
        if USE_ALIBI_SLOPES:
            S += alibi_slope[:, None] * (seq_offset - context_len)
        m_j = tl.maximum(M, tl.max(S, axis=1))
        m_j = tl.where(m_j > float('-inf'), m_j, 0.0)
        P = tl.exp(S - m_j[:, None])
        l_j = tl.sum(P, axis=1)
        alpha = tl.exp(M - m_j)
        acc = acc * alpha[:, None]
        L = L * alpha + l_j
        M = m_j
        acc += tl.dot(P.to(V.dtype), V)
    segm_output_offset = query_offset_0[:, None].to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED) + query_offset_1[:, None] * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED) + segm_idx * HEAD_SIZE_PADDED + tl.arange(0, HEAD_SIZE_PADDED)[None, :]
    tl.store(segm_output_ptr + segm_output_offset, acc, mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None])
    segm_offset = query_offset_0.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ) + query_offset_1 * NUM_SEGMENTS_PER_SEQ + segm_idx
    tl.store(segm_max_ptr + segm_offset, M, mask=query_mask_0 & query_mask_1)
    tl.store(segm_expsum_ptr + segm_offset, L, mask=query_mask_0 & query_mask_1)

@triton.jit
def reduce_segments(output_ptr, segm_output_ptr, segm_max_ptr, segm_expsum_ptr, seq_lens_ptr, num_seqs, num_query_heads: tl.constexpr, output_stride_0: tl.int64, output_stride_1: tl.int64, block_table_stride: tl.int64, BLOCK_SIZE: tl.constexpr, HEAD_SIZE: tl.constexpr, HEAD_SIZE_PADDED: tl.constexpr, query_start_len_ptr, BLOCK_Q: tl.constexpr, NUM_SEGMENTS_PER_SEQ: tl.constexpr):
    query_token_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)
    seq_idx = find_seq_idx(query_start_len_ptr, query_token_idx, num_seqs, BLOCK_Q, False)
    seq_len = tl.load(seq_lens_ptr + seq_idx)
    num_segments = NUM_SEGMENTS_PER_SEQ
    blocks_per_segment = cdiv_fn(seq_len, num_segments * BLOCK_SIZE)
    act_num_segments = cdiv_fn(seq_len, blocks_per_segment * BLOCK_SIZE)
    segm_mask = tl.arange(0, NUM_SEGMENTS_PER_SEQ) < tl.full([NUM_SEGMENTS_PER_SEQ], act_num_segments, dtype=tl.int32)
    dim_mask = tl.where(tl.arange(0, HEAD_SIZE_PADDED) < HEAD_SIZE, 1, 0).to(tl.int1)
    segm_offset = query_token_idx.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ) + query_head_idx * NUM_SEGMENTS_PER_SEQ + tl.arange(0, NUM_SEGMENTS_PER_SEQ)
    segm_max = tl.load(segm_max_ptr + segm_offset, mask=segm_mask, other=float('-inf'))
    overall_max = tl.max(segm_max)
    segm_expsum = tl.load(segm_expsum_ptr + segm_offset, mask=segm_mask, other=0.0)
    segm_expsum = segm_expsum * tl.exp(segm_max - overall_max)
    overall_expsum = tl.sum(segm_expsum)
    segm_output_offset = query_token_idx.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED) + query_head_idx * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED) + tl.arange(0, NUM_SEGMENTS_PER_SEQ)[:, None] * HEAD_SIZE_PADDED + tl.arange(0, HEAD_SIZE_PADDED)[None, :]
    segm_output = tl.load(segm_output_ptr + segm_output_offset, mask=segm_mask[:, None] & dim_mask[None, :], other=0.0)
    segm_output *= tl.exp(segm_max - overall_max)[:, None]
    acc_sum = tl.sum(segm_output, axis=0)
    acc = tl.where(overall_expsum == 0.0, 0.0, acc_sum / overall_expsum)
    output_offset = query_token_idx * output_stride_0 + query_head_idx * output_stride_1 + tl.arange(0, HEAD_SIZE_PADDED)
    tl.store(output_ptr + output_offset, acc, mask=dim_mask)

def unified_attention(q, k, v, out, cu_seqlens_q, max_seqlen_q, seqused_k, max_seqlen_k, softmax_scale, causal, window_size, block_table, softcap, q_descale, k_descale, v_descale, alibi_slopes=None):
    assert causal, 'Only causal attention is supported'
    assert q_descale is None, 'Q scales not supported'
    block_size = v.shape[1]
    assert q.element_size() >= 2 or block_size >= 32, 'Block size must be at least 32 for fp8'
    use_alibi_slopes = alibi_slopes is not None
    block_size = v.shape[1]
    num_seqs = len(seqused_k)
    num_query_heads = q.shape[1]
    num_kv_heads = k.shape[2]
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_size = q.shape[2]
    BLOCK_M = 16
    BLOCK_Q = BLOCK_M // num_queries_per_kv
    total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs
    if max_seqlen_q > 1 or total_num_q_blocks * num_kv_heads > 128:
        kernel_unified_attention_2d[total_num_q_blocks, num_kv_heads](output_ptr=out, query_ptr=q, key_cache_ptr=k, value_cache_ptr=v, block_tables_ptr=block_table, seq_lens_ptr=seqused_k, alibi_slopes_ptr=alibi_slopes, scale=softmax_scale, k_scale=k_descale, v_scale=v_descale, softcap=softcap, num_query_heads=num_query_heads, num_queries_per_kv=num_queries_per_kv, block_table_stride=block_table.stride(0), query_stride_0=q.stride(0), query_stride_1=q.stride(1), output_stride_0=out.stride(0), output_stride_1=out.stride(1), BLOCK_SIZE=block_size, HEAD_SIZE=head_size, HEAD_SIZE_PADDED=triton.next_power_of_2(head_size), USE_ALIBI_SLOPES=use_alibi_slopes, USE_SOFTCAP=softcap > 0, SLIDING_WINDOW=1 + window_size[0], stride_k_cache_0=k.stride(0), stride_k_cache_1=k.stride(1), stride_k_cache_2=k.stride(2), stride_k_cache_3=k.stride(3), stride_v_cache_0=v.stride(0), stride_v_cache_1=v.stride(1), stride_v_cache_2=v.stride(2), stride_v_cache_3=v.stride(3), query_start_len_ptr=cu_seqlens_q, BLOCK_Q=BLOCK_Q, num_seqs=num_seqs, BLOCK_M=BLOCK_M)
    else:
        NUM_SEGMENTS = 16
        segm_output = torch.empty(q.shape[0], num_query_heads, NUM_SEGMENTS, triton.next_power_of_2(head_size), dtype=torch.float32, device=q.device)
        segm_max = torch.empty(q.shape[0], num_query_heads, NUM_SEGMENTS, dtype=torch.float32, device=q.device)
        segm_expsum = torch.empty(q.shape[0], num_query_heads, NUM_SEGMENTS, dtype=torch.float32, device=q.device)
        kernel_unified_attention_3d[total_num_q_blocks, num_kv_heads, NUM_SEGMENTS](segm_output_ptr=segm_output, segm_max_ptr=segm_max, segm_expsum_ptr=segm_expsum, query_ptr=q, key_cache_ptr=k, value_cache_ptr=v, block_tables_ptr=block_table, seq_lens_ptr=seqused_k, alibi_slopes_ptr=alibi_slopes, scale=softmax_scale, k_scale=k_descale, v_scale=v_descale, softcap=softcap, num_query_heads=num_query_heads, num_queries_per_kv=num_queries_per_kv, block_table_stride=block_table.stride(0), query_stride_0=q.stride(0), query_stride_1=q.stride(1), BLOCK_SIZE=block_size, HEAD_SIZE=head_size, HEAD_SIZE_PADDED=triton.next_power_of_2(head_size), USE_ALIBI_SLOPES=use_alibi_slopes, USE_SOFTCAP=softcap > 0, SLIDING_WINDOW=1 + window_size[0], stride_k_cache_0=k.stride(0), stride_k_cache_1=k.stride(1), stride_k_cache_2=k.stride(2), stride_k_cache_3=k.stride(3), stride_v_cache_0=v.stride(0), stride_v_cache_1=v.stride(1), stride_v_cache_2=v.stride(2), stride_v_cache_3=v.stride(3), query_start_len_ptr=cu_seqlens_q, BLOCK_Q=BLOCK_Q, num_seqs=num_seqs, BLOCK_M=BLOCK_M, NUM_SEGMENTS_PER_SEQ=NUM_SEGMENTS)
        reduce_segments[q.shape[0], num_query_heads](output_ptr=out, segm_output_ptr=segm_output, segm_max_ptr=segm_max, segm_expsum_ptr=segm_expsum, seq_lens_ptr=seqused_k, num_seqs=num_seqs, num_query_heads=num_query_heads, output_stride_0=out.stride(0), output_stride_1=out.stride(1), block_table_stride=block_table.stride(0), BLOCK_SIZE=block_size, HEAD_SIZE=head_size, HEAD_SIZE_PADDED=triton.next_power_of_2(head_size), query_start_len_ptr=cu_seqlens_q, BLOCK_Q=BLOCK_Q, NUM_SEGMENTS_PER_SEQ=NUM_SEGMENTS)