from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
import torch
BASE_BLOCK = 128 if current_platform.has_device_capability(80) else 64
NUM_WARPS = 4 if current_platform.is_rocm() else 8
IS_TURING = current_platform.get_device_capability() == (7, 5)

@triton.jit
def _fwd_kernel(Q, K, V, K_cache, V_cache, B_Loc, sm_scale, k_scale, v_scale, B_Start_Loc, B_Seqlen, x: tl.constexpr, Out, stride_b_loc_b, stride_b_loc_s, stride_qbs, stride_qh, stride_qd, stride_kbs, stride_kh, stride_kd, stride_vbs, stride_vh, stride_vd, stride_obs, stride_oh, stride_od, stride_k_cache_bs, stride_k_cache_h, stride_k_cache_d, stride_k_cache_bl: tl.constexpr, stride_k_cache_x, stride_v_cache_bs, stride_v_cache_h, stride_v_cache_d, stride_v_cache_bl, num_queries_per_kv: tl.constexpr, IN_PRECISION: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_DMODEL_PADDED: tl.constexpr, BLOCK_SIZE: tl.constexpr, BLOCK_N: tl.constexpr, SLIDING_WINDOW: tl.constexpr, num_unroll_cache: tl.constexpr, num_unroll_request: tl.constexpr, SKIP_DECODE: tl.constexpr, MAX_Q_LEN: tl.constexpr=0, MAX_CTX_LEN: tl.constexpr=0):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)
    cur_kv_head = cur_head // num_queries_per_kv
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
    cur_batch_in_all_stop_index = tl.load(B_Start_Loc + cur_batch + 1)
    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index
    cur_batch_ctx_len = cur_batch_seq_len - cur_batch_query_len
    if SKIP_DECODE and cur_batch_query_len == 1:
        return
    block_start_loc = BLOCK_M * start_m
    offs_bs_n = tl.arange(0, BLOCK_SIZE)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL_PADDED)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_q = (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs + cur_head * stride_qh + offs_d[None, :] * stride_qd
    dim_mask = tl.where(tl.arange(0, BLOCK_DMODEL_PADDED) < BLOCK_DMODEL, 1, 0).to(tl.int1)
    q = tl.load(Q + off_q, mask=dim_mask[None, :] & (offs_m[:, None] < cur_batch_query_len), other=0.0)
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL_PADDED], dtype=tl.float32)
    for start_n in tl.range(0, cur_batch_ctx_len, BLOCK_SIZE, loop_unroll_factor=num_unroll_cache):
        start_n = tl.multiple_of(start_n, BLOCK_SIZE)
        bn = tl.load(B_Loc + cur_batch * stride_b_loc_b + start_n // BLOCK_SIZE * stride_b_loc_s)
        off_k = bn[None, :] * stride_k_cache_bs + cur_kv_head * stride_k_cache_h + offs_d[:, None] // x * stride_k_cache_d + (start_n + offs_bs_n[None, :]) % BLOCK_SIZE * stride_k_cache_bl + offs_d[:, None] % x * stride_k_cache_x
        off_v = bn[:, None] * stride_v_cache_bs + cur_kv_head * stride_v_cache_h + offs_d[None, :] * stride_v_cache_d + offs_bs_n[:, None] * stride_v_cache_bl
        if start_n + BLOCK_SIZE > cur_batch_ctx_len or BLOCK_DMODEL != BLOCK_DMODEL_PADDED:
            k_load = tl.load(K_cache + off_k, mask=dim_mask[:, None] & (start_n + offs_bs_n[None, :] < cur_batch_ctx_len), other=0.0)
        else:
            k_load = tl.load(K_cache + off_k)
        if k_load.dtype.is_fp8():
            k = (k_load.to(tl.float32) * tl.load(k_scale)).to(q.dtype)
        else:
            k = k_load
        qk = tl.zeros([BLOCK_M, BLOCK_SIZE], dtype=tl.float32)
        qk = tl.dot(q, k, acc=qk, input_precision=IN_PRECISION)
        qk = tl.where(start_n + offs_bs_n[None, :] < cur_batch_ctx_len, qk, float('-inf'))
        qk *= sm_scale
        if SLIDING_WINDOW > 0:
            qk = tl.where(cur_batch_ctx_len + offs_m[:, None] - (start_n + offs_bs_n[None, :]) < SLIDING_WINDOW, qk, -10000)
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)
        alpha = tl.exp(m_i - m_ij)
        acc = acc * alpha[:, None]
        if start_n + BLOCK_SIZE > cur_batch_ctx_len or BLOCK_DMODEL != BLOCK_DMODEL_PADDED:
            v_load = tl.load(V_cache + off_v, mask=dim_mask[None, :] & (start_n + offs_bs_n[:, None] < cur_batch_ctx_len), other=0.0)
        else:
            v_load = tl.load(V_cache + off_v)
        if v_load.dtype.is_fp8():
            v = (v_load.to(tl.float32) * tl.load(v_scale)).to(q.dtype)
        else:
            v = v_load
        p = p.to(v.dtype)
        acc = tl.dot(p, v, acc=acc, input_precision=IN_PRECISION)
        l_i = l_i * alpha + l_ij
        m_i = m_ij
    off_k = offs_n[None, :] * stride_kbs + cur_kv_head * stride_kh + offs_d[:, None] * stride_kd
    off_v = offs_n[:, None] * stride_vbs + cur_kv_head * stride_vh + offs_d[None, :] * stride_vd
    k_ptrs = K + off_k
    v_ptrs = V + off_v
    block_mask = tl.where(block_start_loc < cur_batch_query_len, 1, 0)
    for start_n in tl.range(0, block_mask * (start_m + 1) * BLOCK_M, BLOCK_N, loop_unroll_factor=num_unroll_request):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(k_ptrs + (cur_batch_in_all_start_index + start_n) * stride_kbs, mask=dim_mask[:, None] & (start_n + offs_n[None, :] < cur_batch_query_len), other=0.0)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, k, acc=qk, input_precision=IN_PRECISION)
        qk *= sm_scale
        qk = tl.where(offs_m[:, None] >= start_n + offs_n[None, :], qk, float('-inf'))
        if SLIDING_WINDOW > 0:
            qk = tl.where(offs_m[:, None] - (start_n + offs_n[None, :]) < SLIDING_WINDOW, qk, -10000)
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)
        alpha = tl.exp(m_i - m_ij)
        acc = acc * alpha[:, None]
        v = tl.load(v_ptrs + (cur_batch_in_all_start_index + start_n) * stride_vbs, mask=dim_mask[None, :] & (start_n + offs_n[:, None] < cur_batch_query_len), other=0.0)
        p = p.to(v.dtype)
        acc = tl.dot(p, v, acc=acc, input_precision=IN_PRECISION)
        l_i = l_i * alpha + l_ij
        m_i = m_ij
    acc = acc / l_i[:, None]
    off_o = (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs + cur_head * stride_oh + offs_d[None, :] * stride_od
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc, mask=dim_mask[None, :] & (offs_m[:, None] < cur_batch_query_len))
    return

@triton.jit
def _fwd_kernel_flash_attn_v2(Q, K, V, K_cache, V_cache, B_Loc, sm_scale, B_Start_Loc, B_Seqlen, B_Ctxlen, block_size, x, Out, stride_b_loc_b, stride_b_loc_s, stride_qbs, stride_qh, stride_qd, stride_kbs, stride_kh, stride_kd, stride_vbs, stride_vh, stride_vd, stride_obs, stride_oh, stride_od, stride_k_cache_bs, stride_k_cache_h, stride_k_cache_d, stride_k_cache_bl, stride_k_cache_x, stride_v_cache_bs, stride_v_cache_h, stride_v_cache_d, stride_v_cache_bl, num_queries_per_kv: int, BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)
    cur_kv_head = cur_head // num_queries_per_kv
    cur_batch_ctx_len = tl.load(B_Ctxlen + cur_batch)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
    block_start_loc = BLOCK_M * start_m
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_q = (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs + cur_head * stride_qh + offs_d[None, :] * stride_qd
    q = tl.load(Q + off_q, mask=offs_m[:, None] < cur_batch_seq_len - cur_batch_ctx_len, other=0.0)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    for start_n in range(0, cur_batch_ctx_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        bn = tl.load(B_Loc + cur_batch * stride_b_loc_b + (start_n + offs_n) // block_size * stride_b_loc_s, mask=start_n + offs_n < cur_batch_ctx_len, other=0)
        off_k = bn[None, :] * stride_k_cache_bs + cur_kv_head * stride_k_cache_h + offs_d[:, None] // x * stride_k_cache_d + (start_n + offs_n[None, :]) % block_size * stride_k_cache_bl + offs_d[:, None] % x * stride_k_cache_x
        off_v = bn[:, None] * stride_v_cache_bs + cur_kv_head * stride_v_cache_h + offs_d[None, :] * stride_v_cache_d + (start_n + offs_n[:, None]) % block_size * stride_v_cache_bl
        k = tl.load(K_cache + off_k, mask=start_n + offs_n[None, :] < cur_batch_ctx_len, other=0.0)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk = tl.where(start_n + offs_n[None, :] < cur_batch_ctx_len, qk, float('-inf'))
        qk *= sm_scale
        m_ij = tl.max(qk, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        p = tl.math.exp(qk - m_i_new[:, None])
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp(m_i - m_i_new)
        l_i_new = alpha * l_i + l_ij
        acc_scale = alpha
        acc = acc * acc_scale[:, None]
        v = tl.load(V_cache + off_v, mask=start_n + offs_n[:, None] < cur_batch_ctx_len, other=0.0)
        p = p.to(v.dtype)
        acc += tl.dot(p, v)
        l_i = l_i_new
        m_i = m_i_new
    off_k = offs_n[None, :] * stride_kbs + cur_kv_head * stride_kh + offs_d[:, None] * stride_kd
    off_v = offs_n[:, None] * stride_vbs + cur_kv_head * stride_vh + offs_d[None, :] * stride_vd
    k_ptrs = K + off_k
    v_ptrs = V + off_v
    block_mask = tl.where(block_start_loc < cur_batch_seq_len - cur_batch_ctx_len, 1, 0)
    for start_n in range(0, block_mask * (start_m + 1) * BLOCK_M, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(k_ptrs + (cur_batch_in_all_start_index + start_n) * stride_kbs, mask=start_n + offs_n[None, :] < cur_batch_seq_len - cur_batch_ctx_len, other=0.0)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= sm_scale
        qk = tl.where(offs_m[:, None] >= start_n + offs_n[None, :], qk, float('-inf'))
        m_ij = tl.max(qk, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        p = tl.math.exp(qk - m_i_new[:, None])
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp(m_i - m_i_new)
        l_i_new = alpha * l_i + l_ij
        acc_scale = alpha
        acc = acc * acc_scale[:, None]
        v = tl.load(v_ptrs + (cur_batch_in_all_start_index + start_n) * stride_vbs, mask=start_n + offs_n[:, None] < cur_batch_seq_len - cur_batch_ctx_len, other=0.0)
        p = p.to(v.dtype)
        acc += tl.dot(p, v)
        l_i = l_i_new
        m_i = m_i_new
    off_o = (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs + cur_head * stride_oh + offs_d[None, :] * stride_od
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < cur_batch_seq_len - cur_batch_ctx_len)
    return

@triton.jit
def _fwd_kernel_alibi(Q, K, V, K_cache, V_cache, B_Loc, sm_scale, k_scale, v_scale, B_Start_Loc, B_Seqlen, Alibi_slopes, block_size, x, Out, stride_b_loc_b, stride_b_loc_s, stride_qbs, stride_qh, stride_qd, stride_kbs, stride_kh, stride_kd, stride_vbs, stride_vh, stride_vd, stride_obs, stride_oh, stride_od, stride_k_cache_bs, stride_k_cache_h, stride_k_cache_d, stride_k_cache_bl, stride_k_cache_x, stride_v_cache_bs, stride_v_cache_h, stride_v_cache_d, stride_v_cache_bl, num_queries_per_kv: int, IN_PRECISION: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_DMODEL_PADDED: tl.constexpr, BLOCK_N: tl.constexpr, SKIP_DECODE: tl.constexpr):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)
    cur_kv_head = cur_head // num_queries_per_kv
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
    cur_batch_in_all_stop_index = tl.load(B_Start_Loc + cur_batch + 1)
    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index
    cur_batch_ctx_len = cur_batch_seq_len - cur_batch_query_len
    if SKIP_DECODE and cur_batch_query_len == 1:
        return
    block_start_loc = BLOCK_M * start_m
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL_PADDED)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_q = (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs + cur_head * stride_qh + offs_d[None, :] * stride_qd
    dim_mask = tl.where(tl.arange(0, BLOCK_DMODEL_PADDED) < BLOCK_DMODEL, 1, 0).to(tl.int1)
    q = tl.load(Q + off_q, mask=dim_mask[None, :] & (offs_m[:, None] < cur_batch_seq_len - cur_batch_ctx_len), other=0.0)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL_PADDED], dtype=tl.float32)
    alibi_slope = tl.load(Alibi_slopes + cur_head)
    alibi_start_q = tl.arange(0, BLOCK_M) + block_start_loc + cur_batch_ctx_len
    alibi_start_k = 0
    for start_n in range(0, cur_batch_ctx_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        bn = tl.load(B_Loc + cur_batch * stride_b_loc_b + (start_n + offs_n) // block_size * stride_b_loc_s, mask=start_n + offs_n < cur_batch_ctx_len, other=0)
        off_k = bn[None, :] * stride_k_cache_bs + cur_kv_head * stride_k_cache_h + offs_d[:, None] // x * stride_k_cache_d + (start_n + offs_n[None, :]) % block_size * stride_k_cache_bl + offs_d[:, None] % x * stride_k_cache_x
        off_v = bn[:, None] * stride_v_cache_bs + cur_kv_head * stride_v_cache_h + offs_d[None, :] * stride_v_cache_d + (start_n + offs_n[:, None]) % block_size * stride_v_cache_bl
        k_load = tl.load(K_cache + off_k, mask=dim_mask[:, None] & (start_n + offs_n[None, :] < cur_batch_ctx_len), other=0.0)
        if k_load.dtype.is_fp8():
            k = (k_load.to(tl.float32) * tl.load(k_scale)).to(q.dtype)
        else:
            k = k_load
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, k, acc=qk, input_precision=IN_PRECISION)
        qk = tl.where(start_n + offs_n[None, :] < cur_batch_ctx_len, qk, float('-inf'))
        qk *= sm_scale
        alibi = (tl.arange(0, BLOCK_N)[None, :] + alibi_start_k - alibi_start_q[:, None]) * alibi_slope
        alibi = tl.where((alibi <= 0) & (alibi_start_q[:, None] < cur_batch_seq_len), alibi, float('-inf'))
        qk += alibi
        alibi_start_k += BLOCK_N
        m_ij = tl.max(qk, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        p = tl.math.exp(qk - m_i_new[:, None])
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp(m_i - m_i_new)
        l_i_new = alpha * l_i + l_ij
        acc_scale = alpha
        acc = acc * acc_scale[:, None]
        v_load = tl.load(V_cache + off_v, mask=dim_mask[None, :] & (start_n + offs_n[:, None] < cur_batch_ctx_len), other=0.0)
        if v_load.dtype.is_fp8():
            v = (v_load.to(tl.float32) * tl.load(v_scale)).to(q.dtype)
        else:
            v = v_load
        p = p.to(v.dtype)
        acc = tl.dot(p, v, acc=acc, input_precision='ieee')
        l_i = l_i_new
        m_i = m_i_new
    off_k = offs_n[None, :] * stride_kbs + cur_kv_head * stride_kh + offs_d[:, None] * stride_kd
    off_v = offs_n[:, None] * stride_vbs + cur_kv_head * stride_vh + offs_d[None, :] * stride_vd
    k_ptrs = K + off_k
    v_ptrs = V + off_v
    block_mask = tl.where(block_start_loc < cur_batch_seq_len - cur_batch_ctx_len, 1, 0)
    alibi_slope = tl.load(Alibi_slopes + cur_head)
    alibi_start_q = tl.arange(0, BLOCK_M) + block_start_loc + cur_batch_ctx_len
    alibi_start_k = cur_batch_ctx_len
    for start_n in range(0, block_mask * (start_m + 1) * BLOCK_M, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(k_ptrs + (cur_batch_in_all_start_index + start_n) * stride_kbs, mask=dim_mask[:, None] & (start_n + offs_n[None, :] < cur_batch_seq_len - cur_batch_ctx_len), other=0.0)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, k, acc=qk, input_precision='ieee')
        qk *= sm_scale
        qk = tl.where(offs_m[:, None] >= start_n + offs_n[None, :], qk, float('-inf'))
        alibi = (tl.arange(0, BLOCK_N)[None, :] + alibi_start_k - alibi_start_q[:, None]) * alibi_slope
        alibi = tl.where((alibi <= 0) & (alibi_start_q[:, None] < cur_batch_seq_len), alibi, float('-inf'))
        qk += alibi
        alibi_start_k += BLOCK_N
        m_ij = tl.max(qk, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        p = tl.math.exp(qk - m_i_new[:, None])
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp(m_i - m_i_new)
        l_i_new = alpha * l_i + l_ij
        acc_scale = alpha
        acc = acc * acc_scale[:, None]
        v = tl.load(v_ptrs + (cur_batch_in_all_start_index + start_n) * stride_vbs, mask=dim_mask[None, :] & (start_n + offs_n[:, None] < cur_batch_seq_len - cur_batch_ctx_len), other=0.0)
        p = p.to(v.dtype)
        acc = tl.dot(p, v, acc=acc, input_precision='ieee')
        l_i = l_i_new
        m_i = m_i_new
    acc = acc / l_i[:, None]
    off_o = (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs + cur_head * stride_oh + offs_d[None, :] * stride_od
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc, mask=dim_mask[None, :] & (offs_m[:, None] < cur_batch_seq_len - cur_batch_ctx_len))
    return

@torch.inference_mode()
def context_attention_fwd(q, k, v, o, kv_cache_dtype: str, k_cache, v_cache, b_loc, b_start_loc, b_seq_len, max_seq_len, max_input_len, k_scale: torch.Tensor, v_scale: torch.Tensor, alibi_slopes=None, sliding_window=None, sm_scale=None, skip_decode=False):
    q_dtype_is_f32 = q.dtype is torch.float32
    IN_PRECISION = 'ieee' if IS_TURING and q_dtype_is_f32 else None
    if 'fp8' in kv_cache_dtype:
        assert k_cache.dtype in [torch.uint8, current_platform.fp8_dtype()]
        assert v_cache.dtype in [torch.uint8, current_platform.fp8_dtype()]
        if kv_cache_dtype in ('fp8', 'fp8_e4m3'):
            target_dtype = current_platform.fp8_dtype()
        elif kv_cache_dtype == 'fp8_e5m2':
            target_dtype = torch.float8_e5m2
        else:
            raise ValueError('Unsupported FP8 dtype:', kv_cache_dtype)
        k_cache = k_cache.view(target_dtype)
        v_cache = v_cache.view(target_dtype)
    if k_cache.dtype == torch.uint8 or (v_cache.dtype == torch.uint8 and kv_cache_dtype == 'auto'):
        raise ValueError("kv_cache_dtype='auto' unsupported for            FP8 KV Cache prefill kernel")
    Lq, Lk, Lv = (q.shape[-1], k.shape[-1], v.shape[-1])
    assert Lq == Lk and Lk == Lv
    Lk_padded = triton.next_power_of_2(Lk)
    if sm_scale is None:
        sm_scale = 1.0 / Lq ** 0.5
    batch, head = (b_seq_len.shape[0], q.shape[1])
    num_queries_per_kv = q.shape[1] // k.shape[1]
    assert batch + 1 == len(b_start_loc)
    if sliding_window is None or sliding_window <= 0:
        sliding_window = 0
    if alibi_slopes is not None:
        BLOCK = BASE_BLOCK // 2 if q_dtype_is_f32 else BASE_BLOCK
        grid = (batch, head, triton.cdiv(max_input_len, BLOCK))
        _fwd_kernel_alibi[grid](q, k, v, k_cache, v_cache, b_loc, sm_scale, k_scale, v_scale, b_start_loc, b_seq_len, alibi_slopes, v_cache.shape[3], k_cache.shape[4], o, b_loc.stride(0), b_loc.stride(1), q.stride(0), q.stride(1), q.stride(2), k.stride(0), k.stride(1), k.stride(2), v.stride(0), v.stride(1), v.stride(2), o.stride(0), o.stride(1), o.stride(2), k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3), k_cache.stride(4), v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3), num_queries_per_kv=num_queries_per_kv, IN_PRECISION=IN_PRECISION, BLOCK_M=BLOCK, BLOCK_DMODEL=Lk, BLOCK_DMODEL_PADDED=Lk_padded, BLOCK_N=BLOCK, SKIP_DECODE=skip_decode, num_warps=NUM_WARPS, num_stages=1)
        return
    max_seq_len = 0 if max_seq_len is None else max_seq_len
    extra_kargs = {}
    if current_platform.is_rocm():
        extra_kargs = {'kpack': 2, 'waves_per_eu': 2}
    grid = lambda META: (batch, head, triton.cdiv(max_input_len, META['BLOCK_M']))
    _fwd_kernel[grid](q, k, v, k_cache, v_cache, b_loc, sm_scale, k_scale, v_scale, b_start_loc, b_seq_len, k_cache.shape[4], o, b_loc.stride(0), b_loc.stride(1), q.stride(0), q.stride(1), q.stride(2), k.stride(0), k.stride(1), k.stride(2), v.stride(0), v.stride(1), v.stride(2), o.stride(0), o.stride(1), o.stride(2), k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3), k_cache.stride(4), v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3), BLOCK_SIZE=v_cache.shape[3], num_queries_per_kv=num_queries_per_kv, IN_PRECISION=IN_PRECISION, BLOCK_DMODEL=Lk, BLOCK_DMODEL_PADDED=Lk_padded, SLIDING_WINDOW=sliding_window, SKIP_DECODE=skip_decode, BLOCK_M=128, BLOCK_N=64, num_unroll_cache=4, num_unroll_request=1, num_warps=4, num_stages=1, **extra_kargs)
    return