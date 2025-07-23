from vllm.lora.ops.triton_ops.kernel_utils import do_shrink_kernel
from vllm.lora.ops.triton_ops.utils import _get_lora_a_ptr
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils import direct_register_custom_op
import torch
'\nBased on:\nChen, L., Ye, Z., Wu, Y., Zhuo, D., Ceze, L., & Krishnamurthy, A. (2023). \nPunica: Multi-Tenant LoRA Serving. \nhttps://arxiv.org/abs/2310.18547\n'

@triton.jit
def _lora_shrink_kernel(input_ptr, lora_ptr, out_ptr, M, N, K, token_indices_sorted_by_lora_ids, num_tokens_per_lora, lora_token_start_loc, lora_ids, scaling, input_d0_stride, input_d1_stride, lora_d0_stride, lora_d1_stride, lora_d2_stride, output_d0_stride, output_d1_stride, output_d2_stride, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, EVEN_K: tl.constexpr, SPLIT_K: tl.constexpr, SLICE_NUM: tl.constexpr):
    cta_n_num = tl.cdiv(N, BLOCK_N)
    cta_m_num = tl.cdiv(M, BLOCK_M)
    pid_sk_m_n = tl.program_id(axis=0)
    pid_sk = pid_sk_m_n % SPLIT_K
    pid_m = pid_sk_m_n // SPLIT_K % cta_m_num
    pid_n = pid_sk_m_n // (SPLIT_K * cta_m_num) % cta_n_num
    slice_id = tl.program_id(axis=1)
    lora_idx = tl.program_id(axis=2)
    lora_id = tl.load(lora_ids + lora_idx)
    if lora_id == -1:
        return
    lora_m_size = tl.load(num_tokens_per_lora + lora_idx)
    cta_m_offset = pid_m * BLOCK_M
    if cta_m_offset >= lora_m_size:
        return
    cta_m_len = min(BLOCK_M, lora_m_size - cta_m_offset)
    lora_m_indices_start = tl.load(lora_token_start_loc + lora_idx)
    cta_lora_seq_indices = token_indices_sorted_by_lora_ids + lora_m_indices_start + cta_m_offset
    offset_m = tl.arange(0, BLOCK_M) % cta_m_len
    ram = tl.load(cta_lora_seq_indices + offset_m)
    do_shrink_kernel(pid_n, pid_sk, slice_id, lora_id, input_ptr, lora_ptr, out_ptr, N, K, cta_m_len, ram, input_d0_stride, input_d1_stride, lora_d0_stride, lora_d1_stride, lora_d2_stride, output_d0_stride, output_d1_stride, output_d2_stride, scaling, BLOCK_M, BLOCK_N, BLOCK_K, EVEN_K, SPLIT_K, SLICE_NUM)

@torch.inference_mode()
def _lora_shrink(inputs: torch.Tensor, lora_a_weights: list[torch.Tensor], output_tensor: torch.Tensor, token_lora_mapping: torch.Tensor, token_indices_sorted_by_lora_ids: torch.Tensor, num_tokens_per_lora: torch.Tensor, lora_token_start_loc: torch.Tensor, lora_ids: torch.Tensor, no_lora_flag_cpu: torch.Tensor, scaling: float) -> None:
    """
    Args:
        inputs (torch.Tensor): Input tensor
        lora_a_weights (list[torch.Tensor]): LoRA weights
        output_tensor (torch.Tensor): output tensor
        token_lora_mapping (torch.Tensor): A tensor mapping each input token
            to the lora-id related to that token. A value of -1 indicates that
            LoRA doesn't apply to that token.
        token_indices_sorted_by_lora_ids (torch.Tensor): Row/Token indices from
            the A matrix grouped by LoRA IDs.
        num_tokens_per_lora (torch.Tensor): num_tokens_per_lora[i] is the number
            of tokens that are to be processed by LoRA ID lora_ids[i] 
        lora_token_start_loc (torch.Tensor): A cumulative sum of
            num_tokens_per_lora. lora_token_start_loc[0] is always 0 so that
            lora_token_start_loc[i], along with num_tokens_per_lora[i]
            identifies the region in token_indices_sorted_by_lora_ids that
            LoRA lora_ids[i] should process.
        lora_ids (torch.Tensor): LoRA ids to process.
        no_lora_flag_cpu (torch.Tensor): A CPU tensor of size 1, that indicates
            if there are any requests that require LoRA.
        scaling (float): Scaling factor.
    """
    assert no_lora_flag_cpu.numel() == 1
    if no_lora_flag_cpu.item():
        return
    assert inputs.dtype == lora_a_weights[0].dtype
    assert inputs.dtype in [torch.float16, torch.bfloat16]
    for weight in lora_a_weights:
        assert weight.dtype in [torch.float16, torch.bfloat16]
    assert inputs.size(1) == lora_a_weights[0].size(-1)
    assert inputs.is_contiguous()
    assert output_tensor.is_contiguous()
    M = inputs.size(0)
    assert token_lora_mapping.size(0) == M
    assert token_lora_mapping.size(0) == token_indices_sorted_by_lora_ids.size(0)
    assert lora_ids.size(0) == num_tokens_per_lora.size(0)
    assert lora_token_start_loc.size(0) == lora_ids.size(0) + 1
    lora_ptr_tensor, lora_strides_d0, lora_strides_d1, lora_strides_d2 = _get_lora_a_ptr(lora_a_weights, inputs.device)
    N, K = lora_a_weights[0].shape[-2:]
    NUM_SLICES = len(lora_a_weights)
    MAX_LORAS = lora_ids.size(0)
    BLOCK_M = 32
    BLOCK_N = 16
    BLOCK_K = 256 if M < 128 else 32
    SPLIT_K = 64 if M < 128 else 8
    NUM_WARPS = 4
    NUM_CTAS = 1
    NUM_STAGES = 2
    EVEN_K = K % (BLOCK_K * SPLIT_K) == 0
    grid = (SPLIT_K * triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), NUM_SLICES, MAX_LORAS)
    _lora_shrink_kernel[grid](inputs, lora_ptr_tensor, output_tensor, M, N, K, token_indices_sorted_by_lora_ids, num_tokens_per_lora, lora_token_start_loc, lora_ids, scaling, inputs.stride(0), inputs.stride(1), lora_strides_d0, lora_strides_d1, lora_strides_d2, output_tensor.stride(0), output_tensor.stride(1), output_tensor.stride(2), BLOCK_M, BLOCK_N, BLOCK_K, EVEN_K, SPLIT_K, NUM_SLICES, num_warps=NUM_WARPS, num_ctas=NUM_CTAS, num_stages=NUM_STAGES)
    return

def _lora_shrink_fake(inputs: torch.Tensor, lora_a_weights: list[torch.Tensor], output_tensor: torch.Tensor, token_lora_mapping: torch.Tensor, token_indices_sorted_by_lora_ids: torch.Tensor, num_tokens_per_lora: torch.Tensor, lora_token_start_loc: torch.Tensor, lora_ids: torch.Tensor, no_lora_flag_cpu: torch.Tensor, scaling: float) -> None:
    return
try:
    direct_register_custom_op(op_name='lora_shrink', op_func=_lora_shrink, mutates_args=['output_tensor'], fake_impl=_lora_shrink_fake, dispatch_key=current_platform.dispatch_key)
    lora_shrink = torch.ops.vllm.lora_shrink
except AttributeError:
    lora_shrink = _lora_shrink