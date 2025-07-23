from vllm.lora.ops.triton_ops.kernel_utils import do_expand_kernel
from vllm.lora.ops.triton_ops.utils import _get_lora_b_ptr
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils import direct_register_custom_op
import torch
'\nBased on:\nChen, L., Ye, Z., Wu, Y., Zhuo, D., Ceze, L., & Krishnamurthy, A. (2023).\nPunica: Multi-Tenant LoRA Serving.\nhttps://arxiv.org/abs/2310.18547\n'

@triton.jit
def _lora_expand_kernel(input_ptr, lora_ptr, out_ptr, M, N, K, token_indices_sorted_by_lora_ids, num_tokens_per_lora, lora_token_start_loc, lora_ids, slice_start_loc, input_d0_stride, input_d1_stride, input_d2_stride, ls_d0_ptr, ls_d1_ptr, ls_d2_ptr, output_d0_stride, output_d1_stride, output_hs_ptr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, EVEN_K: tl.constexpr, ADD_INPUTS: tl.constexpr, CAST_TYPE: tl.constexpr, SLICE_NUM: tl.constexpr, SAME_STRIDE: tl.constexpr):
    cta_n_num = tl.cdiv(N, BLOCK_N)
    cta_m_num = tl.cdiv(M, BLOCK_M)
    pid_mn = tl.program_id(axis=0)
    pid_m = pid_mn % cta_m_num
    pid_n = pid_mn // cta_m_num % cta_n_num
    slice_id = tl.program_id(axis=1)
    lora_idx = tl.program_id(axis=2)
    lora_id = tl.load(lora_ids + lora_idx)
    if lora_id == -1:
        return
    lora_m_size = tl.load(num_tokens_per_lora + lora_idx)
    cta_m_offset = pid_m * BLOCK_M
    if cta_m_offset >= lora_m_size:
        return
    curr_N = N if SAME_STRIDE else tl.load(output_hs_ptr + slice_id)
    if pid_n * BLOCK_N >= curr_N:
        return
    cta_m_len = min(BLOCK_M, lora_m_size - cta_m_offset)
    lora_m_indices_start = tl.load(lora_token_start_loc + lora_idx)
    cta_lora_seq_indices = token_indices_sorted_by_lora_ids + lora_m_indices_start + cta_m_offset
    offset_m = tl.arange(0, BLOCK_M) % cta_m_len
    ram = tl.load(cta_lora_seq_indices + offset_m)
    do_expand_kernel(pid_n, lora_id, slice_id, input_ptr, lora_ptr, out_ptr, curr_N, K, cta_m_len, ram, slice_start_loc, input_d0_stride, input_d1_stride, input_d2_stride, ls_d0_ptr, ls_d1_ptr, ls_d2_ptr, output_d0_stride, output_d1_stride, BLOCK_M, BLOCK_N, BLOCK_K, SAME_STRIDE, SLICE_NUM, EVEN_K, CAST_TYPE, ADD_INPUTS)

@torch.inference_mode()
def _lora_expand(inputs: torch.Tensor, lora_b_weights: list[torch.Tensor], output_tensor: torch.Tensor, token_lora_mapping: torch.Tensor, token_indices_sorted_by_lora_ids: torch.Tensor, num_tokens_per_lora: torch.Tensor, lora_token_start_loc: torch.Tensor, lora_ids: torch.Tensor, no_lora_flag_cpu: torch.Tensor, offset_start: int=0, add_inputs: bool=False) -> None:
    """
    Args:
        inputs (torch.Tensor): input tensor
        lora_b_weights (list[torch.Tensor]): lora'b weight
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
        offset_start (int, optional): Offset start for output_tensor. 
            Defaults to 0.
        add_inputs (bool, optional): Whether to add the input tensor to the 
            output tensor. Defaults to False.
    """
    assert no_lora_flag_cpu.numel() == 1
    if no_lora_flag_cpu.item():
        return
    assert inputs.dtype in [torch.float16, torch.bfloat16, torch.float32]
    for weight in lora_b_weights:
        assert weight.dtype in [torch.float16, torch.bfloat16]
    assert inputs.size(0) == len(lora_b_weights)
    assert output_tensor.is_contiguous()
    M = inputs.size(1)
    assert token_lora_mapping.size(0) == M
    assert token_lora_mapping.size(0) == token_indices_sorted_by_lora_ids.size(0)
    assert lora_ids.size(0) == num_tokens_per_lora.size(0)
    assert lora_token_start_loc.size(0) == lora_ids.size(0) + 1
    slice_start_tensor, lora_ptr_tensor, lora_strides_d0_tensor, lora_strides_d1_tensor, lora_strides_d2_tensor, hidden_sizes_tensor, same_stride, MAX_N = _get_lora_b_ptr(lora_b_weights, offset_start, inputs.device)
    K = lora_b_weights[0].shape[-1]
    ADD_INPUTS = add_inputs
    MAX_LORAS = lora_ids.size(0)
    CAST_TYPE = False
    NUM_SLICES = len(lora_b_weights)
    BLOCK_M = 64
    BLOCK_N = 128
    BLOCK_K = 16
    NUM_WARPS = 4
    NUM_CTAS = 1
    NUM_STAGES = 2
    EVEN_K = K % BLOCK_K == 0
    if inputs.dtype == torch.float32 and lora_b_weights[0].dtype in [torch.float16, torch.bfloat16]:
        CAST_TYPE = True
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(MAX_N, BLOCK_N), NUM_SLICES, MAX_LORAS)
    _lora_expand_kernel[grid](inputs, lora_ptr_tensor, output_tensor, M, MAX_N, K, token_indices_sorted_by_lora_ids, num_tokens_per_lora, lora_token_start_loc, lora_ids, slice_start_tensor, inputs.stride(0), inputs.stride(1), inputs.stride(2), lora_strides_d0_tensor, lora_strides_d1_tensor, lora_strides_d2_tensor, output_tensor.stride(0), output_tensor.stride(1), hidden_sizes_tensor, BLOCK_M, BLOCK_N, BLOCK_K, EVEN_K, ADD_INPUTS, CAST_TYPE, NUM_SLICES, same_stride, num_warps=NUM_WARPS, num_ctas=NUM_CTAS, num_stages=NUM_STAGES)
    return

def _lora_expand_fake(inputs: torch.Tensor, lora_b_weights: list[torch.Tensor], output_tensor: torch.Tensor, token_lora_mapping: torch.Tensor, token_indices_sorted_by_lora_ids: torch.Tensor, num_tokens_per_lora: torch.Tensor, lora_token_start_loc: torch.Tensor, lora_ids: torch.Tensor, no_lora_flag_cpu: torch.Tensor, offset_start: int=0, add_inputs: bool=False) -> None:
    return
try:
    direct_register_custom_op(op_name='lora_expand', op_func=_lora_expand, mutates_args=['output_tensor'], fake_impl=_lora_expand_fake, dispatch_key=current_platform.dispatch_key)
    lora_expand = torch.ops.vllm.lora_expand
except AttributeError:
    lora_expand = _lora_expand