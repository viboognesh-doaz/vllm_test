from typing import Optional
from vllm.triton_utils import tl, triton
import torch

def merge_attn_states(output: torch.Tensor, prefix_output: torch.Tensor, prefix_lse: torch.Tensor, suffix_output: torch.Tensor, suffix_lse: torch.Tensor, output_lse: Optional[torch.Tensor]=None) -> None:
    num_tokens = output.shape[0]
    num_query_heads = output.shape[1]
    head_size = output.shape[2]
    padded_head_size = triton.next_power_of_2(head_size)
    merge_attn_states_kernel[num_tokens, num_query_heads](output, output_lse, prefix_output, prefix_lse, suffix_output, suffix_lse, head_size, padded_head_size, output_lse is not None)

@triton.jit
def merge_attn_states_kernel(output, output_lse, prefix_output, prefix_lse, suffix_output, suffix_lse, HEAD_SIZE: tl.constexpr, PADDED_HEAD_SIZE: tl.constexpr, OUTPUT_LSE: tl.constexpr):
    token_idx = tl.program_id(0)
    num_tokens = tl.num_programs(0)
    head_idx = tl.program_id(1)
    num_heads = tl.num_programs(1)
    p_lse = tl.load(prefix_lse + head_idx * num_tokens + token_idx)
    s_lse = tl.load(suffix_lse + head_idx * num_tokens + token_idx)
    p_lse = float('-inf') if p_lse == float('inf') else p_lse
    s_lse = float('-inf') if s_lse == float('inf') else s_lse
    max_lse = tl.maximum(p_lse, s_lse)
    p_lse = p_lse - max_lse
    s_lse = s_lse - max_lse
    p_se = tl.exp(p_lse)
    s_se = tl.exp(s_lse)
    out_se = p_se + s_se
    if OUTPUT_LSE:
        out_lse = tl.log(out_se) + max_lse
        tl.store(output_lse + head_idx * num_tokens + token_idx, out_lse)
    head_arange = tl.arange(0, PADDED_HEAD_SIZE)
    head_mask = head_arange < HEAD_SIZE
    p_out = tl.load(prefix_output + token_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE + head_arange, mask=head_mask)
    s_out = tl.load(suffix_output + token_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE + head_arange, mask=head_mask)
    p_scale = p_se / out_se
    s_scale = s_se / out_se
    out = p_out * p_scale + s_out * s_scale
    tl.store(output + token_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE + head_arange, out, mask=head_mask)