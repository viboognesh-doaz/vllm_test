from typing import Optional
from vllm._custom_ops import merge_attn_states
from vllm.attention.ops.triton_merge_attn_states import merge_attn_states
from vllm.platforms import current_platform
import torch

def merge_attn_states(output: torch.Tensor, prefix_output: torch.Tensor, prefix_lse: torch.Tensor, suffix_output: torch.Tensor, suffix_lse: torch.Tensor, output_lse: Optional[torch.Tensor]=None) -> None:

    def supported_dtypes(o: torch.Tensor) -> bool:
        return o.dtype in [torch.float32, torch.half, torch.bfloat16]

    def supported_headdim(o: torch.Tensor) -> bool:
        headdim = o.shape[2]
        if o.dtype == torch.float32:
            return headdim % 4 == 0
        return headdim % 8 == 0
    if current_platform.is_cuda() and supported_dtypes(output) and supported_headdim(output):
        return merge_attn_states(output, prefix_output, prefix_lse, suffix_output, suffix_lse, output_lse)
    else:
        return merge_attn_states(output, prefix_output, prefix_lse, suffix_output, suffix_lse, output_lse)