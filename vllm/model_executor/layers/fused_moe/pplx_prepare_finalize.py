from typing import Any, Optional
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import TopKWeightAndReduceDelegate
from vllm.model_executor.layers.fused_moe.utils import _validate_scale_shape, moe_kernel_quantize_input
from vllm.utils import cdiv, round_up
import pplx_kernels as pplx
import torch
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
logger = init_logger(__name__)

def pplx_hidden_dim_scale_bytes(max_num_tokens: int, hidden_dim: int, in_dtype: torch.dtype, quant_dtype: Optional[torch.dtype], per_act_token_quant: bool, block_shape: Optional[list[int]]):
    align = 16
    if quant_dtype is not None:
        assert quant_dtype.itemsize == 1
        hidden_dim_bytes = hidden_dim * quant_dtype.itemsize
        elem_size = torch.float32.itemsize
        if per_act_token_quant:
            assert block_shape is None
            hidden_scale_bytes = elem_size
        elif block_shape is not None:
            block_size = block_shape[1]
            num_blocks = cdiv(hidden_dim, block_size)
            hidden_scale_bytes = num_blocks * elem_size
        else:
            hidden_scale_bytes = elem_size
    else:
        hidden_dim_bytes = hidden_dim * in_dtype.itemsize
        hidden_scale_bytes = 0
    return (round_up(hidden_dim_bytes, align), round_up(hidden_scale_bytes, align))

class PplxPrepareAndFinalize(mk.FusedMoEPrepareAndFinalize):

    def __init__(self, a2a: pplx.AllToAll, max_num_tokens: int, num_local_experts: int, num_dispatchers: int):
        super().__init__()
        assert max_num_tokens > 0
        assert num_local_experts > 0
        self.a2a = a2a
        self.max_num_tokens = max_num_tokens
        self.num_local_experts = num_local_experts
        self.num_dispatchers_ = num_dispatchers

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.BatchedExperts

    def max_num_tokens_per_rank(self) -> Optional[int]:
        return self.max_num_tokens

    def topk_indices_dtype(self) -> Optional[torch.dtype]:
        return torch.int32

    def num_dispatchers(self) -> int:
        return self.num_dispatchers_

    def prepare(self, a1: torch.Tensor, a1_scale: Optional[torch.Tensor], a2_scale: Optional[torch.Tensor], topk_weights: torch.Tensor, topk_ids: torch.Tensor, num_experts: int, expert_map: Optional[torch.Tensor], apply_router_weight_on_input: bool, quant_config: FusedMoEQuantConfig, extra_prepare_args: Optional[dict[str, Any]]) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[mk.ExpertTokensMetadata], Optional[torch.Tensor], Optional[torch.Tensor]]:
        num_tokens = a1.size(0)
        hidden_dim = a1.size(-1)
        assert topk_ids.size(0) == num_tokens
        if expert_map is not None:
            logger.warning_once('The PPLX backend does not support expert mapping. The provided `expert_map` will be ignored.')
        expert_map = None
        device = a1.device
        if apply_router_weight_on_input:
            topk = topk_ids.size(1)
            assert topk == 1, 'apply_router_weight_on_input is only implemented for topk=1'
            a1 = a1 * topk_weights.to(a1.dtype)
        repeat_cols = 4
        repeat_rows = 1 if quant_config.per_act_token_quant else a1.size(0)
        a1q, a1q_scale = moe_kernel_quantize_input(a1, None if quant_config.per_act_token_quant else a1_scale, quant_dtype=quant_config.quant_dtype, per_act_token_quant=quant_config.per_act_token_quant, block_shape=quant_config.block_shape)
        _validate_scale_shape(a1q, a1q_scale, quant_config.per_act_token_quant, quant_config.block_shape)
        if a1q_scale is not None:
            scalar_scales = a1q_scale.numel() == 1
            if a1q_scale.dim() <= 1:
                assert scalar_scales
                a1q_scale = a1q_scale.view(1, 1)
            orig_a_scale_block_shape = a1q_scale.shape[-1]
            if not quant_config.is_block_quantized:
                a1q_scale = a1q_scale.repeat(repeat_rows, repeat_cols)
        assert a1q_scale is None or a1q_scale.ndim == 2, f'{(0 if a1q_scale is None else (a1q_scale.ndim, a1q_scale.shape))}'
        expert_num_tokens = torch.empty(self.num_local_experts, dtype=torch.int32, device=device)
        expert_x = torch.empty((self.num_local_experts, self.max_num_tokens * self.num_dispatchers(), hidden_dim), dtype=a1q.dtype, device=device)
        expert_x_scale: Optional[torch.Tensor] = None
        if a1q.dtype.itemsize == 1:
            if quant_config.is_per_act_token:
                final_dim = expert_x.size(2)
            elif quant_config.is_per_tensor:
                final_dim = 1
            else:
                assert quant_config.block_shape is not None
                num_blocks = cdiv(expert_x.size(2), quant_config.block_shape[1])
                final_dim = num_blocks
            expert_x_scale_shape = (self.num_local_experts, expert_x.size(1), round_up(final_dim, 4))
            expert_x_scale = torch.empty(expert_x_scale_shape, dtype=torch.float32, device=expert_x.device)
        bound_m: Optional[torch.Tensor] = None
        self.a2a.dispatch(out_expert_num_tokens=expert_num_tokens, out_expert_x=expert_x, out_expert_x_scale=expert_x_scale, dp_x=a1q, dp_x_scale=a1q_scale, indices=topk_ids.view(dtype=torch.uint32), bound_m=bound_m)
        if expert_x_scale is not None:
            expert_x_scale = expert_x_scale[:, :, :orig_a_scale_block_shape]
            assert expert_x_scale.ndim == 3
        expert_tokens_meta = mk.ExpertTokensMetadata(expert_num_tokens=expert_num_tokens, expert_num_tokens_cpu=None)
        return (expert_x, expert_x_scale, expert_tokens_meta, None, None)

    def finalize(self, output: torch.Tensor, fused_expert_output: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor, apply_router_weight_on_input: bool, weight_and_reduce_impl: mk.TopKWeightAndReduce, extra_finalize_args: Optional[dict[str, Any]]) -> None:
        assert isinstance(weight_and_reduce_impl, TopKWeightAndReduceDelegate), 'Weight application and reduction happens in the combine kernel.'
        bound_m: Optional[torch.Tensor] = None
        assert topk_ids.size() == topk_weights.size(), f'{topk_ids.size()} == {topk_weights.size()}'
        assert output.size(0) <= self.max_num_tokens, f'{output.size(0)} <= {self.max_num_tokens}'
        assert output.size(1) == fused_expert_output.size(-1)
        if apply_router_weight_on_input:
            topk_weights = torch.ones_like(topk_weights)
        self.a2a.combine(out_tokens=output, indices=topk_ids.view(dtype=torch.uint32), weights=topk_weights, expert_y=fused_expert_output, bound_m=bound_m)