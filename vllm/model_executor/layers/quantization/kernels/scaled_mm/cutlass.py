from .ScaledMMLinearKernel import ScaledMMLinearKernel, ScaledMMLinearLayerConfig
from typing import Optional
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.layers.quantization.utils.w8a8_utils import convert_to_channelwise
from vllm.platforms import current_platform
import torch

class CutlassScaledMMLinearKernel(ScaledMMLinearKernel):

    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    @classmethod
    def can_implement(cls, c: ScaledMMLinearLayerConfig) -> tuple[bool, Optional[str]]:
        if not current_platform.is_cuda() and (not current_platform.is_cpu()):
            return (False, 'CutlassScaledMM requires running on CUDA or CPU.')
        return (True, None)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = getattr(layer, self.w_q_name)
        replace_parameter(layer, self.w_q_name, torch.nn.Parameter(weight.t().data, requires_grad=False))
        is_fused_module = len(layer.logical_widths) > 1
        weight_scale = getattr(layer, self.w_s_name)
        if is_fused_module and (not self.config.is_channelwise):
            weight_scale = convert_to_channelwise(weight_scale, layer.logical_widths)
        replace_parameter(layer, self.w_s_name, torch.nn.Parameter(weight_scale.data, requires_grad=False))
        if self.config.is_static_input_scheme:
            input_scale = getattr(layer, self.i_s_name)
            if self.config.input_symmetric:
                replace_parameter(layer, self.i_s_name, torch.nn.Parameter(input_scale.max(), requires_grad=False))
                setattr(layer, self.i_zp_name, None)
            else:
                input_zero_point = getattr(layer, self.i_zp_name)
                int8_traits = torch.iinfo(torch.int8)
                azps = input_zero_point.to(dtype=torch.int32)
                range_max = (input_scale * (int8_traits.max - azps)).max()
                range_min = (input_scale * (int8_traits.min - azps)).min()
                scale = (range_max - range_min) / (int8_traits.max - int8_traits.min)
                replace_parameter(layer, self.i_s_name, torch.nn.Parameter(scale, requires_grad=False))
                azp = (int8_traits.min - range_min / scale).to(dtype=torch.int32)
                replace_parameter(layer, self.i_zp_name, torch.nn.Parameter(azp, requires_grad=False))
        else:
            setattr(layer, self.i_s_name, None)
            setattr(layer, self.i_zp_name, None)
        if not self.config.input_symmetric:
            weight = getattr(layer, self.w_q_name)
            azp_adj = weight.sum(dim=0, keepdim=True, dtype=torch.int32)
            if self.config.is_static_input_scheme:
                azp_adj = getattr(layer, self.i_zp_name) * azp_adj
            setattr(layer, self.azp_adj_name, torch.nn.Parameter(azp_adj, requires_grad=False))
        else:
            setattr(layer, self.azp_adj_name, None)

    def apply_weights(self, layer: torch.nn.Module, x: torch.Tensor, bias: Optional[torch.Tensor]=None) -> torch.Tensor:
        w_q, w_s, i_s, i_zp, azp_adj = self._get_weight_params(layer)
        symmetric = azp_adj is None
        x_q, x_s, x_zp = ops.scaled_int8_quant(x.contiguous(), i_s, i_zp, symmetric=symmetric)
        if x_zp is not None:
            static = i_zp is not None
            azp = None if static else x_zp
            return ops.cutlass_scaled_mm_azp(x_q, w_q, scale_a=x_s, scale_b=w_s, out_dtype=x.dtype, azp_adj=azp_adj, azp=azp, bias=bias)
        return ops.cutlass_scaled_mm(x_q, w_q, scale_a=x_s, scale_b=w_s, out_dtype=x.dtype, bias=bias)