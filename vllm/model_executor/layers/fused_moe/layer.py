from vllm.model_executor.layers.fused_moe.fused_moe import fused_topk
from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import shuffle_weights
from abc import abstractmethod
from collections.abc import Iterable
from enum import Enum
from typing import Callable, Literal, Optional, overload
import torch
import torch.nn.functional as F
from torch.nn.parameter import UninitializedParameter
import vllm.envs as envs
from vllm.config import get_current_vllm_config
from vllm.distributed import get_dp_group, get_ep_group, get_tensor_model_parallel_world_size, tensor_model_parallel_all_reduce
from vllm.distributed.eplb.eplb_state import EplbState
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig, FusedMoEParallelConfig
from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEActivationFormat, FusedMoEModularKernel, FusedMoEPermuteExpertsUnpermute, FusedMoEPrepareAndFinalize
from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import is_rocm_aiter_moe_enabled
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig, QuantizeMethodBase
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.platforms.interface import CpuArchEnum
from vllm.utils import direct_register_custom_op, has_deep_ep, has_pplx
from vllm.utils.flashinfer import has_flashinfer
if current_platform.is_cuda_alike():
    from .fused_batched_moe import BatchedTritonExperts
    from .fused_moe import TritonExperts, fused_experts
    if has_pplx():
        from .pplx_prepare_finalize import PplxPrepareAndFinalize, pplx_hidden_dim_scale_bytes
    if has_deep_ep():
        from .deepep_ht_prepare_finalize import DeepEPHTPrepareAndFinalize
        from .deepep_ll_prepare_finalize import DEEPEP_QUANT_BLOCK_SHAPE, DeepEPLLPrepareAndFinalize
    if has_flashinfer():
        from .flashinfer_cutlass_prepare_finalize import FlashInferCutlassMoEPrepareAndFinalize
else:
    fused_experts = None
    FusedMoEPermuteExpertsUnpermute = None
    FusedMoEPrepareAndFinalize = None
if is_rocm_aiter_moe_enabled():
    from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import rocm_aiter_grouped_topk as grouped_topk
elif current_platform.is_cpu():
    pass
else:
    from vllm.model_executor.layers.fused_moe.fused_moe import grouped_topk
if current_platform.is_tpu():
    from .moe_pallas import fused_moe as fused_moe_pallas
else:
    fused_moe_pallas = None
logger = init_logger(__name__)

class FusedMoeWeightScaleSupported(Enum):
    TENSOR = 'tensor'
    CHANNEL = 'channel'
    GROUP = 'group'
    BLOCK = 'block'

class FusedMoEMethodBase(QuantizeMethodBase):
    moe: FusedMoEConfig

    @abstractmethod
    def create_weights(self, layer: torch.nn.Module, num_experts: int, hidden_size: int, intermediate_size_per_partition: int, params_dtype: torch.dtype, **extra_weight_attrs):
        raise NotImplementedError

    def uses_weight_scale_2_pattern(self) -> bool:
        """
        Returns True if this quantization method uses 'weight_scale_2' pattern
        for per-tensor weight scales (e.g., FP4 variants), False otherwise.

        This method should be overridden by subclasses that use the
        'weight_scale_2' pattern instead of the standard 'weight_scale' pattern.
        """
        return False

    @staticmethod
    def maybe_make_prepare_finalize(moe: FusedMoEConfig) -> Optional[FusedMoEPrepareAndFinalize]:
        all2all_manager = get_ep_group().device_communicator.all2all_manager
        assert all2all_manager is not None
        prepare_finalize: Optional[FusedMoEPrepareAndFinalize] = None
        if moe.use_flashinfer_cutlass_kernels:
            prepare_finalize = FlashInferCutlassMoEPrepareAndFinalize(quant_dtype=moe.quant_dtype)
        if moe.use_pplx_kernels:
            hidden_dim_bytes, hidden_scale_bytes = pplx_hidden_dim_scale_bytes(moe.max_num_tokens, moe.hidden_dim, moe.in_dtype, moe.quant_dtype, per_act_token_quant=moe.per_act_token_quant, block_shape=moe.block_shape)
            all_to_all_args = dict(max_num_tokens=moe.max_num_tokens, num_experts=moe.num_experts, experts_per_token=moe.experts_per_token, rank=all2all_manager.rank, world_size=all2all_manager.world_size, dp_size=all2all_manager.tp_group.world_size, hidden_dim=moe.hidden_dim, hidden_dim_bytes=hidden_dim_bytes, hidden_dim_scale_bytes=hidden_scale_bytes)
            num_dispatchers = all2all_manager.world_size // all2all_manager.tp_group.world_size
            if not all2all_manager.internode:
                all_to_all_args['group_name'] = all2all_manager.cpu_group.group_name
            handle = all2all_manager.get_handle(all_to_all_args)
            prepare_finalize = PplxPrepareAndFinalize(handle, max_num_tokens=moe.max_num_tokens, num_local_experts=moe.num_local_experts, num_dispatchers=num_dispatchers)
        elif moe.use_deepep_ht_kernels:
            assert moe.dp_size == all2all_manager.dp_world_size
            all_to_all_args = dict()
            handle = all2all_manager.get_handle(all_to_all_args)
            prepare_finalize = DeepEPHTPrepareAndFinalize(handle, num_dispatchers=all2all_manager.world_size, dp_size=all2all_manager.dp_world_size, rank_expert_offset=all2all_manager.rank * moe.num_local_experts)
        elif moe.use_deepep_ll_kernels:
            all_to_all_args = dict(max_num_tokens_per_dp_rank=moe.max_num_tokens, token_hidden_size=moe.hidden_dim, num_ep_ranks=all2all_manager.world_size, num_global_experts=moe.num_experts, num_local_experts=moe.num_experts // all2all_manager.world_size)
            handle = all2all_manager.get_handle(all_to_all_args)
            use_fp8_dispatch = moe.quant_config is not None and moe.quant_config.quant_dtype == current_platform.fp8_dtype() and (moe.quant_config.block_shape == DEEPEP_QUANT_BLOCK_SHAPE)
            prepare_finalize = DeepEPLLPrepareAndFinalize(handle, max_tokens_per_rank=moe.max_num_tokens, num_dispatchers=all2all_manager.world_size, use_fp8_dispatch=use_fp8_dispatch)
        return prepare_finalize

    def init_prepare_finalize(self, moe: FusedMoEConfig):
        self.moe = moe
        prepare_finalize = FusedMoEMethodBase.maybe_make_prepare_finalize(self.moe)
        self.topk_indices_dtype = None
        if prepare_finalize is not None:
            logger.debug('%s', prepare_finalize.__class__.__name__)
            self.topk_indices_dtype = prepare_finalize.topk_indices_dtype()
            experts = self.select_gemm_impl(prepare_finalize, self.moe)
            self.fused_experts = FusedMoEModularKernel(prepare_finalize, experts)

    def select_gemm_impl(self, prepare_finalize: FusedMoEPrepareAndFinalize, moe: FusedMoEConfig) -> FusedMoEPermuteExpertsUnpermute:
        raise NotImplementedError(f'{self.__class__.__name__} must select appropriate gemm implementation based on the prepare_finalize')

    def maybe_swap_experts_impl(self, moe_parallel_config: FusedMoEParallelConfig):
        pass

    @abstractmethod
    def apply(self, layer: torch.nn.Module, x: torch.Tensor, router_logits: torch.Tensor, top_k: int, renormalize: bool, use_grouped_topk: bool=False, topk_group: Optional[int]=None, num_expert_group: Optional[int]=None, global_num_experts: int=-1, expert_map: Optional[torch.Tensor]=None, custom_routing_function: Optional[Callable]=None, scoring_func: str='softmax', e_score_correction_bias: Optional[torch.Tensor]=None, apply_router_weight_on_input: bool=False, activation: str='silu', enable_eplb: bool=False, expert_load_view: Optional[torch.Tensor]=None, logical_to_physical_map: Optional[torch.Tensor]=None, logical_replica_count: Optional[torch.Tensor]=None) -> torch.Tensor:
        raise NotImplementedError

@CustomOp.register('unquantized_fused_moe')
class UnquantizedFusedMoEMethod(FusedMoEMethodBase, CustomOp):
    """MoE method without quantization."""

    def __init__(self, moe: FusedMoEConfig):
        super().__init__()
        self.fused_experts = fused_experts
        self.topk_indices_dtype = None
        self.moe = moe
        self.rocm_aiter_moe_enabled = is_rocm_aiter_moe_enabled()
        if self.rocm_aiter_moe_enabled:
            from .rocm_aiter_fused_moe import rocm_aiter_fused_experts
            self.rocm_aiter_fused_experts = rocm_aiter_fused_experts
        else:
            self.rocm_aiter_fused_experts = None

    def select_gemm_impl(self, prepare_finalize: FusedMoEPrepareAndFinalize, moe: FusedMoEConfig) -> FusedMoEPermuteExpertsUnpermute:
        if prepare_finalize.activation_format == FusedMoEActivationFormat.BatchedExperts:
            logger.debug('BatchedTritonExperts %s', self.moe)
            return BatchedTritonExperts(max_num_tokens=self.moe.max_num_tokens, num_dispatchers=prepare_finalize.num_dispatchers())
        else:
            logger.debug('TritonExperts %s', self.moe)
            return TritonExperts()

    def create_weights(self, layer: torch.nn.Module, num_experts: int, hidden_size: int, intermediate_size_per_partition: int, params_dtype: torch.dtype, **extra_weight_attrs):
        w13_weight = torch.nn.Parameter(torch.empty(num_experts, 2 * intermediate_size_per_partition, hidden_size, dtype=params_dtype), requires_grad=False)
        layer.register_parameter('w13_weight', w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)
        w2_weight = torch.nn.Parameter(torch.empty(num_experts, hidden_size, intermediate_size_per_partition, dtype=params_dtype), requires_grad=False)
        layer.register_parameter('w2_weight', w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

    def _maybe_pad_weight(self, weight: torch.Tensor) -> torch.Tensor:
        if envs.VLLM_ROCM_MOE_PADDING and current_platform.is_rocm() and (weight.stride(-1) == 1) and (weight.stride(-2) * weight.element_size() % 512 == 0):
            num_pad = 256 // weight.element_size()
            weight = F.pad(weight, (0, num_pad), 'constant', 0)[..., :-num_pad]
            torch.cuda.empty_cache()
        return weight

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)
        layer.w13_weight.data = self._maybe_pad_weight(layer.w13_weight.data)
        layer.w2_weight.data = self._maybe_pad_weight(layer.w2_weight.data)
        if self.rocm_aiter_moe_enabled:
            shuffled_w13, shuffled_w2 = shuffle_weights(layer.w13_weight.data, layer.w2_weight.data)
            layer.w13_weight.data = shuffled_w13
            layer.w2_weight.data = shuffled_w2
        if current_platform.is_cpu():
            if current_platform.get_cpu_architecture() == CpuArchEnum.X86:
                from vllm.model_executor.layers.fused_moe import cpu_fused_moe
                dtype = layer.w13_weight.dtype
                if envs.VLLM_CPU_SGL_KERNEL and torch._C._cpu._is_amx_tile_supported() and (dtype == torch.bfloat16):
                    packed_w13_weight = torch.ops._C.convert_weight_packed(layer.w13_weight)
                    assert packed_w13_weight.size() == layer.w13_weight.size()
                    layer.w13_weight.copy_(packed_w13_weight)
                    del packed_w13_weight
                    packed_w2_weight = torch.ops._C.convert_weight_packed(layer.w2_weight)
                    assert packed_w2_weight.size() == layer.w2_weight.size()
                    layer.w2_weight.copy_(packed_w2_weight)
                    layer.cpu_fused_moe = cpu_fused_moe.SGLFusedMOE(layer)
                else:
                    layer.cpu_fused_moe = cpu_fused_moe.IPEXFusedMOE(layer)
            else:
                raise NotImplementedError('CPU MOE only supports x86 arch.')

    def apply(self, layer: torch.nn.Module, x: torch.Tensor, router_logits: torch.Tensor, top_k: int, renormalize: bool, use_grouped_topk: bool=False, topk_group: Optional[int]=None, num_expert_group: Optional[int]=None, global_num_experts: int=-1, expert_map: Optional[torch.Tensor]=None, custom_routing_function: Optional[Callable]=None, scoring_func: str='softmax', e_score_correction_bias: Optional[torch.Tensor]=None, apply_router_weight_on_input: bool=False, activation: str='silu', enable_eplb: bool=False, expert_load_view: Optional[torch.Tensor]=None, logical_to_physical_map: Optional[torch.Tensor]=None, logical_replica_count: Optional[torch.Tensor]=None) -> torch.Tensor:
        if enable_eplb:
            assert expert_load_view is not None
            assert logical_to_physical_map is not None
            assert logical_replica_count is not None
            assert isinstance(layer, FusedMoE)
        return self.forward(x=x, layer=layer, router_logits=router_logits, top_k=top_k, renormalize=renormalize, use_grouped_topk=use_grouped_topk, topk_group=topk_group, num_expert_group=num_expert_group, global_num_experts=global_num_experts, expert_map=expert_map, custom_routing_function=custom_routing_function, scoring_func=scoring_func, e_score_correction_bias=e_score_correction_bias, activation=activation, apply_router_weight_on_input=apply_router_weight_on_input, enable_eplb=enable_eplb, expert_load_view=expert_load_view, logical_to_physical_map=logical_to_physical_map, logical_replica_count=logical_replica_count)

    def forward_cuda(self, layer: torch.nn.Module, x: torch.Tensor, use_grouped_topk: bool, top_k: int, router_logits: torch.Tensor, renormalize: bool, topk_group: Optional[int]=None, num_expert_group: Optional[int]=None, global_num_experts: int=-1, expert_map: Optional[torch.Tensor]=None, custom_routing_function: Optional[Callable]=None, scoring_func: str='softmax', e_score_correction_bias: Optional[torch.Tensor]=None, apply_router_weight_on_input: bool=False, activation: str='silu', enable_eplb: bool=False, expert_load_view: Optional[torch.Tensor]=None, logical_to_physical_map: Optional[torch.Tensor]=None, logical_replica_count: Optional[torch.Tensor]=None) -> torch.Tensor:
        topk_weights, topk_ids = FusedMoE.select_experts(hidden_states=x, router_logits=router_logits, use_grouped_topk=use_grouped_topk, top_k=top_k, renormalize=renormalize, topk_group=topk_group, num_expert_group=num_expert_group, custom_routing_function=custom_routing_function, scoring_func=scoring_func, e_score_correction_bias=e_score_correction_bias, indices_type=self.topk_indices_dtype, enable_eplb=enable_eplb, expert_map=expert_map, expert_load_view=expert_load_view, logical_to_physical_map=logical_to_physical_map, logical_replica_count=logical_replica_count)
        if self.rocm_aiter_moe_enabled:
            return self.rocm_aiter_fused_experts(hidden_states=x, w1=layer.w13_weight, w2=layer.w2_weight, topk_weights=topk_weights, topk_ids=topk_ids, expert_map=expert_map, activation=activation, apply_router_weight_on_input=apply_router_weight_on_input)
        else:
            return self.fused_experts(hidden_states=x, w1=layer.w13_weight, w2=layer.w2_weight, topk_weights=topk_weights, topk_ids=topk_ids, inplace=True, activation=activation, apply_router_weight_on_input=apply_router_weight_on_input, global_num_experts=global_num_experts, expert_map=expert_map)

    def forward_cpu(self, layer: torch.nn.Module, x: torch.Tensor, use_grouped_topk: bool, top_k: int, router_logits: torch.Tensor, renormalize: bool, topk_group: Optional[int]=None, num_expert_group: Optional[int]=None, global_num_experts: int=-1, expert_map: Optional[torch.Tensor]=None, custom_routing_function: Optional[Callable]=None, scoring_func: str='softmax', e_score_correction_bias: Optional[torch.Tensor]=None, apply_router_weight_on_input: bool=False, activation: str='silu', **kwargs):
        return layer.cpu_fused_moe(layer, x, use_grouped_topk, top_k, router_logits, renormalize, topk_group, num_expert_group, global_num_experts, expert_map, custom_routing_function, scoring_func, e_score_correction_bias, apply_router_weight_on_input, activation)

    def forward_tpu(self, layer: torch.nn.Module, x: torch.Tensor, use_grouped_topk: bool, top_k: int, router_logits: torch.Tensor, renormalize: bool, topk_group: Optional[int]=None, num_expert_group: Optional[int]=None, global_num_experts: int=-1, expert_map: Optional[torch.Tensor]=None, custom_routing_function: Optional[Callable]=None, scoring_func: str='softmax', e_score_correction_bias: Optional[torch.Tensor]=None, apply_router_weight_on_input: bool=False, activation: str='silu') -> torch.Tensor:
        assert not use_grouped_topk
        assert num_expert_group is None
        assert topk_group is None
        assert custom_routing_function is None
        assert apply_router_weight_on_input is False
        if scoring_func != 'softmax':
            raise NotImplementedError('Only softmax scoring function is supported for TPU.')
        if e_score_correction_bias is not None:
            raise NotImplementedError('Expert score correction bias is not supported for TPU.')
        assert activation == 'silu', f'{activation} is not supported for TPU.'
        return fused_moe_pallas(hidden_states=x, w1=layer.w13_weight, w2=layer.w2_weight, topk=top_k, gating_output=router_logits, global_num_experts=global_num_experts, expert_map=expert_map, renormalize=renormalize)
    if current_platform.is_tpu():
        forward_native = forward_tpu
    elif current_platform.is_cpu():
        forward_native = forward_cpu
    else:
        forward_native = forward_cuda

def determine_expert_map(ep_size: int, ep_rank: int, global_num_experts: int) -> tuple[int, Optional[torch.Tensor]]:
    """
        Calculates how many experts should be assigned to each rank for EP and
        creates a mapping from global to local expert index. Experts are
        distributed evenly across ranks. Any remaining are assigned to the
        last rank.

        Args:
            ep_size (int): The size of the expert parallel group
            global_num_experts (int): The total number of experts in the model.

        Returns:
            tuple[int, Optional[torch.Tensor]]: A tuple containing:
                - local_num_experts (int): The number of experts assigned
                    to the current rank.
                - expert_map (Optional[torch.Tensor]): A tensor of shape
                    (global_num_experts,) mapping from global to local index.
                    Contains -1 for experts not assigned to the current rank.
                    Returns None if ep_size is 1.
        """
    assert ep_size > 0
    if ep_size == 1:
        return (global_num_experts, None)
    local_num_experts = global_num_experts // ep_size
    expert_map = torch.full((global_num_experts,), -1, dtype=torch.int32)
    if ep_rank < ep_size - 1:
        expert_map[ep_rank * local_num_experts:(ep_rank + 1) * local_num_experts] = torch.arange(0, local_num_experts, dtype=torch.int32)
    else:
        local_num_experts = global_num_experts - ep_rank * local_num_experts
        expert_map[-local_num_experts:] = torch.arange(0, local_num_experts, dtype=torch.int32)
    return (local_num_experts, expert_map)

class FusedMoE(torch.nn.Module):
    """FusedMoE layer for MoE models.

    This layer contains both MergedColumnParallel weights (gate_up_proj /
    w13) and RowParallelLinear weights (down_proj/ w2).

    Note: Mixtral uses w1, w2, and w3 for gate, up, and down_proj. We
    copy that naming convention here and handle any remapping in the
    load_weights function in each model implementation.

    Args:
        num_experts: Number of experts in the model
        top_k: Number of experts selected for each token
        hidden_size: Input hidden state size of the transformer
        intermediate_size: Intermediate size of the experts
        params_dtype: Data type for the parameters.
        reduce_results: Whether to all all_reduce on the output of the layer
        renomalize: Whether to renormalize the logits in the fused_moe kernel
        quant_config: Quantization configure.
        enable_eplb: Whether to enable expert parallelism load balancer.
    """

    def __init__(self, num_experts: int, top_k: int, hidden_size: int, intermediate_size: int, params_dtype: Optional[torch.dtype]=None, reduce_results: bool=False, renormalize: bool=True, use_grouped_topk: bool=False, num_expert_group: Optional[int]=None, topk_group: Optional[int]=None, quant_config: Optional[QuantizationConfig]=None, tp_size: Optional[int]=None, ep_size: Optional[int]=None, dp_size: Optional[int]=None, prefix: str='', custom_routing_function: Optional[Callable]=None, scoring_func: str='softmax', e_score_correction_bias: Optional[torch.Tensor]=None, apply_router_weight_on_input: bool=False, activation: str='silu', enable_eplb: bool=False, num_redundant_experts: int=0):
        super().__init__()
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype
        tp_size_ = tp_size if tp_size is not None else get_tensor_model_parallel_world_size()
        dp_size_ = dp_size if dp_size is not None else get_dp_group().world_size
        vllm_config = get_current_vllm_config()
        self.moe_parallel_config: FusedMoEParallelConfig = FusedMoEParallelConfig.make(tp_size_=tp_size_, dp_size_=dp_size_, vllm_parallel_config=vllm_config.parallel_config)
        self.global_num_experts = num_experts + num_redundant_experts
        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError('Duplicate layer name: {}'.format(prefix))
        compilation_config.static_forward_context[prefix] = self
        self.layer_name = prefix
        self.enable_eplb = enable_eplb
        self.expert_load_view: Optional[torch.Tensor] = None
        self.logical_to_physical_map: Optional[torch.Tensor] = None
        self.logical_replica_count: Optional[torch.Tensor] = None
        if self.use_ep:
            if self.enable_eplb:
                assert self.global_num_experts % self.ep_size == 0, 'EPLB currently only supports even distribution of experts across ranks.'
            else:
                assert num_redundant_experts == 0, 'Redundant experts are only supported with EPLB.'
            self.local_num_experts, self.expert_map = determine_expert_map(ep_size=self.ep_size, ep_rank=self.ep_rank, global_num_experts=self.global_num_experts)
        else:
            self.local_num_experts, self.expert_map = (self.global_num_experts, None)
        self.top_k = top_k
        assert intermediate_size % self.tp_size == 0
        self.hidden_size = hidden_size
        self.intermediate_size_per_partition = intermediate_size // self.tp_size
        self.reduce_results = reduce_results
        self.renormalize = renormalize
        self.use_grouped_topk = use_grouped_topk
        if self.use_grouped_topk:
            assert num_expert_group is not None and topk_group is not None
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.custom_routing_function = custom_routing_function
        self.scoring_func = scoring_func
        self.e_score_correction_bias = e_score_correction_bias
        self.apply_router_weight_on_input = apply_router_weight_on_input
        self.activation = activation
        if self.scoring_func != 'softmax' and (not self.use_grouped_topk):
            raise ValueError('Only softmax scoring function is supported for non-grouped topk.')
        if vllm_config.model_config is not None:
            model_dtype = vllm_config.model_config.dtype
        else:
            model_dtype = params_dtype
        moe = FusedMoEConfig.make(num_experts=self.global_num_experts, experts_per_token=top_k, hidden_dim=hidden_size, num_local_experts=self.local_num_experts, moe_parallel_config=self.moe_parallel_config, in_dtype=model_dtype, max_num_tokens=envs.VLLM_MOE_DP_CHUNK_SIZE, quant_config=quant_config)
        self.moe_config = moe
        self.quant_config = quant_config
        quant_method: Optional[QuantizeMethodBase] = None
        quant_method = UnquantizedFusedMoEMethod(moe) if quant_config is None else quant_config.get_quant_method(self, prefix)
        assert quant_method is not None
        assert isinstance(quant_method, FusedMoEMethodBase)
        self.quant_method = quant_method
        if self.enable_eplb:
            from vllm.model_executor.layers.quantization.fp8 import Fp8MoEMethod
            if not isinstance(quant_method, (Fp8MoEMethod, UnquantizedFusedMoEMethod)):
                raise NotImplementedError('EPLB is only supported for FP8 quantization for now.')
        moe_quant_params = {'num_experts': self.local_num_experts, 'hidden_size': hidden_size, 'intermediate_size_per_partition': self.intermediate_size_per_partition, 'params_dtype': params_dtype, 'weight_loader': self.weight_loader}
        if self.quant_method.__class__.__name__ in ('GPTQMarlinMoEMethod', 'CompressedTensorsWNA16MarlinMoEMethod', 'CompressedTensorsWNA16MoEMethod'):
            moe_quant_params['intermediate_size_full'] = intermediate_size
        self.quant_method.create_weights(layer=self, **moe_quant_params)
        if isinstance(self.quant_method, FusedMoEMethodBase):
            self.quant_method.maybe_swap_experts_impl(self.moe_parallel_config)
        self.batched_hidden_states: Optional[torch.Tensor] = None
        self.batched_router_logits: Optional[torch.Tensor] = None
        if self.moe_parallel_config.use_pplx_kernels or self.moe_parallel_config.use_deepep_ll_kernels or self.moe_parallel_config.use_flashinfer_cutlass_kernels:
            self.batched_hidden_states = torch.zeros((moe.max_num_tokens, self.hidden_size), dtype=moe.in_dtype, device=torch.cuda.current_device())
            self.batched_router_logits = torch.zeros((moe.max_num_tokens, num_experts), dtype=moe.in_dtype, device=torch.cuda.current_device())

    @property
    def tp_size(self):
        return self.moe_parallel_config.tp_size

    @property
    def dp_size(self):
        return self.moe_parallel_config.dp_size

    @property
    def ep_size(self):
        return self.moe_parallel_config.ep_size

    @property
    def tp_rank(self):
        return self.moe_parallel_config.tp_rank

    @property
    def dp_rank(self):
        return self.moe_parallel_config.dp_rank

    @property
    def ep_rank(self):
        return self.moe_parallel_config.ep_rank

    @property
    def use_ep(self):
        return self.moe_parallel_config.use_ep

    @property
    def use_pplx_kernels(self):
        return self.moe_parallel_config.use_pplx_kernels

    @property
    def use_deepep_ht_kernels(self):
        return self.moe_parallel_config.use_deepep_ht_kernels

    @property
    def use_deepep_ll_kernels(self):
        return self.moe_parallel_config.use_deepep_ll_kernels

    @property
    def use_flashinfer_cutlass_kernels(self):
        return self.moe_parallel_config.use_flashinfer_cutlass_kernels

    def update_expert_map(self):
        assert self.expert_map is not None
        with self.expert_map.device:
            self.local_num_experts, self.expert_map = determine_expert_map(ep_size=self.ep_size, ep_rank=self.ep_rank, global_num_experts=self.global_num_experts)

    def _load_per_tensor_weight_scale(self, shard_id: str, param: torch.nn.Parameter, loaded_weight: torch.Tensor, expert_id: int):
        param_data = param.data
        if shard_id in ('w1', 'w3'):
            idx = 0 if shard_id == 'w1' else 1
            param_data[expert_id][idx] = loaded_weight
        elif shard_id == 'w2':
            param_data[expert_id] = loaded_weight

    def _load_model_weight_or_group_weight_scale(self, shard_dim: int, expert_data: torch.Tensor, shard_id: str, loaded_weight: torch.Tensor, tp_rank: int, load_full_w2: bool=False):
        """
        Load grouped weight scales for group quantization or model weights
            :param shard_dim: dimension to shard
            :param expert_data: parameter for a particular expert
            :param shard_id: either w1, w2, or w3
            :param loaded_weight: checkpoint weight to load into the param
            :param tp_rank: tensor parallel rank
            :param load_full_w2: whether or not the w2 loaded should be sharded.
        """
        if shard_id == 'w2':
            self._load_w2(shard_dim=shard_dim, loaded_weight=loaded_weight, expert_data=expert_data, tp_rank=tp_rank, load_full=load_full_w2)
        elif shard_id in ('w1', 'w3'):
            self._load_w13(shard_id=shard_id, shard_dim=shard_dim, loaded_weight=loaded_weight, expert_data=expert_data, tp_rank=tp_rank)

    def _load_per_channel_weight_scale(self, expert_data: torch.Tensor, shard_dim: int, shard_id: str, loaded_weight: torch.Tensor, tp_rank: int):
        if shard_id == 'w2':
            expert_data.copy_(loaded_weight)
        elif shard_id in ('w1', 'w3'):
            self._load_w13(shard_id=shard_id, shard_dim=shard_dim, loaded_weight=loaded_weight, expert_data=expert_data, tp_rank=tp_rank)

    def _load_w13(self, expert_data: torch.Tensor, shard_dim: int, shard_id: str, loaded_weight: torch.Tensor, tp_rank: int, load_full: bool=False):
        shard_size = expert_data.shape[shard_dim] // 2
        if not load_full:
            loaded_weight = loaded_weight.narrow(shard_dim, shard_size * tp_rank, shard_size)
        if shard_id == 'w1':
            expert_data = expert_data.narrow(shard_dim, 0, shard_size)
        else:
            assert shard_id == 'w3'
            expert_data = expert_data.narrow(shard_dim, shard_size, shard_size)
        expert_data.copy_(loaded_weight)

    def _load_w2(self, expert_data: torch.Tensor, shard_dim: int, loaded_weight: torch.Tensor, tp_rank: int, load_full: bool=False):
        shard_size = expert_data.shape[shard_dim]
        if not load_full:
            loaded_weight = loaded_weight.narrow(shard_dim, shard_size * tp_rank, shard_size)
        expert_data.copy_(loaded_weight)

    def _load_single_value(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, expert_id: int):
        param_data = param.data
        param_data[expert_id] = loaded_weight

    def _load_g_idx(self, shard_id: str, expert_data: torch.Tensor, shard_dim: int, loaded_weight: torch.Tensor, tp_rank: int):
        if shard_id == 'w2':
            self._load_w2(shard_dim=shard_dim, loaded_weight=loaded_weight, expert_data=expert_data, tp_rank=tp_rank)
        else:
            assert shard_id in ('w1', 'w3')
            expert_data.copy_(loaded_weight)

    def _map_global_expert_id_to_local_expert_id(self, expert_id: int) -> int:
        if self.expert_map is None:
            return expert_id
        return self.expert_map[expert_id].item()

    @overload
    def weight_loader(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, weight_name: str, shard_id: str, expert_id: int, return_success: Literal[False]) -> None:
        ...

    @overload
    def weight_loader(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, weight_name: str, shard_id: str, expert_id: int, return_success: Literal[True]) -> bool:
        ...

    def weight_loader(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, weight_name: str, shard_id: str, expert_id: int, return_success: bool=False) -> Optional[bool]:
        expert_id = self._map_global_expert_id_to_local_expert_id(expert_id)
        if expert_id == -1:
            return False if return_success else None
        quant_method_name = self.quant_method.__class__.__name__
        if self.quant_method.__class__.__name__ in ('CompressedTensorsWNA16MarlinMoEMethod', 'CompressedTensorsWNA16MoEMethod'):
            loaded_weight = loaded_weight.t().contiguous()
        if shard_id not in ('w1', 'w2', 'w3'):
            raise ValueError(f"shard_id must be ['w1','w2','w3'] but got {shard_id}.")
        WEIGHT_SCALE_SUPPORTED = [e.value for e in FusedMoeWeightScaleSupported]
        SHARD_ID_TO_SHARDED_DIM = {'w1': 0, 'w2': 1, 'w3': 0}
        is_gguf_weight = getattr(param, 'is_gguf_weight', False)
        is_gguf_weight_type = getattr(param, 'is_gguf_weight_type', False)
        if is_gguf_weight_type:
            param.weight_type = loaded_weight.item()
            param.data.copy_(loaded_weight)
            return True if return_success else None
        use_bitsandbytes_4bit = getattr(param, 'use_bitsandbytes_4bit', False)
        if use_bitsandbytes_4bit:
            shard_dim = 0
            expert_data = param.data[expert_id]
            if shard_id == 'w2':
                expert_data.copy_(loaded_weight)
            elif shard_id in ('w1', 'w3'):
                full_load = True
                self._load_w13(shard_id=shard_id, shard_dim=shard_dim, loaded_weight=loaded_weight, expert_data=expert_data, tp_rank=self.tp_rank, load_full=full_load)
            return True if return_success else None
        is_transposed = getattr(param, 'is_transposed', False)
        shard_dim = SHARD_ID_TO_SHARDED_DIM[shard_id]
        if is_transposed:
            shard_dim = int(not shard_dim)
        full_load = len(loaded_weight.shape) == 3
        if full_load:
            shard_dim += 1
        if is_gguf_weight and isinstance(param, UninitializedParameter):
            final_shape = list(loaded_weight.shape)
            if shard_id in ['w1', 'w3']:
                final_shape[1] *= 2
            final_shape[shard_dim] = final_shape[shard_dim] // self.tp_size
            param.materialize(final_shape, dtype=loaded_weight.dtype)
        expert_data = param.data if full_load else param.data[expert_id]
        if 'input_scale' in weight_name:
            loaded_weight = loaded_weight.to(param.data.device)
            if 'compressed' in quant_method_name.lower() and param.data[expert_id] != 1 and ((param.data[expert_id] - loaded_weight).abs() > 1e-05):
                raise ValueError(f'input_scales of w1 and w3 of a layer must be equal. But got {param.data[expert_id]} vs. {loaded_weight}')
            self._load_single_value(param=param, loaded_weight=loaded_weight, expert_id=expert_id)
            return True if return_success else None
        if 'g_idx' in weight_name:
            self._load_g_idx(shard_dim=0, shard_id=shard_id, loaded_weight=loaded_weight, expert_data=expert_data, tp_rank=self.tp_rank)
            return True if return_success else None
        if 'ModelOpt' in quant_method_name:
            uses_weight_scale_2 = self.quant_method.uses_weight_scale_2_pattern()
            per_tensor_conditions = ('weight_scale_2' in weight_name if uses_weight_scale_2 else 'weight_scale' in weight_name) or 'input_scale' in weight_name
            if per_tensor_conditions:
                self._load_per_tensor_weight_scale(shard_id=shard_id, param=param, loaded_weight=loaded_weight, expert_id=expert_id)
            elif 'weight' in weight_name:
                self._load_model_weight_or_group_weight_scale(shard_id=shard_id, shard_dim=shard_dim, loaded_weight=loaded_weight, expert_data=expert_data, tp_rank=self.tp_rank)
            return True if return_success else None
        if 'scale' in weight_name or 'zero' in weight_name or 'offset' in weight_name:
            quant_method = getattr(param, 'quant_method', None)
            if quant_method == FusedMoeWeightScaleSupported.CHANNEL.value:
                self._load_per_channel_weight_scale(shard_id=shard_id, shard_dim=shard_dim, loaded_weight=loaded_weight, expert_data=expert_data, tp_rank=self.tp_rank)
            elif quant_method in [FusedMoeWeightScaleSupported.GROUP.value, FusedMoeWeightScaleSupported.BLOCK.value]:
                self._load_model_weight_or_group_weight_scale(shard_id=shard_id, shard_dim=shard_dim, loaded_weight=loaded_weight, expert_data=expert_data, tp_rank=self.tp_rank, load_full_w2=getattr(param, 'load_full_w2', False))
            elif quant_method == FusedMoeWeightScaleSupported.TENSOR.value:
                self._load_per_tensor_weight_scale(shard_id=shard_id, param=param, loaded_weight=loaded_weight, expert_id=expert_id)
            else:
                raise ValueError(f'quant method must be one of {WEIGHT_SCALE_SUPPORTED}')
            return True if return_success else None
        if 'weight_shape' in weight_name:
            self._load_single_value(param=param, loaded_weight=loaded_weight, expert_id=expert_id)
            return True if return_success else None
        if 'weight' in weight_name:
            self._load_model_weight_or_group_weight_scale(shard_id=shard_id, shard_dim=shard_dim, loaded_weight=loaded_weight, expert_data=expert_data, tp_rank=self.tp_rank)
            return True if return_success else None
        return False if return_success else None

    def get_expert_weights(self) -> Iterable[torch.Tensor]:
        weights = list(self.named_parameters())
        assert all((weight.is_contiguous() for _, weight in weights))
        NON_EXPERT_WEIGHTS = {'e_score_correction_bias'}
        return [weight.view(self.local_num_experts, -1) for name, weight in weights if name not in NON_EXPERT_WEIGHTS]

    def set_eplb_state(self, moe_layer_idx: int, expert_load_view: torch.Tensor, logical_to_physical_map: torch.Tensor, logical_replica_count: torch.Tensor) -> None:
        """
        Register the EPLB state in this layer.

        This is used later in forward pass, where we get the expert mapping
        and record the load metrics in `expert_load_view`.
        """
        self.expert_load_view = expert_load_view[moe_layer_idx]
        self.logical_to_physical_map = logical_to_physical_map[moe_layer_idx]
        self.logical_replica_count = logical_replica_count[moe_layer_idx]

    @staticmethod
    def select_experts(hidden_states: torch.Tensor, router_logits: torch.Tensor, top_k: int, use_grouped_topk: bool, renormalize: bool, topk_group: Optional[int]=None, num_expert_group: Optional[int]=None, custom_routing_function: Optional[Callable]=None, scoring_func: str='softmax', e_score_correction_bias: Optional[torch.Tensor]=None, indices_type: Optional[torch.dtype]=None, enable_eplb: bool=False, expert_map: Optional[torch.Tensor]=None, expert_load_view: Optional[torch.Tensor]=None, logical_to_physical_map: Optional[torch.Tensor]=None, logical_replica_count: Optional[torch.Tensor]=None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Route the input hidden states to the top-k experts based on the
        router logits.

        Returns:
            (topk_weights, topk_ids) (tuple[torch.Tensor, torch.Tensor]):
            The weights and *global physical* expert ids of the top-k experts.

            **Compatibility**: When EPLB is not enabled, the returned ids are
            equivalent to global logical ids, so should be compatible with
            plain MoE implementations without redundant experts.
        """
        if use_grouped_topk:
            assert topk_group is not None
            assert num_expert_group is not None
            topk_weights, topk_ids = grouped_topk(hidden_states=hidden_states, gating_output=router_logits, topk=top_k, renormalize=renormalize, num_expert_group=num_expert_group, topk_group=topk_group, scoring_func=scoring_func, e_score_correction_bias=e_score_correction_bias)
            if indices_type is not None:
                topk_ids = topk_ids.to(dtype=indices_type)
        elif custom_routing_function is None:
            topk_weights, topk_ids, token_expert_indices = fused_topk(hidden_states=hidden_states, gating_output=router_logits, topk=top_k, renormalize=renormalize, indices_type=indices_type)
        else:
            topk_weights, topk_ids = custom_routing_function(hidden_states=hidden_states, gating_output=router_logits, topk=top_k, renormalize=renormalize)
            if indices_type is not None:
                topk_ids = topk_ids.to(dtype=indices_type)
        if enable_eplb:
            assert expert_load_view is not None
            assert logical_to_physical_map is not None
            assert logical_replica_count is not None
            topk_ids_long = topk_ids.long()
            replica_indices = (torch.rand_like(topk_ids, dtype=torch.float) * logical_replica_count[topk_ids_long]).long().unsqueeze(-1)
            physical_ids = logical_to_physical_map[topk_ids_long].gather(-1, replica_indices).squeeze(-1)
            topk_ids = physical_ids
            if expert_map is not None:
                topk_ids_local = expert_map[topk_ids]
                topk_ids_flatten = topk_ids_local.flatten()
            else:
                topk_ids_flatten = topk_ids.flatten()
            invalid_mask = topk_ids_flatten < 0
            index = topk_ids_flatten.masked_fill_(invalid_mask, 0)
            src = ~invalid_mask
            expert_load_view.scatter_add_(dim=0, index=index.long(), src=src.to(expert_load_view))
            topk_ids = topk_ids.to(dtype=indices_type)
        assert topk_ids.dtype == indices_type or indices_type is None
        return (topk_weights, topk_ids)

    def must_reduce_shared_expert_outputs(self) -> bool:
        """
        The shared_experts are typically computed using the RowParallelLinear
        layer. The result of this function is typically used as
        the reduce_results argument to the module.
        When just tensor-parallel is used, it is not required to reduce
        the shared_experts results immediately. Instead we reduce at the
        once at the end of the MoE op. (Refer to DeepSeekV2MoE module)
        With EP and all2all kernels - this is no longer viable as all
        GPU ranks in DP, produce the complete set of hidden_states.
        Therefore it is required that we reduce the shared_experts output
        early.
        """
        return self.use_pplx_kernels or self.use_deepep_ht_kernels or self.use_deepep_ll_kernels

    def maybe_all_reduce_tensor_model_parallel(self, final_hidden_states: torch.Tensor):
        """
        The pplx combine kernel reduces across GPU ranks by default.
        """
        if self.use_pplx_kernels or self.use_deepep_ht_kernels or self.use_deepep_ll_kernels:
            return final_hidden_states
        else:
            return tensor_model_parallel_all_reduce(final_hidden_states)

    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor):
        if current_platform.is_tpu():
            return self.forward_impl(hidden_states, router_logits)
        else:
            return torch.ops.vllm.moe_forward(hidden_states, router_logits, self.layer_name)

    def forward_impl_chunked(self, full_hidden_states: torch.Tensor, full_router_logits: torch.Tensor):
        assert self.batched_hidden_states is not None
        assert self.batched_router_logits is not None
        assert self.batched_hidden_states.dtype == full_hidden_states.dtype
        assert self.batched_router_logits.dtype == full_router_logits.dtype
        assert self.batched_hidden_states.size(-1) == full_hidden_states.size(-1)
        assert self.batched_router_logits.size(-1) == full_router_logits.size(-1)
        full_final_hidden_states = torch.empty_like(full_hidden_states)

        def process_chunk(chunk_start, chunk_end, skip_result_store=False):
            chunk_size = chunk_end - chunk_start
            hidden_states = full_hidden_states[chunk_start:chunk_end, :]
            router_logits = full_router_logits[chunk_start:chunk_end, :]
            assert self.batched_hidden_states.size(0) >= chunk_size
            assert self.batched_router_logits.size(0) >= chunk_size
            staged_hidden_states = self.batched_hidden_states[:chunk_size, :]
            staged_router_logits = self.batched_router_logits[:chunk_size, :]
            staged_hidden_states.copy_(hidden_states, non_blocking=True)
            staged_router_logits.copy_(router_logits, non_blocking=True)
            final_hidden_states = self.quant_method.apply(layer=self, x=staged_hidden_states, router_logits=staged_router_logits, top_k=self.top_k, renormalize=self.renormalize, use_grouped_topk=self.use_grouped_topk, global_num_experts=self.global_num_experts, expert_map=self.expert_map, topk_group=self.topk_group, num_expert_group=self.num_expert_group, custom_routing_function=self.custom_routing_function, scoring_func=self.scoring_func, e_score_correction_bias=self.e_score_correction_bias, activation=self.activation, enable_eplb=self.enable_eplb, expert_load_view=self.expert_load_view, logical_to_physical_map=self.logical_to_physical_map, logical_replica_count=self.logical_replica_count)
            if not skip_result_store:
                full_final_hidden_states[chunk_start:chunk_end, :].copy_(final_hidden_states, non_blocking=True)
        ctx = get_forward_context()
        max_tokens_across_dp = ctx.dp_metadata.max_tokens_across_dp_cpu
        moe_dp_chunk_size_per_rank = self.moe_config.max_num_tokens
        num_tokens = full_hidden_states.size(0)
        for chunk_start_ in range(0, max_tokens_across_dp, moe_dp_chunk_size_per_rank):
            chunk_start = chunk_start_
            chunk_end = min(chunk_start + moe_dp_chunk_size_per_rank, max_tokens_across_dp)
            chunk_start = min(chunk_start, num_tokens - 1)
            chunk_end = min(chunk_end, num_tokens)
            process_chunk(chunk_start, chunk_end, skip_result_store=chunk_start_ >= num_tokens)
        return full_final_hidden_states

    def forward_impl(self, hidden_states: torch.Tensor, router_logits: torch.Tensor):
        assert self.quant_method is not None
        use_flashinfer_cutlass_kernels = self.dp_size > 1 and self.moe_parallel_config.use_flashinfer_cutlass_kernels
        if self.moe_parallel_config.use_pplx_kernels or self.moe_parallel_config.use_deepep_ll_kernels or use_flashinfer_cutlass_kernels:
            return self.forward_impl_chunked(hidden_states, router_logits)
        do_naive_dispatch_combine: bool = self.dp_size > 1 and (not self.moe_parallel_config.use_deepep_ht_kernels) and (not self.moe_parallel_config.use_flashinfer_cutlass_kernels)
        if do_naive_dispatch_combine:
            hidden_states, router_logits = get_ep_group().dispatch(hidden_states, router_logits)
        final_hidden_states = self.quant_method.apply(layer=self, x=hidden_states, router_logits=router_logits, top_k=self.top_k, renormalize=self.renormalize, use_grouped_topk=self.use_grouped_topk, global_num_experts=self.global_num_experts, expert_map=self.expert_map, topk_group=self.topk_group, num_expert_group=self.num_expert_group, custom_routing_function=self.custom_routing_function, scoring_func=self.scoring_func, e_score_correction_bias=self.e_score_correction_bias, activation=self.activation, apply_router_weight_on_input=self.apply_router_weight_on_input, enable_eplb=self.enable_eplb, expert_load_view=self.expert_load_view, logical_to_physical_map=self.logical_to_physical_map, logical_replica_count=self.logical_replica_count)
        if do_naive_dispatch_combine:
            final_hidden_states = get_ep_group().combine(final_hidden_states)
        if self.reduce_results and (self.tp_size > 1 or self.ep_size > 1):
            final_hidden_states = self.maybe_all_reduce_tensor_model_parallel(final_hidden_states)
        return final_hidden_states

    @classmethod
    def make_expert_params_mapping(cls, ckpt_gate_proj_name: str, ckpt_down_proj_name: str, ckpt_up_proj_name: str, num_experts: int, num_redundant_experts: int=0) -> list[tuple[str, str, int, str]]:
        num_physical_experts = num_experts + num_redundant_experts
        physical_to_logical_map = EplbState.build_initial_global_physical_to_logical_map(num_experts, num_redundant_experts)
        return [('experts.w13_' if weight_name in [ckpt_gate_proj_name, ckpt_up_proj_name] else 'experts.w2_', f'experts.{physical_to_logical_map[expert_id]}.{weight_name}.', expert_id, shard_id) for expert_id in range(num_physical_experts) for shard_id, weight_name in [('w1', ckpt_gate_proj_name), ('w2', ckpt_down_proj_name), ('w3', ckpt_up_proj_name)]]

    def extra_repr(self) -> str:
        s = f'global_num_experts={self.global_num_experts}, local_num_experts={self.local_num_experts}, top_k={self.top_k}, intermediate_size_per_partition={self.intermediate_size_per_partition}, tp_size={self.tp_size},\nep_size={self.ep_size}, reduce_results={self.reduce_results}, renormalize={self.renormalize}, use_grouped_topk={self.use_grouped_topk}'
        if self.use_grouped_topk:
            s += f', num_expert_group={self.num_expert_group}, topk_group={self.topk_group}'
        s += f", scoring_func='{self.scoring_func}', activation='{self.activation}'"
        return s

def moe_forward(hidden_states: torch.Tensor, router_logits: torch.Tensor, layer_name: str) -> torch.Tensor:
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    assert self.quant_method is not None
    return self.forward_impl(hidden_states, router_logits)

def moe_forward_fake(hidden_states: torch.Tensor, router_logits: torch.Tensor, layer_name: str) -> torch.Tensor:
    return torch.empty_like(hidden_states)
direct_register_custom_op(op_name='moe_forward', op_func=moe_forward, mutates_args=['hidden_states'], fake_impl=moe_forward_fake, dispatch_key=current_platform.dispatch_key, tags=(torch.Tag.needs_fixed_stride_order,))
FusedMoE.weight_loader.supports_moe_loading = True