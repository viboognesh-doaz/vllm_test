from vllm import _custom_ops as ops
from typing import Optional, Union
import torch
from torch import nn
from vllm import envs
from vllm.attention.backends.abstract import AttentionMetadata
from vllm.config import get_current_vllm_config
from vllm.distributed import divide, get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size, tensor_model_parallel_all_gather, tensor_model_parallel_all_reduce
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba2_metadata import Mamba2Metadata, update_metadata
from vllm.model_executor.layers.mamba.mamba_utils import extra_groups_for_head_shards, get_mamba_state_shape
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from vllm.model_executor.layers.mamba.ops.mamba_ssm import selective_state_update
from vllm.model_executor.layers.mamba.ops.ssd_combined import mamba_chunk_scan_combined
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import LoaderFunction, composed_weight_loader, sharded_weight_loader
from vllm.model_executor.models.mamba_cache import MambaCacheParams
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.utils import direct_register_custom_op
from vllm.v1.attention.backends.mamba_attn import Mamba2AttentionMetadata

@CustomOp.register('mixer2_gated_rms_norm')
class Mixer2RMSNormGated(CustomOp):

    def __init__(self, full_hidden_size: int, full_n_groups: int, use_rms_norm: bool=True, eps: float=1e-06):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.full_hidden_size = full_hidden_size
        self.group_size = full_hidden_size // full_n_groups
        self.per_rank_hidden_size = full_hidden_size // self.tp_size
        self.n_groups = full_hidden_size // self.group_size
        self.variance_epsilon = eps
        self.use_rms_norm = use_rms_norm
        if self.use_rms_norm:
            self.weight = nn.Parameter(torch.ones(self.per_rank_hidden_size))
            set_weight_attrs(self.weight, {'weight_loader': sharded_weight_loader(0)})
        else:
            self.register_parameter('weight', None)
        assert self.full_hidden_size % self.tp_size == 0, 'Tensor parallel world size must divide hidden size.'

    def forward_native(self, x: torch.Tensor, gate: torch.Tensor):
        input_dtype = x.dtype
        x = x * nn.functional.silu(gate.to(torch.float32))
        if not self.use_rms_norm:
            return x.to(input_dtype)
        if self.n_groups == 1:
            if self.tp_size > 1:
                local_sums = x.pow(2).sum(dim=-1, keepdim=True)
                global_sums = tensor_model_parallel_all_reduce(local_sums)
                count = self.tp_size * x.shape[-1]
                variance = global_sums / count
            else:
                variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.variance_epsilon)
        else:
            redundant_tp: bool = self.n_groups % self.tp_size != 0
            if redundant_tp:
                x = tensor_model_parallel_all_gather(x, -1)
            *prefix_dims, hidden_dim = x.shape
            group_count = hidden_dim // self.group_size
            x_grouped = x.view(*prefix_dims, group_count, self.group_size)
            variance = x_grouped.pow(2).mean(-1, keepdim=True)
            x_grouped = x_grouped * torch.rsqrt(variance + self.variance_epsilon)
            x = x_grouped.view(*prefix_dims, hidden_dim)
            if redundant_tp:
                start = self.per_rank_hidden_size * self.tp_rank
                end = start + self.per_rank_hidden_size
                x = x[..., start:end]
        return self.weight * x.to(input_dtype)

    def forward_cuda(self, x: torch.Tensor, gate: torch.Tensor) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        input_dtype = x.dtype
        if not self.use_rms_norm:
            return x * nn.functional.silu(gate.to(torch.float32)).to(input_dtype)
        if self.tp_size > 1 or self.n_groups != 1:
            return self.forward_native(x, gate)
        out = torch.empty_like(x)
        y = x * nn.functional.silu(gate.to(torch.float32))
        ops.rms_norm(out, y.to(x.dtype), self.weight.data, self.variance_epsilon)
        return out

def mamba_v2_sharded_weight_loader(shard_spec: list[tuple[int, int, float]], tp_size: int, tp_rank: int) -> LoaderFunction:
    """Create a weight loader for mamba v2. This ensures that the projections
    are correctly sharded so that they can be split into x, B, C. It also
    ensures that all the groups corresponding to a head shard is placed
    together with it.
    """

    def loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        boundary, loaded_boundary = (0, 0)
        for full_dim, extra, duplicate_groups in shard_spec:
            shard_size = full_dim // tp_size
            rank = 0 if duplicate_groups else tp_rank
            loaded_skip = rank * shard_size
            loaded_start_idx = loaded_boundary + loaded_skip
            take = min(shard_size, full_dim - extra - loaded_skip)
            param.data[boundary:boundary + take, ...] = loaded_weight[loaded_start_idx:loaded_start_idx + take]
            boundary += shard_size
            loaded_boundary += full_dim - extra
    return loader

@CustomOp.register('mamba_mixer2')
class MambaMixer2(MambaBase, CustomOp):
    """
    Compute ∆, A, B, C, and D the state space parameters and compute
    the `contextualized_states`. A, D are input independent
    (see Mamba paper [1] Section 3.5.2 "Interpretation of A"
    for why A isn't selective) ∆, B, C are input-dependent
    (this is a key difference between Mamba and the linear time
    invariant S4, and is why Mamba is called
    **selective** state spaces)
    """

    def __init__(self, hidden_size: int, ssm_state_size: int, conv_kernel_size: int, intermediate_size: int, use_conv_bias: bool, use_bias: bool, n_groups: int=1, num_heads: int=128, head_dim: int=64, rms_norm_eps: float=1e-05, activation: str='silu', use_rms_norm: bool=True, quant_config: Optional[QuantizationConfig]=None, prefix: str=''):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        assert num_heads % self.tp_size == 0, 'Tensor parallel world size must divide num heads.'
        assert n_groups % self.tp_size == 0 or n_groups == 1, 'If tensor parallel world size does not divide num_heads, then num_groups must equal 1.'
        assert self.tp_size == 1 or quant_config is None, 'Tensor parallel currently not supported for quantized models.'
        self.ssm_state_size = ssm_state_size
        self.conv_kernel_size = conv_kernel_size
        self.activation = activation
        self.intermediate_size = intermediate_size
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.n_groups = n_groups
        if n_groups % self.tp_size != 0:
            self.n_groups = n_groups + extra_groups_for_head_shards(n_groups, self.tp_size)
        self.conv_dim = intermediate_size + 2 * self.n_groups * ssm_state_size
        self.conv1d = ColumnParallelLinear(input_size=conv_kernel_size, output_size=self.conv_dim, bias=use_conv_bias, quant_config=None)
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)
        self.in_proj = ColumnParallelLinear(input_size=hidden_size, output_size=intermediate_size + self.conv_dim + self.num_heads, bias=use_bias, quant_config=quant_config)
        group_shard_settings = (self.n_groups * self.ssm_state_size, (self.n_groups - n_groups) * self.ssm_state_size, n_groups == 1)
        intermediate_settings = (intermediate_size, 0, False)
        head_settings = (self.num_heads, 0, False)
        delattr(self.conv1d.bias, 'weight_loader')
        set_weight_attrs(self.conv1d.bias, {'weight_loader': mamba_v2_sharded_weight_loader([intermediate_settings, group_shard_settings, group_shard_settings], self.tp_size, tp_rank)})
        delattr(self.conv1d.weight, 'weight_loader')
        set_weight_attrs(self.conv1d.weight, {'weight_loader': mamba_v2_sharded_weight_loader([intermediate_settings, group_shard_settings, group_shard_settings], self.tp_size, tp_rank)})
        if quant_config is None:
            delattr(self.in_proj.weight, 'weight_loader')
            set_weight_attrs(self.in_proj.weight, {'weight_loader': mamba_v2_sharded_weight_loader([intermediate_settings, intermediate_settings, group_shard_settings, group_shard_settings, head_settings], self.tp_size, tp_rank)})
        self.A = nn.Parameter(torch.empty(divide(num_heads, self.tp_size), dtype=torch.float32))
        self.D = nn.Parameter(torch.ones(num_heads // self.tp_size))
        self.dt_bias = nn.Parameter(torch.ones(num_heads // self.tp_size))
        self.use_rms_norm = use_rms_norm
        set_weight_attrs(self.D, {'weight_loader': sharded_weight_loader(0)})
        a_weight_loader = composed_weight_loader(sharded_weight_loader(0), lambda x: -torch.exp(x.float()))
        set_weight_attrs(self.A, {'weight_loader': a_weight_loader})
        set_weight_attrs(self.dt_bias, {'weight_loader': sharded_weight_loader(0)})
        self.out_proj = RowParallelLinear(intermediate_size, hidden_size, bias=use_bias, input_is_parallel=True, quant_config=quant_config)
        self.norm = Mixer2RMSNormGated(intermediate_size, n_groups, self.use_rms_norm, eps=rms_norm_eps)
        if envs.VLLM_USE_V1:
            compilation_config = get_current_vllm_config().compilation_config
            if prefix in compilation_config.static_forward_context:
                raise ValueError(f'Duplicate layer name: {prefix}')
            compilation_config.static_forward_context[prefix] = self
            self.kv_cache = [(torch.tensor([]), torch.tensor([]))]
        self.prefix = prefix

    def forward_native(self, hidden_states: torch.Tensor, output: torch.Tensor, mamba_cache_params: MambaCacheParams, mamba2_metadata: Mamba2Metadata, mup_vector: Optional[torch.Tensor]=None):
        pass

    def forward(self, hidden_states: torch.Tensor, output: torch.Tensor, mamba_cache_params: MambaCacheParams, mamba2_metadata: Mamba2Metadata, mup_vector: Optional[torch.Tensor]=None):
        if not envs.VLLM_USE_V1:
            CustomOp.forward(self, hidden_states, output, mamba_cache_params, mamba2_metadata, mup_vector)
        else:
            torch.ops.vllm.mamba_mixer2(hidden_states, output, self.prefix, mup_vector)

    def forward_cuda(self, hidden_states: torch.Tensor, output: torch.Tensor, mamba_cache_params: MambaCacheParams, mamba2_metadata: Mamba2Metadata, mup_vector: Optional[torch.Tensor]=None):
        forward_context = get_forward_context()
        attn_metadata: AttentionMetadata = forward_context.attn_metadata
        if envs.VLLM_USE_V1:
            if attn_metadata is not None:
                assert isinstance(attn_metadata, dict)
                attn_metadata = attn_metadata[self.prefix]
                mamba2_metadata = attn_metadata
                assert isinstance(attn_metadata, Mamba2AttentionMetadata)
                self_kv_cache = self.kv_cache[forward_context.virtual_engine]
                conv_state = self_kv_cache[0].transpose(-1, -2)
                ssm_state = self_kv_cache[1]
                state_indices_tensor = attn_metadata.state_indices_tensor
                has_initial_states_p = attn_metadata.has_initial_states
                prep_initial_states = attn_metadata.prep_initial_states
                chunk_size = attn_metadata.chunk_size
                seq_idx_p = attn_metadata.seq_idx
                chunk_indices_p = attn_metadata.chunk_indices
                chunk_offsets_p = attn_metadata.chunk_offsets
        else:
            conv_state = mamba_cache_params.conv_state
            ssm_state = mamba_cache_params.ssm_state
            state_indices_tensor = mamba_cache_params.state_indices_tensor
            has_initial_states_p = mamba2_metadata.has_initial_states
            prep_initial_states = mamba2_metadata.prep_initial_states
            chunk_size = mamba2_metadata.chunk_size
            seq_idx_p = mamba2_metadata.seq_idx
            chunk_indices_p = mamba2_metadata.chunk_indices
            chunk_offsets_p = mamba2_metadata.chunk_offsets
        groups_time_state_size = self.n_groups * self.ssm_state_size
        projected_states, _ = self.in_proj(hidden_states)
        if mup_vector is not None:
            projected_states = projected_states * mup_vector
        gate, hidden_states_B_C, dt = torch.split(projected_states, [self.intermediate_size // self.tp_size, self.conv_dim // self.tp_size, self.num_heads // self.tp_size], dim=-1)
        conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
        split_hidden_states_B_C_fn = lambda hidden_states_B_C: torch.split(hidden_states_B_C, [self.intermediate_size // self.tp_size, groups_time_state_size // self.tp_size, groups_time_state_size // self.tp_size], dim=-1)
        if envs.VLLM_USE_V1 and attn_metadata is None:
            hidden_states_B_C = hidden_states_B_C.transpose(0, 1).clone().transpose(0, 1).contiguous()
            hidden_states, _B, _C = split_hidden_states_B_C_fn(hidden_states_B_C)
            hidden_states = self.norm(hidden_states, gate)
            out, _ = self.out_proj(hidden_states)
            return out
        num_prefills = attn_metadata.num_prefills
        num_decodes = attn_metadata.num_decode_tokens
        num_prefill_tokens = attn_metadata.num_prefill_tokens
        has_prefill = num_prefills > 0
        has_decode = num_decodes > 0
        num_actual_tokens = num_prefill_tokens + num_decodes
        if envs.VLLM_USE_V1:
            hidden_states_B_C_d, hidden_states_B_C_p = torch.split(hidden_states_B_C[:num_actual_tokens], [num_decodes, num_prefill_tokens], dim=0)
            dt_d, dt_p = torch.split(dt[:num_actual_tokens], [num_decodes, num_prefill_tokens], dim=0)
            state_indices_tensor_d, state_indices_tensor_p = torch.split(state_indices_tensor[:num_actual_tokens], [num_decodes, num_prefills], dim=0)
            query_start_loc_p = attn_metadata.query_start_loc[-num_prefills - 1:] - num_decodes if has_prefill else None
        else:
            hidden_states_B_C_p, hidden_states_B_C_d = torch.split(hidden_states_B_C, [num_prefill_tokens, num_decodes], dim=0)
            dt_p, dt_d = torch.split(dt, [num_prefill_tokens, num_decodes], dim=0)
            state_indices_tensor_p, state_indices_tensor_d = torch.split(state_indices_tensor, [num_prefills, num_decodes], dim=0)
            query_start_loc_p = attn_metadata.query_start_loc[:num_prefills + 1] if has_prefill else None
        ssd_output_list = []
        if has_prefill:
            x = hidden_states_B_C_p.transpose(0, 1)
            if mamba2_metadata.cu_seqlen is None:
                mamba2_metadata = update_metadata(x, query_start_loc_p, mamba2_metadata)
            hidden_states_B_C_p = causal_conv1d_fn(x, conv_weights, self.conv1d.bias, activation=self.activation, conv_states=conv_state, has_initial_state=has_initial_states_p, cache_indices=state_indices_tensor_p, metadata=mamba2_metadata, query_start_loc=query_start_loc_p).transpose(0, 1)[:num_prefill_tokens]
            hidden_states_p, B_p, C_p = split_hidden_states_B_C_fn(hidden_states_B_C_p)
            initial_states = None
            if has_initial_states_p is not None and prep_initial_states:
                if envs.VLLM_USE_V1:
                    initial_states = torch.where(has_initial_states_p[:, None, None, None], ssm_state[state_indices_tensor_p], 0)
                else:
                    initial_states = torch.where(has_initial_states_p[:num_prefills, None, None, None], ssm_state[state_indices_tensor_p], 0)
            scan_output, varlen_state = mamba_chunk_scan_combined(hidden_states_p.view(1, num_prefill_tokens, self.num_heads // self.tp_size, self.head_dim), dt_p.unsqueeze(0), self.A, B_p.view(1, num_prefill_tokens, self.n_groups // self.tp_size, -1), C_p.view(1, num_prefill_tokens, self.n_groups // self.tp_size, -1), chunk_size=chunk_size, D=self.D, z=None, dt_bias=self.dt_bias, seq_idx=seq_idx_p, chunk_indices=chunk_indices_p, chunk_offsets=chunk_offsets_p, cu_seqlens=query_start_loc_p, initial_states=initial_states, return_varlen_states=True, return_final_states=False, dt_softplus=True, dt_limit=(0.0, float('inf')))
            ssm_state[state_indices_tensor_p] = varlen_state
            ssd_output_list.append(scan_output.view(num_prefill_tokens, -1))
        if has_decode:
            hidden_states_B_C_d = causal_conv1d_update(hidden_states_B_C_d, conv_state, conv_weights, self.conv1d.bias, self.activation, conv_state_indices=state_indices_tensor_d)
            hidden_states_d, B_d, C_d = split_hidden_states_B_C_fn(hidden_states_B_C_d)
            n_groups = self.n_groups // self.tp_size
            A_d = self.A[:, None, ...][:, :, None].expand(-1, self.head_dim, self.ssm_state_size).to(dtype=torch.float32)
            dt_d = dt_d[:, :, None].expand(-1, -1, self.head_dim)
            dt_bias = self.dt_bias[:, None, ...].expand(-1, self.head_dim)
            D_d = self.D[:, None, ...].expand(-1, self.head_dim)
            B_d = B_d.view(-1, n_groups, B_d.shape[1] // n_groups)
            C_d = C_d.view(-1, n_groups, C_d.shape[1] // n_groups)
            hidden_states_d = hidden_states_d.view(-1, self.num_heads // self.tp_size, self.head_dim)
            hidden_states_d = selective_state_update(ssm_state, hidden_states_d, dt_d, A_d, B_d, C_d, D_d, z=None, dt_bias=dt_bias, dt_softplus=True, state_batch_indices=state_indices_tensor_d)
            if envs.VLLM_USE_V1:
                ssd_output_list.insert(0, hidden_states_d.view(-1, self.num_heads // self.tp_size * self.head_dim))
            else:
                ssd_output_list.append(hidden_states_d.view(-1, self.num_heads // self.tp_size * self.head_dim))
        hidden_states = torch.vstack(ssd_output_list)
        hidden_states = self.norm(hidden_states, gate[:num_actual_tokens])
        output[:num_actual_tokens], _ = self.out_proj(hidden_states)

    def get_state_shape(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return get_mamba_state_shape(intermediate_size=self.intermediate_size, tp_world_size=get_tensor_model_parallel_world_size(), n_groups=self.n_groups, num_heads=self.num_heads, head_dim=self.head_dim, state_size=self.ssm_state_size, conv_kernel=self.conv_kernel_size)

def mamba_mixer2(hidden_states: torch.Tensor, output: torch.Tensor, layer_name: str, mup_vector: Optional[torch.Tensor]=None) -> None:
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    self.forward_cuda(hidden_states=hidden_states, output=output, mamba_cache_params=None, mamba2_metadata=None, mup_vector=mup_vector)

def mamba_mixer2_fake(hidden_states: torch.Tensor, output: torch.Tensor, layer_name: str, mup_vector: Optional[torch.Tensor]=None) -> None:
    return
direct_register_custom_op(op_name='mamba_mixer2', op_func=mamba_mixer2, mutates_args=['output'], fake_impl=mamba_mixer2_fake, dispatch_key=current_platform.dispatch_key)