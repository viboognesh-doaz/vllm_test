from importlib.util import find_spec
from torch_xla.experimental.custom_kernel import flash_attention
from typing import List, Optional
from vllm.attention import AttentionType
from vllm.attention.selector import backend_name_to_enum, get_attn_backend
from vllm.attention.utils.kv_sharing_utils import validate_kv_sharing_target
from vllm.config import CacheConfig, get_current_vllm_config
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group, is_v1_kv_transfer_group
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
from vllm.platforms import _Backend, current_platform
from vllm.utils import direct_register_custom_op
from xformers import ops as xops
import torch
import torch.nn as nn
import torch.nn.functional as F
import vllm.envs as envs
'Attention layer.'
logger = init_logger(__name__)
USE_XFORMERS_OPS = None

def check_xformers_availability():
    global USE_XFORMERS_OPS
    if USE_XFORMERS_OPS is not None:
        return USE_XFORMERS_OPS
    if current_platform.is_cuda() and current_platform.has_device_capability(100):
        USE_XFORMERS_OPS = False
    else:
        try:
            find_spec('xformers.ops')
            USE_XFORMERS_OPS = True
        except ImportError:
            USE_XFORMERS_OPS = False
    if not USE_XFORMERS_OPS:
        logger.warning('Xformers is not available, falling back.')
    return USE_XFORMERS_OPS

class Attention(nn.Module):
    """Attention layer.

    This class takes query, key, and value tensors as input. The input tensors
    can either contain prompt tokens or generation tokens.
    The class does the following:

    1. Store the input key and value tensors in the KV cache.
    2. Perform (multi-head/multi-query/grouped-query) attention.
    3. Return the output tensor.
    """

    def __init__(self, num_heads: int, head_size: int, scale: float, num_kv_heads: Optional[int]=None, alibi_slopes: Optional[List[float]]=None, cache_config: Optional[CacheConfig]=None, quant_config: Optional[QuantizationConfig]=None, logits_soft_cap: Optional[float]=None, per_layer_sliding_window: Optional[int]=None, use_mla: bool=False, prefix: str='', attn_type: str=AttentionType.DECODER, kv_sharing_target_layer_name: Optional[str]=None, **extra_impl_args) -> None:
        """
        The KV cache is stored inside this class and is accessed via
        `self.kv_cache`.
        """
        super().__init__()
        if per_layer_sliding_window is not None:
            sliding_window = per_layer_sliding_window
        elif cache_config is not None:
            sliding_window = cache_config.sliding_window
        else:
            sliding_window = None
        if cache_config is not None:
            kv_cache_dtype = cache_config.cache_dtype
            block_size = cache_config.block_size
            is_attention_free = cache_config.is_attention_free
            calculate_kv_scales = cache_config.calculate_kv_scales
        else:
            kv_cache_dtype = 'auto'
            block_size = 16
            is_attention_free = False
            calculate_kv_scales = False
        if num_kv_heads is None:
            num_kv_heads = num_heads
        assert num_heads % num_kv_heads == 0, f'num_heads ({num_heads}) is not divisible by num_kv_heads ({num_kv_heads})'
        self.kv_cache_dtype = kv_cache_dtype
        self.calculate_kv_scales = calculate_kv_scales
        self._k_scale = torch.tensor(1.0, dtype=torch.float32)
        self._v_scale = torch.tensor(1.0, dtype=torch.float32)
        self._q_scale = torch.tensor(1.0, dtype=torch.float32)
        self._prob_scale = torch.tensor(1.0, dtype=torch.float32)
        self._k_scale_float = 1.0
        self._v_scale_float = 1.0
        self.use_mla = use_mla
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_heads = num_kv_heads
        self.sliding_window = sliding_window
        quant_method = quant_config.get_quant_method(self, prefix=prefix) if quant_config else None
        if quant_method is not None and (not isinstance(quant_method, UnquantizedLinearMethod)):
            assert isinstance(quant_method, BaseKVCacheMethod)
            if self.kv_cache_dtype == 'fp8_e5m2':
                raise ValueError('fp8_e5m2 kv-cache is not supported with fp8 checkpoints.')
            self.quant_method = quant_method
            self.quant_method.create_weights(self)
        dtype = torch.get_default_dtype()
        attn_backend = get_attn_backend(head_size, dtype, kv_cache_dtype, block_size, is_attention_free, use_mla=use_mla)
        impl_cls = attn_backend.get_impl_cls()
        self.impl = impl_cls(num_heads, head_size, scale, num_kv_heads, alibi_slopes, sliding_window, kv_cache_dtype, logits_soft_cap, attn_type, kv_sharing_target_layer_name, **extra_impl_args)
        self.backend = backend_name_to_enum(attn_backend.get_name())
        self.dtype = dtype
        self.use_irope = extra_impl_args.get('use_irope', False)
        self.use_direct_call = not current_platform.is_cuda_alike() and (not current_platform.is_cpu())
        self.use_output = attn_backend.accept_output_buffer
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f'Duplicate layer name: {prefix}')
        compilation_config.static_forward_context[prefix] = self
        self.layer_name = prefix
        self.attn_type = attn_type
        if kv_sharing_target_layer_name is not None:
            validate_kv_sharing_target(prefix, kv_sharing_target_layer_name, compilation_config.static_forward_context)
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        self.kv_cache = [torch.tensor([]) for _ in range(get_current_vllm_config().parallel_config.pipeline_parallel_size)]
        self.q_range = torch.tensor(envs.Q_SCALE_CONSTANT, dtype=torch.float32)
        self.k_range = torch.tensor(envs.K_SCALE_CONSTANT, dtype=torch.float32)
        self.v_range = torch.tensor(envs.V_SCALE_CONSTANT, dtype=torch.float32)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, output_shape: Optional[torch.Size]=None) -> torch.Tensor:
        """
        The KV cache is stored inside this class and is accessed via
        `self.kv_cache`.

        Attention metadata (`attn_metadata`) is set using a context manager in
        the model runner's `execute_model` method. It is accessed via forward
        context using
        `vllm.forward_context.get_forward_context().attn_metadata`.
        """
        if self.calculate_kv_scales:
            attn_metadata = get_forward_context().attn_metadata
            if attn_metadata.enable_kv_scales_calculation:
                self.calc_kv_scales(query, key, value)
        if self.use_output:
            output_shape = output_shape if output_shape is not None else query.shape
            output = torch.zeros(output_shape, dtype=query.dtype, device=query.device)
            hidden_size = output_shape[-1]
            if not self.use_mla:
                query = query.view(-1, self.num_heads, self.head_size)
                output = output.view(-1, self.num_heads, self.head_size)
                if key is not None:
                    key = key.view(-1, self.num_kv_heads, self.head_size)
                if value is not None:
                    value = value.view(-1, self.num_kv_heads, self.head_size)
            if self.use_direct_call:
                forward_context: ForwardContext = get_forward_context()
                attn_metadata = forward_context.attn_metadata
                if isinstance(attn_metadata, dict):
                    attn_metadata = attn_metadata[self.layer_name]
                self_kv_cache = self.kv_cache[forward_context.virtual_engine]
                self.impl.forward(self, query, key, value, self_kv_cache, attn_metadata, output=output)
            else:
                torch.ops.vllm.unified_attention_with_output(query, key, value, output, self.layer_name)
            return output.view(-1, hidden_size)
        elif self.use_direct_call:
            forward_context = get_forward_context()
            attn_metadata = forward_context.attn_metadata
            if isinstance(attn_metadata, dict):
                attn_metadata = attn_metadata[self.layer_name]
            self_kv_cache = self.kv_cache[forward_context.virtual_engine]
            return self.impl.forward(self, query, key, value, self_kv_cache, attn_metadata)
        else:
            return torch.ops.vllm.unified_attention(query, key, value, self.layer_name)

    def calc_kv_scales(self, query, key, value):
        self._q_scale.copy_(torch.abs(query).max() / self.q_range)
        self._k_scale.copy_(torch.abs(key).max() / self.k_range)
        self._v_scale.copy_(torch.abs(value).max() / self.v_range)
        self._k_scale_float = self._k_scale.item()
        self._v_scale_float = self._v_scale.item()
        self.calculate_kv_scales = False

    def extra_repr(self) -> str:
        s = f'head_size={self.impl.head_size}'
        s += f', num_heads={self.impl.num_heads}'
        s += f', num_kv_heads={self.impl.num_kv_heads}'
        s += f', scale={self.impl.scale}'
        s += f', backend={self.impl.__class__.__name__}'
        return s

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        if hasattr(self.impl, 'process_weights_after_loading'):
            self.impl.process_weights_after_loading(act_dtype)

class MultiHeadAttention(nn.Module):
    """Multi-headed attention without any cache, used for ViT."""

    def __init__(self, num_heads: int, head_size: int, scale: float, num_kv_heads: Optional[int]=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        assert self.num_heads % self.num_kv_heads == 0, f'num_heads ({self.num_heads}) is not divisible by num_kv_heads ({self.num_kv_heads})'
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        dtype = torch.get_default_dtype()
        attn_backend = get_attn_backend(head_size, dtype, kv_cache_dtype=None, block_size=16, is_attention_free=False)
        backend = backend_name_to_enum(attn_backend.get_name())
        if current_platform.is_rocm():
            self.attn_backend = _Backend.TORCH_SDPA
        else:
            if backend in (_Backend.FLASH_ATTN, _Backend.FLASH_ATTN_VLLM_V1, _Backend.FLEX_ATTENTION):
                backend = _Backend.XFORMERS
            self.attn_backend = backend if backend in {_Backend.TORCH_SDPA, _Backend.XFORMERS, _Backend.PALLAS_VLLM_V1} else _Backend.TORCH_SDPA
        if self.attn_backend == _Backend.XFORMERS and (not check_xformers_availability()):
            self.attn_backend = _Backend.TORCH_SDPA

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """Input shape: batch_size x seq_len x hidden_size"""
        bsz, q_len, _ = query.size()
        kv_len = key.size(1)
        query = query.view(bsz, q_len, self.num_heads, self.head_size)
        key = key.view(bsz, kv_len, self.num_kv_heads, self.head_size)
        value = value.view(bsz, kv_len, self.num_kv_heads, self.head_size)
        if (num_repeat := self.num_queries_per_kv) > 1:
            key = torch.repeat_interleave(key, num_repeat, dim=2)
            value = torch.repeat_interleave(value, num_repeat, dim=2)
        if self.attn_backend == _Backend.XFORMERS:
            out = xops.memory_efficient_attention_forward(query, key, value, scale=self.scale)
        elif self.attn_backend == _Backend.TORCH_SDPA:
            query, key, value = (x.transpose(1, 2) for x in (query, key, value))
            out = F.scaled_dot_product_attention(query, key, value, scale=self.scale)
            out = out.transpose(1, 2)
        elif self.attn_backend == _Backend.PALLAS_VLLM_V1:
            query, key, value = (x.transpose(1, 2) for x in (query, key, value))
            out = flash_attention(query, key, value, sm_scale=self.scale)
            out = out.transpose(1, 2)
        return out.reshape(bsz, q_len, -1)

def wait_for_kv_layer_from_connector(layer_name: str):
    if not has_kv_transfer_group() or not is_v1_kv_transfer_group():
        return
    connector = get_kv_transfer_group()
    forward_context: ForwardContext = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    if attn_metadata is None:
        return
    assert isinstance(attn_metadata, dict)
    connector.wait_for_layer_load(layer_name)

def maybe_save_kv_layer_to_connector(layer_name: str, kv_cache_layer: List[torch.Tensor]):
    if not has_kv_transfer_group() or not is_v1_kv_transfer_group():
        return
    connector = get_kv_transfer_group()
    forward_context: ForwardContext = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    if attn_metadata is None:
        return
    assert isinstance(attn_metadata, dict)
    connector.save_kv_layer(layer_name, kv_cache_layer, attn_metadata[layer_name])

def unified_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, layer_name: str) -> torch.Tensor:
    wait_for_kv_layer_from_connector(layer_name)
    forward_context: ForwardContext = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    if isinstance(attn_metadata, dict):
        attn_metadata = attn_metadata[layer_name]
    self = forward_context.no_compile_layers[layer_name]
    kv_cache = self.kv_cache[forward_context.virtual_engine]
    output = self.impl.forward(self, query, key, value, kv_cache, attn_metadata)
    maybe_save_kv_layer_to_connector(layer_name, kv_cache)
    return output

def unified_attention_fake(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, layer_name: str) -> torch.Tensor:
    return torch.empty_like(query).contiguous()
direct_register_custom_op(op_name='unified_attention', op_func=unified_attention, mutates_args=[], fake_impl=unified_attention_fake, dispatch_key=current_platform.dispatch_key)

def unified_attention_with_output(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, output: torch.Tensor, layer_name: str, output_scale: Optional[torch.Tensor]=None) -> None:
    wait_for_kv_layer_from_connector(layer_name)
    forward_context: ForwardContext = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    if isinstance(attn_metadata, dict):
        attn_metadata = attn_metadata[layer_name]
    self = forward_context.no_compile_layers[layer_name]
    kv_cache = self.kv_cache[forward_context.virtual_engine]
    self.impl.forward(self, query, key, value, kv_cache, attn_metadata, output=output, output_scale=output_scale)
    maybe_save_kv_layer_to_connector(layer_name, kv_cache)

def unified_attention_with_output_fake(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, output: torch.Tensor, layer_name: str, output_scale: Optional[torch.Tensor]=None) -> None:
    return
direct_register_custom_op(op_name='unified_attention_with_output', op_func=unified_attention_with_output, mutates_args=['output'], fake_impl=unified_attention_with_output_fake, dispatch_key=current_platform.dispatch_key)