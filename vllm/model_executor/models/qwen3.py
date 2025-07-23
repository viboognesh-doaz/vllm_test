from .interfaces import SupportsLoRA, SupportsPP
from .qwen2 import Qwen2MLP as Qwen3MLP
from .qwen2 import Qwen2Model
from .utils import AutoWeightsLoader, PPMissingLayer, maybe_prefix
from collections.abc import Iterable
from torch import nn
from transformers import Qwen3Config
from typing import Optional, Union
from vllm.attention import Attention, AttentionType
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors
import torch
'Inference-only Qwen3 model compatible with HuggingFace weights.'
logger = init_logger(__name__)

class Qwen3Attention(nn.Module):

    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int, max_position: int=4096 * 32, head_dim: Optional[int]=None, rms_norm_eps: float=1e-06, qkv_bias: bool=False, rope_theta: float=10000, cache_config: Optional[CacheConfig]=None, quant_config: Optional[QuantizationConfig]=None, rope_scaling: Optional[tuple]=None, prefix: str='', attn_type: str=AttentionType.DECODER) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** (-0.5)
        self.rope_theta = rope_theta
        self.qkv_proj = QKVParallelLinear(hidden_size, self.head_dim, self.total_num_heads, self.total_num_kv_heads, bias=qkv_bias, quant_config=quant_config, prefix=f'{prefix}.qkv_proj')
        self.o_proj = RowParallelLinear(self.total_num_heads * self.head_dim, hidden_size, bias=False, quant_config=quant_config, prefix=f'{prefix}.o_proj')
        self.rotary_emb = get_rope(self.head_dim, rotary_dim=self.head_dim, max_position=max_position, base=self.rope_theta, rope_scaling=rope_scaling)
        self.attn = Attention(self.num_heads, self.head_dim, self.scaling, num_kv_heads=self.num_kv_heads, cache_config=cache_config, quant_config=quant_config, prefix=f'{prefix}.attn', attn_type=attn_type)
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)
        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output

class Qwen3DecoderLayer(nn.Module):

    def __init__(self, config: Qwen3Config, cache_config: Optional[CacheConfig]=None, quant_config: Optional[QuantizationConfig]=None, prefix: str='') -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, 'rope_theta', 1000000)
        rope_scaling = getattr(config, 'rope_scaling', None)
        if getattr(config, 'is_causal', True):
            attn_type = AttentionType.DECODER
        else:
            attn_type = AttentionType.ENCODER_ONLY
        self.self_attn = Qwen3Attention(hidden_size=self.hidden_size, num_heads=config.num_attention_heads, max_position=config.max_position_embeddings, num_kv_heads=config.num_key_value_heads, rope_theta=rope_theta, rms_norm_eps=config.rms_norm_eps, qkv_bias=getattr(config, 'attention_bias', False), head_dim=getattr(config, 'head_dim', None), cache_config=cache_config, quant_config=quant_config, rope_scaling=rope_scaling, prefix=f'{prefix}.self_attn', attn_type=attn_type)
        self.mlp = Qwen3MLP(hidden_size=self.hidden_size, intermediate_size=config.intermediate_size, hidden_act=config.hidden_act, quant_config=quant_config, prefix=f'{prefix}.mlp')
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor, residual: Optional[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return (hidden_states, residual)
ALL_DECODER_LAYER_TYPES = {'attention': Qwen3DecoderLayer}

@support_torch_compile(dynamic_arg_dims={'input_ids': 0, 'positions': -1, 'intermediate_tensors': 0, 'inputs_embeds': 0})
class Qwen3Model(Qwen2Model):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str=''):
        super().__init__(vllm_config=vllm_config, prefix=prefix, decoder_layer_type=Qwen3DecoderLayer)

class Qwen3ForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
    packed_modules_mapping = {'qkv_proj': ['q_proj', 'k_proj', 'v_proj'], 'gate_up_proj': ['gate_proj', 'up_proj']}

    def __init__(self, *, vllm_config: VllmConfig, prefix: str=''):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config
        self.config = config
        self.lora_config = lora_config
        self.quant_config = quant_config
        self.model = Qwen3Model(vllm_config=vllm_config, prefix=maybe_prefix(prefix, 'model'))
        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size, quant_config=quant_config, prefix=maybe_prefix(prefix, 'lm_head'))
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, intermediate_tensors: Optional[IntermediateTensors]=None, inputs_embeds: Optional[torch.Tensor]=None) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = self.model(input_ids, positions, intermediate_tensors, inputs_embeds)
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor, sampling_metadata: SamplingMetadata) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states, sampling_metadata)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self, skip_prefixes=['lm_head.'] if self.config.tie_word_embeddings else None)
        return loader.load_weights(weights)