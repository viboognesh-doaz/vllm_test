from .interfaces import SupportsLoRA, SupportsPP
from .utils import AutoWeightsLoader, PPMissingLayer, is_pp_missing_parameter, make_empty_intermediate_tensors_factory, make_layers, maybe_prefix
from collections.abc import Iterable
from torch import nn
from typing import Any, Optional, Union
from vllm.attention import Attention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import MergedColumnParallelLinear, QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader, maybe_remap_kv_scale_name
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.exaone import ExaoneConfig
import torch
'Inference-only Exaone model compatible with HuggingFace weights.'

class ExaoneGatedMLP(nn.Module):

    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str, quant_config: Optional[QuantizationConfig]=None, bias: bool=False, prefix: str='') -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(input_size=hidden_size, output_sizes=[intermediate_size] * 2, bias=bias, quant_config=quant_config, prefix=f'{prefix}.gate_up_proj')
        self.c_proj = RowParallelLinear(input_size=intermediate_size, output_size=hidden_size, bias=bias, quant_config=quant_config, prefix=f'{prefix}.c_proj')
        if hidden_act != 'silu':
            raise ValueError(f'Unsupported activation: {hidden_act}. Only silu is supported for now.')
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.c_proj(x)
        return x

class ExaoneAttention(nn.Module):

    def __init__(self, config: ExaoneConfig, hidden_size: int, num_heads: int, num_kv_heads: int, rope_theta: float=10000, rope_scaling: Optional[dict[str, Any]]=None, max_position_embeddings: int=8192, quant_config: Optional[QuantizationConfig]=None, bias: bool=False, cache_config: Optional[CacheConfig]=None, prefix: str='') -> None:
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
        self.head_dim = getattr(config, 'head_dim', None)
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** (-0.5)
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.qkv_proj = QKVParallelLinear(hidden_size=hidden_size, head_size=self.head_dim, total_num_heads=self.total_num_heads, total_num_kv_heads=self.total_num_kv_heads, bias=bias, quant_config=quant_config, prefix=f'{prefix}.qkv_proj')
        self.out_proj = RowParallelLinear(input_size=self.total_num_heads * self.head_dim, output_size=hidden_size, bias=bias, quant_config=quant_config, prefix=f'{prefix}.out_proj')
        is_neox_style = True
        if quant_config is not None and quant_config.get_name() == 'gguf':
            is_neox_style = False
        self.rotary_emb = get_rope(self.head_dim, rotary_dim=self.head_dim, max_position=max_position_embeddings, base=rope_theta, rope_scaling=rope_scaling, is_neox_style=is_neox_style)
        self.attn = Attention(self.num_heads, self.head_dim, self.scaling, num_kv_heads=self.num_kv_heads, cache_config=cache_config, quant_config=quant_config, prefix=f'{prefix}.attn')

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.out_proj(attn_output)
        return output

class ExaoneBlockAttention(nn.Module):

    def __init__(self, config: ExaoneConfig, hidden_size: int, num_heads: int, num_kv_heads: int, rope_theta: float=10000, rope_scaling: Optional[dict[str, Any]]=None, max_position_embeddings: int=8192, quant_config: Optional[QuantizationConfig]=None, bias: bool=False, cache_config: Optional[CacheConfig]=None, prefix: str='') -> None:
        super().__init__()
        self.attention = ExaoneAttention(config=config, hidden_size=hidden_size, num_heads=num_heads, num_kv_heads=num_kv_heads, rope_theta=rope_theta, rope_scaling=rope_scaling, max_position_embeddings=max_position_embeddings, quant_config=quant_config, bias=bias, cache_config=cache_config, prefix=f'{prefix}.attention')

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.attention(positions=positions, hidden_states=hidden_states)

class ExaoneDecoderLayer(nn.Module):

    def __init__(self, config: ExaoneConfig, cache_config: Optional[CacheConfig]=None, quant_config: Optional[QuantizationConfig]=None, prefix: str='') -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, 'rope_theta', 10000)
        rope_scaling = getattr(config, 'rope_scaling', None)
        if rope_scaling is not None and getattr(config, 'original_max_position_embeddings', None):
            rope_scaling['original_max_position_embeddings'] = config.original_max_position_embeddings
        max_position_embeddings = getattr(config, 'max_position_embeddings', 8192)
        attention_bias = getattr(config, 'attention_bias', False) or getattr(config, 'bias', False)
        self.attn = ExaoneBlockAttention(config=config, hidden_size=self.hidden_size, num_heads=config.num_attention_heads, num_kv_heads=getattr(config, 'num_key_value_heads', config.num_attention_heads), rope_theta=rope_theta, rope_scaling=rope_scaling, max_position_embeddings=max_position_embeddings, quant_config=quant_config, bias=attention_bias, cache_config=cache_config, prefix=f'{prefix}.attn')
        self.mlp = ExaoneGatedMLP(hidden_size=self.hidden_size, intermediate_size=config.intermediate_size, hidden_act=config.activation_function, quant_config=quant_config, bias=getattr(config, 'mlp_bias', False), prefix=f'{prefix}.mlp')
        self.ln_1 = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.ln_2 = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor, residual: Optional[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.ln_1(hidden_states)
        else:
            hidden_states, residual = self.ln_1(hidden_states, residual)
        hidden_states = self.attn(positions=positions, hidden_states=hidden_states)
        hidden_states, residual = self.ln_2(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return (hidden_states, residual)

@support_torch_compile
class ExaoneModel(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str=''):
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config
        self.config = config
        self.quant_config = quant_config
        lora_vocab = lora_config.lora_extra_vocab_size * (lora_config.max_loras or 1) if lora_config else 0
        self.vocab_size = config.vocab_size + lora_vocab
        self.wte = config.vocab_size
        if get_pp_group().is_first_rank or (config.tie_word_embeddings and get_pp_group().is_last_rank):
            self.wte = VocabParallelEmbedding(self.vocab_size, config.hidden_size, org_num_embeddings=config.vocab_size, quant_config=quant_config)
        else:
            self.wte = PPMissingLayer()
        self.start_layer, self.end_layer, self.h = make_layers(config.num_hidden_layers, lambda prefix: ExaoneDecoderLayer(config=config, cache_config=cache_config, quant_config=quant_config, prefix=prefix), prefix=f'{prefix}.h')
        if get_pp_group().is_last_rank:
            self.ln_f = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        else:
            self.ln_f = PPMissingLayer()
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(['hidden_states', 'residual'], config.hidden_size)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.wte(input_ids)

    def forward(self, input_ids: Optional[torch.Tensor], positions: torch.Tensor, intermediate_tensors: Optional[IntermediateTensors], inputs_embeds: Optional[torch.Tensor]=None) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors['hidden_states']
            residual = intermediate_tensors['residual']
        for layer in self.h[self.start_layer:self.end_layer]:
            hidden_states, residual = layer(positions, hidden_states, residual)
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({'hidden_states': hidden_states, 'residual': residual})
        hidden_states, _ = self.ln_f(hidden_states, residual)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [('.qkv_proj', '.q_proj', 'q'), ('.qkv_proj', '.k_proj', 'k'), ('.qkv_proj', '.v_proj', 'v'), ('.gate_up_proj', '.c_fc_0', 0), ('.gate_up_proj', '.c_fc_1', 1)]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if 'rotary_emb.inv_freq' in name:
                continue
            if 'rotary_emb.cos_cached' in name or 'rotary_emb.sin_cached' in name:
                continue
            if self.quant_config is not None and (scale_name := self.quant_config.get_cache_scale(name)):
                param = params_dict[scale_name]
                weight_loader = getattr(param, 'weight_loader', default_weight_loader)
                loaded_weight = loaded_weight if loaded_weight.dim() == 0 else loaded_weight[0]
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith('.bias') and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name.endswith('.bias') and name not in params_dict:
                    continue
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, 'weight_loader', default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params

class ExaoneForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
    packed_modules_mapping = {'qkv_proj': ['q_proj', 'k_proj', 'v_proj'], 'gate_up_proj': ['c_fc_0', 'c_fc_1']}
    embedding_modules = {'wte': 'input_embeddings', 'lm_head': 'output_embeddings'}
    embedding_padding_modules = ['lm_head']

    def __init__(self, *, vllm_config: VllmConfig, prefix: str=''):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config
        self.config = config
        self.lora_config = lora_config
        self.quant_config = quant_config
        self.transformer = ExaoneModel(vllm_config=vllm_config, prefix=maybe_prefix(prefix, 'model'))
        if get_pp_group().is_last_rank:
            self.unpadded_vocab_size = config.vocab_size
            if lora_config:
                self.unpadded_vocab_size += lora_config.lora_extra_vocab_size
            self.lm_head = ParallelLMHead(self.unpadded_vocab_size, config.hidden_size, org_num_embeddings=config.vocab_size, padding_size=DEFAULT_VOCAB_PADDING_SIZE if not lora_config else lora_config.lora_vocab_padding_size, quant_config=quant_config)
            if config.tie_word_embeddings:
                self.lm_head.weight = self.transformer.wte.weight
            logit_scale = getattr(config, 'logit_scale', 1.0)
            self.logits_processor = LogitsProcessor(self.unpadded_vocab_size, config.vocab_size, logit_scale)
        else:
            self.lm_head = PPMissingLayer()
        self.make_empty_intermediate_tensors = self.transformer.make_empty_intermediate_tensors

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, intermediate_tensors: Optional[IntermediateTensors]=None, inputs_embeds: Optional[torch.Tensor]=None) -> Union[torch.Tensor, IntermediateTensors]:
        model_output = self.transformer(input_ids, positions, intermediate_tensors, inputs_embeds)
        return model_output

    def compute_logits(self, hidden_states: torch.Tensor, sampling_metadata: SamplingMetadata) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states, sampling_metadata)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self, skip_prefixes=['lm_head.'] if self.config.tie_word_embeddings else None)
        return loader.load_weights(weights)