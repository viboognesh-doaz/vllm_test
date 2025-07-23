from .llama import LlamaDecoderLayer
from .utils import AutoWeightsLoader, PPMissingLayer, WeightsMapper, is_pp_missing_parameter
from collections.abc import Iterable
from vllm.config import VllmConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.llama import LlamaForCausalLM, LlamaModel
import torch
import torch.nn as nn

class TeleChat2Model(LlamaModel):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str=''):
        vllm_config.model_config.hf_config.bias = True
        vllm_config.model_config.hf_config.mlp_bias = True
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        for layer in self.layers:
            if not isinstance(layer, PPMissingLayer):
                layer.self_attn.qkv_proj.bias = None
                layer.self_attn.qkv_proj.skip_bias_add = True
                layer.mlp.gate_up_proj.bias = None
                layer.mlp.gate_up_proj.skip_bias_add = True

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [('gate_up_proj', 'gate_proj', 0), ('gate_up_proj', 'up_proj', 1)]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        total_num_heads = self.config.n_head
        head_dim = self.config.hidden_size // total_num_heads
        for name, loaded_weight in weights:
            if 'self_attn.key_value' in name:
                k_weight = []
                v_weight = []
                for i in range(total_num_heads):
                    start = i * head_dim * 2
                    k_weight.append(loaded_weight[start:start + head_dim, :])
                    v_weight.append(loaded_weight[start + head_dim:start + 2 * head_dim])
                k_weight = torch.cat(k_weight, dim=0)
                v_weight = torch.cat(v_weight, dim=0)
                name = name.replace('key_value', 'qkv_proj')
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, k_weight, 'k')
                weight_loader(param, v_weight, 'v')
            elif 'query' in name:
                name = name.replace('query', 'qkv_proj')
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, 'q')
            else:
                for param_name, weight_name, shard_id in stacked_params_mapping:
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    if is_pp_missing_parameter(name, self):
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, shard_id)
                    break
                else:
                    if is_pp_missing_parameter(name, self):
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(param, 'weight_loader', default_weight_loader)
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params

class TeleChat2ForCausalLM(LlamaForCausalLM):
    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={'transformer.': 'model.'}, orig_to_new_substr={'.h.': '.layers.', '.self_attention.': '.self_attn.', '.word_embeddings.': '.embed_tokens.', '.dense.': '.o_proj.', '.ln_f.': '.norm.'})

    def _init_model(self, vllm_config: VllmConfig, prefix: str='', layer_type: type[nn.Module]=LlamaDecoderLayer):
        return TeleChat2Model(vllm_config=vllm_config, prefix=prefix)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self, skip_prefixes=['lm_head.'] if self.config.tie_word_embeddings else None)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)