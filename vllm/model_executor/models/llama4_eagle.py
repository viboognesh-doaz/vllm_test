from .utils import AutoWeightsLoader, maybe_prefix
from collections.abc import Iterable
from typing import Optional
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.torchao import TorchAOConfig
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.llama4 import Llama4DecoderLayer, Llama4ForCausalLM
from vllm.model_executor.models.utils import extract_layer_index
import torch
import torch.nn as nn
logger = init_logger(__name__)

@support_torch_compile
class LlamaModel(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str='', start_layer_id: int=0, quant_config: Optional[QuantizationConfig]=None) -> None:
        super().__init__()
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        self.validate_and_update_config(start_layer_id, quant_config)
        self.vocab_size = self.config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(self.config.vocab_size, self.config.hidden_size, prefix=maybe_prefix(prefix, 'embed_tokens'))
        self.layers = nn.ModuleList([Llama4DecoderLayer(self.config, quant_config=quant_config, prefix=maybe_prefix(prefix, f'layers.{i + start_layer_id}')) for i in range(self.config.num_hidden_layers)])
        self.fc = torch.nn.Linear(self.config.hidden_size * 2, self.config.hidden_size, bias=False)
        self.norm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)

    def forward(self, input_ids: Optional[torch.Tensor], positions: torch.Tensor, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        input_embeds = self.embed_tokens(input_ids)
        hidden_states = self.fc(torch.cat((input_embeds, hidden_states), dim=-1))
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return (hidden_states, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [('.qkv_proj', '.q_proj', 'q'), ('.qkv_proj', '.k_proj', 'k'), ('.qkv_proj', '.v_proj', 'v'), ('.gate_up_proj', '.gate_proj', 0), ('.gate_up_proj', '.up_proj', 1)]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            name = name.removeprefix('model.')
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if get_pp_group().world_size == 1 and 'embed_tokens.' in name:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, 'weight_loader', default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        for name in params_dict:
            if get_pp_group().world_size == 1 and 'embed_tokens.' in name:
                continue
            assert name in loaded_params, f'{name} is not loaded!'
        return loaded_params

    def validate_and_update_config(self, start_layer_id: int, quant_config: Optional[QuantizationConfig]=None) -> None:
        assert self.config.yoco_global_kv_layer is None
        assert self.config.yoco_local_kv_layer is None
        assert len(self.config.moe_layers) == 0
        self.config.no_rope_layers = [0] * start_layer_id + self.config.no_rope_layers
        if isinstance(quant_config, TorchAOConfig):

            def pad_layer_name(layer: str) -> str:
                layer_index = extract_layer_index(layer)
                return layer.replace(str(layer_index), str(layer_index + start_layer_id))
            quant_config.torchao_config.module_fqn_to_config = {pad_layer_name(layer): quantization for layer, quantization in quant_config.torchao_config.module_fqn_to_config.items()}

class EagleLlama4ForCausalLM(Llama4ForCausalLM):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str=''):
        nn.Module.__init__(self)
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        target_layer_num = vllm_config.model_config.get_num_layers(vllm_config.parallel_config)
        quant_config = VllmConfig.get_quantization_config(vllm_config.speculative_config.draft_model_config, vllm_config.load_config)
        self.model = LlamaModel(vllm_config=vllm_config, prefix='model', start_layer_id=target_layer_num, quant_config=quant_config)
        logit_scale = getattr(self.config, 'logit_scale', 1.0)
        self.logits_processor = LogitsProcessor(self.config.vocab_size, scale=logit_scale)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(input_ids, positions, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        loader = AutoWeightsLoader(self, skip_prefixes=['lm_head.'])
        model_weights = {}
        weights = [self.permute_qk_weight_for_rotary(name, loaded_weight) for name, loaded_weight in weights]
        for name, loaded_weight in weights:
            if 'lm_head' not in name:
                name = 'model.' + name
            model_weights[name] = loaded_weight
        loader.load_weights(model_weights.items())