from .utils import PPMissingLayer
from vllm.config import VllmConfig
from vllm.model_executor.models.llama import LlamaForCausalLM
'Inference-only HF format GLM-4 model compatible with THUDM weights.'

class GlmForCausalLM(LlamaForCausalLM):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str=''):
        vllm_config.model_config.hf_config.partial_rotary_factor = 0.5
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        for layer in self.model.layers:
            if not isinstance(layer, PPMissingLayer):
                layer.self_attn.rotary_emb.is_neox_style = False
                layer.self_attn.o_proj.bias = None
                layer.self_attn.o_proj.skip_bias_add = True