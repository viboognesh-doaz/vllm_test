from copy import deepcopy
from typing import Optional, Union
from vllm.config import QuantizationConfig
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, UnquantizedEmbeddingMethod
import regex as re
import torch

def override_config(config: QuantizationConfig, prefix: str):
    weight_bits = get_dynamic_override(config, prefix, 'bits', config.weight_bits)
    if isinstance(weight_bits, int):
        config.weight_bits = weight_bits
    group_size = get_dynamic_override(config, prefix, 'group_size', config.group_size)
    if isinstance(group_size, int):
        config.group_size = group_size
    desc_act = get_dynamic_override(config, prefix, 'desc_act', config.desc_act)
    if isinstance(desc_act, bool):
        config.desc_act = desc_act
    config.pack_factor = 32 // config.weight_bits
    if config.get_name() == 'gptq_marlin':
        is_sym = get_dynamic_override(config, prefix, 'sym', config.is_sym)
        if isinstance(is_sym, bool):
            config.is_sym = is_sym
        if (config.weight_bits, config.is_sym) not in config.TYPE_MAP:
            raise ValueError(f'Unsupported quantization config: bits={config.weight_bits}, sym={config.is_sym}')
        config.quant_type = config.TYPE_MAP[config.weight_bits, config.is_sym]
    elif config.get_name() == 'gptq':
        if config.weight_bits not in [2, 3, 4, 8]:
            raise ValueError(f'Currently, only 2/3/4/8-bit weight quantization is supported for GPTQ, but got {config.weight_bits} bits.')

def get_dynamic_override(config: QuantizationConfig, layer_name: str, key: Optional[str]=None, default_value: Union[int, bool, None]=None) -> Union[dict, int, bool, None]:
    for pattern, pattern_dict in config.dynamic.items():
        if pattern.startswith('-:'):
            if re.match(pattern.removeprefix('-:'), layer_name):
                return False
        elif re.match(pattern.removeprefix('+:'), layer_name):
            if key is None:
                return pattern_dict
            else:
                return pattern_dict.get(key, default_value)
    return default_value

def get_linear_quant_method(config: QuantizationConfig, layer: torch.nn.Module, prefix: str, linear_method_cls: type):
    cloned_config = deepcopy(config)
    parallel_lm_head_quantized = isinstance(layer, ParallelLMHead) and cloned_config.lm_head_quantized
    if isinstance(layer, LinearBase) or parallel_lm_head_quantized:
        if get_dynamic_override(cloned_config, layer_name=prefix) == False:
            if parallel_lm_head_quantized:
                return UnquantizedEmbeddingMethod()
            return UnquantizedLinearMethod()
        if prefix:
            override_config(cloned_config, prefix=prefix)
        return linear_method_cls(cloned_config)
    return None