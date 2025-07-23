from pydantic import BaseModel, ConfigDict, ValidationInfo, model_validator
from typing import Optional
'\nThis file contains the Pydantic schemas for various quantization-related\nparameters. When a relevant quantization technique is specified, these\nparameters are loaded in the form of a JSON alongside the model weights\nand augment the model with additional information needed for use of that\ntechnique. The format of this JSON should be specified by one or more\nschemas contained here.\n\nFor example, when the KV cache is quantized to FP8-E4M3 (currently only\npossible on ROCm), the model can be optionally augmented with KV cache\nscaling factors.\n'

class KVCacheQuantSchema(BaseModel):
    dtype: str
    scaling_factor: dict[int, dict[int, float]]

    @model_validator(mode='after')
    def check_is_fp8(self) -> 'KVCacheQuantSchema':
        assert self.dtype == 'float8_e4m3fn', f'Loaded scaling factors intended for KV cache dtype = {self.dtype} rather than float8_e4m3fn!'
        return self

    @model_validator(mode='after')
    def check_tp_ranks(self, info: ValidationInfo) -> 'KVCacheQuantSchema':
        context = info.context
        if context:
            tp_size = context['tp_size']
            num_hidden_layers = context['num_hidden_layers']
            assert len(self.scaling_factor) == tp_size, f'Loaded dictionary has TP size {len(self.scaling_factor)} but LLM engine is currently running with TP size {tp_size}.'
            for tp_rank, layer_maps in self.scaling_factor.items():
                assert len(layer_maps) == num_hidden_layers, f'KV cache scales map for TP rank {tp_rank} is malformed. Expected {num_hidden_layers} layers, got {len(layer_maps)}.'
            for i in range(tp_size):
                assert i in self.scaling_factor, f'KV cache scales map for TP rank {i} not found.'
        return self

    @model_validator(mode='after')
    def check_current_rank(self, info: ValidationInfo) -> 'KVCacheQuantSchema':
        context = info.context
        if context:
            tp_rank = context['tp_rank']
            num_hidden_layers = context['num_hidden_layers']
            layer_scales_map = self.scaling_factor[tp_rank]
            for i in range(num_hidden_layers):
                assert i in layer_scales_map, f'Could not find KV cache scales for layer {i} in TP rank {tp_rank}.'
        return self

class QuantParamSchema(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_type: Optional[str]
    kv_cache: KVCacheQuantSchema

    @model_validator(mode='after')
    def check_model_type(self, info: ValidationInfo) -> 'QuantParamSchema':
        context = info.context
        if context:
            model_type = context.get('model_type', None)
            if model_type is not None:
                assert model_type == self.model_type, f'Model type is {model_type} but loaded scaling factors belonging to different model type {self.model_type}!'
        return self