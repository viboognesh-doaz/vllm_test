from .utils import is_pp_missing_parameter, maybe_prefix
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from torch import nn
from transformers import BatchFeature
from transformers.activations import GELUActivation
from typing import Any, Literal, Optional, TypedDict, Union
from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead
from vllm.model_executor.model_loader.weight_utils import default_weight_loader, maybe_remap_kv_scale_name
from vllm.model_executor.models.deepseek_v2 import DeepseekV2Model
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.moonvit import MoonVitPretrainedModel
from vllm.model_executor.models.utils import merge_multimodal_embeddings
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalDataDict, MultiModalFieldConfig, MultiModalKwargs, NestedTensors
from vllm.multimodal.parse import ImageEmbeddingItems, ImageProcessorItems, MultiModalDataItems
from vllm.multimodal.processing import BaseMultiModalProcessor, BaseProcessingInfo, PromptReplacement, PromptUpdate
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs import KimiVLConfig, MoonViTConfig
from vllm.transformers_utils.configs.deepseek_vl2 import DeepseekV2Config
import copy
import math
import torch

@dataclass
class MaxImageTokenMeta:
    width: int = 1024
    height: int = 1024

class KimiVLMultiModalProjector(nn.Module):

    def __init__(self, config: KimiVLConfig):
        super().__init__()
        self.hidden_size = config.vision_config.hidden_size * config.vision_config.merge_kernel_size[0] * config.vision_config.merge_kernel_size[1]
        self.pre_norm = torch.nn.LayerNorm(config.vision_config.hidden_size, eps=1e-05)
        self.linear_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.act = GELUActivation()
        self.linear_2 = nn.Linear(self.hidden_size, config.text_config.hidden_size, bias=True)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        hidden_states = self.pre_norm(image_features).view(-1, self.hidden_size)
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states

class KimiVLImagePixelInputs(TypedDict):
    type: Literal['pixel_values']
    pixel_values: Union[torch.Tensor, list[torch.Tensor]]
    '\n    Shape:`(num_patches, num_channels, patch_size, patch_size)`\n    '
    image_grid_hws: torch.Tensor
    'Shape:`(num_images, 2)`'
KimiVLImageInputs = KimiVLImagePixelInputs

class KimiVLProcessingInfo(BaseProcessingInfo):

    def get_hf_config(self):
        return self.ctx.get_hf_config(KimiVLConfig)

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {'image': None}

    def get_num_image_tokens(self, *, image_width: int, image_height: int) -> int:
        hf_processor = self.get_hf_processor()
        patch_size = hf_processor.image_processor.patch_size
        kernel_size = hf_processor.image_processor.merge_kernel_size
        in_token_limit = hf_processor.image_processor.in_token_limit
        height = image_height
        width = image_width
        assert isinstance(height, int), f'height must be int, current height {height}'
        assert isinstance(width, int), f'width must be int, current width {width}'
        assert kernel_size is not None, 'kernel_size must be specified'
        if width // patch_size * (height // patch_size) > in_token_limit:
            scale = math.sqrt(in_token_limit / (width // patch_size * (height // patch_size)))
            new_w, new_h = (int(width * scale), int(height * scale))
            width, height = (new_w, new_h)
        kernel_height, kernel_width = kernel_size
        pad_height = (kernel_height * patch_size - height % (kernel_height * patch_size)) % (kernel_height * patch_size)
        pad_width = (kernel_width * patch_size - width % (kernel_width * patch_size)) % (kernel_width * patch_size)
        token_height = (height + pad_height) // (kernel_size[0] * patch_size)
        token_width = (width + pad_width) // (kernel_size[1] * patch_size)
        return int(token_height * token_width)

    @property
    def image_token_id(self) -> int:
        return self.get_hf_config().media_placeholder_token_id

class KimiVLDummyInputsBuilder(BaseDummyInputsBuilder[KimiVLProcessingInfo]):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get('image', 0)
        processor = self.info.get_hf_processor()
        image_token = processor.image_token
        return image_token * num_images

    def get_dummy_mm_data(self, seq_len: int, mm_counts: Mapping[str, int]) -> MultiModalDataDict:
        num_images = mm_counts.get('image', 0)
        return {'image': self._get_dummy_images(width=MaxImageTokenMeta.width, height=MaxImageTokenMeta.height, num_images=num_images)}

class KimiVLMultiModalProcessor(BaseMultiModalProcessor[KimiVLProcessingInfo]):

    def _get_mm_fields_config(self, hf_inputs: BatchFeature, hf_processor_mm_kwargs: Mapping[str, object]) -> Mapping[str, MultiModalFieldConfig]:
        image_grid_hws = hf_inputs.get('image_grid_hws', torch.empty((0, 2)))
        image_grid_sizes = image_grid_hws.prod(-1)
        return dict(pixel_values=MultiModalFieldConfig.flat_from_sizes('image', image_grid_sizes), image_grid_hws=MultiModalFieldConfig.batched('image'))

    def _get_prompt_updates(self, mm_items: MultiModalDataItems, hf_processor_mm_kwargs: Mapping[str, Any], out_mm_kwargs: MultiModalKwargs) -> Sequence[PromptUpdate]:
        image_token_id = self.info.image_token_id

        def get_replacement(item_idx: int):
            images = mm_items.get_items('image', (ImageEmbeddingItems, ImageProcessorItems))
            if isinstance(images, ImageEmbeddingItems):
                num_image_tokens = images.get_feature_size(item_idx)
            else:
                image_size = images.get_image_size(item_idx)
                num_image_tokens = self.info.get_num_image_tokens(image_width=image_size.width, image_height=image_size.height)
            return [image_token_id] * num_image_tokens
        return [PromptReplacement(modality='image', target=[image_token_id], replacement=get_replacement)]

@MULTIMODAL_REGISTRY.register_processor(KimiVLMultiModalProcessor, info=KimiVLProcessingInfo, dummy_inputs=KimiVLDummyInputsBuilder)
class KimiVLForConditionalGeneration(nn.Module, SupportsMultiModal):

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> Optional[str]:
        if modality.startswith('image'):
            return '<|media_start|>image<|media_content|><|media_pad|><|media_end|>'
        raise ValueError('Only image modality is supported')

    def __init__(self, vllm_config: VllmConfig, prefix: str='') -> None:
        super().__init__()
        model_config = vllm_config.model_config
        config: KimiVLConfig = model_config.hf_config
        self.config = config
        quant_config = vllm_config.quant_config
        assert isinstance(config.vision_config, MoonViTConfig)
        self.vision_tower = MoonVitPretrainedModel(config.vision_config)
        self.multi_modal_projector = KimiVLMultiModalProjector(config=config)
        self.quant_config = quant_config
        sub_vllm_config = copy.deepcopy(vllm_config)
        sub_vllm_config.model_config.hf_config = sub_vllm_config.model_config.hf_config.text_config
        self.language_model = DeepseekV2Model(vllm_config=sub_vllm_config, prefix=maybe_prefix(prefix, 'language_model'))
        self.unpadded_vocab_size = config.text_config.vocab_size
        self.lm_head = ParallelLMHead(self.unpadded_vocab_size, config.text_config.hidden_size, org_num_embeddings=self.config.text_config.vocab_size, padding_size=DEFAULT_VOCAB_PADDING_SIZE)
        logit_scale = getattr(config, 'logit_scale', 1.0)
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size, config.vocab_size, logit_scale)
        self.media_placeholder: int = self.config.media_placeholder_token_id
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_world_size = get_tensor_model_parallel_world_size()

    def _validate_and_reshape_mm_tensor(self, mm_input: object, name: str) -> torch.Tensor:
        if not isinstance(mm_input, (torch.Tensor, list)):
            raise ValueError(f'Incorrect type of {name}. Got type: {type(mm_input)}')
        if isinstance(mm_input, torch.Tensor):
            if mm_input.ndim == 2:
                return mm_input
            if mm_input.ndim != 3:
                raise ValueError(f'{name} should be 2D or batched 3D tensor. Got ndim: {mm_input.ndim} (shape={mm_input.shape})')
            return mm_input.reshape(-1, mm_input.shape[-1])
        else:
            return torch.concat(mm_input)

    def _parse_and_validate_image_input(self, **kwargs: object) -> Optional[KimiVLImageInputs]:
        pixel_values = kwargs.pop('pixel_values', None)
        image_grid_hws = kwargs.pop('image_grid_hws', None)
        if pixel_values is None:
            return None
        image_grid_hws = self._validate_and_reshape_mm_tensor(image_grid_hws, 'image grid hws')
        num_channels = 3
        patch_size = self.config.vision_config.patch_size
        if isinstance(pixel_values, list):
            pixel_values = torch.cat([x.reshape(-1, num_channels, patch_size, patch_size) for x in pixel_values])
        else:
            pixel_values = pixel_values.reshape(-1, num_channels, patch_size, patch_size)
        pixel_values = pixel_values.to(self.vision_tower.dtype)
        assert image_grid_hws.ndim == 2, f'unexpected shape for image_grid_hws: {image_grid_hws.shape}'
        return KimiVLImagePixelInputs(type='pixel_values', pixel_values=pixel_values, image_grid_hws=image_grid_hws)

    @torch.inference_mode()
    def _process_image_pixels(self, inputs: KimiVLImagePixelInputs) -> torch.Tensor:
        assert self.vision_tower is not None
        pixel_values = inputs['pixel_values']
        image_grid_hws = inputs['image_grid_hws']
        return self.vision_tower(pixel_values, image_grid_hws)

    def _process_image_input(self, image_input: KimiVLImageInputs) -> torch.Tensor:
        assert image_input['type'] == 'pixel_values'
        image_features = self._process_image_pixels(image_input)
        assert isinstance(image_features, list)
        lengths = [x.shape[0] for x in image_features]
        return self.multi_modal_projector(torch.cat(image_features)).split(lengths)

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def get_multimodal_embeddings(self, **kwargs: object) -> Optional[NestedTensors]:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return None
        vision_embeddings = self._process_image_input(image_input)
        return vision_embeddings

    def get_input_embeddings(self, input_ids: torch.Tensor, multimodal_embeddings: Optional[NestedTensors]=None) -> torch.Tensor:
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None and len(multimodal_embeddings) != 0:
            inputs_embeds = merge_multimodal_embeddings(input_ids=input_ids, inputs_embeds=inputs_embeds, multimodal_embeddings=multimodal_embeddings, placeholder_token_id=self.config.media_placeholder_token_id)
        return inputs_embeds

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, intermediate_tensors: Optional[IntermediateTensors]=None, inputs_embeds: Optional[torch.Tensor]=None, **kwargs: object) -> IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None
        elif inputs_embeds is None:
            image_input = self._parse_and_validate_image_input(**kwargs)
            if image_input is None:
                inputs_embeds = None
            else:
                inputs_embeds = self.get_input_embeddings(input_ids)
                image_embeds = self._process_image_input(image_input)
                inputs_embeds = merge_multimodal_embeddings(input_ids, inputs_embeds, image_embeds, placeholder_token_id=self.config.media_placeholder_token_id)
                input_ids = None
        hidden_states = self.language_model(input_ids=input_ids, positions=positions, intermediate_tensors=intermediate_tensors, inputs_embeds=inputs_embeds)
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor, sampling_metadata: SamplingMetadata, **kwargs) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head, hidden_states, sampling_metadata, **kwargs)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        config = self.config.text_config
        _KEYS_TO_MODIFY_MAPPING = {'language_model.lm_head': 'lm_head', 'language_model.model': 'language_model'}
        stacked_params_mapping = [('.gate_up_proj', '.gate_proj', 0), ('.gate_up_proj', '.up_proj', 1)]
        if not config.use_mla:
            stacked_params_mapping += [('.qkv_proj', '.q_proj', 'q'), ('.qkv_proj', '.k_proj', 'k'), ('.qkv_proj', '.v_proj', 'v')]
        if getattr(config, 'n_routed_experts', None):
            expert_params_mapping = FusedMoE.make_expert_params_mapping(ckpt_gate_proj_name='gate_proj', ckpt_down_proj_name='down_proj', ckpt_up_proj_name='up_proj', num_experts=config.n_routed_experts)
        else:
            expert_params_mapping = []
        params_dict = dict(self.named_parameters())
        for args in weights:
            name, loaded_weight = args[:2]
            kwargs = args[2] if len(args) > 2 else {}
            if 'rotary_emb.inv_freq' in name:
                continue
            spec_layer = get_spec_layer_idx_from_weight_name(config, name)
            if spec_layer is not None:
                continue
            if 'rotary_emb.cos_cached' in name or 'rotary_emb.sin_cached' in name:
                continue
            for key_to_modify, new_key in _KEYS_TO_MODIFY_MAPPING.items():
                if key_to_modify in name:
                    name = name.replace(key_to_modify, new_key)
            use_default_weight_loading = False
            if 'vision' in name:
                if self.vision_tower is not None:
                    use_default_weight_loading = True
            else:
                for param_name, weight_name, shard_id in stacked_params_mapping:
                    if weight_name not in name:
                        continue
                    if 'mlp.experts.' in name and name not in params_dict:
                        continue
                    name = name.replace(weight_name, param_name)
                    if name.endswith('.bias') and name not in params_dict:
                        continue
                    if is_pp_missing_parameter(name, self):
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, shard_id, **kwargs)
                    break
                else:
                    for idx, (param_name, weight_name, expert_id, shard_id) in enumerate(expert_params_mapping):
                        if weight_name not in name:
                            continue
                        name = name.replace(weight_name, param_name)
                        if is_pp_missing_parameter(name, self):
                            continue
                        param = params_dict[name]
                        weight_loader = param.weight_loader
                        weight_loader(param, loaded_weight, name, expert_id=expert_id, shard_id=shard_id, **kwargs)
                        break
                    else:
                        use_default_weight_loading = True
            if use_default_weight_loading:
                if name.endswith('.bias') and name not in params_dict:
                    continue
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, 'weight_loader', default_weight_loader)
                weight_loader(param, loaded_weight, **kwargs)

def get_spec_layer_idx_from_weight_name(config: DeepseekV2Config, weight_name: str) -> Optional[int]:
    if hasattr(config, 'num_nextn_predict_layers') and config.num_nextn_predict_layers > 0:
        layer_idx = config.num_hidden_layers
        for i in range(config.num_nextn_predict_layers):
            if weight_name.startswith(f'model.layers.{layer_idx + i}.'):
                return layer_idx + i
    return None