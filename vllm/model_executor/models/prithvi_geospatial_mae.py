from collections.abc import Iterable, Mapping, Sequence
from terratorch.cli_tools import SemanticSegmentationTask
from transformers import BatchFeature
from typing import Optional, Union
from vllm.config import VllmConfig
from vllm.model_executor.layers.pooler import AllPool, PoolerHead, PoolerIdentity, SimplePooler
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import IsAttentionFree, SupportsMultiModal, SupportsV0Only
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalDataDict, MultiModalFieldConfig, MultiModalInputs, MultiModalKwargs
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import BaseMultiModalProcessor, BaseProcessingInfo, PromptUpdate
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors
import torch
import torch.nn as nn
'Inference-only IBM/NASA Prithvi Geospatial model.'

class PrithviGeoSpatialMAEProcessingInfo(BaseProcessingInfo):

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {'image': None}

class PrithviGeoSpatialMAEInputBuilder(BaseDummyInputsBuilder[PrithviGeoSpatialMAEProcessingInfo]):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return ''

    def get_dummy_mm_data(self, seq_len: int, mm_counts: Mapping[str, int]) -> MultiModalDataDict:
        return {'pixel_values': torch.full((1, 6, 512, 512), 1.0), 'location_coords': torch.full((1, 2), 1.0)}

class PrithviGeoSpatialMAEMultiModalProcessor(BaseMultiModalProcessor):

    def _get_mm_fields_config(self, hf_inputs: BatchFeature, hf_processor_mm_kwargs: Mapping[str, object]) -> Mapping[str, MultiModalFieldConfig]:
        return dict(pixel_values=MultiModalFieldConfig.batched('image'), location_coords=MultiModalFieldConfig.batched('image'))

    def _get_prompt_updates(self, mm_items: MultiModalDataItems, hf_processor_mm_kwargs: Mapping[str, object], out_mm_kwargs: MultiModalKwargs) -> Sequence[PromptUpdate]:
        return []

    def apply(self, prompt: Union[str, list[int]], mm_data: MultiModalDataDict, hf_processor_mm_kwargs: Mapping[str, object], tokenization_kwargs: Optional[Mapping[str, object]]=None, return_mm_hashes: bool=False) -> MultiModalInputs:
        mm_kwargs = {}
        for k, v in mm_data.items():
            mm_kwargs[k] = v
        return MultiModalInputs(type='multimodal', prompt=prompt, prompt_token_ids=[1], mm_kwargs=MultiModalKwargs(mm_kwargs), mm_hashes=None, mm_placeholders={})

@MULTIMODAL_REGISTRY.register_processor(PrithviGeoSpatialMAEMultiModalProcessor, info=PrithviGeoSpatialMAEProcessingInfo, dummy_inputs=PrithviGeoSpatialMAEInputBuilder)
class PrithviGeoSpatialMAE(nn.Module, IsAttentionFree, SupportsMultiModal, SupportsV0Only):
    """Prithvi Masked Autoencoder"""
    is_pooling_model = True

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> Optional[str]:
        if modality.startswith('image'):
            return None
        raise ValueError('Only image modality is supported')

    def _instantiate_model(self, config: dict) -> Optional[nn.Module]:
        if config['task_args']['task'] == 'SemanticSegmentationTask':
            task = SemanticSegmentationTask(config['model_args'], config['task_args']['model_factory'], loss=config['task_args']['loss'], lr=config['task_args']['lr'], ignore_index=config['task_args']['ignore_index'], optimizer=config['task_args']['optimizer'], optimizer_hparams=config['optimizer_params'], scheduler=config['task_args']['scheduler'], scheduler_hparams=config['scheduler_params'], plot_on_val=config['task_args']['plot_on_val'], freeze_decoder=config['task_args']['freeze_decoder'], freeze_backbone=config['task_args']['freeze_backbone'])
            return task.model
        else:
            return None

    def __init__(self, vllm_config: VllmConfig, prefix: str=''):
        super().__init__()
        self.model = self._instantiate_model(vllm_config.model_config.hf_config.to_dict()['pretrained_cfg'])
        if self.model is None:
            raise ValueError('Unsupported task. Only SemanticSegmentationTask is supported for now by PrithviGeospatialMAE.')
        self.pooler = SimplePooler(AllPool(), PoolerHead(PoolerIdentity()))

    def _parse_and_validate_multimodal_data(self, **kwargs) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        pixel_values = kwargs.pop('pixel_values', None)
        if not isinstance(pixel_values, torch.Tensor):
            raise ValueError(f'Incorrect type of pixel_values. Got type: {type(pixel_values)}')
        pixel_values = torch.unbind(pixel_values, dim=0)[0]
        location_coords = kwargs.pop('location_coords', None)
        if not isinstance(location_coords, torch.Tensor):
            raise ValueError(f'Incorrect type of location_coords. Got type: {type(location_coords)}')
        location_coords = torch.unbind(location_coords, dim=0)[0]
        if location_coords.shape == torch.Size([0]):
            location_coords = None
        return (pixel_values, location_coords)

    def forward(self, input_ids: Optional[torch.Tensor], positions: torch.Tensor, intermediate_tensors: Optional[IntermediateTensors]=None, inputs_embeds: Optional[torch.Tensor]=None, **kwargs: object):
        pixel_values, location_coords = self._parse_and_validate_multimodal_data(**kwargs)
        model_output = self.model(pixel_values, location_coords=location_coords)
        return model_output.output

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_list = []
        model_buffers = dict(self.named_buffers())
        loaded_buffers = []
        for key, value in weights:
            if key == 'state_dict':
                weights_to_parse = value
                for name, weight in weights_to_parse.items():
                    if 'pos_embed' in name:
                        continue
                    if '_timm_module.' in name:
                        name = name.replace('_timm_module.', '')
                    if name in model_buffers:
                        if '_timm_module.' in name:
                            name = name.replace('_timm_module.', '')
                        buffer = model_buffers[name]
                        weight_loader = getattr(buffer, 'weight_loader', default_weight_loader)
                        weight_loader(buffer, weight)
                        loaded_buffers.append(name)
                    else:
                        params_list.append((name, weight))
                break
        loader = AutoWeightsLoader(self)
        autoloaded_weights = loader.load_weights(params_list)
        return autoloaded_weights.union(set(loaded_buffers))