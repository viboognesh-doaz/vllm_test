from transformers import AutoFeatureExtractor
from transformers import AutoImageProcessor
from transformers import AutoProcessor
from transformers.feature_extraction_utils import FeatureExtractionMixin
from transformers.image_processing_utils import BaseImageProcessor
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Optional, Union, cast
from transformers.processing_utils import ProcessorMixin
from typing_extensions import TypeVar
if TYPE_CHECKING:
    from vllm.config import ModelConfig
_P = TypeVar('_P', bound=ProcessorMixin, default=ProcessorMixin)

class HashableDict(dict):
    """
    A dictionary that can be hashed by lru_cache.
    """

    def __hash__(self) -> int:
        return hash(frozenset(self.items()))

class HashableList(list):
    """
    A list that can be hashed by lru_cache.
    """

    def __hash__(self) -> int:
        return hash(tuple(self))

def _merge_mm_kwargs(model_config: 'ModelConfig', **kwargs):
    mm_config = model_config.get_multimodal_config()
    base_kwargs = mm_config.mm_processor_kwargs
    if base_kwargs is None:
        base_kwargs = {}
    merged_kwargs = {**base_kwargs, **kwargs}
    for key, value in merged_kwargs.items():
        if isinstance(value, dict):
            merged_kwargs[key] = HashableDict(value)
        if isinstance(value, list):
            merged_kwargs[key] = HashableList(value)
    return merged_kwargs

def get_processor(processor_name: str, *args: Any, revision: Optional[str]=None, trust_remote_code: bool=False, processor_cls: Union[type[_P], tuple[type[_P], ...]]=ProcessorMixin, **kwargs: Any) -> _P:
    """Load a processor for the given model name via HuggingFace."""
    processor_factory = AutoProcessor if processor_cls == ProcessorMixin or isinstance(processor_cls, tuple) else processor_cls
    try:
        processor = processor_factory.from_pretrained(processor_name, *args, revision=revision, trust_remote_code=trust_remote_code, **kwargs)
    except ValueError as e:
        if not trust_remote_code:
            err_msg = 'Failed to load the processor. If the processor is a custom processor not yet available in the HuggingFace transformers library, consider setting `trust_remote_code=True` in LLM or using the `--trust-remote-code` flag in the CLI.'
            raise RuntimeError(err_msg) from e
        else:
            raise e
    if not isinstance(processor, processor_cls):
        raise TypeError(f'Invalid type of HuggingFace processor. Expected type: {processor_cls}, but found type: {type(processor)}')
    return processor
cached_get_processor = lru_cache(get_processor)

def cached_processor_from_config(model_config: 'ModelConfig', processor_cls: Union[type[_P], tuple[type[_P], ...]]=ProcessorMixin, **kwargs: Any) -> _P:
    return cached_get_processor(model_config.model, revision=model_config.revision, trust_remote_code=model_config.trust_remote_code, processor_cls=processor_cls, **_merge_mm_kwargs(model_config, **kwargs))

def get_feature_extractor(processor_name: str, *args: Any, revision: Optional[str]=None, trust_remote_code: bool=False, **kwargs: Any):
    """Load an audio feature extractor for the given model name 
    via HuggingFace."""
    try:
        feature_extractor = AutoFeatureExtractor.from_pretrained(processor_name, *args, revision=revision, trust_remote_code=trust_remote_code, **kwargs)
    except ValueError as e:
        if not trust_remote_code:
            err_msg = 'Failed to load the feature extractor. If the feature extractor is a custom extractor not yet available in the HuggingFace transformers library, consider setting `trust_remote_code=True` in LLM or using the `--trust-remote-code` flag in the CLI.'
            raise RuntimeError(err_msg) from e
        else:
            raise e
    return cast(FeatureExtractionMixin, feature_extractor)
cached_get_feature_extractor = lru_cache(get_feature_extractor)

def cached_feature_extractor_from_config(model_config: 'ModelConfig', **kwargs: Any):
    return cached_get_feature_extractor(model_config.model, revision=model_config.revision, trust_remote_code=model_config.trust_remote_code, **_merge_mm_kwargs(model_config, **kwargs))

def get_image_processor(processor_name: str, *args: Any, revision: Optional[str]=None, trust_remote_code: bool=False, **kwargs: Any):
    """Load an image processor for the given model name via HuggingFace."""
    try:
        processor = AutoImageProcessor.from_pretrained(processor_name, *args, revision=revision, trust_remote_code=trust_remote_code, **kwargs)
    except ValueError as e:
        if not trust_remote_code:
            err_msg = 'Failed to load the image processor. If the image processor is a custom processor not yet available in the HuggingFace transformers library, consider setting `trust_remote_code=True` in LLM or using the `--trust-remote-code` flag in the CLI.'
            raise RuntimeError(err_msg) from e
        else:
            raise e
    return cast(BaseImageProcessor, processor)
cached_get_image_processor = lru_cache(get_image_processor)

def cached_image_processor_from_config(model_config: 'ModelConfig', **kwargs: Any):
    return cached_get_image_processor(model_config.model, revision=model_config.revision, trust_remote_code=model_config.trust_remote_code, **_merge_mm_kwargs(model_config, **kwargs))