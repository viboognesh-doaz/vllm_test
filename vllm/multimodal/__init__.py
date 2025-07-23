from .base import MultiModalPlaceholderMap
from .hasher import MultiModalHashDict, MultiModalHasher
from .inputs import BatchedTensorInputs, ModalityData, MultiModalDataBuiltins, MultiModalDataDict, MultiModalKwargs, MultiModalPlaceholderDict, NestedTensors
from .registry import MultiModalRegistry
MULTIMODAL_REGISTRY = MultiModalRegistry()
'\nThe global [`MultiModalRegistry`][vllm.multimodal.registry.MultiModalRegistry]\nis used by model runners to dispatch data processing according to the target\nmodel.\n\nInfo:\n    [mm_processing](../../../design/mm_processing.html)\n'
__all__ = ['BatchedTensorInputs', 'ModalityData', 'MultiModalDataBuiltins', 'MultiModalDataDict', 'MultiModalHashDict', 'MultiModalHasher', 'MultiModalKwargs', 'MultiModalPlaceholderDict', 'MultiModalPlaceholderMap', 'NestedTensors', 'MULTIMODAL_REGISTRY', 'MultiModalRegistry']