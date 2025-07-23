from typing import Any, Optional
from vllm.platforms import current_platform
from vllm.platforms import current_platform
import copy
import torch
'Utils for model executor.'

def set_random_seed(seed: int) -> None:
    current_platform.seed_everything(seed)

def set_weight_attrs(weight: torch.Tensor, weight_attrs: Optional[dict[str, Any]]):
    """Set attributes on a weight tensor.

    This method is used to set attributes on a weight tensor. This method
    will not overwrite existing attributes.

    Args:
        weight: The weight tensor.
        weight_attrs: A dictionary of attributes to set on the weight tensor.
    """
    if weight_attrs is None:
        return
    for key, value in weight_attrs.items():
        assert not hasattr(weight, key), f'Overwriting existing tensor attribute: {key}'
        if current_platform.is_tpu() and key == 'weight_loader':
            value = _make_synced_weight_loader(value)
        setattr(weight, key, value)

def _make_synced_weight_loader(original_weight_loader):

    def _synced_weight_loader(param, *args, **kwargs):
        original_weight_loader(param, *args, **kwargs)
        if param.device != torch.device('cpu'):
            torch._sync(param)
    return _synced_weight_loader

def get_packed_modules_mapping(model: torch.nn.Module) -> dict[str, list[str]]:
    parent_map = getattr(model, 'packed_modules_mapping', None)
    parent_map = copy.deepcopy(parent_map) if parent_map is not None else {}
    if parent_map:
        return parent_map
    for child in model.children():
        child_map = getattr(child, 'packed_modules_mapping', None)
        child_map = copy.deepcopy(child_map) if child_map is not None else {}
        if any((k in parent_map and parent_map[k] != v for k, v in child_map.items())):
            raise ValueError(f"Can't update {type(model).__name__}'s packed_modules_mapping safely because of conflicts from {type(child).__name__}.")
        else:
            parent_map.update(child_map)
    return parent_map