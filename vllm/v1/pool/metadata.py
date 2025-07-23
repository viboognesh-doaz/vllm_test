from dataclasses import dataclass
from typing import Optional
from vllm.pooling_params import PoolingParams
import torch

@dataclass
class PoolingMetadata:
    """Tensors for pooling."""
    prompt_lens: torch.Tensor
    prompt_token_ids: Optional[torch.Tensor]
    pooling_params: list[PoolingParams]