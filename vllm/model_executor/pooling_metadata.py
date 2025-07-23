from dataclasses import dataclass
from typing import Any
from vllm.pooling_params import PoolingParams
from vllm.utils import is_pin_memory_available
import torch

class PoolingMetadata:
    """Metadata for pooling operations in the Pooler layer.

    This class holds the necessary information for pooling operations,
    providing context for how to perform pooling and other related operations.

    Attributes:
        seq_groups: List of (seq_ids, pooling_params).
        seq_data: A mapping of sequence ID to additional sequence data.
        prompt_lens: List of the lengths of each prompt.
    """

    def __init__(self, seq_groups: list[tuple[list[int], PoolingParams]], seq_data: dict[int, Any], prompt_lens: list[int]) -> None:
        self.seq_groups = seq_groups
        self.seq_data = seq_data
        self.prompt_lens = prompt_lens

    def __repr__(self) -> str:
        return f'PoolingMetadata(seq_groups={self.seq_groups}, seq_data={self.seq_data}, prompt_lens={self.prompt_lens})'

@dataclass
class PoolingTensors:
    """Tensors for pooling."""
    prompt_lens: torch.Tensor

    @classmethod
    def from_pooling_metadata(cls, pooling_metadata: 'PoolingMetadata', device: torch.device) -> 'PoolingTensors':
        """
        Create PoolingTensors from PoolingMetadata.

        Args:
            pooling_metadata: PoolingMetadata instance to convert.
            device: Device to store the tensors.
        """
        pin_memory = is_pin_memory_available()
        prompt_lens_t = torch.tensor(pooling_metadata.prompt_lens, device='cpu', dtype=torch.long, pin_memory=pin_memory)
        return cls(prompt_lens=prompt_lens_t.to(device=device, non_blocking=True))