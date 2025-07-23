from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
import torch

@RotaryEmbedding.register_oot
class DummyRotaryEmbedding(RotaryEmbedding):
    """Original rotary positional embedding."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.addition_config = True

    def forward_oot(self, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        return super().forward_oot(*args, **kwargs)