from dataclasses import dataclass
from typing import Optional
from vllm.v1.sample.logits_processor import LogitsProcessorManager
import torch

@dataclass
class SamplingMetadata:
    temperature: Optional[torch.Tensor]
    all_greedy: bool
    all_random: bool
    top_p: Optional[torch.Tensor]
    top_k: Optional[torch.Tensor]
    generators: dict[int, torch.Generator]
    max_num_logprobs: Optional[int]
    no_penalties: bool
    prompt_token_ids: Optional[torch.Tensor]
    frequency_penalties: torch.Tensor
    presence_penalties: torch.Tensor
    repetition_penalties: torch.Tensor
    output_token_ids: list[list[int]]
    allowed_token_ids_mask: Optional[torch.Tensor]
    bad_words_token_ids: dict[int, list[list[int]]]
    logitsprocs: LogitsProcessorManager