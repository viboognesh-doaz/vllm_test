from dataclasses import dataclass
from typing import NamedTuple, Optional
import torch

class LogprobsLists(NamedTuple):
    logprob_token_ids: list[list[int]]
    logprobs: list[list[float]]
    sampled_token_ranks: list[int]

    def slice(self, start: int, end: int):
        return LogprobsLists(self.logprob_token_ids[start:end], self.logprobs[start:end], self.sampled_token_ranks[start:end])

class LogprobsTensors(NamedTuple):
    logprob_token_ids: torch.Tensor
    logprobs: torch.Tensor
    selected_token_ranks: torch.Tensor

    def tolists(self):
        return LogprobsLists(self.logprob_token_ids.tolist(), self.logprobs.tolist(), self.selected_token_ranks.tolist())

    @staticmethod
    def empty_cpu(num_positions: int, num_tokens_per_position: int) -> 'LogprobsTensors':
        """Create empty LogprobsTensors on CPU."""
        logprob_token_ids = torch.empty((num_positions, num_tokens_per_position), dtype=torch.int32, device='cpu')
        logprobs = torch.empty_like(logprob_token_ids, dtype=torch.float32)
        selected_token_ranks = torch.empty(num_positions, dtype=torch.int32, device='cpu')
        return LogprobsTensors(logprob_token_ids=logprob_token_ids, logprobs=logprobs, selected_token_ranks=selected_token_ranks)

@dataclass
class SamplerOutput:
    sampled_token_ids: torch.Tensor
    logprobs_tensors: Optional[LogprobsTensors]

@dataclass
class ModelRunnerOutput:
    req_ids: list[str]
    req_id_to_index: dict[str, int]
    sampled_token_ids: list[list[int]]
    spec_token_ids: Optional[list[list[int]]]
    logprobs: Optional[LogprobsLists]
    prompt_logprobs_dict: dict[str, Optional[LogprobsTensors]]
    pooler_output: list[Optional[torch.Tensor]]
    finished_sending: Optional[set[str]] = None
    finished_recving: Optional[set[str]] = None
    num_nans_in_logits: Optional[dict[str, int]] = None
EMPTY_MODEL_RUNNER_OUTPUT = ModelRunnerOutput(req_ids=[], req_id_to_index={}, sampled_token_ids=[], spec_token_ids=None, logprobs=None, prompt_logprobs_dict={}, pooler_output=[], finished_sending=None, finished_recving=None, num_nans_in_logits=None)