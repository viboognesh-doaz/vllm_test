from vllm.v1.outputs import LogprobsTensors, SamplerOutput
from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler
from vllm.v1.sample.tpu.metadata import TPUSupportedSamplingMetadata
import torch
import torch.nn as nn
'Sampler layer implementing TPU supported operations.'
_SAMPLING_EPS = 1e-05

class Sampler(nn.Module):

    def __init__(self):
        super().__init__()
        self.topk_topp_sampler = TopKTopPSampler()

    def forward(self, logits: torch.Tensor, sampling_metadata: TPUSupportedSamplingMetadata) -> SamplerOutput:
        logits = logits.to(torch.float32)
        sampled = self.sample(logits, sampling_metadata)
        sampler_output = SamplerOutput(sampled_token_ids=sampled.unsqueeze(-1), logprobs_tensors=None)
        return sampler_output

    def apply_temperature(self, logits: torch.Tensor, temp: torch.Tensor) -> torch.Tensor:
        return logits.div_(temp.unsqueeze(dim=1))

    def greedy_sample(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(dim=-1).view(-1)

    def sample(self, logits: torch.Tensor, sampling_metadata: TPUSupportedSamplingMetadata) -> torch.Tensor:
        greedy_sampled = self.greedy_sample(logits)
        assert sampling_metadata.temperature is not None
        logits = self.apply_temperature(logits, sampling_metadata.temperature)
        if sampling_metadata.min_p is not None:
            logits = self.apply_min_p(logits, sampling_metadata.min_p)
        random_sampled = self.topk_topp_sampler(logits, sampling_metadata.generators, sampling_metadata.top_k, sampling_metadata.top_p)
        sampled = torch.where(sampling_metadata.temperature < _SAMPLING_EPS, greedy_sampled, random_sampled)
        return sampled

    def compute_logprobs(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.log_softmax(dim=-1, dtype=torch.float32)

    def gather_logprobs(self, logprobs: torch.Tensor, num_logprobs: int, token_ids: torch.Tensor) -> LogprobsTensors:
        """
        Gather logprobs for topk and sampled/prompt token.

        Args:
          logits: (num tokens) x (vocab) tensor
          num_logprobs: minimum number of logprobs to
                        retain per token
          token_ids: prompt tokens (if prompt logprobs)
                     or sampled tokens (if sampled
                     logprobs); 1D token ID tensor
                     with (num tokens) elements

        Returns:
          Top-k int indices tensor, (num tokens) x (num_logprobs + 1)
          Top-k float logprobs tensor, (num tokens) x (num_logprobs + 1)
          Sampled token rank tensor, (num tokens)
        """
        topk_logprobs, topk_indices = torch.topk(logprobs, num_logprobs, dim=-1)
        token_ids = token_ids.unsqueeze(-1)
        token_logprobs = logprobs.gather(-1, token_ids)
        token_ranks = (logprobs >= token_logprobs).sum(-1)
        indices = torch.cat((token_ids, topk_indices), dim=1)
        logprobs = torch.cat((token_logprobs, topk_logprobs), dim=1)
        indices = indices.to(torch.int32)
        return LogprobsTensors(indices, logprobs, token_ranks)

    def apply_min_p(self, logits: torch.Tensor, min_p: torch.Tensor) -> torch.Tensor:
        """
        Filters logits using adaptive probability thresholding.
        """
        probability_values = torch.nn.functional.softmax(logits, dim=-1)
        max_probabilities = torch.amax(probability_values, dim=-1, keepdim=True)
        adjusted_min_p = min_p.unsqueeze(1) * max_probabilities
        valid_token_mask = probability_values >= adjusted_min_p
        logits.masked_fill_(~valid_token_mask, -float('inf'))
        return logits