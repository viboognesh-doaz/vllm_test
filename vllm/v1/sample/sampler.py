from vllm.utils import is_pin_memory_available
from vllm.v1.outputs import LogprobsTensors, SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.bad_words import apply_bad_words
from vllm.v1.sample.ops.penalties import apply_all_penalties
from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler
import torch
import torch.nn as nn
"A layer that samples the next tokens from the model's outputs."
_SAMPLING_EPS = 1e-05

class Sampler(nn.Module):

    def __init__(self):
        super().__init__()
        self.topk_topp_sampler = TopKTopPSampler()
        self.pin_memory = is_pin_memory_available()

    def forward(self, logits: torch.Tensor, sampling_metadata: SamplingMetadata) -> SamplerOutput:
        num_logprobs = sampling_metadata.max_num_logprobs
        if num_logprobs is not None:
            raw_logprobs = self.compute_logprobs(logits)
        logits = logits.to(torch.float32)
        logits = self.apply_allowed_token_ids(logits, sampling_metadata)
        logits = self.apply_bad_words(logits, sampling_metadata)
        for processor in sampling_metadata.logitsprocs.non_argmax_invariant:
            logits = processor.apply(logits)
        logits = self.apply_penalties(logits, sampling_metadata)
        sampled = self.sample(logits, sampling_metadata)
        sampled = sampled.long()
        logprobs_tensors = None if num_logprobs is None else self.gather_logprobs(raw_logprobs, num_logprobs, token_ids=sampled)
        sampled = sampled.to(torch.int32)
        sampler_output = SamplerOutput(sampled_token_ids=sampled.unsqueeze(-1), logprobs_tensors=logprobs_tensors)
        return sampler_output

    def apply_temperature(self, logits: torch.Tensor, temp: torch.Tensor) -> torch.Tensor:
        return logits.div_(temp.unsqueeze(dim=1))

    def greedy_sample(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(dim=-1).view(-1)

    def sample(self, logits: torch.Tensor, sampling_metadata: SamplingMetadata) -> torch.Tensor:
        """Sample logits based on sampling metadata.

        The various logits processing functions called in this method
        may update the logits tensor in-place.
        """
        assert not (sampling_metadata.all_greedy and sampling_metadata.all_random)
        if sampling_metadata.all_random:
            greedy_sampled = None
        else:
            greedy_sampled = self.greedy_sample(logits)
            if sampling_metadata.all_greedy:
                return greedy_sampled
        assert sampling_metadata.temperature is not None
        logits = self.apply_temperature(logits, sampling_metadata.temperature)
        for processor in sampling_metadata.logitsprocs.argmax_invariant:
            logits = processor.apply(logits)
        random_sampled = self.topk_topp_sampler(logits, sampling_metadata.generators, sampling_metadata.top_k, sampling_metadata.top_p)
        if greedy_sampled is None:
            return random_sampled
        sampled = torch.where(sampling_metadata.temperature < _SAMPLING_EPS, greedy_sampled, random_sampled, out=greedy_sampled)
        return sampled

    def compute_logprobs(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.log_softmax(dim=-1, dtype=torch.float32)

    def gather_logprobs(self, logprobs: torch.Tensor, num_logprobs: int, token_ids: torch.Tensor) -> LogprobsTensors:
        """
        Gather logprobs for topk and sampled/prompt token.

        Args:
          logprobs: (num tokens) x (vocab) tensor
          num_logprobs: minimum number of logprobs to
                        retain per token
          token_ids: prompt tokens (if prompt logprobs)
                     or sampled tokens (if sampled
                     logprobs); 1D token ID tensor
                     with (num tokens) elements
                     Must be int64.

        Returns:
          Top-k int indices tensor, (num tokens) x (num_logprobs + 1)
          Top-k float logprobs tensor, (num tokens) x (num_logprobs + 1)
          Sampled token rank tensor, (num tokens)
        """
        assert token_ids.dtype == torch.int64
        topk_logprobs, topk_indices = torch.topk(logprobs, num_logprobs, dim=-1)
        token_ids = token_ids.unsqueeze(-1)
        token_logprobs = logprobs.gather(-1, token_ids)
        token_ranks = (logprobs >= token_logprobs).sum(-1)
        indices = torch.cat((token_ids, topk_indices), dim=1)
        logprobs = torch.cat((token_logprobs, topk_logprobs), dim=1)
        indices = indices.to(torch.int32)
        return LogprobsTensors(indices, logprobs, token_ranks)

    def apply_penalties(self, logits: torch.Tensor, sampling_metadata: SamplingMetadata) -> torch.Tensor:
        if not sampling_metadata.no_penalties:
            assert sampling_metadata.prompt_token_ids is not None
            logits = apply_all_penalties(logits, sampling_metadata.prompt_token_ids, sampling_metadata.presence_penalties, sampling_metadata.frequency_penalties, sampling_metadata.repetition_penalties, sampling_metadata.output_token_ids)
        return logits

    def apply_allowed_token_ids(self, logits: torch.Tensor, sampling_metadata: SamplingMetadata) -> torch.Tensor:
        if sampling_metadata.allowed_token_ids_mask is not None:
            logits.masked_fill_(sampling_metadata.allowed_token_ids_mask, float('-inf'))
        return logits

    def apply_bad_words(self, logits: torch.Tensor, sampling_metadata: SamplingMetadata) -> torch.Tensor:
        if sampling_metadata.bad_words_token_ids:
            apply_bad_words(logits, sampling_metadata.bad_words_token_ids, sampling_metadata.output_token_ids)
        return logits