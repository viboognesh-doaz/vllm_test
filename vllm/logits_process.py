from typing import Callable, Union
from vllm.transformers_utils.tokenizer import AnyTokenizer
import torch
LogitsProcessor = Union[Callable[[list[int], torch.Tensor], torch.Tensor], Callable[[list[int], list[int], torch.Tensor], torch.Tensor]]
'LogitsProcessor is a function that takes a list\nof previously generated tokens, the logits tensor\nfor the next token and, optionally, prompt tokens as a\nfirst argument, and returns a modified tensor of logits\nto sample from.'

def get_bad_words_logits_processors(bad_words: list[str], tokenizer: AnyTokenizer) -> list[LogitsProcessor]:
    bad_words_ids: list[list[int]] = list()
    for bad_word in bad_words:
        for add_prefix_space in [False, True]:
            prefix = ' ' if add_prefix_space else ''
            prompt = prefix + bad_word.lstrip()
            prompt_token_ids = tokenizer.encode(text=prompt, add_special_tokens=False)
            if not add_prefix_space or (add_prefix_space and prompt_token_ids[0] != bad_words_ids[-1][0] and (len(prompt_token_ids) == len(bad_words_ids[-1]))):
                bad_words_ids.append(prompt_token_ids)
    return [NoBadWordsLogitsProcessor(bad_words_ids=bad_words_ids)]

class NoBadWordsLogitsProcessor:
    _SMALLEST_LOGIT = float('-inf')
    _NEUTRAL_LOGIT = 0.0

    def __init__(self, bad_words_ids: list[list[int]]):
        self.bad_words_ids = bad_words_ids
        self.word_bias: torch.FloatTensor = None

    def __call__(self, past_tokens_ids: Union[list[int], tuple[int]], logits: torch.FloatTensor) -> torch.Tensor:
        if self.word_bias is None:
            self._init_word_bias(logits=logits)
        last_token_bias = torch.zeros_like(logits)
        for bad_word_ids in self.bad_words_ids:
            if len(bad_word_ids) == 1:
                continue
            if len(bad_word_ids) > len(past_tokens_ids) + 1:
                continue
            prefix_length = len(bad_word_ids) - 1
            last_token_id = bad_word_ids[-1]
            actual_prefix = past_tokens_ids[-prefix_length:]
            expected_prefix = bad_word_ids[:prefix_length]
            assert len(actual_prefix) == len(expected_prefix)
            is_match = tuple(actual_prefix) == tuple(expected_prefix)
            last_token_bias[last_token_id] += self._SMALLEST_LOGIT if is_match else self._NEUTRAL_LOGIT
        logits = logits + self.word_bias + last_token_bias
        return logits

    def _init_word_bias(self, logits: torch.FloatTensor) -> None:
        vocab_size = logits.shape[-1]
        self._check_token_ids_bounds(vocab_size=vocab_size)
        self.word_bias = torch.zeros((vocab_size,), dtype=torch.float, device=logits.device)
        for bad_word_ids in self.bad_words_ids:
            if len(bad_word_ids) == 1:
                bad_word_id = bad_word_ids[-1]
                self.word_bias[bad_word_id] = self._SMALLEST_LOGIT

    def _check_token_ids_bounds(self, vocab_size: int) -> None:
        invalid_token_ids = []
        for bad_word_ids in self.bad_words_ids:
            for token_id in bad_word_ids:
                if token_id < 0 or token_id >= vocab_size:
                    invalid_token_ids.append(token_id)
        if len(invalid_token_ids) > 0:
            raise ValueError(f'The model vocabulary size is {vocab_size}, but the following tokens were specified as bad: {invalid_token_ids}. All token id values should be integers satisfying: 0 <= token_id < {vocab_size}.')