from dataclasses import dataclass
from enum import Enum, IntEnum
from functools import cached_property
from pydantic import BaseModel
from typing import Annotated, Any, Optional, Union
from vllm.logger import init_logger
from vllm.logits_process import LogitsProcessor
from vllm.transformers_utils.tokenizer import AnyTokenizer
import copy
import msgspec
'Sampling parameters for text generation.'
logger = init_logger(__name__)
_SAMPLING_EPS = 1e-05
_MAX_TEMP = 0.01

class SamplingType(IntEnum):
    GREEDY = 0
    RANDOM = 1
    RANDOM_SEED = 2

@dataclass
class GuidedDecodingParams:
    """One of these fields will be used to build a logit processor."""
    json: Optional[Union[str, dict]] = None
    regex: Optional[str] = None
    choice: Optional[list[str]] = None
    grammar: Optional[str] = None
    json_object: Optional[bool] = None
    'These are other options that can be set'
    backend: Optional[str] = None
    backend_was_auto: bool = False
    disable_fallback: bool = False
    disable_any_whitespace: bool = False
    disable_additional_properties: bool = False
    whitespace_pattern: Optional[str] = None
    structural_tag: Optional[str] = None

    @staticmethod
    def from_optional(json: Optional[Union[dict, BaseModel, str]]=None, regex: Optional[str]=None, choice: Optional[list[str]]=None, grammar: Optional[str]=None, json_object: Optional[bool]=None, backend: Optional[str]=None, whitespace_pattern: Optional[str]=None, structural_tag: Optional[str]=None) -> Optional['GuidedDecodingParams']:
        if all((arg is None for arg in (json, regex, choice, grammar, json_object, structural_tag))):
            return None
        if isinstance(json, (BaseModel, type(BaseModel))):
            json = json.model_json_schema()
        return GuidedDecodingParams(json=json, regex=regex, choice=choice, grammar=grammar, json_object=json_object, backend=backend, whitespace_pattern=whitespace_pattern, structural_tag=structural_tag)

    def __post_init__(self):
        """Validate that some fields are mutually exclusive."""
        guide_count = sum([self.json is not None, self.regex is not None, self.choice is not None, self.grammar is not None, self.json_object is not None])
        if guide_count > 1:
            raise ValueError(f'You can only use one kind of guided decoding but multiple are specified: {self.__dict__}')

class RequestOutputKind(Enum):
    CUMULATIVE = 0
    DELTA = 1
    FINAL_ONLY = 2

class SamplingParams(msgspec.Struct, omit_defaults=True, dict=True):
    """Sampling parameters for text generation.

    Overall, we follow the sampling parameters from the OpenAI text completion
    API (https://platform.openai.com/docs/api-reference/completions/create).
    In addition, we support beam search, which is not supported by OpenAI.

    Args:
        n: Number of output sequences to return for the given prompt.
        best_of: Number of output sequences that are generated from the prompt.
            From these `best_of` sequences, the top `n` sequences are returned.
            `best_of` must be greater than or equal to `n`. By default,
            `best_of` is set to `n`. Warning, this is only supported in V0.
        presence_penalty: Float that penalizes new tokens based on whether they
            appear in the generated text so far. Values > 0 encourage the model
            to use new tokens, while values < 0 encourage the model to repeat
            tokens.
        frequency_penalty: Float that penalizes new tokens based on their
            frequency in the generated text so far. Values > 0 encourage the
            model to use new tokens, while values < 0 encourage the model to
            repeat tokens.
        repetition_penalty: Float that penalizes new tokens based on whether
            they appear in the prompt and the generated text so far. Values > 1
            encourage the model to use new tokens, while values < 1 encourage
            the model to repeat tokens.
        temperature: Float that controls the randomness of the sampling. Lower
            values make the model more deterministic, while higher values make
            the model more random. Zero means greedy sampling.
        top_p: Float that controls the cumulative probability of the top tokens
            to consider. Must be in (0, 1]. Set to 1 to consider all tokens.
        top_k: Integer that controls the number of top tokens to consider. Set
            to 0 (or -1) to consider all tokens.
        min_p: Float that represents the minimum probability for a token to be
            considered, relative to the probability of the most likely token.
            Must be in [0, 1]. Set to 0 to disable this.
        seed: Random seed to use for the generation.
        stop: list of strings that stop the generation when they are generated.
            The returned output will not contain the stop strings.
        stop_token_ids: list of tokens that stop the generation when they are
            generated. The returned output will contain the stop tokens unless
            the stop tokens are special tokens.
        bad_words: list of words that are not allowed to be generated.
            More precisely, only the last token of a corresponding
            token sequence is not allowed when the next generated token
            can complete the sequence.
        include_stop_str_in_output: Whether to include the stop strings in
            output text. Defaults to False.
        ignore_eos: Whether to ignore the EOS token and continue generating
            tokens after the EOS token is generated.
        max_tokens: Maximum number of tokens to generate per output sequence.
        min_tokens: Minimum number of tokens to generate per output sequence
            before EOS or stop_token_ids can be generated
        logprobs: Number of log probabilities to return per output token.
            When set to None, no probability is returned. If set to a non-None
            value, the result includes the log probabilities of the specified
            number of most likely tokens, as well as the chosen tokens.
            Note that the implementation follows the OpenAI API: The API will
            always return the log probability of the sampled token, so there
            may be up to `logprobs+1` elements in the response.
        prompt_logprobs: Number of log probabilities to return per prompt token.
        detokenize: Whether to detokenize the output. Defaults to True.
        skip_special_tokens: Whether to skip special tokens in the output.
        spaces_between_special_tokens: Whether to add spaces between special
            tokens in the output.  Defaults to True.
        logits_processors: list of functions that modify logits based on
            previously generated tokens, and optionally prompt tokens as
            a first argument.
        truncate_prompt_tokens: If set to -1, will use the truncation size
            supported by the model. If set to an integer k, will use only
            the last k tokens from the prompt (i.e., left truncation).
            Defaults to None (i.e., no truncation).
        guided_decoding: If provided, the engine will construct a guided
            decoding logits processor from these parameters. Defaults to None.
        logit_bias: If provided, the engine will construct a logits processor
            that applies these logit biases. Defaults to None.
        allowed_token_ids: If provided, the engine will construct a logits
            processor which only retains scores for the given token ids.
            Defaults to None.
        extra_args: Arbitrary additional args, that can be used by custom
            sampling implementations, plugins, etc. Not used by any in-tree
            sampling implementations.
    """
    n: int = 1
    best_of: Optional[int] = None
    _real_n: Optional[int] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    seed: Optional[int] = None
    stop: Optional[Union[str, list[str]]] = None
    stop_token_ids: Optional[list[int]] = None
    ignore_eos: bool = False
    max_tokens: Optional[int] = 16
    min_tokens: int = 0
    logprobs: Optional[int] = None
    prompt_logprobs: Optional[int] = None
    detokenize: bool = True
    skip_special_tokens: bool = True
    spaces_between_special_tokens: bool = True
    logits_processors: Optional[Any] = None
    include_stop_str_in_output: bool = False
    truncate_prompt_tokens: Optional[Annotated[int, msgspec.Meta(ge=1)]] = None
    output_kind: RequestOutputKind = RequestOutputKind.CUMULATIVE
    output_text_buffer_length: int = 0
    _all_stop_token_ids: set[int] = msgspec.field(default_factory=set)
    guided_decoding: Optional[GuidedDecodingParams] = None
    logit_bias: Optional[dict[int, float]] = None
    allowed_token_ids: Optional[list[int]] = None
    extra_args: Optional[dict[str, Any]] = None
    bad_words: Optional[list[str]] = None
    _bad_words_token_ids: Optional[list[list[int]]] = None

    @staticmethod
    def from_optional(n: Optional[int]=1, best_of: Optional[int]=None, presence_penalty: Optional[float]=0.0, frequency_penalty: Optional[float]=0.0, repetition_penalty: Optional[float]=1.0, temperature: Optional[float]=1.0, top_p: Optional[float]=1.0, top_k: int=0, min_p: float=0.0, seed: Optional[int]=None, stop: Optional[Union[str, list[str]]]=None, stop_token_ids: Optional[list[int]]=None, bad_words: Optional[list[str]]=None, include_stop_str_in_output: bool=False, ignore_eos: bool=False, max_tokens: Optional[int]=16, min_tokens: int=0, logprobs: Optional[int]=None, prompt_logprobs: Optional[int]=None, detokenize: bool=True, skip_special_tokens: bool=True, spaces_between_special_tokens: bool=True, logits_processors: Optional[list[LogitsProcessor]]=None, truncate_prompt_tokens: Optional[Annotated[int, msgspec.Meta(ge=1)]]=None, output_kind: RequestOutputKind=RequestOutputKind.CUMULATIVE, guided_decoding: Optional[GuidedDecodingParams]=None, logit_bias: Optional[Union[dict[int, float], dict[str, float]]]=None, allowed_token_ids: Optional[list[int]]=None, extra_args: Optional[dict[str, Any]]=None) -> 'SamplingParams':
        if logit_bias is not None:
            logit_bias = {int(token): min(100.0, max(-100.0, bias)) for token, bias in logit_bias.items()}
        return SamplingParams(n=1 if n is None else n, best_of=best_of, presence_penalty=0.0 if presence_penalty is None else presence_penalty, frequency_penalty=0.0 if frequency_penalty is None else frequency_penalty, repetition_penalty=1.0 if repetition_penalty is None else repetition_penalty, temperature=1.0 if temperature is None else temperature, top_p=1.0 if top_p is None else top_p, top_k=top_k, min_p=min_p, seed=seed, stop=stop, stop_token_ids=stop_token_ids, bad_words=bad_words, include_stop_str_in_output=include_stop_str_in_output, ignore_eos=ignore_eos, max_tokens=max_tokens, min_tokens=min_tokens, logprobs=logprobs, prompt_logprobs=prompt_logprobs, detokenize=detokenize, skip_special_tokens=skip_special_tokens, spaces_between_special_tokens=spaces_between_special_tokens, logits_processors=logits_processors, truncate_prompt_tokens=truncate_prompt_tokens, output_kind=output_kind, guided_decoding=guided_decoding, logit_bias=logit_bias, allowed_token_ids=allowed_token_ids, extra_args=extra_args)

    def __post_init__(self) -> None:
        if self.best_of:
            if self.best_of < self.n:
                raise ValueError(f'best_of must be greater than or equal to n, got n={self.n} and best_of={self.best_of}.')
            if not self._real_n:
                self._real_n = self.n
                self.n = self.best_of
        if 0 < self.temperature < _MAX_TEMP:
            logger.warning('temperature %s is less than %s, which may cause numerical errors nan or inf in tensors. We have maxed it out to %s.', self.temperature, _MAX_TEMP, _MAX_TEMP)
            self.temperature = max(self.temperature, _MAX_TEMP)
        if self.seed == -1:
            self.seed = None
        if self.stop is None:
            self.stop = []
        elif isinstance(self.stop, str):
            self.stop = [self.stop]
        if self.stop_token_ids is None:
            self.stop_token_ids = []
        if self.bad_words is None:
            self.bad_words = []
        if self.logprobs is True:
            self.logprobs = 1
        if self.prompt_logprobs is True:
            self.prompt_logprobs = 1
        if self.stop and (not self.include_stop_str_in_output):
            self.output_text_buffer_length = max((len(s) for s in self.stop)) - 1
        self._verify_args()
        if self.temperature < _SAMPLING_EPS:
            self.top_p = 1.0
            self.top_k = 0
            self.min_p = 0.0
            self._verify_greedy_sampling()
        self._all_stop_token_ids.update(self.stop_token_ids)

    def _verify_args(self) -> None:
        if not isinstance(self.n, int):
            raise ValueError(f'n must be an int, but is of type {type(self.n)}')
        if self.n < 1:
            raise ValueError(f'n must be at least 1, got {self.n}.')
        if self.best_of is not None:
            if not isinstance(self.best_of, int):
                raise ValueError(f'best_of must be an integer, got {type(self.best_of)}')
            if self.best_of < 1:
                raise ValueError(f'best_of must be at least 1, got {self.best_of}')
            if self.best_of < self.n:
                raise ValueError(f'best_of must be greater than or equal to n, got n={self.n} and best_of={self.best_of}.')
        if not -2.0 <= self.presence_penalty <= 2.0:
            raise ValueError(f'presence_penalty must be in [-2, 2], got {self.presence_penalty}.')
        if not -2.0 <= self.frequency_penalty <= 2.0:
            raise ValueError(f'frequency_penalty must be in [-2, 2], got {self.frequency_penalty}.')
        if self.repetition_penalty <= 0.0:
            raise ValueError(f'repetition_penalty must be greater than zero, got {self.repetition_penalty}.')
        if self.temperature < 0.0:
            raise ValueError(f'temperature must be non-negative, got {self.temperature}.')
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f'top_p must be in (0, 1], got {self.top_p}.')
        if self.top_k < -1:
            raise ValueError(f'top_k must be 0 (disable), or at least 1, got {self.top_k}.')
        if not isinstance(self.top_k, int):
            raise TypeError(f'top_k must be an integer, got {type(self.top_k).__name__}')
        if not 0.0 <= self.min_p <= 1.0:
            raise ValueError(f'min_p must be in [0, 1], got {self.min_p}.')
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError(f'max_tokens must be at least 1, got {self.max_tokens}.')
        if self.min_tokens < 0:
            raise ValueError(f'min_tokens must be greater than or equal to 0, got {self.min_tokens}.')
        if self.max_tokens is not None and self.min_tokens > self.max_tokens:
            raise ValueError(f'min_tokens must be less than or equal to max_tokens={self.max_tokens}, got {self.min_tokens}.')
        if self.logprobs is not None and self.logprobs < 0:
            raise ValueError(f'logprobs must be non-negative, got {self.logprobs}.')
        if self.prompt_logprobs is not None and self.prompt_logprobs < 0:
            raise ValueError(f'prompt_logprobs must be non-negative, got {self.prompt_logprobs}.')
        if self.truncate_prompt_tokens is not None and self.truncate_prompt_tokens < 1:
            raise ValueError(f'truncate_prompt_tokens must be >= 1, got {self.truncate_prompt_tokens}')
        assert isinstance(self.stop_token_ids, list)
        if not all((isinstance(st_id, int) for st_id in self.stop_token_ids)):
            raise ValueError(f'stop_token_ids must contain only integers, got {self.stop_token_ids}.')
        assert isinstance(self.stop, list)
        if any((not stop_str for stop_str in self.stop)):
            raise ValueError('stop cannot contain an empty string.')
        if self.stop and (not self.detokenize):
            raise ValueError('stop strings are only supported when detokenize is True. Set detokenize=True to use stop.')
        if self.best_of != self._real_n and self.output_kind == RequestOutputKind.DELTA:
            raise ValueError('best_of must equal n to use output_kind=DELTA')

    def _verify_greedy_sampling(self) -> None:
        if self.n > 1:
            raise ValueError(f'n must be 1 when using greedy sampling, got {self.n}.')

    def update_from_generation_config(self, generation_config: dict[str, Any], model_eos_token_id: Optional[int]=None) -> None:
        """Update if there are non-default values from generation_config"""
        if model_eos_token_id is not None:
            self._all_stop_token_ids.add(model_eos_token_id)
        if (eos_ids := generation_config.get('eos_token_id')) is not None:
            eos_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids)
            if model_eos_token_id is not None:
                eos_ids.discard(model_eos_token_id)
            if eos_ids:
                self._all_stop_token_ids.update(eos_ids)
                if not self.ignore_eos:
                    eos_ids.update(self.stop_token_ids)
                    self.stop_token_ids = list(eos_ids)

    def update_from_tokenizer(self, tokenizer: AnyTokenizer) -> None:
        if not self.bad_words:
            return
        self._bad_words_token_ids = []
        for bad_word in self.bad_words:
            for add_prefix_space in [False, True]:
                prefix = ' ' if add_prefix_space else ''
                prompt = prefix + bad_word.lstrip()
                prompt_token_ids = tokenizer.encode(text=prompt, add_special_tokens=False)
                if not add_prefix_space or (add_prefix_space and prompt_token_ids[0] != self._bad_words_token_ids[-1][0] and (len(prompt_token_ids) == len(self._bad_words_token_ids[-1]))):
                    self._bad_words_token_ids.append(prompt_token_ids)
        invalid_token_ids = [token_id for bad_words_token_ids in self._bad_words_token_ids for token_id in bad_words_token_ids if token_id < 0 or token_id > tokenizer.max_token_id]
        if len(invalid_token_ids) > 0:
            raise ValueError(f'The model vocabulary size is {tokenizer.max_token_id + 1}, but the following tokens were specified as bad: {invalid_token_ids}. All token id values should be integers satisfying: 0 <= token_id <= {tokenizer.max_token_id}.')

    @cached_property
    def sampling_type(self) -> SamplingType:
        if self.temperature < _SAMPLING_EPS:
            return SamplingType.GREEDY
        if self.seed is not None:
            return SamplingType.RANDOM_SEED
        return SamplingType.RANDOM

    @property
    def all_stop_token_ids(self) -> set[int]:
        return self._all_stop_token_ids

    @property
    def bad_words_token_ids(self) -> Optional[list[list[int]]]:
        return self._bad_words_token_ids

    def clone(self) -> 'SamplingParams':
        """Deep copy, but maybe not the LogitsProcessor objects.

        LogitsProcessor objects may contain an arbitrary, nontrivial amount of
        data that is expensive to copy. However, if not copied, the processor
        needs to support parallel decoding for multiple sequences
        See https://github.com/vllm-project/vllm/issues/3087
        """
        logit_processor_refs = None if self.logits_processors is None else {id(lp): lp.clone() if hasattr(lp, 'clone') else lp for lp in self.logits_processors}
        return copy.deepcopy(self, memo=logit_processor_refs)

    def __repr__(self) -> str:
        return f'SamplingParams(n={self.n}, presence_penalty={self.presence_penalty}, frequency_penalty={self.frequency_penalty}, repetition_penalty={self.repetition_penalty}, temperature={self.temperature}, top_p={self.top_p}, top_k={self.top_k}, min_p={self.min_p}, seed={self.seed}, stop={self.stop}, stop_token_ids={self.stop_token_ids}, bad_words={self.bad_words}, include_stop_str_in_output={self.include_stop_str_in_output}, ignore_eos={self.ignore_eos}, max_tokens={self.max_tokens}, min_tokens={self.min_tokens}, logprobs={self.logprobs}, prompt_logprobs={self.prompt_logprobs}, skip_special_tokens={self.skip_special_tokens}, spaces_between_special_tokens={self.spaces_between_special_tokens}, truncate_prompt_tokens={self.truncate_prompt_tokens}, guided_decoding={self.guided_decoding}, extra_args={self.extra_args})'

class BeamSearchParams(msgspec.Struct, omit_defaults=True, dict=True):
    """Beam search parameters for text generation."""
    beam_width: int
    max_tokens: int
    ignore_eos: bool = False
    temperature: float = 0.0
    length_penalty: float = 1.0
    include_stop_str_in_output: bool = False