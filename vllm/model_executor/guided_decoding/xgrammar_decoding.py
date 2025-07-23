from __future__ import annotations
from dataclasses import dataclass, field
from transformers import PreTrainedTokenizer
from typing import TYPE_CHECKING, Any
from vllm.config import ModelConfig
from vllm.logger import init_logger
from vllm.model_executor.guided_decoding.utils import convert_lark_to_gbnf, grammar_is_likely_lark
from vllm.reasoning import ReasoningParser
from vllm.sampling_params import GuidedDecodingParams
from vllm.transformers_utils.tokenizers.mistral import MistralTokenizer
import json
import regex as re
import torch
import vllm.envs
import xgrammar as xgr
try:
    xgr_installed = True
except ImportError:
    xgr_installed = False
    pass
if TYPE_CHECKING:
logger = init_logger(__name__)

def get_local_xgrammar_guided_decoding_logits_processor(guided_params: GuidedDecodingParams, tokenizer: PreTrainedTokenizer, model_config: ModelConfig, reasoner: ReasoningParser | None, max_threads: int=8):
    config = GrammarConfig.from_guided_params(guided_params=guided_params, model_config=model_config, tokenizer=tokenizer, max_threads=max_threads)
    return XGrammarLogitsProcessor(config, reasoner)

@dataclass(frozen=True)
class TokenizerData:
    """Immutable container for cached tokenizer data."""
    metadata: str
    encoded_vocab: list[str] = field(default_factory=list)

class TokenizerDataCache:
    """Cache manager for tokenizer data to avoid repeated processing."""
    _cache: dict[int, TokenizerData] = {}

    @classmethod
    def get_tokenizer_data(cls, tokenizer: PreTrainedTokenizer, /, *, tokenizer_hash: int, vocab_size: int) -> TokenizerData:
        if tokenizer_hash not in cls._cache:
            tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab_size)
            metadata = json.loads(tokenizer_info.dump_metadata())
            try:
                vocab_dict = tokenizer.get_vocab()
            except AttributeError as e:
                raise ValueError(f'Cannot get the vocabulary of the tokenizer {type(tokenizer)}. The tokenizer should have a get_vocab method.') from e
            encoded_vocab = [''] * tokenizer_info.vocab_size
            for token, idx in vocab_dict.items():
                if idx < tokenizer_info.vocab_size:
                    encoded_vocab[idx] = token
            if isinstance(tokenizer, MistralTokenizer):
                metadata.update({'vocab_type': xgr.VocabType.BYTE_FALLBACK, 'add_prefix_space': True})
            cls._cache[tokenizer_hash] = TokenizerData(encoded_vocab=encoded_vocab, metadata=json.dumps(metadata))
        return cls._cache[tokenizer_hash]

class GrammarCompilerCache:
    """
    Cache for GrammarCompiler instances based on tokenizer.

    This cache reduces the overhead of creating new compiler instances when
    using the same tokenizer configuration.
    """
    _cache: dict[str, xgr.GrammarCompiler] = {}

    @classmethod
    def get_compiler(cls, config: GrammarConfig) -> xgr.GrammarCompiler:
        cache_key = str(config.tokenizer_hash)
        if cache_key not in cls._cache:
            config_data = config.tokenizer_data
            tokenizer_info = xgr.TokenizerInfo.from_vocab_and_metadata(encoded_vocab=config_data.encoded_vocab, metadata=config_data.metadata)
            cache_size = vllm.envs.VLLM_XGRAMMAR_CACHE_MB * 1024 * 1024
            cls._cache[cache_key] = xgr.GrammarCompiler(tokenizer_info, max_threads=config.max_threads, cache_enabled=True, cache_limit_bytes=cache_size)
        return cls._cache[cache_key]

@dataclass
class GrammarConfig:
    """Serializable configuration for grammar compilation"""
    tokenizer_hash: int
    tokenizer_data: TokenizerData
    json_str: str | None = None
    grammar_str: str | None = None
    json_object: bool | None = None
    any_whitespace: bool = True
    regex_str: str | None = None
    max_threads: int = 8

    @classmethod
    def from_guided_params(cls, guided_params: GuidedDecodingParams, model_config: ModelConfig, tokenizer: PreTrainedTokenizer, max_threads: int=8) -> GrammarConfig:
        tokenizer_hash = hash(tokenizer)
        tokenizer_data = TokenizerDataCache.get_tokenizer_data(tokenizer, tokenizer_hash=tokenizer_hash, vocab_size=model_config.hf_text_config.vocab_size)
        if guided_params.json:
            if not isinstance(guided_params.json, str):
                json_str = json.dumps(guided_params.json)
            else:
                json_str = guided_params.json
            any_whitespace = not guided_params.disable_any_whitespace
            model_with_warn = None
            if 'Mistral' in model_config.model:
                model_with_warn = 'Mistral'
            elif 'Qwen' in model_config.model:
                model_with_warn = 'Qwen'
            if model_with_warn is not None and any_whitespace:
                logger.info_once('%s model detected, consider setting `disable_any_whitespace` to prevent runaway generation of whitespaces.', model_with_warn)
            try:
                xgr.Grammar.from_json_schema(json_str, any_whitespace=any_whitespace)
            except RuntimeError as err:
                raise ValueError(str(err)) from err
            return cls(json_str=json_str, tokenizer_hash=tokenizer_hash, max_threads=max_threads, tokenizer_data=tokenizer_data, any_whitespace=any_whitespace)
        elif guided_params.grammar:
            if grammar_is_likely_lark(guided_params.grammar):
                try:
                    grammar_str = convert_lark_to_gbnf(guided_params.grammar)
                except ValueError as e:
                    raise ValueError(f'Failed to convert the grammar from Lark to GBNF. Please either use GBNF grammar directly or specify --guided-decoding-backend=outlines.\nConversion error: {str(e)}') from e
            else:
                grammar_str = guided_params.grammar
            try:
                xgr.Grammar.from_ebnf(grammar_str)
            except RuntimeError as err:
                raise ValueError(str(err)) from err
            return cls(grammar_str=grammar_str, tokenizer_hash=tokenizer_hash, max_threads=max_threads, tokenizer_data=tokenizer_data)
        elif guided_params.json_object:
            return cls(json_object=True, tokenizer_hash=tokenizer_hash, max_threads=max_threads, tokenizer_data=tokenizer_data)
        elif guided_params.choice:
            choice_str = GrammarConfig.choice_as_grammar(guided_params.choice)
            try:
                xgr.Grammar.from_ebnf(choice_str)
            except RuntimeError as err:
                raise ValueError(str(err)) from err
            return cls(grammar_str=choice_str, tokenizer_hash=tokenizer_hash, max_threads=max_threads, tokenizer_data=tokenizer_data)
        elif guided_params.regex:
            return cls(regex_str=guided_params.regex, tokenizer_hash=tokenizer_hash, max_threads=max_threads, tokenizer_data=tokenizer_data)
        else:
            raise ValueError('Currently only support JSON and EBNF grammar mode for xgrammar')

    @staticmethod
    def escape_ebnf_string(s: str) -> str:
        """Escape special characters in a EBNF string."""
        return re.sub('(["\\\\])', '\\\\\\1', s)

    @staticmethod
    def choice_as_grammar(choice: list[str] | None) -> str:
        if choice is None:
            raise ValueError('Choice is not set')
        escaped_choices = (GrammarConfig.escape_ebnf_string(c) for c in choice)
        grammar = 'root ::= ' + ' | '.join((f'"{c}"' for c in escaped_choices))
        return grammar

    @staticmethod
    def tokenizer_info(tokenizer_data: TokenizerData) -> xgr.TokenizerInfo:
        return xgr.TokenizerInfo.from_vocab_and_metadata(encoded_vocab=tokenizer_data.encoded_vocab, metadata=tokenizer_data.metadata)

@dataclass
class XGrammarLogitsProcessor:
    """Wrapper class to support pickle protocol"""
    config: GrammarConfig
    reasoner: ReasoningParser | None = None
    ctx: xgr.CompiledGrammar | None = None
    tokenizer_info: xgr.TokenizerInfo = None
    token_bitmask: torch.Tensor = None
    matchers: list[xgr.GrammarMatcher] = field(default_factory=list)
    batch_size: int = field(default=1)
    prefilled: bool = field(default=False)

    def __post_init__(self):
        if self.tokenizer_info is None:
            self.tokenizer_info = self.config.tokenizer_info(self.config.tokenizer_data)

    def __getstate__(self) -> dict[str, Any]:
        return {'config': self.config, 'reasoner': self.reasoner}

    def __setstate__(self, state: dict[str, Any]):
        self.config = state['config']
        self.reasoner = state['reasoner']
        self.tokenizer_info = GrammarConfig.tokenizer_info(self.config.tokenizer_data)
        self.ctx = None
        self.matchers = []
        self.batch_size = 1
        self.token_bitmask = None
        self.prefilled = False

    def _ensure_ctx(self):
        """Lazily initialize the processor in the worker process"""
        if self.ctx is None:
            compiler = GrammarCompilerCache.get_compiler(self.config)
            if self.config.json_str is not None:
                any_whitespace = self.config.any_whitespace
                self.ctx = compiler.compile_json_schema(self.config.json_str, any_whitespace=any_whitespace)
            elif self.config.grammar_str is not None:
                self.ctx = compiler.compile_grammar(self.config.grammar_str)
            elif self.config.json_object:
                any_whitespace = self.config.any_whitespace
                self.ctx = compiler.compile_json_schema('{"type": "object"}', any_whitespace=any_whitespace)
            elif self.config.regex_str:
                self.ctx = compiler.compile_regex(self.config.regex_str)
            else:
                raise ValueError('Invalid configuration for xgrammar logits processor')

    def __call__(self, input_ids: list[int], scores: torch.Tensor) -> torch.Tensor:
        if self.reasoner is not None and (not self.reasoner.is_reasoning_end(input_ids)):
            return scores
        if self.ctx is None:
            self._ensure_ctx()
        if len(self.matchers) == 0:
            self.matchers = [xgr.GrammarMatcher(self.ctx) for _ in range(self.batch_size)]
            self.token_bitmask = xgr.allocate_token_bitmask(self.batch_size, self.tokenizer_info.vocab_size)
        if not self.prefilled:
            self.prefilled = True
        else:
            for i, matcher in enumerate(self.matchers):
                if not matcher.is_terminated():
                    sampled_token = input_ids[-1]
                    assert self.matchers[i].accept_token(sampled_token)
        for i, matcher in enumerate(self.matchers):
            if not matcher.is_terminated():
                matcher.fill_next_token_bitmask(self.token_bitmask, i)
        device_type = scores.device.type
        dtype = scores.dtype
        if device_type != 'cuda':
            scores = scores.to('cpu').float().unsqueeze(0)
        xgr.apply_token_bitmask_inplace(scores, self.token_bitmask.to(scores.device, non_blocking=True))
        if device_type != 'cuda':
            scores = scores.to(dtype).to(device_type).squeeze()
        return scores

    def clone(self) -> XGrammarLogitsProcessor:
        """Create a new instance with shared compiled grammar
          but separate state"""
        new_processor = XGrammarLogitsProcessor(self.config, self.reasoner, None, self.tokenizer_info)
        new_processor.ctx = self.ctx
        if self.ctx is not None:
            new_processor.matchers = [xgr.GrammarMatcher(self.ctx) for _ in range(self.batch_size)]
        if hasattr(self, 'token_bitmask') and self.token_bitmask is not None:
            new_processor.token_bitmask = self.token_bitmask
        new_processor.batch_size = self.batch_size
        new_processor.prefilled = False
        return new_processor