from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.tokenizers.mistral import MistralTokenizer
from vllm.utils import LazyLoader
from vllm.v1.structured_output.backend_types import StructuredOutputBackend, StructuredOutputGrammar, StructuredOutputOptions
from vllm.v1.structured_output.utils import choice_as_grammar, convert_lark_to_ebnf, grammar_is_likely_lark
import json
import torch
import vllm.envs
import xgrammar as xgr
if TYPE_CHECKING:
else:
    xgr = LazyLoader('xgr', globals(), 'xgrammar')
logger = init_logger(__name__)

@dataclass
class XgrammarBackend(StructuredOutputBackend):

    def __post_init__(self):
        self.disable_any_whitespace = self.vllm_config.decoding_config.disable_any_whitespace
        if isinstance(self.tokenizer, MistralTokenizer):
            try:
                if self.tokenizer.is_tekken:
                    encoded_vocab = self.tokenizer._vocab
                else:
                    encoded_vocab = [token for token, _ in sorted(self.tokenizer.get_vocab().items(), key=lambda x: x[1])]
                stop_token_ids = None
                if hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id is not None:
                    stop_token_ids = [self.tokenizer.eos_token_id]
            except AttributeError as e:
                raise ValueError(f'Cannot get the vocabulary of the tokenizer {type(self.tokenizer)}. The tokenizer should have a get_vocab method.') from e
            tokenizer_info = xgr.TokenizerInfo(encoded_vocab=encoded_vocab, vocab_type=xgr.VocabType.RAW if self.tokenizer.is_tekken else xgr.VocabType.BYTE_FALLBACK, vocab_size=self.vocab_size, stop_token_ids=stop_token_ids, add_prefix_space=True)
        else:
            tokenizer_info = xgr.TokenizerInfo.from_huggingface(self.tokenizer, vocab_size=self.vocab_size)
        self.compiler = xgr.GrammarCompiler(tokenizer_info, max_threads=8, cache_enabled=True, cache_limit_bytes=vllm.envs.VLLM_XGRAMMAR_CACHE_MB * 1024 * 1024)
        self.num_speculative_tokens = 0
        if self.vllm_config.speculative_config is not None:
            self.num_speculative_tokens = self.vllm_config.speculative_config.num_speculative_tokens

    def compile_grammar(self, request_type: StructuredOutputOptions, grammar_spec: str) -> StructuredOutputGrammar:
        if request_type == StructuredOutputOptions.JSON:
            ctx = self.compiler.compile_json_schema(grammar_spec, any_whitespace=not self.disable_any_whitespace)
        elif request_type == StructuredOutputOptions.JSON_OBJECT:
            ctx = self.compiler.compile_json_schema('{"type": "object"}', any_whitespace=not self.disable_any_whitespace)
        elif request_type == StructuredOutputOptions.GRAMMAR:
            ctx = self.compiler.compile_grammar(grammar_spec)
        elif request_type == StructuredOutputOptions.REGEX:
            ctx = self.compiler.compile_regex(grammar_spec)
        elif request_type == StructuredOutputOptions.STRUCTURAL_TAG:
            s_tag = json.loads(grammar_spec)
            tags = [xgr.StructuralTagItem(begin=s['begin'], schema=json.dumps(s['schema']), end=s['end']) for s in s_tag['structures']]
            ctx = self.compiler.compile_structural_tag(tags, s_tag['triggers'])
        else:
            logger.error('Validation should have already occurred. Please file an issue.')
            raise ValueError(f'grammar is not of valid supported types. ({request_type!s})')
        return XgrammarGrammar(matcher=xgr.GrammarMatcher(ctx, max_rollback_tokens=self.num_speculative_tokens), vocab_size=self.vocab_size, ctx=ctx)

    def allocate_token_bitmask(self, max_num_seqs: int):
        return xgr.allocate_token_bitmask(max_num_seqs, self.vocab_size)

    def destroy(self):
        del self.compiler

@dataclass
class XgrammarGrammar(StructuredOutputGrammar):
    vocab_size: int
    matcher: xgr.GrammarMatcher = field(hash=False)
    ctx: xgr.CompiledGrammar = field(hash=False)
    num_processed_tokens: int = field(default_factory=lambda: 0, repr=False, hash=False, init=False)

    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
        """Accepts a list of tokens and advances the FSM.

        Returns True if the FSM was advanced successfully.
        Returns False if the FSM failed to advance.
        """
        for token in tokens:
            if not self.matcher.accept_token(token):
                logger.error('Failed to advance FSM for request %s for tokens %s. Please file an issue.', request_id, token)
                return False
            self.num_processed_tokens += 1
        return True

    def validate_tokens(self, tokens: list[int]) -> list[int]:
        """Checks if the list of tokens are accepted by the FSM in sequence.
        Will not advance the FSM.

        Returns the prefix list of tokens that are accepted by the FSM.
        """
        accepted_tokens = []
        for token in tokens:
            if self.matcher.accept_token(token):
                accepted_tokens.append(token)
            else:
                break
        if len(accepted_tokens) > 0:
            self.matcher.rollback(len(accepted_tokens))
        return accepted_tokens

    def rollback(self, num_tokens: int) -> None:
        self.matcher.rollback(num_tokens)
        self.num_processed_tokens -= num_tokens

    def fill_bitmask(self, bitmask: torch.Tensor, idx: int) -> None:
        self.matcher.fill_next_token_bitmask(bitmask, idx)

    def is_terminated(self) -> bool:
        return self.matcher.is_terminated()

    def reset(self):
        self.num_processed_tokens = 0
        self.matcher.reset()

def has_xgrammar_unsupported_json_features(schema: dict[str, Any]) -> bool:
    """Check if JSON schema contains features unsupported by xgrammar."""

    def check_object(obj: dict[str, Any]) -> bool:
        if not isinstance(obj, dict):
            return False
        if obj.get('type') in ('integer', 'number') and 'multipleOf' in obj:
            return True
        if obj.get('type') == 'array' and any((key in obj for key in ('uniqueItems', 'contains', 'minContains', 'maxContains'))):
            return True
        if obj.get('type') == 'string' and 'format' in obj:
            return True
        if obj.get('type') == 'object' and any((key in obj for key in ('minProperties', 'maxProperties', 'propertyNames', 'patternProperties'))):
            return True
        for value in obj.values():
            if isinstance(value, dict):
                if check_object(value):
                    return True
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict) and check_object(item):
                        return True
        return False
    return check_object(schema)

def validate_xgrammar_grammar(sampling_params: SamplingParams) -> None:
    """Validate that the request is supported by structured output.

    Raises ValueError if the request is not supported.
    """
    if sampling_params.guided_decoding is None:
        return
    gd_params = sampling_params.guided_decoding
    if gd_params.regex:
        try:
            xgr.Grammar.from_regex(gd_params.regex)
        except Exception as err:
            raise ValueError(f'Failed to transform regex into a grammar: {err}') from err
    if gd_params.choice:
        choice_grammar = choice_as_grammar(gd_params.choice)
        try:
            xgr.Grammar.from_ebnf(choice_grammar)
        except Exception as err:
            raise ValueError('Failed to transform choices into a grammar: {err}') from err
        gd_params.choice = None
        gd_params.grammar = choice_grammar
        return
    if gd_params.json:
        if isinstance(gd_params.json, str):
            try:
                schema = json.loads(gd_params.json)
            except json.JSONDecodeError as e:
                raise ValueError('Invalid JSON grammar specification.') from e
        else:
            schema = gd_params.json
        try:
            xgr.Grammar.from_json_schema(schema)
        except Exception as err:
            raise ValueError(f'Failed to transform json schema into a grammar: {err}') from err
        if has_xgrammar_unsupported_json_features(schema):
            raise ValueError('The provided JSON schema contains features not supported by xgrammar.')
        return
    if gd_params.grammar:
        if grammar_is_likely_lark(gd_params.grammar):
            try:
                gd_params.grammar = convert_lark_to_ebnf(gd_params.grammar)
            except ValueError as e:
                raise ValueError('Failed to convert the grammar from Lark to EBNF. ') from e
        try:
            xgr.Grammar.from_ebnf(gd_params.grammar)
        except Exception as e:
            raise ValueError('Invalid grammar specification.') from e
        return
    if gd_params.structural_tag:
        try:
            s_tag = json.loads(gd_params.structural_tag)
            tags = [xgr.StructuralTagItem(begin=s['begin'], schema=json.dumps(s['schema']), end=s['end']) for s in s_tag['structures']]
            xgr.Grammar.from_structural_tag(tags, s_tag['triggers'])
        except Exception as e:
            raise ValueError('Invalid structural tag specification.') from e