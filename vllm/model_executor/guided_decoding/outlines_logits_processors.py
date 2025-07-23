from __future__ import annotations
from cachetools import LRUCache
from diskcache import Cache
from outlines_core import Guide, Index, Vocabulary
from outlines_core.json_schema import build_regex_from_schema
from outlines_core.kernels.torch import _apply_token_bitmask_inplace_kernel, allocate_token_bitmask
from pydantic import BaseModel
from transformers import PreTrainedTokenizerBase
from transformers.file_utils import SPIECE_UNDERLINE
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
from typing import Optional, Union
from vllm.logger import init_logger
from vllm.reasoning import ReasoningParser
from vllm.transformers_utils.tokenizer import AnyTokenizer
import copy
import hashlib
import importlib.metadata
import json
import os
import regex as re
import tempfile
import torch
import vllm.envs as envs
logger = init_logger(__name__)
CACHE = None

class BaseLogitsProcessor:

    def __init__(self, guide: Guide, eos_token_id: int, reasoner: Optional[ReasoningParser]) -> None:
        self._guide: Guide = guide
        self._eos_token_id: int = eos_token_id
        self._reasoner: Optional[ReasoningParser] = reasoner
        self._mask: Optional[torch.Tensor] = None

    def __call__(self, input_ids: list[int], scores: torch.Tensor) -> torch.Tensor:
        if self._mask is None:
            self._mask = allocate_token_bitmask(scores.size(-1))
        if self._reasoner is not None and (not self._reasoner.is_reasoning_end(input_ids)):
            return scores
        input_ids = self._reasoner.extract_content_ids(input_ids) if self._reasoner is not None else input_ids
        if input_ids and input_ids[-1] != self._eos_token_id:
            self._guide.advance(token_id=input_ids[-1], return_tokens=False)
        self._guide.write_mask_into(data_ptr=self._mask.data_ptr(), numel=self._mask.numel(), element_size=self._mask.element_size())
        _apply_token_bitmask_inplace_kernel(logits=scores.unsqueeze(dim=0), mask=self._mask.to(scores.device, non_blocking=True))
        self._mask.to('cpu', non_blocking=True)
        return scores

    def clone(self) -> BaseLogitsProcessor:
        guide = copy.deepcopy(self._guide)
        guide.reset()
        return BaseLogitsProcessor(guide=guide, eos_token_id=self._eos_token_id, reasoner=self._reasoner)

class RegexLogitsProcessor(BaseLogitsProcessor):

    @classmethod
    def _get_guide(cls, regex_string: str, tokenizer: PreTrainedTokenizerBase) -> Guide:
        global CACHE
        if CACHE is None:
            CACHE = get_cache()
        vocabulary = get_vocabulary(tokenizer)
        cache_key = f'{vocabulary._hash}_{regex_string}'
        if CACHE is not None and cache_key in CACHE:
            return Guide(CACHE[cache_key])
        index = Index(regex_string, vocabulary.inner)
        if CACHE is not None:
            CACHE[cache_key] = index
        return Guide(index)

    def __init__(self, regex_string: str, tokenizer: PreTrainedTokenizerBase, reasoner: Optional[ReasoningParser]) -> None:
        super().__init__(guide=RegexLogitsProcessor._get_guide(regex_string, tokenizer), eos_token_id=tokenizer.eos_token_id, reasoner=reasoner)

class JSONLogitsProcessor(RegexLogitsProcessor):

    def __init__(self, schema: Union[str, dict, BaseModel], tokenizer: PreTrainedTokenizerBase, whitespace_pattern: Union[str, None], reasoner: Optional[ReasoningParser]) -> None:
        if isinstance(schema, type(BaseModel)):
            schema_str = json.dumps(schema.model_json_schema())
        elif isinstance(schema, dict):
            schema_str = json.dumps(schema)
        elif isinstance(schema, str):
            schema_str = schema
        else:
            raise ValueError(f'Cannot parse schema {schema}. The schema must be either a Pydantic object, a dictionary or a string that contains the JSON Schema specification')
        regex_string = build_regex_from_schema(schema_str, whitespace_pattern)
        super().__init__(regex_string, tokenizer, reasoner)

class OutlinesVocabulary:
    """
    Wrapper class for `outlines_core.Vocabulary`,
    which allows us to store a hash with the vocabulary
    """

    def __init__(self, vocabulary: Vocabulary) -> None:
        self.inner = vocabulary
        hex_str = hashlib.sha256(vocabulary.__repr__().encode('utf-8')).hexdigest()
        hash_int = int(hex_str, 16)
        self._hash = hash_int
re_llama_byte_token = re.compile('^<0x[0-9A-F]{2}>$')
re_replacement_seq = re.compile('^.{0,6}�+.{0,6}$')

def _reduced_vocabulary(tokenizer: AnyTokenizer, eos_token_id: int) -> dict[bytes, list[int]]:
    """Create a map from vocabulary tokens to lists of equivalent token ids.
    
    Returns:
        A Dict of token string -> equivalent token ids
    """
    unicode_to_bytes = {v: k for k, v in bytes_to_unicode().items()}

    def convert_token_to_string(token: str) -> str:
        string = tokenizer.convert_tokens_to_string([token])
        if type(token) is str and token.startswith(SPIECE_UNDERLINE) or token == '<0x20>':
            return ' ' + string
        return string
    vocabulary: dict[bytes, list[int]] = {}
    empty_token_ids: list[int] = []
    for token, token_idx in tokenizer.get_vocab().items():
        if token in tokenizer.all_special_tokens:
            continue
        token_str = convert_token_to_string(token)
        if token_str:
            if isinstance(token, (bytes, bytearray)):
                token_bytes = bytes(token_str)
            elif '�' in token_str and (not re_replacement_seq.match(token_str)):
                if re_llama_byte_token.match(token):
                    token_bytes = bytes([int(token[3:5], 16)])
                else:
                    byte_vals = [unicode_to_bytes.get(c) for c in token]
                    if None in byte_vals:
                        raise RuntimeError(f'Cannot convert token `{token}` ({token_idx}) to bytes: {token_str}')
                    token_bytes = bytes(byte_vals)
            else:
                token_bytes = token_str.encode('utf-8')
            if token_idx != eos_token_id:
                vocabulary.setdefault(token_bytes, []).append(token_idx)
        else:
            empty_token_ids.append(token_idx)
    return vocabulary

def get_vocabulary(tokenizer: AnyTokenizer) -> Vocabulary:
    """Get the `Vocabulary` object for a given tokenizer.
    """
    if hasattr(tokenizer, '_outlines_vocabulary'):
        return tokenizer._outlines_vocabulary
    try:
        if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
            eos_token_id = tokenizer.eos_token_id
        else:
            raise ValueError(f'Error during guided decoding setup: Tokenizer ({type(tokenizer)}) has no `eos_token_id` property, but `eos_token_id` is required for guided decoding to work properly.')
        reduced_vocab = _reduced_vocabulary(tokenizer, eos_token_id)
        vocabulary = OutlinesVocabulary(Vocabulary(eos_token_id, reduced_vocab))
        tokenizer._outlines_vocabulary = vocabulary
        return vocabulary
    except AttributeError as e:
        raise ValueError(f'Cannot get the vocabulary of the tokenizer ({type(tokenizer)}). The tokenizer should have a get_vocab method.') from e

def get_cache_path() -> str:
    """Get the context object that contains previously-computed return values"""
    outlines_cache_dir = os.getenv('OUTLINES_CACHE_DIR')
    xdg_cache_home = os.getenv('XDG_CACHE_HOME')
    home_dir = os.path.expanduser('~')
    if outlines_cache_dir:
        return outlines_cache_dir
    elif xdg_cache_home:
        return os.path.join(xdg_cache_home, '.cache', 'outlines')
    elif os.path.isdir(home_dir) and home_dir != '/':
        return os.path.join(home_dir, '.cache', 'outlines')
    else:
        tempdir = tempfile.gettempdir()
        return os.path.join(tempdir, '.cache', 'outlines')

def get_cache():
    """Get the Cache instance to be used for index caching"""
    cache_dir = get_cache_path()
    if envs.VLLM_V0_USE_OUTLINES_CACHE:
        logger.warning('Enabling outlines cache. This is an unbounded on-disk cache. It may consume a lot of disk space and should not be used with untrusted clients.')
        cache = Cache(cache_dir, eviction_policy='none', cull_limit=0)
        outlines_version = importlib.metadata.version('outlines_core')
        cached_version = cache.get('__version__', None)
        if cached_version != outlines_version:
            cache.clear()
        cache.set('__version__', outlines_version)
        return cache
    else:
        return LRUCache(maxsize=128)