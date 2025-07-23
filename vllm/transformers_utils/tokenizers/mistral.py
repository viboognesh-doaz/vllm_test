from dataclasses import dataclass
from huggingface_hub import HfApi, hf_hub_download
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.base import SpecialTokens
from mistral_common.tokens.tokenizers.base import SpecialTokens
from mistral_common.tokens.tokenizers.base import SpecialTokens
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer as PublicMistralTokenizer
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer as PublicMistralTokenizer
from mistral_common.tokens.tokenizers.sentencepiece import SentencePieceTokenizer
from mistral_common.tokens.tokenizers.tekken import SpecialTokenPolicy, Tekkenizer
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union, cast
from vllm.entrypoints.chat_utils import ChatCompletionMessageParam
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer_base import TokenizerBase
from vllm.utils import is_list_of
import huggingface_hub
import os
import regex as re
if TYPE_CHECKING:
logger = init_logger(__name__)

@dataclass
class Encoding:
    input_ids: Union[list[int], list[list[int]]]

def maybe_serialize_tool_calls(request: 'ChatCompletionRequest'):
    for i, message in enumerate(request.messages):
        if message.get('role') == 'assistant':
            tool_calls_validator = message.get('tool_calls', ().__iter__())
            validated_tool_calls = []
            while True:
                try:
                    tool_call = next(tool_calls_validator)
                    validated_tool_calls.append(tool_call)
                except StopIteration:
                    break
            request.messages[i]['tool_calls'] = validated_tool_calls

def truncate_tool_call_ids(request: 'ChatCompletionRequest'):
    """Truncates tool call IDs for Mistral's ID requirements."""
    for i, message in enumerate(request.messages):
        if message.get('role') == 'assistant':
            tool_calls = message.get('tool_calls', [])
            for tool_call in tool_calls:
                if len(tool_call['id']) > 9:
                    logger.warning('Truncating tool call ID: %s to %s', tool_call['id'], tool_call['id'][-9:])
                    tool_call['id'] = tool_call['id'][-9:]
            request.messages[i]['tool_calls'] = tool_calls
        elif message.get('role') in {'tool_results', 'tool'}:
            if 'tool_call_id' in message:
                tool_call_id = message['tool_call_id']
                if len(tool_call_id) > 9:
                    logger.warning('Truncating tool_call_id: %s to %s', tool_call_id, tool_call_id[-9:])
                    tool_call_id = tool_call_id[-9:]
                request.messages[i]['tool_call_id'] = tool_call_id

def validate_request_params(request: 'ChatCompletionRequest'):
    if request.skip_special_tokens is not None and (not request.skip_special_tokens):
        raise ValueError('skip_special_tokens=False is not supported for Mistral tokenizers.')

def list_local_repo_files(repo_id: str, revision: Optional[str]) -> list[str]:
    repo_cache = os.path.join(huggingface_hub.constants.HF_HUB_CACHE, huggingface_hub.constants.REPO_ID_SEPARATOR.join(['models', *repo_id.split('/')]))
    if revision is None:
        revision_file = os.path.join(repo_cache, 'refs', 'main')
        if os.path.isfile(revision_file):
            with open(revision_file) as file:
                revision = file.read()
    if revision:
        revision_dir = os.path.join(repo_cache, 'snapshots', revision)
        if os.path.isdir(revision_dir):
            return os.listdir(revision_dir)
    return []

def find_tokenizer_file(files: list[str]):
    file_pattern = re.compile('^tokenizer\\.model\\.v.*$|^tekken\\.json$|^tokenizer\\.mm\\.model\\.v.*$')
    matched_files = [file for file in files if file_pattern.match(file)]
    if len(matched_files) > 1:
        raise OSError(f'Found {len(matched_files)} files matching the pattern: `{file_pattern.pattern}`. Make sure only one Mistral tokenizer is present in {files}.')
    elif len(matched_files) == 0:
        raise OSError(f'Found {len(matched_files)} files matching the pattern: `{file_pattern.pattern}`. Make sure that a Mistral tokenizer is present in {files}.')
    return matched_files[0]

def make_mistral_chat_completion_request(messages: list['ChatCompletionMessageParam'], tools: Optional[list[dict[str, Any]]]=None) -> 'ChatCompletionRequest':
    last_message = cast(dict[str, Any], messages[-1])
    if last_message['role'] == 'assistant':
        last_message['prefix'] = True
    for message in messages:
        _ = message.pop('reasoning_content', None)
        if message.get('role') in ('assistant', 'tool'):
            content = message.get('content')
            if isinstance(content, list):
                content = '\n'.join((chunk.get('text') for chunk in content))
                message['content'] = content
    if tools:
        for function in [tool['function'] for tool in tools if tool['type'] == 'function']:
            if function.get('parameters') is None:
                function['parameters'] = {}
    return ChatCompletionRequest(messages=messages, tools=tools)

class MistralTokenizer(TokenizerBase):

    def __init__(self, tokenizer: 'PublicMistralTokenizer') -> None:
        self.mistral = tokenizer
        self.instruct = tokenizer.instruct_tokenizer
        _mistral_version_str = self.instruct.tokenizer.version.value
        self.version: int = int(_mistral_version_str.split('v')[-1])
        tokenizer_ = tokenizer.instruct_tokenizer.tokenizer
        self.is_tekken = isinstance(tokenizer_, Tekkenizer)
        self.is_spm = isinstance(tokenizer_, SentencePieceTokenizer)
        if self.is_tekken:
            tokenizer_.special_token_policy = SpecialTokenPolicy.IGNORE
        elif self.is_spm:
            pass
        else:
            raise TypeError(f'Unsupported tokenizer: {type(tokenizer_)}')
        self._vocab = tokenizer_.vocab()
        self._vocab_dict = {token: idx for idx, token in enumerate(self._vocab)}
        self.tokenizer = tokenizer_
        self._max_token_id = self.vocab_size - 1

    @classmethod
    def from_pretrained(cls, path_or_repo_id: str, *, revision: Optional[str]=None) -> 'MistralTokenizer':
        if not Path(path_or_repo_id).exists():
            assert len(path_or_repo_id.split('/')) == 2, 'You have either provided a non-existent path: {path_or_repo_id} or an invalid HF Hub repo id.'
            tokenizer_file = cls._download_mistral_tokenizer_from_hf(path_or_repo_id, revision)
        elif Path(path_or_repo_id).is_dir():
            tokenizer_file_name = find_tokenizer_file(os.listdir(path_or_repo_id))
            tokenizer_file = str(Path(path_or_repo_id) / tokenizer_file_name)
        else:
            assert Path(path_or_repo_id).is_file(), f'Invalid path: {path_or_repo_id}'
            tokenizer_file = str(Path(path_or_repo_id))
        mistral_tokenizer = PublicMistralTokenizer.from_file(tokenizer_file)
        return cls(mistral_tokenizer)

    @staticmethod
    def _download_mistral_tokenizer_from_hf(tokenizer_name: str, revision: Optional[str]) -> str:
        try:
            hf_api = HfApi()
            files = hf_api.list_repo_files(repo_id=tokenizer_name, revision=revision)
        except ConnectionError as exc:
            files = list_local_repo_files(repo_id=tokenizer_name, revision=revision)
            if len(files) == 0:
                raise exc
        filename = find_tokenizer_file(files)
        tokenizer_file = hf_hub_download(tokenizer_name, filename=filename, revision=revision)
        return tokenizer_file

    @property
    def all_special_tokens_extended(self) -> list[str]:
        if hasattr(self.tokenizer, 'SPECIAL_TOKENS'):
            special_tokens = self.tokenizer.SPECIAL_TOKENS
        else:
            special_tokens = list(SpecialTokens)
        return [s.value if isinstance(s, SpecialTokens) else s for s in special_tokens]

    @property
    def all_special_tokens(self) -> list[str]:
        return self.all_special_tokens_extended

    @property
    def all_special_ids(self) -> list[int]:
        return [self.all_special_tokens.index(t) for t in self.all_special_tokens]

    @property
    def bos_token_id(self) -> int:
        return self.tokenizer.bos_id

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_id

    @property
    def sep_token(self) -> str:
        raise NotImplementedError()

    @property
    def pad_token(self) -> str:
        raise NotImplementedError()

    @property
    def is_fast(self) -> bool:
        return True

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    @property
    def max_token_id(self) -> int:
        return self._max_token_id

    def __len__(self) -> int:
        return self.vocab_size

    def __call__(self, text: Union[str, list[str], list[int]], text_pair: Optional[str]=None, add_special_tokens: bool=False, truncation: bool=False, max_length: Optional[int]=None):
        input_ids: Union[list[int], list[list[int]]]
        if is_list_of(text, str):
            input_ids_: list[list[int]] = []
            for p in text:
                each_input_ids = self.encode_one(p, truncation, max_length)
                input_ids_.append(each_input_ids)
            input_ids = input_ids_
        elif is_list_of(text, int):
            input_ids = text
        else:
            input_ids = self.encode_one(text, truncation, max_length)
        return Encoding(input_ids=input_ids)

    def get_vocab(self) -> dict[str, int]:
        return self._vocab_dict

    def get_added_vocab(self) -> dict[str, int]:
        return {}

    def encode_one(self, text: str, truncation: bool=False, max_length: Optional[int]=None) -> list[int]:
        input_ids = self.encode(text)
        if truncation:
            input_ids = input_ids[:max_length]
        return input_ids

    def encode(self, text: str, truncation: Optional[bool]=None, max_length: Optional[int]=None, add_special_tokens: Optional[bool]=None) -> list[int]:
        if add_special_tokens is not None:
            return self.tokenizer.encode(text, bos=add_special_tokens, eos=add_special_tokens)
        else:
            return self.tokenizer.encode(text, bos=True, eos=False)

    def apply_chat_template(self, messages: list['ChatCompletionMessageParam'], tools: Optional[list[dict[str, Any]]]=None, **kwargs) -> list[int]:
        request = make_mistral_chat_completion_request(messages, tools)
        encoded = self.mistral.encode_chat_completion(request)
        return encoded.tokens

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        if self.is_tekken:
            tokens = [t for t in tokens if t is SpecialTokens.tool_calls or t not in self.tokenizer._all_special_tokens]
            if any((isinstance(t, bytes) for t in tokens)):
                shift = self.tokenizer.num_special_tokens

                def _token_to_id(t: str):
                    t_bytes = t.encode('utf-8') if not isinstance(t, bytes) else t
                    try:
                        return shift + self.tokenizer._tekken_token2id_nospecial[t_bytes]
                    except KeyError:
                        logger.warning('Failed to convert token %s to id, replacing with <unk>', t_bytes)
                        return self.tokenizer.unk_id
                ids = [_token_to_id(t) for t in tokens]
                decoded = self.tokenizer.decode(ids)
            else:
                decoded = ''.join(tokens)
        else:
            special_tokens = {SpecialTokens.tool_calls}
            regular_tokens: list[str] = []
            decoded_list = []
            for token in tokens:
                if token in special_tokens:
                    if regular_tokens:
                        decoded_list.append(self.tokenizer.decode(regular_tokens))
                        regular_tokens = []
                    decoded_list.append(token)
                else:
                    regular_tokens.append(token)
            if regular_tokens:
                decoded_list.append(self.tokenizer.decode(regular_tokens))
            decoded = ''.join(decoded_list)
        return decoded

    def decode(self, ids: Union[list[int], int], skip_special_tokens: bool=True) -> str:
        assert skip_special_tokens, 'skip_special_tokens=False is not supported for Mistral tokenizers.'
        if isinstance(ids, int):
            ids = [ids]
        return self.tokenizer.decode(ids)

    def convert_ids_to_tokens(self, ids: list[int], skip_special_tokens: bool=True) -> list[str]:
        assert skip_special_tokens, 'skip_special_tokens=False is not supported for Mistral tokenizers.'
        assert self.is_tekken or self.is_spm, type(self.tokenizer)
        if self.is_tekken:
            ids = [i for i in ids if i > self.tokenizer.num_special_tokens or i == self.tokenizer.get_control_token(SpecialTokens.tool_calls)]
        tokens = [self.tokenizer.id_to_piece(id) for id in ids]
        if any(('�' in t for t in tokens)) and self.is_tekken:
            tokens = [self.tokenizer.id_to_byte_piece(id) for id in ids]
        return tokens