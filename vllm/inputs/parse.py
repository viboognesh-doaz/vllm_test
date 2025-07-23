from .data import EmbedsPrompt, ExplicitEncoderDecoderPrompt, ProcessorInputs, PromptType, SingletonInputs, SingletonPrompt, TextPrompt, TokensPrompt
from collections.abc import Sequence
from typing import Literal, Optional, TypedDict, Union, cast, overload
from typing_extensions import TypeIs
from vllm.utils import is_list_of

class ParsedText(TypedDict):
    content: str
    is_tokens: Literal[False]

class ParsedTokens(TypedDict):
    content: list[int]
    is_tokens: Literal[True]

@overload
def parse_and_batch_prompt(prompt: Union[str, list[str]]) -> Sequence[ParsedText]:
    ...

@overload
def parse_and_batch_prompt(prompt: Union[list[int], list[list[int]]]) -> Sequence[ParsedTokens]:
    ...

def parse_and_batch_prompt(prompt: Union[str, list[str], list[int], list[list[int]]]) -> Union[Sequence[ParsedText], Sequence[ParsedTokens]]:
    if isinstance(prompt, str):
        return [ParsedText(content=prompt, is_tokens=False)]
    if isinstance(prompt, list):
        if len(prompt) == 0:
            raise ValueError('please provide at least one prompt')
        if is_list_of(prompt, str):
            prompt = cast(list[str], prompt)
            return [ParsedText(content=elem, is_tokens=False) for elem in prompt]
        if is_list_of(prompt, int):
            prompt = cast(list[int], prompt)
            return [ParsedTokens(content=prompt, is_tokens=True)]
        if is_list_of(prompt, list):
            prompt = cast(list[list[int]], prompt)
            if len(prompt[0]) == 0:
                raise ValueError('please provide at least one prompt')
            if is_list_of(prompt[0], int):
                return [ParsedTokens(content=elem, is_tokens=True) for elem in prompt]
    raise TypeError('prompt must be a string, array of strings, array of tokens, or array of token arrays')

class ParsedStrPrompt(TypedDict):
    type: Literal['str']
    content: str

class ParsedTextPrompt(TypedDict):
    type: Literal['text']
    content: TextPrompt

class ParsedTokensPrompt(TypedDict):
    type: Literal['tokens']
    content: TokensPrompt

class ParsedEmbedsPrompt(TypedDict):
    type: Literal['embeds']
    content: EmbedsPrompt
ParsedSingletonPrompt = Union[ParsedStrPrompt, ParsedTextPrompt, ParsedTokensPrompt, ParsedEmbedsPrompt]

@overload
def parse_singleton_prompt(prompt: str) -> ParsedStrPrompt:
    ...

@overload
def parse_singleton_prompt(prompt: TextPrompt) -> ParsedTextPrompt:
    ...

@overload
def parse_singleton_prompt(prompt: TokensPrompt) -> ParsedTokensPrompt:
    ...

@overload
def parse_singleton_prompt(prompt: EmbedsPrompt) -> ParsedEmbedsPrompt:
    ...

def parse_singleton_prompt(prompt: SingletonPrompt) -> ParsedSingletonPrompt:
    if isinstance(prompt, str):
        return ParsedStrPrompt(type='str', content=prompt)
    elif isinstance(prompt, dict):
        if 'prompt_embeds' in prompt:
            return ParsedEmbedsPrompt(type='embeds', content=prompt)
        elif 'prompt_token_ids' in prompt:
            return ParsedTokensPrompt(type='tokens', content=prompt)
        elif 'prompt' in prompt:
            return ParsedTextPrompt(type='text', content=prompt)
    raise TypeError('inputs must be a string, TextPrompt, TokensPrompt, or EmbedsPrompt')

def is_explicit_encoder_decoder_prompt(prompt: PromptType) -> TypeIs[ExplicitEncoderDecoderPrompt]:
    return isinstance(prompt, dict) and 'encoder_prompt' in prompt

def split_enc_dec_inputs(inputs: ProcessorInputs) -> tuple[Optional[SingletonInputs], SingletonInputs]:
    if 'encoder' in inputs and 'decoder' in inputs:
        return (inputs['encoder'], inputs['decoder'])
    return (None, inputs)