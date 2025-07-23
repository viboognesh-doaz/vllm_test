from .tokenizer import AnyTokenizer
from typing import Optional

def _replace_none_with_empty(tokens: list[Optional[str]]):
    for i, token in enumerate(tokens):
        if token is None:
            tokens[i] = ''

def _convert_tokens_to_string_with_added_encoders(tokenizer: AnyTokenizer, output_tokens: list[str], skip_special_tokens: bool, spaces_between_special_tokens: bool) -> str:
    sub_texts: list[str] = []
    current_sub_text: list[str] = []
    all_special_tokens = set(tokenizer.all_special_tokens)
    for token in output_tokens:
        if skip_special_tokens and token in all_special_tokens:
            continue
        if token in tokenizer.get_added_vocab():
            if current_sub_text:
                sub_text = tokenizer.convert_tokens_to_string(current_sub_text)
                sub_texts.append(sub_text)
                current_sub_text = []
            sub_texts.append(token)
        else:
            current_sub_text.append(token)
    if current_sub_text:
        sub_text = tokenizer.convert_tokens_to_string(current_sub_text)
        sub_texts.append(sub_text)
    if spaces_between_special_tokens:
        return ' '.join(sub_texts)
    else:
        return ''.join(sub_texts)
INITIAL_INCREMENTAL_DETOKENIZATION_OFFSET = 5

def convert_prompt_ids_to_tokens(tokenizer: AnyTokenizer, prompt_ids: list[int], skip_special_tokens: bool=False) -> tuple[list[str], int, int]:
    """Converts the prompt ids to tokens and returns the tokens and offsets
    for incremental detokenization.

    Note that not all tokens are converted to strings. Only the tokens that
    are necessary for incremental detokenization are converted to strings.
    """
    new_tokens = tokenizer.convert_ids_to_tokens(prompt_ids[-INITIAL_INCREMENTAL_DETOKENIZATION_OFFSET - 2:], skip_special_tokens=skip_special_tokens)
    read_offset = len(new_tokens)
    prefix_offset = max(read_offset - INITIAL_INCREMENTAL_DETOKENIZATION_OFFSET, 0)
    _replace_none_with_empty(new_tokens)
    return (new_tokens, prefix_offset, read_offset)

def convert_ids_list_to_tokens(tokenizer: AnyTokenizer, token_ids: list[int]) -> list[str]:
    """Detokenize the input ids individually.

    Args:
      tokenizer: tokenizer used by model under test
      token_ids: convert these tokens (Python list form)

    Returns:
      Python list of token string representations
    
    """
    token_str_lst = []
    for token_id in token_ids:
        token_str = tokenizer.decode([token_id])
        if token_str is None:
            token_str = ''
        token_str_lst.append(token_str)
    return token_str_lst

def detokenize_incrementally(tokenizer: AnyTokenizer, all_input_ids: list[int], prev_tokens: Optional[list[str]], prefix_offset: int, read_offset: int, skip_special_tokens: bool=False, spaces_between_special_tokens: bool=True) -> tuple[list[str], str, int, int]:
    """Detokenizes the input ids incrementally and returns the new tokens
    and the new text.

    If `prev_tokens` is None, this function will convert the input ids to
    tokens and return the tokens and the new text. Otherwise, it will return the
    new tokens and the new text.

    This function will also return the new prefix offset and the new read
    offset to be used in the next iteration.

    The offsets are necessary to defeat cleanup algorithms in the decode which
    decide to add a space or not depending on the surrounding ids.

    Args:
        tokenizer: The tokenizer to use.
        all_input_ids: The input ids. The last id is the new token id.
        prev_tokens: The previous tokens. If None, this function will convert
            the input ids to tokens and return the tokens and the new text.
        prefix_offset: The prefix offset.
        read_offset: The read offset.
        skip_special_tokens: Whether to skip special tokens.
        spaces_between_special_tokens: Whether to add spaces between special
            tokens.
    """
    new_token_id = all_input_ids[-1]
    is_first_iter = prev_tokens is None
    if is_first_iter:
        prev_tokens, prefix_offset, read_offset = convert_prompt_ids_to_tokens(tokenizer, all_input_ids[:-1], skip_special_tokens=skip_special_tokens)
    assert prev_tokens is not None
    if 0 <= new_token_id < len(tokenizer):
        new_tokens = tokenizer.convert_ids_to_tokens([new_token_id], skip_special_tokens=skip_special_tokens)
        if isinstance(new_tokens, str):
            new_tokens = [new_tokens]
    else:
        new_tokens = ['']
    output_tokens = prev_tokens + new_tokens
    if is_first_iter:
        new_tokens = output_tokens
    if tokenizer.is_fast or not tokenizer.get_added_vocab():
        prefix_text = tokenizer.convert_tokens_to_string(output_tokens[prefix_offset:read_offset])
        new_text = tokenizer.convert_tokens_to_string(output_tokens[prefix_offset:])
    else:
        prefix_text = _convert_tokens_to_string_with_added_encoders(tokenizer, output_tokens[prefix_offset:read_offset], skip_special_tokens=skip_special_tokens, spaces_between_special_tokens=spaces_between_special_tokens)
        new_text = _convert_tokens_to_string_with_added_encoders(tokenizer, output_tokens[prefix_offset:], skip_special_tokens=skip_special_tokens, spaces_between_special_tokens=spaces_between_special_tokens)
    if len(new_text) <= len(prefix_text) or new_text.endswith('�'):
        return (new_tokens, '', prefix_offset, read_offset)
    new_text = new_text[len(prefix_text):]
    return (new_tokens, new_text, read_offset, len(output_tokens))