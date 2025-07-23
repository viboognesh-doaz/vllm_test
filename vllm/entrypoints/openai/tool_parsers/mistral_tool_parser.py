from collections.abc import Sequence
from partial_json_parser.core.options import Allow
from pydantic import Field
from random import choices
from string import ascii_letters, digits
from typing import Union
from vllm.entrypoints.openai.protocol import ChatCompletionRequest, DeltaFunctionCall, DeltaMessage, DeltaToolCall, ExtractedToolCallInformation, FunctionCall, ToolCall
from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import ToolParser, ToolParserManager
from vllm.entrypoints.openai.tool_parsers.utils import extract_intermediate_diff
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import AnyTokenizer, MistralTokenizer
import json
import partial_json_parser
import regex as re
logger = init_logger(__name__)
ALPHANUMERIC = ascii_letters + digits

class MistralToolCall(ToolCall):
    id: str = Field(default_factory=lambda: MistralToolCall.generate_random_id())

    @staticmethod
    def generate_random_id():
        return ''.join(choices(ALPHANUMERIC, k=9))

    @staticmethod
    def is_valid_id(id: str) -> bool:
        return id.isalnum() and len(id) == 9

def _is_fn_name_regex_support(model_tokenizer: AnyTokenizer) -> bool:
    return isinstance(model_tokenizer, MistralTokenizer) and model_tokenizer.version >= 11

@ToolParserManager.register_module('mistral')
class MistralToolParser(ToolParser):
    """
    Tool call parser for Mistral 7B Instruct v0.3, intended for use with
    - [`mistral_common`](https://github.com/mistralai/mistral-common/)
    - the examples/tool_chat_template_mistral.jinja template.

    Used when --enable-auto-tool-choice --tool-call-parser mistral are all set
    """

    def __init__(self, tokenizer: AnyTokenizer):
        super().__init__(tokenizer)
        if not isinstance(self.model_tokenizer, MistralTokenizer):
            logger.info('Non-Mistral tokenizer detected when using a Mistral model...')
        self.prev_tool_call_arr: list[dict] = []
        self.current_tool_id: int = -1
        self.current_tool_name_sent: bool = False
        self.streamed_args_for_tool: list[str] = []
        self.bot_token = '[TOOL_CALLS]'
        self.bot_token_id = self.vocab.get(self.bot_token)
        self.tool_call_regex = re.compile('\\[{.*}\\]', re.DOTALL)
        if _is_fn_name_regex_support(self.model_tokenizer):
            self.fn_name_regex = re.compile('([a-zA-Z0-9_-]+)(\\{[\\s\\S]*?\\})(?=\\s*$|,|\\s)', re.DOTALL)
        else:
            self.fn_name_regex = None
        if self.bot_token_id is None:
            raise RuntimeError('Mistral Tool Parser could not locate the tool call token in the tokenizer!')

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        if not isinstance(self.model_tokenizer, MistralTokenizer) and request.tools and (request.tool_choice != 'none'):
            request.skip_special_tokens = False
        return request

    def extract_tool_calls(self, model_output: str, request: ChatCompletionRequest) -> ExtractedToolCallInformation:
        """
        Extract the tool calls from a complete model response. Requires
        find-and-replacing single quotes with double quotes for JSON parsing,
        make sure your tool call arguments don't ever include quotes!
        """
        if self.bot_token not in model_output:
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)
        tool_content = model_output.replace(self.bot_token, '').strip()
        try:
            try:
                if self.fn_name_regex:
                    matches = self.fn_name_regex.findall(tool_content)
                    function_call_arr = []
                    for match in matches:
                        fn_name = match[0]
                        args = match[1]
                        function_call_arr.append({'name': fn_name, 'arguments': json.loads(args)})
                else:
                    function_call_arr = json.loads(tool_content)
            except json.JSONDecodeError:
                raw_tool_call = self.tool_call_regex.findall(tool_content)[0]
                function_call_arr = json.loads(raw_tool_call)
            tool_calls: list[MistralToolCall] = [MistralToolCall(type='function', function=FunctionCall(name=raw_function_call['name'], arguments=json.dumps(raw_function_call['arguments'], ensure_ascii=False))) for raw_function_call in function_call_arr]
            content = model_output.split(self.bot_token)[0]
            return ExtractedToolCallInformation(tools_called=True, tool_calls=tool_calls, content=content if len(content) > 0 else None)
        except Exception:
            logger.exception('Error in extracting tool call from response.')
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=tool_content)

    def extract_tool_calls_streaming(self, previous_text: str, current_text: str, delta_text: str, previous_token_ids: Sequence[int], current_token_ids: Sequence[int], delta_token_ids: Sequence[int], request: ChatCompletionRequest) -> Union[DeltaMessage, None]:
        if self.bot_token not in current_text:
            return DeltaMessage(content=delta_text)
        if self.bot_token_id in delta_token_ids and len(delta_token_ids) == 1:
            return None
        flags = Allow.ALL if self.current_tool_name_sent else Allow.ALL & ~Allow.STR
        try:
            parsable_arr = current_text.split(self.bot_token)[-1]
            try:
                tool_call_arr: list[dict] = partial_json_parser.loads(parsable_arr, flags)
            except partial_json_parser.core.exceptions.MalformedJSON:
                logger.debug('not enough tokens to parse into JSON yet')
                return None
            current_tool_call: dict = tool_call_arr[self.current_tool_id] if len(tool_call_arr) > 0 else {}
            if len(tool_call_arr) == 0:
                return None
            elif len(tool_call_arr) > 0 and len(tool_call_arr) > self.current_tool_id + 1:
                if self.current_tool_id >= 0:
                    diff: Union[str, None] = current_tool_call.get('arguments')
                    if diff:
                        diff = json.dumps(diff, ensure_ascii=False).replace(self.streamed_args_for_tool[self.current_tool_id], '')
                        delta = DeltaMessage(tool_calls=[DeltaToolCall(index=self.current_tool_id, function=DeltaFunctionCall(arguments=diff).model_dump(exclude_none=True))])
                        self.streamed_args_for_tool[self.current_tool_id] += diff
                    else:
                        delta = None
                else:
                    delta = None
                self.current_tool_id = len(tool_call_arr) - 1
                self.current_tool_name_sent = False
                self.streamed_args_for_tool.append('')
                logger.debug('starting on new tool %d', self.current_tool_id)
                return delta
            if not self.current_tool_name_sent:
                function_name = current_tool_call.get('name')
                if function_name:
                    delta = DeltaMessage(tool_calls=[DeltaToolCall(index=self.current_tool_id, type='function', id=MistralToolCall.generate_random_id(), function=DeltaFunctionCall(name=function_name).model_dump(exclude_none=True))])
                    self.current_tool_name_sent = True
                else:
                    delta = None
            else:
                prev_arguments = self.prev_tool_call_arr[self.current_tool_id].get('arguments')
                cur_arguments = current_tool_call.get('arguments')
                new_text = delta_text.replace("'", '"')
                if '"}' in new_text:
                    new_text = new_text[:new_text.rindex('"}')]
                if not cur_arguments and (not prev_arguments):
                    delta = None
                elif not cur_arguments and prev_arguments:
                    logger.error('INVARIANT - impossible to have arguments reset mid-arguments')
                    delta = None
                elif cur_arguments and (not prev_arguments):
                    cur_arguments_json = json.dumps(cur_arguments, ensure_ascii=False)[:-2]
                    logger.debug('finding %s in %s', new_text, cur_arguments_json)
                    if new_text not in cur_arguments_json:
                        return None
                    arguments_delta = cur_arguments_json[:cur_arguments_json.rindex(new_text) + len(new_text)]
                    logger.debug('First tokens in arguments received: %s', arguments_delta)
                    delta = DeltaMessage(tool_calls=[DeltaToolCall(index=self.current_tool_id, function=DeltaFunctionCall(arguments=arguments_delta).model_dump(exclude_none=True))])
                    self.streamed_args_for_tool[self.current_tool_id] += arguments_delta
                elif cur_arguments and prev_arguments:
                    cur_args_json = json.dumps(cur_arguments, ensure_ascii=False)
                    prev_args_json = json.dumps(prev_arguments, ensure_ascii=False)
                    logger.debug('Searching for diff between \n%s\n%s', cur_args_json, prev_args_json)
                    argument_diff = extract_intermediate_diff(cur_args_json, prev_args_json)
                    logger.debug('got arguments diff: %s', argument_diff)
                    delta = DeltaMessage(tool_calls=[DeltaToolCall(index=self.current_tool_id, function=DeltaFunctionCall(arguments=argument_diff).model_dump(exclude_none=True))])
                    self.streamed_args_for_tool[self.current_tool_id] += argument_diff
                else:
                    delta = None
            self.prev_tool_call_arr = tool_call_arr
            return delta
        except Exception:
            logger.exception('Error trying to handle streaming tool call.')
            logger.debug('Skipping chunk as a result of tool streaming extraction error')
            return None