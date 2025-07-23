from collections.abc import Sequence
from json import JSONDecoder
from partial_json_parser.core.options import Allow
from transformers import PreTrainedTokenizerBase
from typing import Union
from vllm.entrypoints.chat_utils import random_tool_call_id
from vllm.entrypoints.openai.protocol import ChatCompletionRequest, DeltaFunctionCall, DeltaMessage, DeltaToolCall, ExtractedToolCallInformation, FunctionCall, ToolCall
from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import ToolParser, ToolParserManager
from vllm.entrypoints.openai.tool_parsers.utils import find_common_prefix, is_complete_json, partial_json_loads
from vllm.logger import init_logger
import json
import partial_json_parser
import regex as re
logger = init_logger(__name__)

@ToolParserManager.register_module('llama3_json')
@ToolParserManager.register_module('llama4_json')
class Llama3JsonToolParser(ToolParser):
    """
    Tool call parser for Llama 3.1 models intended for use with the
    examples/tool_chat_template_llama.jinja template.

    Used when --enable-auto-tool-choice --tool-call-parser llama3_json 
    are all set
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        super().__init__(tokenizer)
        self.prev_tool_call_arr: list[dict] = []
        self.current_tool_id: int = -1
        self.current_tool_name_sent: bool = False
        self.streamed_args_for_tool: list[str] = []
        self.bot_token = '<|python_tag|>'
        self.bot_token_id = tokenizer.encode(self.bot_token, add_special_tokens=False)[0]
        self.tool_call_regex = re.compile('\\[{.*?}\\]', re.DOTALL)

    def extract_tool_calls(self, model_output: str, request: ChatCompletionRequest) -> ExtractedToolCallInformation:
        """
        Extract the tool calls from a complete model response.
        """
        if not (model_output.startswith(self.bot_token) or model_output.startswith('{')):
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)
        try:
            dec = JSONDecoder()
            function_call_arr = []
            start_idx = len(self.bot_token) if model_output.startswith(self.bot_token) else 0
            while start_idx < len(model_output):
                obj, end_idx = dec.raw_decode(model_output[start_idx:])
                start_idx += end_idx + len('; ')
                function_call_arr.append(obj)
            tool_calls: list[ToolCall] = [ToolCall(type='function', function=FunctionCall(name=raw_function_call['name'], arguments=json.dumps(raw_function_call['arguments'] if 'arguments' in raw_function_call else raw_function_call['parameters'], ensure_ascii=False))) for raw_function_call in function_call_arr]
            ret = ExtractedToolCallInformation(tools_called=True, tool_calls=tool_calls, content=None)
            return ret
        except Exception:
            logger.exception('Error in extracting tool call from response.')
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

    def extract_tool_calls_streaming(self, previous_text: str, current_text: str, delta_text: str, previous_token_ids: Sequence[int], current_token_ids: Sequence[int], delta_token_ids: Sequence[int], request: ChatCompletionRequest) -> Union[DeltaMessage, None]:
        if not (current_text.startswith(self.bot_token) or current_text.startswith('{')):
            return DeltaMessage(content=delta_text)
        flags = Allow.ALL if self.current_tool_name_sent else Allow.ALL & ~Allow.STR
        try:
            tool_call_arr = []
            is_complete = []
            try:
                start_idx = len(self.bot_token) if current_text.startswith(self.bot_token) else 0
                while start_idx < len(current_text):
                    obj, end_idx = partial_json_loads(current_text[start_idx:], flags)
                    is_complete.append(is_complete_json(current_text[start_idx:start_idx + end_idx]))
                    start_idx += end_idx + len('; ')
                    if 'parameters' in obj:
                        assert 'arguments' not in obj, 'model generated both parameters and arguments'
                        obj['arguments'] = obj['parameters']
                    tool_call_arr.append(obj)
            except partial_json_parser.core.exceptions.MalformedJSON:
                logger.debug('not enough tokens to parse into JSON yet')
                return None
            current_tool_call: dict = tool_call_arr[self.current_tool_id] if len(tool_call_arr) > 0 else {}
            if len(tool_call_arr) == 0:
                return None
            elif len(tool_call_arr) > 0 and len(tool_call_arr) > self.current_tool_id + 1:
                if self.current_tool_id >= 0:
                    cur_arguments = current_tool_call.get('arguments')
                    if cur_arguments:
                        cur_args_json = json.dumps(cur_arguments, ensure_ascii=False)
                        sent = len(self.streamed_args_for_tool[self.current_tool_id])
                        argument_diff = cur_args_json[sent:]
                        logger.debug('got arguments diff: %s', argument_diff)
                        delta = DeltaMessage(tool_calls=[DeltaToolCall(index=self.current_tool_id, function=DeltaFunctionCall(arguments=argument_diff).model_dump(exclude_none=True))])
                        self.streamed_args_for_tool[self.current_tool_id] += argument_diff
                    else:
                        delta = None
                else:
                    delta = None
                self.current_tool_id = len(tool_call_arr) - 1
                self.current_tool_name_sent = False
                self.streamed_args_for_tool.append('')
                logger.debug('starting on new tool %d', self.current_tool_id)
                return delta
            elif not self.current_tool_name_sent:
                function_name = current_tool_call.get('name')
                if function_name:
                    delta = DeltaMessage(tool_calls=[DeltaToolCall(index=self.current_tool_id, type='function', id=random_tool_call_id(), function=DeltaFunctionCall(name=function_name).model_dump(exclude_none=True))])
                    self.current_tool_name_sent = True
                else:
                    delta = None
            else:
                cur_arguments = current_tool_call.get('arguments')
                delta = None
                if cur_arguments:
                    sent = len(self.streamed_args_for_tool[self.current_tool_id])
                    cur_args_json = json.dumps(cur_arguments, ensure_ascii=False)
                    prev_arguments = self.prev_tool_call_arr[self.current_tool_id].get('arguments')
                    argument_diff = None
                    if is_complete[self.current_tool_id]:
                        argument_diff = cur_args_json[sent:]
                    elif prev_arguments:
                        prev_args_json = json.dumps(prev_arguments, ensure_ascii=False)
                        if cur_args_json != prev_args_json:
                            prefix = find_common_prefix(prev_args_json, cur_args_json)
                            argument_diff = prefix[sent:]
                    if argument_diff is not None:
                        delta = DeltaMessage(tool_calls=[DeltaToolCall(index=self.current_tool_id, function=DeltaFunctionCall(arguments=argument_diff).model_dump(exclude_none=True))])
                        self.streamed_args_for_tool[self.current_tool_id] += argument_diff
            self.prev_tool_call_arr = tool_call_arr
            return delta
        except Exception:
            logger.exception('Error trying to handle streaming tool call.')
            logger.debug('Skipping chunk as a result of tool streaming extraction error')
            return None