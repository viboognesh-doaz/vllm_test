from collections.abc import Sequence
from partial_json_parser.core.options import Allow
from typing import Union
from vllm.entrypoints.chat_utils import random_tool_call_id
from vllm.entrypoints.openai.protocol import ChatCompletionRequest, DeltaFunctionCall, DeltaMessage, DeltaToolCall, ExtractedToolCallInformation, FunctionCall, ToolCall
from vllm.entrypoints.openai.tool_parsers import ToolParser, ToolParserManager
from vllm.entrypoints.openai.tool_parsers.utils import extract_intermediate_diff
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.transformers_utils.tokenizers import MistralTokenizer
import json
import partial_json_parser
import regex as re
logger = init_logger(__name__)

@ToolParserManager.register_module('jamba')
class JambaToolParser(ToolParser):

    def __init__(self, tokenizer: AnyTokenizer):
        super().__init__(tokenizer)
        if isinstance(self.model_tokenizer, MistralTokenizer):
            raise ValueError('Detected a MistralTokenizer tokenizer when using a Jamba model')
        self.current_tool_name_sent: bool = False
        self.prev_tool_call_arr: list[dict] = []
        self.current_tool_id: int = -1
        self.streamed_args_for_tool: list[str] = []
        self.tool_calls_start_token: str = '<tool_calls>'
        self.tool_calls_end_token: str = '</tool_calls>'
        self.tool_calls_regex = re.compile(f'{self.tool_calls_start_token}(.*?){self.tool_calls_end_token}', re.DOTALL)
        if not self.model_tokenizer:
            raise ValueError('The model tokenizer must be passed to the ToolParser constructor during construction.')
        self.tool_calls_start_token_id = self.vocab.get(self.tool_calls_start_token)
        self.tool_calls_end_token_id = self.vocab.get(self.tool_calls_end_token)
        if self.tool_calls_start_token_id is None or self.tool_calls_end_token_id is None:
            raise RuntimeError('Jamba Tool parser could not locate tool calls start/end tokens in the tokenizer!')

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        if request.tools and request.tool_choice != 'none':
            request.skip_special_tokens = False
        return request

    def extract_tool_calls(self, model_output: str, request: ChatCompletionRequest) -> ExtractedToolCallInformation:
        if self.tool_calls_start_token not in model_output:
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)
        else:
            try:
                function_calls = self.tool_calls_regex.findall(model_output)[0]
                raw_function_calls = json.loads(function_calls)
                tool_calls = [ToolCall(type='function', function=FunctionCall(name=function_call['name'], arguments=json.dumps(function_call['arguments'], ensure_ascii=False))) for function_call in raw_function_calls]
                content = model_output[:model_output.find(self.tool_calls_start_token)]
                return ExtractedToolCallInformation(tools_called=True, tool_calls=tool_calls, content=content if len(content) > 0 and content != ' ' else None)
            except Exception:
                logger.exception('Error in extracting tool call from response.')
                return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

    def extract_tool_calls_streaming(self, previous_text: str, current_text: str, delta_text: str, previous_token_ids: Sequence[int], current_token_ids: Sequence[int], delta_token_ids: Sequence[int], request: ChatCompletionRequest) -> Union[DeltaMessage, None]:
        if self.tool_calls_start_token not in current_text:
            return DeltaMessage(content=delta_text)
        if self.tool_calls_start_token_id in delta_token_ids and len(delta_token_ids) == 1:
            return None
        flags = Allow.ALL if self.current_tool_name_sent else Allow.ALL & ~Allow.STR
        try:
            parsable_arr = current_text.split(self.tool_calls_start_token)[-1].split(self.tool_calls_end_token)[0]
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
                    delta = DeltaMessage(tool_calls=[DeltaToolCall(index=self.current_tool_id, type='function', id=random_tool_call_id(), function=DeltaFunctionCall(name=function_name).model_dump(exclude_none=True))])
                    self.current_tool_name_sent = True
                else:
                    delta = None
            else:
                prev_arguments = self.prev_tool_call_arr[self.current_tool_id].get('arguments')
                cur_arguments = current_tool_call.get('arguments')
                new_text = delta_text.replace("'", '"')
                if not cur_arguments and (not prev_arguments):
                    delta = None
                elif not cur_arguments and prev_arguments:
                    logger.error('INVARIANT - impossible to have arguments reset mid-arguments')
                    delta = None
                elif cur_arguments and (not prev_arguments):
                    cur_arguments_json = json.dumps(cur_arguments, ensure_ascii=False)
                    logger.debug('finding %s in %s', new_text, cur_arguments_json)
                    arguments_delta = cur_arguments_json[:cur_arguments_json.index(new_text) + len(new_text)]
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