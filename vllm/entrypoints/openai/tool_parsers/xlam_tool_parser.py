from collections.abc import Sequence
from typing import Any, Optional, Union
from vllm.entrypoints.chat_utils import random_tool_call_id
from vllm.entrypoints.openai.protocol import ChatCompletionRequest, DeltaFunctionCall, DeltaMessage, DeltaToolCall, ExtractedToolCallInformation, FunctionCall, ToolCall
from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import ToolParser, ToolParserManager
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.utils import random_uuid
import json
import regex as re
logger = init_logger(__name__)

@ToolParserManager.register_module('xlam')
class xLAMToolParser(ToolParser):

    def __init__(self, tokenizer: AnyTokenizer):
        super().__init__(tokenizer)
        self.prev_tool_calls: list[dict] = []
        self.current_tool_id = -1
        self.current_tool_name_sent = False
        self.streamed_args: list[str] = []
        self.current_tools_sent: list[bool] = []
        self.prev_tool_call_arr = []
        self.json_code_block_patterns = ['```(?:json)?\\s*([\\s\\S]*?)```', '\\[TOOL_CALLS\\]([\\s\\S]*?)(?=\\n|$)', '<tool_call>([\\s\\S]*?)</tool_call>']
        self.thinking_tag_pattern = '</think>([\\s\\S]*)'
        self.streaming_state: dict[str, Any] = {'current_tool_index': -1, 'tool_ids': [], 'sent_tools': []}

    def preprocess_model_output(self, model_output: str) -> tuple[Optional[str], Optional[str]]:
        """
        Preprocess the model output to extract content and potential tool calls.
        Returns:
            Tuple of (content, potential_tool_calls_json)
        """
        thinking_match = re.search(self.thinking_tag_pattern, model_output)
        if thinking_match:
            content = model_output[:thinking_match.start() + len('</think>')].strip()
            thinking_content = thinking_match.group(1).strip()
            try:
                json.loads(thinking_content)
                return (content, thinking_content)
            except json.JSONDecodeError:
                for json_pattern in self.json_code_block_patterns:
                    json_matches = re.findall(json_pattern, thinking_content)
                    if json_matches:
                        for json_str in json_matches:
                            try:
                                json.loads(json_str)
                                return (content, json_str)
                            except json.JSONDecodeError:
                                continue
        for json_pattern in self.json_code_block_patterns:
            json_matches = re.findall(json_pattern, model_output)
            if json_matches:
                for json_str in json_matches:
                    try:
                        json.loads(json_str)
                        content = re.sub(json_pattern, '', model_output).strip()
                        return (content, json_str)
                    except json.JSONDecodeError:
                        continue
        if model_output.strip().startswith('['):
            try:
                json.loads(model_output)
                return (None, model_output)
            except json.JSONDecodeError:
                if '{' in model_output and 'name' in model_output and ('arguments' in model_output):
                    return (None, model_output)
        return (model_output, None)

    def extract_tool_calls(self, model_output: str, request: ChatCompletionRequest) -> ExtractedToolCallInformation:
        """
        Extract tool calls from a complete model output.
        """
        try:
            content, potential_tool_calls = self.preprocess_model_output(model_output)
            if not potential_tool_calls:
                return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=content)
            tool_calls_data = json.loads(potential_tool_calls)
            if not isinstance(tool_calls_data, list):
                logger.debug('Tool calls data is not an array')
                return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=content or model_output)
            tool_calls: list[ToolCall] = []
            for idx, call in enumerate(tool_calls_data):
                if not isinstance(call, dict) or 'name' not in call or 'arguments' not in call:
                    logger.debug('Invalid tool call format at index %d', idx)
                    continue
                tool_call = ToolCall(id=f'call_{idx}_{random_uuid()}', type='function', function=FunctionCall(name=call['name'], arguments=json.dumps(call['arguments']) if isinstance(call['arguments'], dict) else call['arguments']))
                tool_calls.append(tool_call)
            return ExtractedToolCallInformation(tools_called=len(tool_calls) > 0, tool_calls=tool_calls, content=content)
        except Exception as e:
            logger.exception('Error extracting tool calls: %s', str(e))
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

    def extract_tool_calls_streaming(self, previous_text: str, current_text: str, delta_text: str, previous_token_ids: Sequence[int], current_token_ids: Sequence[int], delta_token_ids: Sequence[int], request: ChatCompletionRequest) -> Union[DeltaMessage, None]:
        """
        Extract tool calls for streaming mode.
        """
        is_function_call = current_text.strip().startswith('[')
        if not is_function_call:
            return DeltaMessage(content=delta_text)
        try:
            if not hasattr(self, 'streaming_state'):
                self.streaming_state = {'current_tool_index': -1, 'tool_ids': [], 'sent_tools': []}
            try:
                parsed_tools = json.loads(current_text)
                if isinstance(parsed_tools, list):
                    self.prev_tool_call_arr = parsed_tools
            except json.JSONDecodeError:
                pass
            if hasattr(self, 'current_tools_sent') and len(self.current_tools_sent) > 0:
                if len(self.current_tools_sent) == 1 and self.current_tools_sent[0] is False:
                    name_pattern = '"name"\\s*:\\s*"([^"]+)"'
                    name_match = re.search(name_pattern, current_text)
                    if name_match:
                        function_name = name_match.group(1)
                        tool_id = random_tool_call_id()
                        delta = DeltaMessage(tool_calls=[DeltaToolCall(index=0, type='function', id=tool_id, function=DeltaFunctionCall(name=function_name).model_dump(exclude_none=True))])
                        self.current_tools_sent = [True]
                        self.current_tool_id = 0
                        self.streaming_state['current_tool_index'] = 0
                        if len(self.streaming_state['sent_tools']) == 0:
                            self.streaming_state['sent_tools'].append({'sent_name': True, 'sent_arguments_prefix': False, 'sent_arguments': ''})
                        else:
                            self.streaming_state['sent_tools'][0]['sent_name'] = True
                        self.current_tool_name_sent = True
                        return delta
            name_pattern = '"name"\\s*:\\s*"([^"]+)"'
            name_matches = list(re.finditer(name_pattern, current_text))
            tool_count = len(name_matches)
            if tool_count == 0:
                return None
            while len(self.streaming_state['sent_tools']) < tool_count:
                self.streaming_state['sent_tools'].append({'sent_name': False, 'sent_arguments_prefix': False, 'sent_arguments': ''})
            while len(self.streaming_state['tool_ids']) < tool_count:
                self.streaming_state['tool_ids'].append(None)
            current_idx = self.streaming_state['current_tool_index']
            if current_idx == -1 or current_idx < tool_count - 1:
                next_idx = current_idx + 1
                if next_idx < tool_count and (not self.streaming_state['sent_tools'][next_idx]['sent_name']):
                    self.streaming_state['current_tool_index'] = next_idx
                    self.current_tool_id = next_idx
                    current_idx = next_idx
                    tool_name = name_matches[current_idx].group(1)
                    tool_id = f'call_{current_idx}_{random_uuid()}'
                    self.streaming_state['tool_ids'][current_idx] = tool_id
                    delta = DeltaMessage(tool_calls=[DeltaToolCall(index=current_idx, type='function', id=tool_id, function=DeltaFunctionCall(name=tool_name).model_dump(exclude_none=True))])
                    self.streaming_state['sent_tools'][current_idx]['sent_name'] = True
                    self.current_tool_name_sent = True
                    while len(self.streamed_args) <= current_idx:
                        self.streamed_args.append('')
                    return delta
            if current_idx >= 0 and current_idx < tool_count:
                empty_args_pattern = '"name"\\s*:\\s*"[^"]+"\\s*,\\s*"arguments"\\s*:\\s*\\{\\s*\\}'
                empty_args_match = re.search(empty_args_pattern, current_text)
                if empty_args_match and empty_args_match.start() > 0:
                    empty_args_tool_idx = 0
                    for i in range(tool_count):
                        if i == current_idx:
                            if not self.streaming_state['sent_tools'][current_idx]['sent_arguments_prefix']:
                                self.streaming_state['sent_tools'][current_idx]['sent_arguments_prefix'] = True
                                self.streaming_state['sent_tools'][current_idx]['sent_arguments'] = '{}'
                                while len(self.streamed_args) <= current_idx:
                                    self.streamed_args.append('')
                                self.streamed_args[current_idx] += '{}'
                                delta = DeltaMessage(tool_calls=[DeltaToolCall(index=current_idx, function=DeltaFunctionCall(arguments='{}').model_dump(exclude_none=True))])
                                if current_idx < tool_count - 1:
                                    self.streaming_state['current_tool_index'] += 1
                                    self.current_tool_id = self.streaming_state['current_tool_index']
                                return delta
                args_pattern = '"name"\\s*:\\s*"[^"]+"\\s*,\\s*"arguments"\\s*:\\s*(\\{(?:[^{}]|(?:\\{[^{}]*\\}))*\\})'
                args_matches = list(re.finditer(args_pattern, current_text))
                if current_idx < len(args_matches):
                    args_text = args_matches[current_idx].group(1)
                    is_last_tool = current_idx == tool_count - 1
                    if not is_last_tool:
                        next_tool_pos = current_text.find('},{', args_matches[current_idx].start())
                        if next_tool_pos != -1:
                            args_end_pos = next_tool_pos + 1
                            args_text = current_text[args_matches[current_idx].start():args_end_pos].split('"arguments":')[1].strip()
                    sent_args = self.streaming_state['sent_tools'][current_idx]['sent_arguments']
                    if not self.streaming_state['sent_tools'][current_idx]['sent_arguments_prefix'] and args_text.startswith('{'):
                        self.streaming_state['sent_tools'][current_idx]['sent_arguments_prefix'] = True
                        self.streaming_state['sent_tools'][current_idx]['sent_arguments'] = '{'
                        while len(self.streamed_args) <= current_idx:
                            self.streamed_args.append('')
                        self.streamed_args[current_idx] += '{'
                        delta = DeltaMessage(tool_calls=[DeltaToolCall(index=current_idx, function=DeltaFunctionCall(arguments='{').model_dump(exclude_none=True))])
                        return delta
                    if args_text.startswith(sent_args):
                        args_diff = args_text[len(sent_args):]
                        if args_diff:
                            self.streaming_state['sent_tools'][current_idx]['sent_arguments'] = args_text
                            while len(self.streamed_args) <= current_idx:
                                self.streamed_args.append('')
                            self.streamed_args[current_idx] += args_diff
                            delta = DeltaMessage(tool_calls=[DeltaToolCall(index=current_idx, function=DeltaFunctionCall(arguments=args_diff).model_dump(exclude_none=True))])
                            return delta
                    if args_text.endswith('}') and args_text == sent_args:
                        if current_idx < tool_count - 1:
                            self.streaming_state['current_tool_index'] += 1
                            self.current_tool_id = self.streaming_state['current_tool_index']
            return None
        except Exception as e:
            logger.exception(f'Error in streaming tool calls: {e}')
            return DeltaMessage(content=delta_text)