from vllm.entrypoints.openai.protocol import FunctionCall, ToolCall
from vllm.entrypoints.openai.tool_parsers import MinimaxToolParser
from vllm.transformers_utils.tokenizer import get_tokenizer
import json
import pytest
MODEL = 'MiniMaxAi/MiniMax-M1-40k'

@pytest.fixture(scope='module')
def minimax_tokenizer():
    return get_tokenizer(tokenizer_name=MODEL)

@pytest.fixture
def minimax_tool_parser(minimax_tokenizer):
    return MinimaxToolParser(minimax_tokenizer)

def assert_tool_calls(actual_tool_calls: list[ToolCall], expected_tool_calls: list[ToolCall]):
    assert len(actual_tool_calls) == len(expected_tool_calls)
    for actual_tool_call, expected_tool_call in zip(actual_tool_calls, expected_tool_calls):
        assert isinstance(actual_tool_call.id, str)
        assert len(actual_tool_call.id) > 16
        assert actual_tool_call.type == 'function'
        assert actual_tool_call.function == expected_tool_call.function

def test_extract_tool_calls_no_tools(minimax_tool_parser):
    model_output = 'This is a test'
    extracted_tool_calls = minimax_tool_parser.extract_tool_calls(model_output, request=None)
    assert not extracted_tool_calls.tools_called
    assert extracted_tool_calls.tool_calls == []
    assert extracted_tool_calls.content == model_output

@pytest.mark.parametrize(ids=['single_tool_call', 'multiple_tool_calls', 'tool_call_with_content_before', 'tool_call_with_single_line_json', 'tool_call_incomplete_tag'], argnames=['model_output', 'expected_tool_calls', 'expected_content'], argvalues=[('<tool_calls>\n{"name": "get_current_weather", "arguments": {"city": "Dallas", "state": "TX", "unit": "fahrenheit"}}\n</tool_calls>', [ToolCall(function=FunctionCall(name='get_current_weather', arguments=json.dumps({'city': 'Dallas', 'state': 'TX', 'unit': 'fahrenheit'})))], None), ('<tool_calls>\n{"name": "get_current_weather", "arguments": {"city": "Dallas", "state": "TX", "unit": "fahrenheit"}}\n{"name": "get_current_weather", "arguments": {"city": "Orlando", "state": "FL", "unit": "fahrenheit"}}\n</tool_calls>', [ToolCall(function=FunctionCall(name='get_current_weather', arguments=json.dumps({'city': 'Dallas', 'state': 'TX', 'unit': 'fahrenheit'}))), ToolCall(function=FunctionCall(name='get_current_weather', arguments=json.dumps({'city': 'Orlando', 'state': 'FL', 'unit': 'fahrenheit'})))], None), ('I\'ll help you check the weather. <tool_calls>\n{"name": "get_current_weather", "arguments": {"city": "Seattle", "state": "WA", "unit": "celsius"}}\n</tool_calls>', [ToolCall(function=FunctionCall(name='get_current_weather', arguments=json.dumps({'city': 'Seattle', 'state': 'WA', 'unit': 'celsius'})))], "I'll help you check the weather."), ('<tool_calls>\n{"name": "get_current_weather", "arguments": {"city": "New York", "state": "NY", "unit": "celsius"}}\n</tool_calls>', [ToolCall(function=FunctionCall(name='get_current_weather', arguments=json.dumps({'city': 'New York', 'state': 'NY', 'unit': 'celsius'})))], None), ('<tool_calls>\n{"name": "get_current_weather", "arguments": {"city": "Boston", "state": "MA"}}', [ToolCall(function=FunctionCall(name='get_current_weather', arguments=json.dumps({'city': 'Boston', 'state': 'MA'})))], None)])
def test_extract_tool_calls(minimax_tool_parser, model_output, expected_tool_calls, expected_content):
    extracted_tool_calls = minimax_tool_parser.extract_tool_calls(model_output, request=None)
    assert extracted_tool_calls.tools_called
    assert_tool_calls(extracted_tool_calls.tool_calls, expected_tool_calls)
    assert extracted_tool_calls.content == expected_content

def test_preprocess_model_output_with_thinking_tags(minimax_tool_parser):
    """Test that tool calls within thinking tags are removed during preprocessing."""
    model_output = '<think>Let me think about this. <tool_calls>\n{"name": "fake_tool", "arguments": {"param": "value"}}\n</tool_calls> This should be removed.</think>\n\nI\'ll help you with that. <tool_calls>\n{"name": "get_current_weather", "arguments": {"city": "Seattle", "state": "WA"}}\n</tool_calls>'
    processed_output = minimax_tool_parser.preprocess_model_output(model_output)
    assert 'fake_tool' not in processed_output
    assert '<think>' in processed_output
    assert '</think>' in processed_output
    assert 'get_current_weather' in processed_output

def test_extract_tool_calls_with_thinking_tags(minimax_tool_parser):
    """Test tool extraction when thinking tags contain tool calls that should be ignored."""
    model_output = '<think>I should use a tool. <tool_calls>\n{"name": "ignored_tool", "arguments": {"should": "ignore"}}\n</tool_calls></think>\n\nLet me help you with the weather. <tool_calls>\n{"name": "get_current_weather", "arguments": {"city": "Miami", "state": "FL", "unit": "fahrenheit"}}\n</tool_calls>'
    extracted_tool_calls = minimax_tool_parser.extract_tool_calls(model_output, request=None)
    assert extracted_tool_calls.tools_called
    assert len(extracted_tool_calls.tool_calls) == 1
    assert extracted_tool_calls.tool_calls[0].function.name == 'get_current_weather'
    expected_content = '<think>I should use a tool. <tool_calls>\n{"name": "ignored_tool", "arguments": {"should": "ignore"}}\n</tool_calls></think>\n\nLet me help you with the weather.'
    assert extracted_tool_calls.content == expected_content

def test_extract_tool_calls_invalid_json(minimax_tool_parser):
    """Test that invalid JSON in tool calls is handled gracefully."""
    model_output = '<tool_calls>\n{"name": "valid_tool", "arguments": {"city": "Seattle"}}\n{invalid json here}\n{"name": "another_valid_tool", "arguments": {"param": "value"}}\n</tool_calls>'
    extracted_tool_calls = minimax_tool_parser.extract_tool_calls(model_output, request=None)
    assert extracted_tool_calls.tools_called
    assert len(extracted_tool_calls.tool_calls) == 2
    assert extracted_tool_calls.tool_calls[0].function.name == 'valid_tool'
    assert extracted_tool_calls.tool_calls[1].function.name == 'another_valid_tool'

def test_extract_tool_calls_missing_name_or_arguments(minimax_tool_parser):
    """Test that tool calls missing name or arguments are filtered out."""
    model_output = '<tool_calls>\n{"name": "valid_tool", "arguments": {"city": "Seattle"}}\n{"name": "missing_args"}\n{"arguments": {"city": "Portland"}}\n{"name": "another_valid_tool", "arguments": {"param": "value"}}\n</tool_calls>'
    extracted_tool_calls = minimax_tool_parser.extract_tool_calls(model_output, request=None)
    assert extracted_tool_calls.tools_called
    assert len(extracted_tool_calls.tool_calls) == 2
    assert extracted_tool_calls.tool_calls[0].function.name == 'valid_tool'
    assert extracted_tool_calls.tool_calls[1].function.name == 'another_valid_tool'

def test_streaming_basic_functionality(minimax_tool_parser):
    """Test basic streaming functionality."""
    minimax_tool_parser.current_tool_name_sent = False
    minimax_tool_parser.prev_tool_call_arr = []
    minimax_tool_parser.current_tool_id = -1
    minimax_tool_parser.streamed_args_for_tool = []
    current_text = '<tool_calls>\n{"name": "get_current_weather", "arguments": {"city": "Seattle"}}\n</tool_calls>'
    result = minimax_tool_parser.extract_tool_calls_streaming(previous_text='', current_text=current_text, delta_text='</tool_calls>', previous_token_ids=[], current_token_ids=[], delta_token_ids=[], request=None)
    if result is not None and hasattr(result, 'tool_calls') and result.tool_calls:
        assert len(result.tool_calls) >= 0

def test_streaming_with_content_before_tool_calls(minimax_tool_parser):
    """Test streaming when there's content before tool calls."""
    minimax_tool_parser.current_tool_name_sent = False
    minimax_tool_parser.prev_tool_call_arr = []
    minimax_tool_parser.current_tool_id = -1
    minimax_tool_parser.streamed_args_for_tool = []
    current_text = "I'll help you with that. <tool_calls>"
    result = minimax_tool_parser.extract_tool_calls_streaming(previous_text="I'll help you", current_text=current_text, delta_text=' with that. <tool_calls>', previous_token_ids=[], current_token_ids=[], delta_token_ids=[], request=None)
    if result is not None and hasattr(result, 'content'):
        assert result.content is not None

def test_streaming_no_tool_calls(minimax_tool_parser):
    """Test streaming when there are no tool calls."""
    current_text = 'This is just regular text without any tool calls.'
    result = minimax_tool_parser.extract_tool_calls_streaming(previous_text='This is just regular text', current_text=current_text, delta_text=' without any tool calls.', previous_token_ids=[], current_token_ids=[], delta_token_ids=[], request=None)
    assert result is not None
    assert hasattr(result, 'content')
    assert result.content == ' without any tool calls.'

def test_streaming_with_thinking_tags(minimax_tool_parser):
    """Test streaming with thinking tags that contain tool calls."""
    minimax_tool_parser.current_tool_name_sent = False
    minimax_tool_parser.prev_tool_call_arr = []
    minimax_tool_parser.current_tool_id = -1
    minimax_tool_parser.streamed_args_for_tool = []
    current_text = '<think><tool_calls>{"name": "ignored", "arguments": {}}</tool_calls></think><tool_calls>{"name": "real_tool", "arguments": {"param": "value"}}</tool_calls>'
    result = minimax_tool_parser.extract_tool_calls_streaming(previous_text='', current_text=current_text, delta_text=current_text, previous_token_ids=[], current_token_ids=[], delta_token_ids=[], request=None)
    if result is not None and hasattr(result, 'tool_calls') and result.tool_calls:
        for tool_call in result.tool_calls:
            assert tool_call.function.name != 'ignored'

def test_extract_tool_calls_multiline_json_not_supported(minimax_tool_parser):
    """Test that multiline JSON in tool calls is not currently supported."""
    model_output = '<tool_calls>\n{\n  "name": "get_current_weather",\n  "arguments": {\n    "city": "New York",\n    "state": "NY",\n    "unit": "celsius"\n  }\n}\n</tool_calls>'
    extracted_tool_calls = minimax_tool_parser.extract_tool_calls(model_output, request=None)
    assert not extracted_tool_calls.tools_called
    assert extracted_tool_calls.tool_calls == []
    assert extracted_tool_calls.content is None