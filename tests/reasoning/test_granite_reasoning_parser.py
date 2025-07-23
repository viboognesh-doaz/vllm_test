from tests.reasoning.utils import DeltaMessage, run_reasoning_extraction
from transformers import AutoTokenizer
from vllm.reasoning import ReasoningParser, ReasoningParserManager
import pytest
parser_name = 'granite'
START_REASONING = 'Here is my thought process:'
START_RESPONSE = 'Here is my response:'
SIMPLE_REASONING = {'output': f'{START_REASONING}This is a reasoning section{START_RESPONSE}This is the rest', 'reasoning_content': 'This is a reasoning section', 'content': 'This is the rest'}
COMPLETE_REASONING = {'output': f'{START_REASONING}This is a reasoning section{START_RESPONSE}', 'reasoning_content': 'This is a reasoning section', 'content': None}
NO_REASONING = {'output': 'This is content', 'reasoning_content': None, 'content': 'This is content'}
MULTIPLE_LINES = {'output': f'{START_REASONING}This\nThat{START_RESPONSE}This is the rest\nThat', 'reasoning_content': 'This\nThat', 'content': 'This is the rest\nThat'}
REASONING_WITH_THINK = {'output': f'{START_REASONING}This is a reasoning section{START_RESPONSE}This is the rest', 'reasoning_content': 'This is a reasoning section', 'content': 'This is the rest'}
COMPLETE_REASONING_WITH_THINK = {'output': f'{START_REASONING}This is a reasoning section{START_RESPONSE}', 'reasoning_content': 'This is a reasoning section', 'content': None}
MULTIPLE_LINES_WITH_THINK = {'output': f'{START_REASONING}This\nThat{START_RESPONSE}This is the rest\nThat', 'reasoning_content': 'This\nThat', 'content': 'This is the rest\nThat'}
TEST_CASES = [pytest.param(False, SIMPLE_REASONING, id='simple_reasoning'), pytest.param(False, COMPLETE_REASONING, id='complete_reasoning'), pytest.param(False, NO_REASONING, id='no_reasoning'), pytest.param(False, MULTIPLE_LINES, id='multiple_lines'), pytest.param(False, REASONING_WITH_THINK, id='reasoning_with_think'), pytest.param(False, COMPLETE_REASONING_WITH_THINK, id='complete_reasoning_with_think'), pytest.param(False, MULTIPLE_LINES_WITH_THINK, id='multiple_lines_with_think'), pytest.param(True, SIMPLE_REASONING, id='simple_reasoning_streaming'), pytest.param(True, COMPLETE_REASONING, id='complete_reasoning_streaming'), pytest.param(True, NO_REASONING, id='no_reasoning_streaming'), pytest.param(True, MULTIPLE_LINES, id='multiple_lines_streaming'), pytest.param(True, REASONING_WITH_THINK, id='reasoning_with_think_streaming'), pytest.param(True, COMPLETE_REASONING_WITH_THINK, id='complete_reasoning_with_think_streaming'), pytest.param(True, MULTIPLE_LINES_WITH_THINK, id='multiple_lines_with_think_streaming')]
tokenizer = AutoTokenizer.from_pretrained('facebook/opt-125m')

@pytest.mark.parametrize('streaming, param_dict', TEST_CASES)
def test_reasoning(streaming: bool, param_dict: dict):
    output = tokenizer.tokenize(param_dict['output'])
    output_tokens: list[str] = [tokenizer.convert_tokens_to_string([token]) for token in output]
    parser: ReasoningParser = ReasoningParserManager.get_reasoning_parser(parser_name)(tokenizer)
    reasoning, content = run_reasoning_extraction(parser, output_tokens, streaming=streaming)
    assert reasoning == param_dict['reasoning_content']
    assert content == param_dict['content']
STREAMING_1 = {'previous_text': None, 'current_text': 'Here', 'delta_text': 'Here', 'reasoning_content': None, 'content': None}
STREAMING_2 = {'previous_text': 'Here is my thought', 'current_text': 'Here is my thought failure', 'delta_text': ' failure', 'reasoning_content': None, 'content': 'Here is my thought failure'}
STREAMING_3 = {'previous_text': 'Here wrong', 'current_text': ' words', 'delta_text': ' Here wrong words', 'reasoning_content': None, 'content': ' words'}
STREAMING_4 = {'previous_text': 'Here is my thought', 'current_text': 'Here is my thought process:', 'delta_text': ' process:', 'reasoning_content': None, 'content': None}
STREAMING_5 = {'previous_text': 'Here is my thought process:', 'current_text': 'Here is my thought process: foo', 'delta_text': ' foo', 'reasoning_content': ' foo', 'content': None}
STREAMING_6 = {'previous_text': 'Here is my thought process: foo', 'current_text': 'Here is my thought process: foo Here is', 'delta_text': ' Here is', 'reasoning_content': ' ', 'content': None}
STREAMING_7 = {'previous_text': 'Here is my thought process: foo Here is', 'current_text': 'Here is my thought process: foo Here is Here', 'delta_text': ' Here', 'reasoning_content': 'Here is ', 'content': None}
STREAMING_8 = {'previous_text': 'Here is my thought process: foo Here is my response:', 'current_text': 'Here is my thought process: foo Here is my response: bar', 'delta_text': ' bar', 'reasoning_content': None, 'content': ' bar'}
STREAMING_9 = {'previous_text': None, 'current_text': 'Here is my thought process: foo Here is my response: bar', 'delta_text': 'Here is my thought process: foo Here is my response: bar', 'reasoning_content': ' foo ', 'content': ' bar'}
STREAMING_10 = {'previous_text': 'Here is my thought process: foo', 'current_text': 'Here is my thought process: foo bar Here is my response: baz', 'delta_text': ' bar Here is my response: baz', 'reasoning_content': ' bar ', 'content': ' baz'}
STREAMING_11 = {'previous_text': 'Here is my thought process: This is a reasoning section ', 'current_text': 'Here is my thought process: This is a reasoning section Here', 'delta_text': 'Here', 'reasoning_content': None, 'content': None}
STREAMING_12 = {'previous_text': 'Here is my thought process: foo Here is my response', 'current_text': 'Here is my thought process: foo Here is my response:', 'delta_text': ':', 'reasoning_content': None, 'content': None}
STREAMING_13 = {'previous_text': 'Here is my thought process: foo Here', 'current_text': 'Here is my thought process: foo Here was', 'delta_text': ' was', 'reasoning_content': 'Here was', 'content': None}
STREAMING_SUBCASES = [pytest.param(STREAMING_1, id='Starting reasoning special sequence'), pytest.param(STREAMING_2, id='Unexpected start reasoning sequence'), pytest.param(STREAMING_3, id='Continuing unexpected start reasoning sequence'), pytest.param(STREAMING_4, id='Only start reasoning sequence and nothing else'), pytest.param(STREAMING_5, id='Reasoning content has started'), pytest.param(STREAMING_6, id='Response special sequence has started'), pytest.param(STREAMING_7, id='Response special sequence reset'), pytest.param(STREAMING_8, id='Response text has started'), pytest.param(STREAMING_9, id='Delta contains everything'), pytest.param(STREAMING_10, id='Delta contains some reasoning and response'), pytest.param(STREAMING_11, id='Delta starts response sequence'), pytest.param(STREAMING_12, id='Delta finishes response sequence'), pytest.param(STREAMING_13, id='Delta breaks potential responise sequence')]

@pytest.mark.parametrize('param_dict', STREAMING_SUBCASES)
def test_streaming_subcases(param_dict):
    previous_token_ids = tokenizer.encode(param_dict['previous_text']) if param_dict['previous_text'] is not None else []
    current_token_ids = tokenizer.encode(param_dict['current_text'])
    delta_token_ids = tokenizer.encode(param_dict['delta_text'])
    parser: ReasoningParser = ReasoningParserManager.get_reasoning_parser(parser_name)(tokenizer)
    response = parser.extract_reasoning_content_streaming(previous_text=param_dict['previous_text'], current_text=param_dict['current_text'], delta_text=param_dict['delta_text'], previous_token_ids=previous_token_ids, current_token_ids=current_token_ids, delta_token_ids=delta_token_ids)
    if param_dict['reasoning_content'] is None and param_dict['content'] is None:
        assert response is None
    else:
        assert isinstance(response, DeltaMessage)
        assert param_dict['reasoning_content'] == response.reasoning_content
        assert param_dict['content'] == response.content