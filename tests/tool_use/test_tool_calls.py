from .utils import MESSAGES_ASKING_FOR_TOOLS, MESSAGES_WITH_TOOL_RESPONSE, SEARCH_TOOL, WEATHER_TOOL
from typing import Optional
import json
import openai
import pytest

@pytest.mark.asyncio
async def test_tool_call_and_choice(client: openai.AsyncOpenAI):
    models = await client.models.list()
    model_name: str = models.data[0].id
    chat_completion = await client.chat.completions.create(messages=MESSAGES_ASKING_FOR_TOOLS, temperature=0, max_completion_tokens=100, model=model_name, tools=[WEATHER_TOOL, SEARCH_TOOL], logprobs=False)
    choice = chat_completion.choices[0]
    stop_reason = chat_completion.choices[0].finish_reason
    tool_calls = chat_completion.choices[0].message.tool_calls
    assert choice.message.role == 'assistant'
    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0].type == 'function'
    assert tool_calls[0].function is not None
    assert isinstance(tool_calls[0].id, str)
    assert len(tool_calls[0].id) >= 9
    assert tool_calls[0].function.name == WEATHER_TOOL['function']['name']
    assert tool_calls[0].function.arguments is not None
    assert isinstance(tool_calls[0].function.arguments, str)
    parsed_arguments = json.loads(tool_calls[0].function.arguments)
    assert isinstance(parsed_arguments, dict)
    assert isinstance(parsed_arguments.get('city'), str)
    assert isinstance(parsed_arguments.get('state'), str)
    assert parsed_arguments.get('city') == 'Dallas'
    assert parsed_arguments.get('state') == 'TX'
    assert stop_reason == 'tool_calls'
    function_name: Optional[str] = None
    function_args_str: str = ''
    tool_call_id: Optional[str] = None
    role_name: Optional[str] = None
    finish_reason_count: int = 0
    stream = await client.chat.completions.create(model=model_name, messages=MESSAGES_ASKING_FOR_TOOLS, temperature=0, max_completion_tokens=100, tools=[WEATHER_TOOL, SEARCH_TOOL], logprobs=False, stream=True)
    async for chunk in stream:
        assert chunk.choices[0].index == 0
        if chunk.choices[0].finish_reason:
            finish_reason_count += 1
            assert chunk.choices[0].finish_reason == 'tool_calls'
        if chunk.choices[0].delta.role:
            assert not role_name or role_name == 'assistant'
            role_name = 'assistant'
        streamed_tool_calls = chunk.choices[0].delta.tool_calls
        if streamed_tool_calls and len(streamed_tool_calls) > 0:
            assert len(streamed_tool_calls) == 1
            tool_call = streamed_tool_calls[0]
            if tool_call.id:
                assert not tool_call_id
                tool_call_id = tool_call.id
            if tool_call.function:
                if tool_call.function.name:
                    assert function_name is None
                    assert isinstance(tool_call.function.name, str)
                    function_name = tool_call.function.name
                if tool_call.function.arguments:
                    assert isinstance(tool_call.function.arguments, str)
                    function_args_str += tool_call.function.arguments
    assert finish_reason_count == 1
    assert role_name == 'assistant'
    assert isinstance(tool_call_id, str) and len(tool_call_id) >= 9
    assert function_name == WEATHER_TOOL['function']['name']
    assert function_name == tool_calls[0].function.name
    assert isinstance(function_args_str, str)
    streamed_args = json.loads(function_args_str)
    assert isinstance(streamed_args, dict)
    assert isinstance(streamed_args.get('city'), str)
    assert isinstance(streamed_args.get('state'), str)
    assert streamed_args.get('city') == 'Dallas'
    assert streamed_args.get('state') == 'TX'
    assert function_name == tool_calls[0].function.name
    assert choice.message.role == role_name
    assert choice.message.tool_calls[0].function.name == function_name
    assert parsed_arguments == streamed_args

@pytest.mark.asyncio
async def test_tool_call_with_results(client: openai.AsyncOpenAI):
    models = await client.models.list()
    model_name: str = models.data[0].id
    chat_completion = await client.chat.completions.create(messages=MESSAGES_WITH_TOOL_RESPONSE, temperature=0, max_completion_tokens=100, model=model_name, tools=[WEATHER_TOOL, SEARCH_TOOL], logprobs=False)
    choice = chat_completion.choices[0]
    assert choice.finish_reason != 'tool_calls'
    assert choice.message.role == 'assistant'
    assert choice.message.tool_calls is None or len(choice.message.tool_calls) == 0
    assert choice.message.content is not None
    assert '98' in choice.message.content
    stream = await client.chat.completions.create(messages=MESSAGES_WITH_TOOL_RESPONSE, temperature=0, max_completion_tokens=100, model=model_name, tools=[WEATHER_TOOL, SEARCH_TOOL], logprobs=False, stream=True)
    chunks: list[str] = []
    finish_reason_count = 0
    role_sent: bool = False
    async for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.role:
            assert not role_sent
            assert delta.role == 'assistant'
            role_sent = True
        if delta.content:
            chunks.append(delta.content)
        if chunk.choices[0].finish_reason is not None:
            finish_reason_count += 1
            assert chunk.choices[0].finish_reason == choice.finish_reason
        assert not delta.tool_calls or len(delta.tool_calls) == 0
    assert role_sent
    assert finish_reason_count == 1
    assert len(chunks)
    assert ''.join(chunks) == choice.message.content