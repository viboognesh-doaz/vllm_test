from .utils import MESSAGES_WITHOUT_TOOLS, WEATHER_TOOL, ServerConfig, ensure_system_prompt
import openai
import pytest

@pytest.mark.asyncio
async def test_chat_completion_without_tools(client: openai.AsyncOpenAI, server_config: ServerConfig):
    models = await client.models.list()
    model_name: str = models.data[0].id
    chat_completion = await client.chat.completions.create(messages=ensure_system_prompt(MESSAGES_WITHOUT_TOOLS, server_config), temperature=0, max_completion_tokens=150, model=model_name, logprobs=False)
    choice = chat_completion.choices[0]
    stop_reason = chat_completion.choices[0].finish_reason
    output_text = chat_completion.choices[0].message.content
    assert output_text is not None
    assert len(output_text) > 0
    assert stop_reason != 'tool_calls'
    assert choice.message.tool_calls is None or len(choice.message.tool_calls) == 0
    stream = await client.chat.completions.create(messages=ensure_system_prompt(MESSAGES_WITHOUT_TOOLS, server_config), temperature=0, max_completion_tokens=150, model=model_name, logprobs=False, stream=True)
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
    assert ''.join(chunks) == output_text

@pytest.mark.asyncio
async def test_chat_completion_with_tools(client: openai.AsyncOpenAI, server_config: ServerConfig):
    models = await client.models.list()
    model_name: str = models.data[0].id
    chat_completion = await client.chat.completions.create(messages=ensure_system_prompt(MESSAGES_WITHOUT_TOOLS, server_config), temperature=0, max_completion_tokens=150, model=model_name, tools=[WEATHER_TOOL], logprobs=False)
    choice = chat_completion.choices[0]
    stop_reason = chat_completion.choices[0].finish_reason
    output_text = chat_completion.choices[0].message.content
    assert output_text is not None
    assert stop_reason != 'tool_calls'
    assert len(output_text) > 0
    assert choice.message.tool_calls is None or len(choice.message.tool_calls) == 0
    stream = await client.chat.completions.create(messages=ensure_system_prompt(MESSAGES_WITHOUT_TOOLS, server_config), temperature=0, max_completion_tokens=150, model=model_name, logprobs=False, tools=[WEATHER_TOOL], stream=True)
    chunks: list[str] = []
    finish_reason_count = 0
    role_sent: bool = False
    async for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.role:
            assert delta.role == 'assistant'
            role_sent = True
        if delta.content:
            chunks.append(delta.content)
        if chunk.choices[0].finish_reason is not None:
            finish_reason_count += 1
        assert not delta.tool_calls or len(delta.tool_calls) == 0
    assert role_sent
    assert finish_reason_count == 1
    assert chunk.choices[0].finish_reason == stop_reason
    assert chunk.choices[0].finish_reason != 'tool_calls'
    assert len(chunks)
    assert ''.join(chunks) == output_text