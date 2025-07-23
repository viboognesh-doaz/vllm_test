from ...utils import RemoteOpenAIServer
from typing import NamedTuple
import openai
import pytest
import pytest_asyncio
MODEL_NAME = 'Qwen/Qwen2-1.5B-Instruct'

@pytest.fixture(scope='module')
def server():
    args = ['--dtype', 'float16', '--enforce-eager', '--max-model-len', '4080']
    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server

@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client

class TestCase(NamedTuple):
    model_name: str
    echo: bool

@pytest.mark.asyncio
@pytest.mark.parametrize('test_case', [TestCase(model_name=MODEL_NAME, echo=True), TestCase(model_name=MODEL_NAME, echo=False)])
async def test_chat_session_with_echo_and_continue_final_message(client: openai.AsyncOpenAI, test_case: TestCase):
    saying: str = 'Here is a common saying about apple. An apple a day, keeps'
    chat_completion = await client.chat.completions.create(model=test_case.model_name, messages=[{'role': 'user', 'content': 'tell me a common saying'}, {'role': 'assistant', 'content': saying}], extra_body={'echo': test_case.echo, 'continue_final_message': True, 'add_generation_prompt': False})
    assert chat_completion.id is not None
    assert len(chat_completion.choices) == 1
    choice = chat_completion.choices[0]
    assert choice.finish_reason == 'stop'
    message = choice.message
    if test_case.echo:
        assert message.content is not None and saying in message.content
    else:
        assert message.content is not None and saying not in message.content
    assert message.role == 'assistant'