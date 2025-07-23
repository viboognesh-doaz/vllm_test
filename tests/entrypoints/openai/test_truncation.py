from tests.utils import RemoteOpenAIServer
from typing import Any
import openai
import pytest
import pytest_asyncio
MODEL_NAME = 'sentence-transformers/all-MiniLM-L12-v2'
max_model_len = 128
input = "Immerse yourself in the enchanting chronicle of calculus, a \n    mathematical domain that has radically transformed our comprehension of \n    change and motion. Despite its roots in ancient civilizations, the \n    formal birth of calculus predominantly occurred in the 17th century, \n    primarily under the influential guidance of Sir Isaac Newton and Gottfried \n    Wilhelm Leibniz. The earliest traces of calculus concepts are found in \n    ancient Greek mathematics,most notably in the works of Eudoxus and \n    Archimedes, around 300 BCE. They utilized the 'method of exhaustion'—a \n    technique for computing areas and volumes through the use of finite sums. \n    This methodology laid crucial foundational work for integral calculus. \n    In the 17th century, both Newton and Leibniz independently pioneered \n    calculus, each contributing unique perspectives that would shape this new \n    field."

@pytest.fixture(scope='module')
def server():
    args = ['--task', 'embed', '--dtype', 'bfloat16', '--enforce-eager', '--max-model-len', str(max_model_len)]
    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server

@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client

@pytest.mark.asyncio
async def test_smaller_truncation_size(client: openai.AsyncOpenAI):
    truncation_size = 10
    kwargs: dict[str, Any] = {'model': MODEL_NAME, 'input': input, 'truncate_prompt_tokens': truncation_size}
    response = await client.post(path='embeddings', cast_to=object, body={**kwargs})
    assert response['usage']['prompt_tokens'] == truncation_size

@pytest.mark.asyncio
async def test_bigger_truncation_size(client: openai.AsyncOpenAI):
    truncation_size = max_model_len + 1
    kwargs: dict[str, Any] = {'model': MODEL_NAME, 'input': input, 'truncate_prompt_tokens': truncation_size}
    with pytest.raises(openai.BadRequestError) as err:
        err = await client.post(path='embeddings', cast_to=object, body={**kwargs})
        assert str(err) == f"openai.BadRequestError: \n                    Error code: 400 - {{'object': 'error', \n                    'message': 'truncate_prompt_tokens value \n                    ({truncation_size}) \n                    is greater than max_model_len ({max_model_len}). \n                    Please, select a smaller truncation size.', \n                    'type': 'BadRequestError', \n                    'param': None, 'code': 400}}"

@pytest.mark.asyncio
async def test_max_truncation_size(client: openai.AsyncOpenAI):
    truncation_size = -1
    kwargs: dict[str, Any] = {'model': MODEL_NAME, 'input': input, 'truncate_prompt_tokens': truncation_size}
    response = await client.post(path='embeddings', cast_to=object, body={**kwargs})
    assert response['usage']['prompt_tokens'] == max_model_len