from ...utils import RemoteOpenAIServer
from .test_completion import MODEL_NAME
from .test_completion import default_server_args
from .test_completion import zephyr_lora_added_tokens_files
from .test_completion import zephyr_lora_files
from .test_completion import zephyr_pa_files
from vllm.transformers_utils.tokenizer import get_tokenizer
import pytest

@pytest.fixture(scope='module')
def server_fixture(request, default_server_args):
    use_server_flag = request.param
    if use_server_flag:
        args_with_flag = default_server_args + ['--return-tokens-as-token-ids']
        with RemoteOpenAIServer(MODEL_NAME, args_with_flag) as remote_server:
            yield (remote_server, True)
    else:
        with RemoteOpenAIServer(MODEL_NAME, default_server_args) as remote_server:
            yield (remote_server, False)

@pytest.mark.asyncio
@pytest.mark.parametrize('server_fixture', [True, False], indirect=True)
async def test_completion_return_tokens_as_token_ids_completion(server_fixture):
    server, use_server_flag = server_fixture
    request_args = {}
    if not use_server_flag:
        request_args['return_tokens_as_token_ids'] = True
    async with server.get_async_client() as client:
        completion = await client.completions.create(model=MODEL_NAME, prompt="Say 'Hello, world! 🎉'", echo=True, temperature=0, max_tokens=10, logprobs=1, extra_body=request_args)
        text = completion.choices[0].text
        token_strs = completion.choices[0].logprobs.tokens
        tokenizer = get_tokenizer(tokenizer_name=MODEL_NAME)
        top_logprobs = completion.choices[0].logprobs.top_logprobs[1:]
        top_logprob_keys = [next(iter(logprob_by_tokens)) for logprob_by_tokens in top_logprobs]
        assert token_strs[1:] == top_logprob_keys
        tokens = [int(token.removeprefix('token_id:')) for token in token_strs]
        assert text == tokenizer.decode(tokens, skip_special_tokens=True)

@pytest.mark.asyncio
@pytest.mark.parametrize('server_fixture', [True, False], indirect=True)
async def test_chat_return_tokens_as_token_ids_completion(server_fixture):
    server, use_server_flag = server_fixture
    request_args = {}
    if not use_server_flag:
        request_args['return_tokens_as_token_ids'] = True
    async with server.get_async_client() as client:
        response = await client.chat.completions.create(model=MODEL_NAME, messages=[{'role': 'system', 'content': 'You like to respond in only emojis, like 🎉'}, {'role': 'user', 'content': 'Please write some emojis: 🐱🐶🎉'}], temperature=0, max_tokens=8, logprobs=True, extra_body=request_args)
        text = response.choices[0].message.content
        tokenizer = get_tokenizer(tokenizer_name=MODEL_NAME)
        token_ids = []
        for logprob_content in response.choices[0].logprobs.content:
            token_ids.append(int(logprob_content.token.removeprefix('token_id:')))
        assert tokenizer.decode(token_ids, skip_special_tokens=True) == text