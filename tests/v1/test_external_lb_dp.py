from contextlib import AsyncExitStack
from tests.utils import RemoteOpenAIServer
from vllm.platforms import Platform
import asyncio
import openai
import os
import pytest
import pytest_asyncio
import threading
import time
MODEL_NAME = 'ibm-research/PowerMoE-3b'
DP_SIZE = int(os.getenv('DP_SIZE', '2'))
TP_SIZE = int(os.getenv('TP_SIZE', '1'))

class ExternalLBServerManager:
    """Manages data parallel vLLM server instances for external
    load balancer testing."""

    def __init__(self, model_name: str, dp_size: int, api_server_count: int, base_server_args: list, tp_size: int=TP_SIZE):
        self.model_name = model_name
        self.dp_size = dp_size
        self.tp_size = tp_size
        self.api_server_count = api_server_count
        self.base_server_args = base_server_args
        self.servers: list[tuple[RemoteOpenAIServer, list[str]]] = []
        self.server_threads: list[threading.Thread] = []

    def __enter__(self) -> list[tuple[RemoteOpenAIServer, list[str]]]:
        """Start all server instances for external LB mode."""
        for rank in range(self.dp_size):
            server_args = self.base_server_args.copy()
            server_args.extend(['--data-parallel-size', str(self.dp_size), '--data-parallel-rank', str(rank), '--data-parallel-size-local', '1', '--tensor-parallel-size', str(self.tp_size), '--port', str(8000 + rank), '--api-server-count', str(self.api_server_count)])

            def start_server(r: int, sargs: list[str]):
                try:
                    server = RemoteOpenAIServer(self.model_name, sargs, auto_port=False, env_dict={'CUDA_VISIBLE_DEVICES': ','.join((str(Platform.device_id_to_physical_device_id(i)) for i in range(r * TP_SIZE, (r + 1) * TP_SIZE)))})
                    server.__enter__()
                    print(f'Server rank {r} started successfully with {self.api_server_count} API servers')
                    self.servers.append((server, sargs))
                except Exception as e:
                    print(f'Failed to start server rank {r}: {e}')
                    raise
            thread = threading.Thread(target=start_server, args=(rank, server_args))
            thread.start()
            self.server_threads.append(thread)
        for thread in self.server_threads:
            thread.join()
        time.sleep(2)
        if len(self.servers) != self.dp_size:
            raise Exception('Servers failed to start')
        return self.servers

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop all server instances."""
        while self.servers:
            try:
                self.servers.pop()[0].__exit__(exc_type, exc_val, exc_tb)
            except Exception as e:
                print(f'Error stopping server: {e}')

@pytest.fixture(scope='module')
def default_server_args():
    return ['--dtype', 'bfloat16', '--max-model-len', '2048', '--max-num-seqs', '128', '--enforce-eager']

@pytest.fixture(scope='module', params=[1, 4])
def servers(request, default_server_args):
    api_server_count = request.param
    with ExternalLBServerManager(MODEL_NAME, DP_SIZE, api_server_count, default_server_args) as server_list:
        yield server_list

@pytest_asyncio.fixture
async def clients(servers: list[tuple[RemoteOpenAIServer, list[str]]]):
    async with AsyncExitStack() as stack:
        yield [await stack.enter_async_context(server.get_async_client()) for server, _ in servers]

@pytest.mark.asyncio
@pytest.mark.parametrize('model_name', [MODEL_NAME])
async def test_external_lb_single_completion(clients: list[openai.AsyncOpenAI], servers: list[tuple[RemoteOpenAIServer, list[str]]], model_name: str) -> None:

    async def make_request(client: openai.AsyncOpenAI):
        completion = await client.completions.create(model=model_name, prompt='Hello, my name is', max_tokens=10, temperature=1.0)
        assert completion.id is not None
        assert completion.choices is not None and len(completion.choices) == 1
        choice = completion.choices[0]
        assert len(choice.text) >= 1
        assert choice.finish_reason in ('length', 'stop')
        assert completion.usage.completion_tokens > 0
        assert completion.usage.prompt_tokens > 0
        assert completion.usage.total_tokens > 0
        return completion
    for i, client in enumerate(clients):
        result = await make_request(client)
        assert result is not None
        print(f'Server {i} handled single completion request successfully')
    await asyncio.sleep(0.5)
    num_requests_per_server = 25
    all_tasks = []
    for i, client in enumerate(clients):
        tasks = [make_request(client) for _ in range(num_requests_per_server)]
        all_tasks.extend(tasks)
    results = await asyncio.gather(*all_tasks)
    assert len(results) == num_requests_per_server * len(clients)
    assert all((completion is not None for completion in results))
    await asyncio.sleep(0.5)
    all_tasks = []
    for i, client in enumerate(clients):
        tasks = [make_request(client) for _ in range(num_requests_per_server)]
        all_tasks.extend(tasks)
    results = await asyncio.gather(*all_tasks)
    assert len(results) == num_requests_per_server * len(clients)
    assert all((completion is not None for completion in results))
    _, server_args = servers[0]
    api_server_count = server_args.count('--api-server-count') and server_args[server_args.index('--api-server-count') + 1] or 1
    print(f'Successfully completed external LB test with {len(clients)} servers (API server count: {api_server_count})')

@pytest.mark.asyncio
@pytest.mark.parametrize('model_name', [MODEL_NAME])
async def test_external_lb_completion_streaming(clients: list[openai.AsyncOpenAI], servers: list[tuple[RemoteOpenAIServer, list[str]]], model_name: str) -> None:
    prompt = 'What is an LLM?'

    async def make_streaming_request(client: openai.AsyncOpenAI):
        single_completion = await client.completions.create(model=model_name, prompt=prompt, max_tokens=5, temperature=0.0)
        single_output = single_completion.choices[0].text
        stream = await client.completions.create(model=model_name, prompt=prompt, max_tokens=5, temperature=0.0, stream=True)
        chunks: list[str] = []
        finish_reason_count = 0
        last_chunk = None
        async for chunk in stream:
            chunks.append(chunk.choices[0].text)
            if chunk.choices[0].finish_reason is not None:
                finish_reason_count += 1
            last_chunk = chunk
        assert finish_reason_count == 1, 'Finish reason should appear exactly once.'
        assert last_chunk is not None, 'Stream should have yielded at least one chunk.'
        assert last_chunk.choices[0].finish_reason == 'length', "Finish reason should be 'length'."
        assert ''.join(chunks) == single_output, 'Streamed output should match non-streamed output.'
        return True
    for i, client in enumerate(clients):
        result = await make_streaming_request(client)
        assert result is not None
        print(f'Server {i} handled single streaming request successfully')
    await asyncio.sleep(0.5)
    num_requests_per_server = 25
    all_tasks = []
    for i, client in enumerate(clients):
        tasks = [make_streaming_request(client) for _ in range(num_requests_per_server)]
        all_tasks.extend(tasks)
    results = await asyncio.gather(*all_tasks)
    assert len(results) == num_requests_per_server * len(clients)
    assert all(results), 'Not all streaming requests completed successfully.'
    await asyncio.sleep(0.5)
    all_tasks = []
    for i, client in enumerate(clients):
        tasks = [make_streaming_request(client) for _ in range(num_requests_per_server)]
        all_tasks.extend(tasks)
    results = await asyncio.gather(*all_tasks)
    assert len(results) == num_requests_per_server * len(clients)
    assert all(results), 'Not all streaming requests completed successfully.'
    _, server_args = servers[0]
    api_server_count = server_args.count('--api-server-count') and server_args[server_args.index('--api-server-count') + 1] or 1
    print(f'Successfully completed external LB streaming test with {len(clients)} servers (API server count: {api_server_count})')