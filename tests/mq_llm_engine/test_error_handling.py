import torch
'Test that various errors are handled properly.'
import asyncio
import tempfile
import time
import uuid
from unittest.mock import Mock
import pytest
from tests.mq_llm_engine.utils import RemoteMQLLMEngine
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.engine.multiprocessing import MQEngineDeadError
from vllm.engine.multiprocessing.engine import MQLLMEngine
from vllm.entrypoints.openai.api_server import build_async_engine_client
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.lora.request import LoRARequest
from vllm.sequence import SequenceGroupMetadata
from vllm.usage.usage_lib import UsageContext
from vllm.utils import FlexibleArgumentParser
MODEL = 'google/gemma-1.1-2b-it'
ENGINE_ARGS = AsyncEngineArgs(model=MODEL, enforce_eager=True)
RAISED_ERROR = KeyError
RAISED_VALUE = 'foo'

@pytest.fixture(scope='function')
def tmp_socket():
    with tempfile.TemporaryDirectory() as td:
        yield f'ipc://{td}/{uuid.uuid4()}'

def run_with_evil_forward(engine_args: AsyncEngineArgs, ipc_path: str):
    engine = MQLLMEngine.from_engine_args(engine_args=engine_args, usage_context=UsageContext.UNKNOWN_CONTEXT, ipc_path=ipc_path)
    engine.engine.model_executor.execute_model = Mock(side_effect=RAISED_ERROR(RAISED_VALUE))
    engine.start()

@pytest.mark.asyncio
async def test_evil_forward(tmp_socket):
    with RemoteMQLLMEngine(engine_args=ENGINE_ARGS, ipc_path=tmp_socket, run_fn=run_with_evil_forward) as engine:
        client = await engine.make_client()
        await asyncio.sleep(2.0)
        await client.check_health()
        with pytest.raises(MQEngineDeadError):
            async for _ in client.generate(prompt='Hello my name is', sampling_params=SamplingParams(), request_id=str(uuid.uuid4())):
                pass
        assert client.errored
        await asyncio.sleep(1.0)
        with pytest.raises(RAISED_ERROR):
            await client.check_health()
        assert client.errored
        client.close()

def run_with_evil_model_executor_health(engine_args: AsyncEngineArgs, ipc_path: str):
    engine = MQLLMEngine.from_engine_args(engine_args=engine_args, usage_context=UsageContext.UNKNOWN_CONTEXT, ipc_path=ipc_path)
    engine.engine.model_executor.check_health = Mock(side_effect=RAISED_ERROR)
    engine.start()

@pytest.mark.asyncio
async def test_failed_health_check(tmp_socket):
    with RemoteMQLLMEngine(engine_args=ENGINE_ARGS, ipc_path=tmp_socket, run_fn=run_with_evil_model_executor_health) as engine:
        client = await engine.make_client()
        assert client.is_running
        await asyncio.sleep(15.0)
        with pytest.raises(RAISED_ERROR):
            await client.check_health()
        assert client.errored
        with pytest.raises(MQEngineDeadError):
            async for _ in client.generate(prompt='Hello my name is', sampling_params=SamplingParams(), request_id=str(uuid.uuid4())):
                pass
        client.close()

def run_with_evil_abort(engine_args: AsyncEngineArgs, ipc_path: str):
    engine = MQLLMEngine.from_engine_args(engine_args=engine_args, usage_context=UsageContext.UNKNOWN_CONTEXT, ipc_path=ipc_path)
    engine.engine.abort_request = Mock(side_effect=RAISED_ERROR)
    engine.start()

@pytest.mark.asyncio
async def test_failed_abort(tmp_socket):
    with RemoteMQLLMEngine(engine_args=ENGINE_ARGS, ipc_path=tmp_socket, run_fn=run_with_evil_abort) as engine:
        client = await engine.make_client()
        assert client.is_running
        await client.check_health()
        await client.abort(request_id='foo')
        with pytest.raises(MQEngineDeadError) as execinfo:
            async for _ in client.generate(prompt='Hello my name is', sampling_params=SamplingParams(max_tokens=10), request_id=str(uuid.uuid4())):
                pass
        assert 'KeyError' in repr(execinfo.value)
        assert client.errored
        with pytest.raises(RAISED_ERROR):
            await client.check_health()
        client.close()

@pytest.mark.asyncio
async def test_batch_error(tmp_socket):
    with RemoteMQLLMEngine(engine_args=ENGINE_ARGS, ipc_path=tmp_socket, run_fn=run_with_evil_abort) as engine:
        client = await engine.make_client()
        assert client.is_running
        await client.check_health()

        async def do_generate(client):
            params = SamplingParams(min_tokens=2048, max_tokens=2048)
            async for _ in client.generate(prompt='Hello my name is', sampling_params=params, request_id=str(uuid.uuid4())):
                pass
        tasks = [asyncio.create_task(do_generate(client)) for _ in range(10)]
        await client.abort(request_id='foo')
        errors = await asyncio.gather(*tasks, return_exceptions=True)
        for e in errors:
            assert isinstance(e, MQEngineDeadError)
            assert 'KeyError' in repr(e)
        client.close()

@pytest.mark.asyncio
async def test_bad_request(tmp_socket):
    with RemoteMQLLMEngine(engine_args=ENGINE_ARGS, ipc_path=tmp_socket) as engine:
        client = await engine.make_client()
        with pytest.raises(ValueError):
            async for _ in client.generate(prompt='Hello my name is', sampling_params=SamplingParams(), request_id='abcd-1', lora_request=LoRARequest('invalid-lora', 1, 'invalid-path')):
                pass
        async for _ in client.generate(prompt='Hello my name is', sampling_params=SamplingParams(), request_id='abcd-2'):
            pass
        client.close()

@pytest.mark.asyncio
async def test_mp_crash_detection(monkeypatch: pytest.MonkeyPatch):
    with monkeypatch.context() as m:
        parser = FlexibleArgumentParser(description="vLLM's remote OpenAI server.")
        parser = make_arg_parser(parser)
        args = parser.parse_args([])

        def mock_init():
            raise ValueError
        m.setattr(LLMEngine, '__init__', mock_init)
        start = time.perf_counter()
        async with build_async_engine_client(args):
            pass
        end = time.perf_counter()
        assert end - start < 60, 'Expected vLLM to gracefully shutdown in <60s if there is an error in the startup.'

@pytest.mark.asyncio
async def test_mp_cuda_init():
    torch.cuda.init()
    parser = FlexibleArgumentParser(description="vLLM's remote OpenAI server.")
    parser = make_arg_parser(parser)
    args = parser.parse_args([])
    async with build_async_engine_client(args):
        pass

@pytest.mark.asyncio
async def test_engine_process_death(tmp_socket):
    with RemoteMQLLMEngine(engine_args=ENGINE_ARGS, ipc_path=tmp_socket) as engine:
        client = await engine.make_client()
        assert client.is_running
        engine.proc.kill()
        with pytest.raises(MQEngineDeadError):
            async for _ in client.generate(prompt='Hello my name is', sampling_params=SamplingParams(), request_id=str(uuid.uuid4())):
                pass
        with pytest.raises(RuntimeError, match='Engine process .* died'):
            await client.check_health()
        client.close()

def run_with_evil_input_processing(engine_args: AsyncEngineArgs, ipc_path: str):
    """Simulate an exception while preparing inputs for the model.
    In the wild, this could be something like a multimodal input processor
    failing on invalid image data."""
    engine = MQLLMEngine.from_engine_args(engine_args=engine_args, usage_context=UsageContext.UNKNOWN_CONTEXT, ipc_path=ipc_path)
    runner = engine.engine.model_executor.driver_worker.worker.model_runner

    def raiser(_, seq_group_metadata: SequenceGroupMetadata):
        if seq_group_metadata.request_id.startswith('evil'):
            raise RAISED_ERROR(RAISED_VALUE)
    runner.builder.per_seq_group_compute_fns.append(raiser)
    engine.start()

@pytest.mark.asyncio
async def test_failed_inputs(tmp_socket):
    with RemoteMQLLMEngine(engine_args=ENGINE_ARGS, ipc_path=tmp_socket, run_fn=run_with_evil_input_processing) as engine:
        client = await engine.make_client()
        assert client.is_running
        await client.check_health()

        async def run_failing_request():
            async for _ in client.generate(prompt='Hello my name is', sampling_params=SamplingParams(max_tokens=10), request_id='evil' + str(uuid.uuid4())):
                pass

        async def run_passing_request():
            async for _ in client.generate(prompt='Hello my name is', sampling_params=SamplingParams(max_tokens=10), request_id=str(uuid.uuid4())):
                pass
        passing_tasks = [asyncio.create_task(run_passing_request()) for _ in range(10)]
        failing_tasks = [asyncio.create_task(run_failing_request()) for _ in range(10)]
        await asyncio.gather(*failing_tasks, return_exceptions=True)
        await asyncio.gather(*passing_tasks)
        for task in failing_tasks:
            with pytest.raises(RAISED_ERROR):
                task.result()
        for task in passing_tasks:
            task.result()
        assert not client.errored
        await client.check_health()
        client.close()