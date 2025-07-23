from ...utils import create_new_process_for_each_test, multi_gpu_test
from concurrent.futures import Future, ThreadPoolExecutor
from transformers import AutoTokenizer
from vllm import SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.platforms import current_platform
from vllm.utils import set_default_torch_num_threads
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core import EngineCore
from vllm.v1.executor.abstract import Executor, UniProcExecutor
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import ModelRunnerOutput
import copy
import pytest
import time
import uuid
if not current_platform.is_cuda():
    pytest.skip(reason='V1 currently only supported on CUDA.', allow_module_level=True)
MODEL_NAME = 'meta-llama/Llama-3.2-1B-Instruct'
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
PROMPT = 'Hello my name is Robert and I love quantization kernels'
PROMPT_TOKENS = TOKENIZER(PROMPT).input_ids

def make_request() -> EngineCoreRequest:
    return EngineCoreRequest(request_id=str(uuid.uuid4()), prompt_token_ids=PROMPT_TOKENS, mm_inputs=None, mm_hashes=None, mm_placeholders=None, sampling_params=SamplingParams(), pooling_params=None, eos_token_id=None, arrival_time=time.time(), lora_request=None, cache_salt=None, data_parallel_rank=None)

@create_new_process_for_each_test()
def test_engine_core(monkeypatch: pytest.MonkeyPatch):
    with monkeypatch.context() as m:
        m.setenv('VLLM_USE_V1', '1')
        'Setup the EngineCore.'
        engine_args = EngineArgs(model=MODEL_NAME)
        vllm_config = engine_args.create_engine_config()
        executor_class = Executor.get_class(vllm_config)
        with set_default_torch_num_threads(1):
            engine_core = EngineCore(vllm_config=vllm_config, executor_class=executor_class, log_stats=True)
        'Test basic request lifecycle.'
        engine_core.add_request(make_request())
        assert len(engine_core.scheduler.waiting) == 1
        assert len(engine_core.scheduler.running) == 0
        _ = engine_core.step()
        assert len(engine_core.scheduler.waiting) == 0
        assert len(engine_core.scheduler.running) == 1
        engine_core.add_request(make_request())
        assert len(engine_core.scheduler.waiting) == 1
        assert len(engine_core.scheduler.running) == 1
        _ = engine_core.step()
        assert len(engine_core.scheduler.waiting) == 0
        assert len(engine_core.scheduler.running) == 2
        engine_core.add_request(make_request())
        engine_core.add_request(make_request())
        assert len(engine_core.scheduler.waiting) == 2
        assert len(engine_core.scheduler.running) == 2
        _ = engine_core.step()
        assert len(engine_core.scheduler.waiting) == 0
        assert len(engine_core.scheduler.running) == 4
        while (outs := engine_core.step()[0].get(0)) and outs.outputs:
            pass
        assert len(engine_core.scheduler.waiting) == 0
        assert len(engine_core.scheduler.running) == 0
        'Test abort cycle.'
        req = make_request()
        request_id = req.request_id
        engine_core.add_request(req)
        assert len(engine_core.scheduler.waiting) == 1
        assert len(engine_core.scheduler.running) == 0
        assert engine_core.scheduler.has_unfinished_requests()
        assert not engine_core.scheduler.has_finished_requests()
        _ = engine_core.step()
        assert len(engine_core.scheduler.waiting) == 0
        assert len(engine_core.scheduler.running) == 1
        assert engine_core.scheduler.has_unfinished_requests()
        assert not engine_core.scheduler.has_finished_requests()
        engine_core.abort_requests([request_id])
        assert len(engine_core.scheduler.waiting) == 0
        assert len(engine_core.scheduler.running) == 0
        assert not engine_core.scheduler.has_unfinished_requests()
        assert engine_core.scheduler.has_finished_requests()
        _ = engine_core.step()
        assert not engine_core.scheduler.has_unfinished_requests()
        assert not engine_core.scheduler.has_finished_requests()
        req0 = make_request()
        req1 = make_request()
        req2 = make_request()
        engine_core.add_request(req0)
        engine_core.add_request(req1)
        assert len(engine_core.scheduler.waiting) == 2
        assert len(engine_core.scheduler.running) == 0
        _ = engine_core.step()
        assert len(engine_core.scheduler.waiting) == 0
        assert len(engine_core.scheduler.running) == 2
        engine_core.add_request(req2)
        assert len(engine_core.scheduler.waiting) == 1
        assert len(engine_core.scheduler.running) == 2
        _ = engine_core.step()
        assert len(engine_core.scheduler.waiting) == 0
        assert len(engine_core.scheduler.running) == 3
        engine_core.abort_requests([req1.request_id])
        assert len(engine_core.scheduler.waiting) == 0
        assert len(engine_core.scheduler.running) == 2
        _ = engine_core.step()
        assert len(engine_core.scheduler.waiting) == 0
        assert len(engine_core.scheduler.running) == 2
        engine_core.abort_requests([req2.request_id, req0.request_id])
        assert len(engine_core.scheduler.waiting) == 0
        assert len(engine_core.scheduler.running) == 0
        req0 = make_request()
        req1 = make_request()
        req0.request_id = req1.request_id = 'test'
        engine_core.add_request(req0)
        while (outs := engine_core.step()[0].get(0)) and outs.outputs:
            pass
        engine_core.add_request(req1)
        while (outs := engine_core.step()[0].get(0)) and outs.outputs:
            pass
        assert len(engine_core.scheduler.waiting) == 0
        assert len(engine_core.scheduler.running) == 0

@create_new_process_for_each_test()
def test_engine_core_advanced_sampling(monkeypatch: pytest.MonkeyPatch):
    """
    A basic end-to-end test to verify that the engine functions correctly
    when additional sampling parameters, such as top_p, min_tokens, and
    presence_penalty, are set.
    """
    with monkeypatch.context() as m:
        m.setenv('VLLM_USE_V1', '1')
        'Setup the EngineCore.'
        engine_args = EngineArgs(model=MODEL_NAME)
        vllm_config = engine_args.create_engine_config()
        executor_class = Executor.get_class(vllm_config)
        with set_default_torch_num_threads(1):
            engine_core = EngineCore(vllm_config=vllm_config, executor_class=executor_class, log_stats=True)
        'Test basic request lifecycle.'
        request: EngineCoreRequest = make_request()
        request.sampling_params = SamplingParams(min_tokens=4, presence_penalty=1.0, frequency_penalty=1.0, repetition_penalty=0.1, stop_token_ids=[1001, 1002])
        engine_core.add_request(request)

        def _check_engine_state():
            assert len(engine_core.scheduler.waiting) == 1
            assert len(engine_core.scheduler.running) == 0
            while (outs := engine_core.step()[0].get(0)) and outs.outputs:
                pass
            assert len(engine_core.scheduler.waiting) == 0
            assert len(engine_core.scheduler.running) == 0
        _check_engine_state()
        request2 = make_request()
        request2.sampling_params = SamplingParams(top_p=0.99, top_k=50)
        engine_core.add_request(request2)
        _check_engine_state()

@create_new_process_for_each_test()
def test_engine_core_concurrent_batches(monkeypatch: pytest.MonkeyPatch):
    """
    Test that the engine can handle multiple concurrent batches.
    """

    def make_request_with_max_tokens(req_id: int, max_tokens: int) -> EngineCoreRequest:
        request = make_request()
        request.request_id = req_id
        request.sampling_params.max_tokens = max_tokens
        return request

    class DummyExecutor(UniProcExecutor):

        def initialize_from_config(self, kv_cache_configs: list[KVCacheConfig]) -> None:
            super().initialize_from_config(kv_cache_configs)
            self.thread_pool = ThreadPoolExecutor(max_workers=1)

        def execute_model(self, scheduler_output) -> Future[ModelRunnerOutput]:
            """Make execute_model non-blocking."""

            def _execute():
                output = self.collective_rpc('execute_model', args=(scheduler_output,))
                return copy.deepcopy(output[0])
            return self.thread_pool.submit(_execute)

        @property
        def max_concurrent_batches(self) -> int:
            return 2

        def shutdown(self):
            if hasattr(self, 'thread_pool'):
                self.thread_pool.shutdown(wait=False)
    with monkeypatch.context() as m:
        m.setenv('VLLM_USE_V1', '1')
        engine_args = EngineArgs(model=MODEL_NAME, max_num_seqs=2, enable_prefix_caching=False, max_num_batched_tokens=10, enforce_eager=True)
        vllm_config = engine_args.create_engine_config()
        with set_default_torch_num_threads(1):
            engine_core = EngineCore(vllm_config=vllm_config, log_stats=False, executor_class=DummyExecutor)
        assert engine_core.batch_queue is not None
        req0 = make_request_with_max_tokens(0, 5)
        engine_core.add_request(req0)
        req1 = make_request_with_max_tokens(1, 5)
        engine_core.add_request(req1)
        assert engine_core.step_with_batch_queue()[0] is None
        assert engine_core.batch_queue.qsize() == 1
        scheduler_output = engine_core.batch_queue.queue[-1][1]
        assert scheduler_output.num_scheduled_tokens[0] == 10
        assert engine_core.scheduler.requests[req0.request_id].num_computed_tokens == 10
        assert engine_core.step_with_batch_queue()[0] is None
        assert engine_core.batch_queue.qsize() == 2
        scheduler_output = engine_core.batch_queue.queue[-1][1]
        assert scheduler_output.num_scheduled_tokens[0] == 2
        assert scheduler_output.num_scheduled_tokens[1] == 8
        assert engine_core.scheduler.requests[0].num_computed_tokens == 12
        assert engine_core.scheduler.requests[1].num_computed_tokens == 8
        assert engine_core.scheduler.get_num_unfinished_requests() == 2
        engine_core.step_with_batch_queue()
        engine_core.step_with_batch_queue()
        assert engine_core.batch_queue.qsize() == 2
        scheduler_output = engine_core.batch_queue.queue[-1][1]
        assert scheduler_output.num_scheduled_tokens[1] == 4
        output = engine_core.step_with_batch_queue()[0].get(0)
        assert output is not None
        assert len(output.outputs) == 1
        assert engine_core.scheduler.requests[req0.request_id].num_tokens == 13
        engine_core.step_with_batch_queue()
        assert engine_core.batch_queue.qsize() == 2
        scheduler_output = engine_core.batch_queue.queue[-1][1]
        assert scheduler_output.num_scheduled_tokens[0] == 1
        output = engine_core.step_with_batch_queue()[0].get(0)
        assert output is not None
        assert len(output.outputs) == 1
        assert engine_core.scheduler.requests[req1.request_id].num_tokens == 13
        engine_core.step_with_batch_queue()
        assert engine_core.batch_queue.qsize() == 2
        scheduler_output = engine_core.batch_queue.queue[-1][1]
        assert scheduler_output.num_scheduled_tokens[1] == 1
        step = 0
        req_id = 0
        expected_num_tokens = [engine_core.scheduler.requests[0].num_tokens + 1, engine_core.scheduler.requests[1].num_tokens + 1]
        while engine_core.scheduler.get_num_unfinished_requests() == 2:
            output = engine_core.step_with_batch_queue()[0]
            if step % 2 == 0:
                assert output is not None
                assert len(output[0].outputs) == 1
                if req_id in engine_core.scheduler.requests:
                    assert engine_core.scheduler.requests[req_id].num_tokens == expected_num_tokens[req_id]
                expected_num_tokens[req_id] += 1
                req_id = (req_id + 1) % 2
            else:
                assert output is None
            step += 1

@multi_gpu_test(num_gpus=2)
def test_engine_core_tp(monkeypatch: pytest.MonkeyPatch):
    """
    Test engine can initialize worker in tp properly
    """
    with monkeypatch.context() as m:
        m.setenv('VLLM_USE_V1', '1')
        'Setup the EngineCore.'
        engine_args = EngineArgs(model=MODEL_NAME, tensor_parallel_size=2, enforce_eager=True)
        vllm_config = engine_args.create_engine_config()
        executor_class = Executor.get_class(vllm_config)
        with set_default_torch_num_threads(1):
            engine_core = EngineCore(vllm_config=vllm_config, executor_class=executor_class, log_stats=True)

        def get_worker_cache_config_field(worker, key: str):
            return getattr(worker.cache_config, key)
        num_gpu_blocks = engine_core.collective_rpc(get_worker_cache_config_field, args=('num_gpu_blocks',))
        num_cpu_blocks = engine_core.collective_rpc(get_worker_cache_config_field, args=('num_cpu_blocks',))
        assert all((x is not None for x in num_gpu_blocks))
        assert all((x is not None for x in num_cpu_blocks))