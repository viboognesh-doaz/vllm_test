from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum, auto
from functools import partial
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from threading import Thread
from typing import Any, Callable, Optional, Union, cast
from vllm.config import VllmConfig
from vllm.distributed import destroy_distributed_environment, destroy_model_parallel
from vllm.distributed.device_communicators.shm_broadcast import Handle, MessageQueue
from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.executor.multiproc_worker_utils import _add_prefix, set_multiprocessing_worker_envs
from vllm.logger import init_logger
from vllm.utils import get_distributed_init_method, get_loopback_ip, get_mp_context, get_open_port
from vllm.v1.executor.abstract import Executor, FailureCallback
from vllm.v1.outputs import ModelRunnerOutput
from vllm.worker.worker_base import WorkerWrapperBase
import cloudpickle
import multiprocessing
import os
import pickle
import signal
import sys
import threading
import time
import traceback
import vllm.envs as envs
import weakref
logger = init_logger(__name__)

class MultiprocExecutor(Executor):

    def _init_executor(self) -> None:
        self._finalizer = weakref.finalize(self, self.shutdown)
        self.is_failed = False
        self.shutdown_event = threading.Event()
        self.failure_callback: Optional[FailureCallback] = None
        self.io_thread_pool: Optional[ThreadPoolExecutor] = None
        self.world_size = self.parallel_config.world_size
        tensor_parallel_size = self.parallel_config.tensor_parallel_size
        pp_parallel_size = self.parallel_config.pipeline_parallel_size
        assert self.world_size == tensor_parallel_size * pp_parallel_size, f'world_size ({self.world_size}) must be equal to the tensor_parallel_size ({tensor_parallel_size}) x pipeline_parallel_size ({pp_parallel_size}). '
        set_multiprocessing_worker_envs(self.parallel_config)
        distributed_init_method = get_distributed_init_method(get_loopback_ip(), get_open_port())
        max_chunk_bytes = envs.VLLM_MQ_MAX_CHUNK_BYTES_MB * 1024 * 1024
        self.rpc_broadcast_mq = MessageQueue(self.world_size, self.world_size, max_chunk_bytes=max_chunk_bytes)
        scheduler_output_handle = self.rpc_broadcast_mq.export_handle()
        unready_workers: list[UnreadyWorkerProcHandle] = []
        success = False
        try:
            for rank in range(self.world_size):
                unready_workers.append(WorkerProc.make_worker_process(vllm_config=self.vllm_config, local_rank=rank, rank=rank, distributed_init_method=distributed_init_method, input_shm_handle=scheduler_output_handle))
            self.workers = WorkerProc.wait_for_ready(unready_workers)
            self.rpc_broadcast_mq.wait_until_ready()
            for w in self.workers:
                w.worker_response_mq.wait_until_ready()
            self.start_worker_monitor()
            success = True
        finally:
            if not success:
                self._ensure_worker_termination([w.proc for w in unready_workers])
        if self.max_concurrent_batches > 1:
            self.io_thread_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='mp_exec_io')
        self.output_rank = self._get_output_rank()
        self.has_connector = self.vllm_config.kv_transfer_config is not None
        self.kv_output_aggregator = KVOutputAggregator(self.parallel_config.world_size)

    def start_worker_monitor(self):
        workers = self.workers
        self_ref = weakref.ref(self)

        def monitor_workers():
            sentinels = [h.proc.sentinel for h in workers]
            died = multiprocessing.connection.wait(sentinels)
            _self = self_ref()
            if not _self or getattr(_self, 'shutting_down', False):
                return
            _self.is_failed = True
            proc_name = next((h.proc.name for h in workers if h.proc.sentinel == died[0]))
            logger.error('Worker proc %s died unexpectedly, shutting down executor.', proc_name)
            _self.shutdown()
            callback = _self.failure_callback
            if callback is not None:
                _self.failure_callback = None
                callback()
        Thread(target=monitor_workers, daemon=True, name='MultiprocWorkerMonitor').start()

    def register_failure_callback(self, callback: FailureCallback):
        if self.is_failed:
            callback()
        else:
            self.failure_callback = callback

    def execute_model(self, scheduler_output) -> Union[ModelRunnerOutput, Future[ModelRunnerOutput]]:
        non_block = self.max_concurrent_batches > 1
        if not self.has_connector:
            output, = self.collective_rpc('execute_model', args=(scheduler_output,), unique_reply_rank=self.output_rank, non_block=non_block, timeout=envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS)
            return output
        outputs = self.collective_rpc('execute_model', args=(scheduler_output,), non_block=non_block, timeout=envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS)
        if non_block:
            return self.kv_output_aggregator.async_aggregate(outputs, self.output_rank)
        return self.kv_output_aggregator.aggregate(outputs, self.output_rank)

    def collective_rpc(self, method: Union[str, Callable], timeout: Optional[float]=None, args: tuple=(), kwargs: Optional[dict]=None, non_block: bool=False, unique_reply_rank: Optional[int]=None) -> list[Any]:
        if self.is_failed:
            raise RuntimeError('Executor failed.')
        deadline = None if timeout is None else time.monotonic() + timeout
        kwargs = kwargs or {}
        try:
            if isinstance(method, str):
                send_method = method
            else:
                send_method = cloudpickle.dumps(method, protocol=pickle.HIGHEST_PROTOCOL)
            self.rpc_broadcast_mq.enqueue((send_method, args, kwargs, unique_reply_rank))
            workers = (self.workers[unique_reply_rank],) if unique_reply_rank is not None else self.workers
            responses = []

            def get_response(w: WorkerProcHandle, dequeue_timeout: Optional[float]=None, cancel_event: Optional[threading.Event]=None):
                status, result = w.worker_response_mq.dequeue(timeout=dequeue_timeout, cancel=cancel_event)
                if status != WorkerProc.ResponseStatus.SUCCESS:
                    raise RuntimeError(f"Worker failed with error '{result}', please check the stack trace above for the root cause")
                return result
            for w in workers:
                dequeue_timeout = None if deadline is None else deadline - time.monotonic()
                if non_block:
                    result = self.io_thread_pool.submit(get_response, w, dequeue_timeout, self.shutdown_event)
                else:
                    result = get_response(w, dequeue_timeout)
                responses.append(result)
            return responses
        except TimeoutError as e:
            raise TimeoutError(f'RPC call to {method} timed out.') from e

    @staticmethod
    def _ensure_worker_termination(worker_procs: list[BaseProcess]):
        """Ensure that all worker processes are terminated. Assumes workers have
        received termination requests. Waits for processing, then sends
        termination and kill signals if needed."""

        def wait_for_termination(procs, timeout):
            if not time:
                return all((not proc.is_alive() for proc in procs))
            start_time = time.time()
            while time.time() - start_time < timeout:
                if all((not proc.is_alive() for proc in procs)):
                    return True
                time.sleep(0.1)
            return False
        active_procs = [proc for proc in worker_procs if proc.is_alive()]
        for p in active_procs:
            p.terminate()
        if not wait_for_termination(active_procs, 4):
            active_procs = [p for p in active_procs if p.is_alive()]
            for p in active_procs:
                p.kill()

    def shutdown(self):
        """Properly shut down the executor and its workers"""
        if not getattr(self, 'shutting_down', False):
            self.shutting_down = True
            self.shutdown_event.set()
            if self.io_thread_pool is not None:
                self.io_thread_pool.shutdown(wait=False, cancel_futures=True)
                self.io_thread_pool = None
            if (workers := getattr(self, 'workers', None)):
                for w in workers:
                    w.worker_response_mq = None
                self._ensure_worker_termination([w.proc for w in workers])
        self.rpc_broadcast_mq = None

    def check_health(self) -> None:
        self.collective_rpc('check_health', timeout=10)
        return

    @property
    def max_concurrent_batches(self) -> int:
        if self.scheduler_config.async_scheduling:
            return 2
        return self.parallel_config.pipeline_parallel_size

    def _get_output_rank(self) -> int:
        return self.world_size - self.parallel_config.tensor_parallel_size

@dataclass
class UnreadyWorkerProcHandle:
    """WorkerProcess handle before READY."""
    proc: BaseProcess
    rank: int
    ready_pipe: Connection

@dataclass
class WorkerProcHandle:
    proc: BaseProcess
    rank: int
    worker_response_mq: MessageQueue

    @classmethod
    def from_unready_handle(cls, unready_handle: UnreadyWorkerProcHandle, worker_response_mq: MessageQueue) -> 'WorkerProcHandle':
        return cls(proc=unready_handle.proc, rank=unready_handle.rank, worker_response_mq=worker_response_mq)

class WorkerProc:
    """Wrapper that runs one Worker in a separate process."""
    READY_STR = 'READY'

    def __init__(self, vllm_config: VllmConfig, local_rank: int, rank: int, distributed_init_method: str, input_shm_handle: Handle):
        self.rank = rank
        wrapper = WorkerWrapperBase(vllm_config=vllm_config, rpc_rank=rank)
        all_kwargs: list[dict] = [{} for _ in range(vllm_config.parallel_config.world_size)]
        is_driver_worker = rank % vllm_config.parallel_config.tensor_parallel_size == 0
        all_kwargs[rank] = {'vllm_config': vllm_config, 'local_rank': local_rank, 'rank': rank, 'distributed_init_method': distributed_init_method, 'is_driver_worker': is_driver_worker}
        wrapper.init_worker(all_kwargs)
        self.worker = wrapper
        pid = os.getpid()
        _add_prefix(sys.stdout, f'VllmWorker rank={rank}', pid)
        _add_prefix(sys.stderr, f'VllmWorker rank={rank}', pid)
        self.rpc_broadcast_mq = MessageQueue.create_from_handle(input_shm_handle, self.worker.rank)
        self.worker_response_mq = MessageQueue(1, 1)
        self.worker.init_device()
        self.worker.load_model()

    @staticmethod
    def make_worker_process(vllm_config: VllmConfig, local_rank: int, rank: int, distributed_init_method: str, input_shm_handle) -> UnreadyWorkerProcHandle:
        context = get_mp_context()
        reader, writer = context.Pipe(duplex=False)
        process_kwargs = {'vllm_config': vllm_config, 'local_rank': local_rank, 'rank': rank, 'distributed_init_method': distributed_init_method, 'input_shm_handle': input_shm_handle, 'ready_pipe': (reader, writer)}
        proc = context.Process(target=WorkerProc.worker_main, kwargs=process_kwargs, name=f'VllmWorker-{rank}', daemon=True)
        proc.start()
        writer.close()
        return UnreadyWorkerProcHandle(proc, rank, reader)

    @staticmethod
    def wait_for_ready(unready_proc_handles: list[UnreadyWorkerProcHandle]) -> list[WorkerProcHandle]:
        e = Exception('WorkerProc initialization failed due to an exception in a background process. See stack trace for root cause.')
        pipes = {handle.ready_pipe: handle for handle in unready_proc_handles}
        ready_proc_handles: list[Optional[WorkerProcHandle]] = [None] * len(unready_proc_handles)
        while pipes:
            ready = multiprocessing.connection.wait(pipes.keys())
            for pipe in ready:
                assert isinstance(pipe, Connection)
                try:
                    unready_proc_handle = pipes.pop(pipe)
                    response: dict[str, Any] = pipe.recv()
                    if response['status'] != 'READY':
                        raise e
                    worker_response_mq = MessageQueue.create_from_handle(response['handle'], 0)
                    ready_proc_handles[unready_proc_handle.rank] = WorkerProcHandle.from_unready_handle(unready_proc_handle, worker_response_mq)
                except EOFError:
                    e.__suppress_context__ = True
                    raise e from None
                finally:
                    pipe.close()
        return cast(list[WorkerProcHandle], ready_proc_handles)

    def shutdown(self):
        self.rpc_broadcast_mq = None
        self.worker_response_mq = None
        destroy_model_parallel()
        destroy_distributed_environment()

    @staticmethod
    def worker_main(*args, **kwargs):
        """ Worker initialization and execution loops.
        This runs a background process """
        shutdown_requested = False

        def signal_handler(signum, frame):
            nonlocal shutdown_requested
            if not shutdown_requested:
                shutdown_requested = True
                raise SystemExit()
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        worker = None
        reader, ready_writer = kwargs.pop('ready_pipe')
        try:
            reader.close()
            worker = WorkerProc(*args, **kwargs)
            ready_writer.send({'status': WorkerProc.READY_STR, 'handle': worker.worker_response_mq.export_handle()})
            worker.rpc_broadcast_mq.wait_until_ready()
            worker.worker_response_mq.wait_until_ready()
            ready_writer.close()
            ready_writer = None
            worker.worker_busy_loop()
        except Exception:
            if ready_writer is not None:
                logger.exception('WorkerProc failed to start.')
            else:
                logger.exception('WorkerProc failed.')
            shutdown_requested = True
        finally:
            if ready_writer is not None:
                ready_writer.close()
            if worker is not None:
                worker.shutdown()

    class ResponseStatus(Enum):
        SUCCESS = auto()
        FAILURE = auto()

    def worker_busy_loop(self):
        """Main busy loop for Multiprocessing Workers"""
        while True:
            method, args, kwargs, output_rank = self.rpc_broadcast_mq.dequeue()
            try:
                if isinstance(method, str):
                    func = getattr(self.worker, method)
                elif isinstance(method, bytes):
                    func = partial(cloudpickle.loads(method), self.worker)
                output = func(*args, **kwargs)
            except Exception as e:
                if hasattr(e, 'add_note'):
                    e.add_note(traceback.format_exc())
                logger.exception('WorkerProc hit an exception.')
                if output_rank is None or self.rank == output_rank:
                    self.worker_response_mq.enqueue((WorkerProc.ResponseStatus.FAILURE, str(e)))
                continue
            if output_rank is None or self.rank == output_rank:
                self.worker_response_mq.enqueue((WorkerProc.ResponseStatus.SUCCESS, output))