from typing import Any, Callable, List, Optional, Union
from vllm.executor.executor_base import DistributedExecutorBase
from vllm.executor.multiproc_worker_utils import ProcessWorkerWrapper, ResultHandler, WorkerMonitor, set_multiprocessing_worker_envs
from vllm.logger import init_logger
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.platforms import current_platform
from vllm.sequence import ExecuteModelRequest
from vllm.utils import _run_task_with_lock, cuda_device_count_stateless, get_distributed_init_method, get_ip, get_open_port, make_async, run_method, update_environment_variables
from vllm.worker.worker_base import WorkerWrapperBase
import asyncio
import cloudpickle
import os
logger = init_logger(__name__)

class MultiprocessingDistributedExecutor(DistributedExecutorBase):
    """Python multiprocessing-based distributed executor"""
    uses_ray: bool = False

    def _check_cuda(self) -> None:
        """Check that the number of GPUs is sufficient for the parallel
        configuration. Separate from _init_executor to reduce the number of
        indented blocks.
        """
        parallel_config = self.parallel_config
        world_size = parallel_config.world_size
        tensor_parallel_size = parallel_config.tensor_parallel_size
        cuda_device_count = cuda_device_count_stateless()
        if tensor_parallel_size > cuda_device_count:
            raise RuntimeError(f'please set tensor_parallel_size ({tensor_parallel_size}) to less than max local gpu count ({cuda_device_count})')
        if world_size > cuda_device_count:
            raise RuntimeError(f'please ensure that world_size ({world_size}) is less than than max local gpu count ({cuda_device_count})')
        if 'CUDA_VISIBLE_DEVICES' not in os.environ:
            update_environment_variables({'CUDA_VISIBLE_DEVICES': ','.join(map(str, range(world_size)))})

    def _init_executor(self) -> None:
        if current_platform.is_cuda_alike():
            self._check_cuda()
        world_size = self.parallel_config.world_size
        tensor_parallel_size = self.parallel_config.tensor_parallel_size
        set_multiprocessing_worker_envs(self.parallel_config)
        distributed_init_method = get_distributed_init_method('127.0.0.1', get_open_port())
        self.workers: List[ProcessWorkerWrapper] = []
        self.tp_driver_workers: List[ProcessWorkerWrapper] = []
        self.non_driver_workers: List[ProcessWorkerWrapper] = []
        if world_size == 1:
            self.worker_monitor = None
        else:
            result_handler = ResultHandler()
            for rank in range(1, world_size):
                worker = ProcessWorkerWrapper(result_handler, WorkerWrapperBase, self.vllm_config, rank)
                self.workers.append(worker)
                if rank % tensor_parallel_size == 0:
                    self.tp_driver_workers.append(worker)
                else:
                    self.non_driver_workers.append(worker)
            self.worker_monitor = WorkerMonitor(self.workers, result_handler)
            result_handler.start()
            self.worker_monitor.start()
        self.driver_worker = WorkerWrapperBase(self.vllm_config, 0)
        all_kwargs = []
        distributed_init_method = get_distributed_init_method(get_ip(), get_open_port())
        for i in range(world_size):
            local_rank = i
            rank = i
            kwargs = dict(vllm_config=self.vllm_config, local_rank=local_rank, rank=rank, distributed_init_method=distributed_init_method, is_driver_worker=not self.parallel_config or rank % self.parallel_config.tensor_parallel_size == 0)
            all_kwargs.append(kwargs)
        self._run_workers('init_worker', all_kwargs)
        self._run_workers('init_device')
        self._run_workers('load_model', max_concurrent_workers=self.parallel_config.max_parallel_loading_workers)
        self.driver_exec_model = make_async(self.driver_worker.execute_model)
        self.pp_locks: Optional[List[asyncio.Lock]] = None

    def shutdown(self):
        if (worker_monitor := getattr(self, 'worker_monitor', None)) is not None:
            worker_monitor.close()

    def _driver_execute_model(self, execute_model_req: Optional[ExecuteModelRequest]) -> Optional[List[SamplerOutput]]:
        """Run execute_model in the driver worker.

        Passing None will cause the driver to stop the model execution
        loop running in each of the remote workers.
        """
        return self.driver_worker.execute_model(execute_model_req)

    def _run_workers(self, method: Union[str, Callable], *args, async_run_tensor_parallel_workers_only: bool=False, max_concurrent_workers: Optional[int]=None, **kwargs) -> List[Any]:
        """Runs the given method on all workers.

        Args:
            async_run_tensor_parallel_workers_only: If True the method will be
                run only in the remote TP workers, not the driver worker.
                It will also be run asynchronously and return a list of futures
                rather than blocking on the results.
        """
        if isinstance(method, str):
            sent_method = method
        else:
            sent_method = cloudpickle.dumps(method)
        del method
        if max_concurrent_workers:
            raise NotImplementedError('max_concurrent_workers is not supported yet.')
        if async_run_tensor_parallel_workers_only:
            return [worker.execute_method(sent_method, *args, **kwargs) for worker in self.non_driver_workers]
        worker_outputs = [worker.execute_method(sent_method, *args, **kwargs) for worker in self.workers]
        driver_worker_output = run_method(self.driver_worker, sent_method, args, kwargs)
        return [driver_worker_output] + [output.get() for output in worker_outputs]

    def check_health(self) -> None:
        """Raises an error if engine is unhealthy."""
        if self.worker_monitor is not None and (not self.worker_monitor.is_alive()):
            raise RuntimeError('Worker processes are not running')

    def _wait_for_tasks_completion(self, parallel_worker_tasks: Any) -> None:
        """Wait for futures returned from _run_workers() with
        async_run_remote_workers_only to complete."""
        for result in parallel_worker_tasks:
            result.get()

    async def _driver_execute_model_async(self, execute_model_req: Optional[ExecuteModelRequest]=None) -> List[SamplerOutput]:
        if not self.tp_driver_workers:
            return await self.driver_exec_model(execute_model_req)
        if self.pp_locks is None:
            self.pp_locks = [asyncio.Lock() for _ in range(self.parallel_config.pipeline_parallel_size)]
        tasks = [asyncio.create_task(_run_task_with_lock(self.driver_exec_model, self.pp_locks[0], execute_model_req))]
        for pp_rank, driver_worker in enumerate(self.tp_driver_workers, start=1):
            tasks.append(asyncio.create_task(_run_task_with_lock(driver_worker.execute_method_async, self.pp_locks[pp_rank], 'execute_model', execute_model_req)))
        results = await asyncio.gather(*tasks)
        return results[-1]

    async def _start_worker_execution_loop(self):
        coros = [worker.execute_method_async('start_worker_execution_loop') for worker in self.non_driver_workers]
        return await asyncio.gather(*coros)