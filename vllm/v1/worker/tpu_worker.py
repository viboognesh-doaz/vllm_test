from tpu_commons.worker import TPUWorker as TPUCommonsWorker
from typing import Any, Optional
from vllm.config import ParallelConfig, VllmConfig
from vllm.distributed import ensure_model_parallel_initialized, init_distributed_environment
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor import set_random_seed
from vllm.platforms import current_platform
from vllm.pooling_params import PoolingTask
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE, cdiv
from vllm.utils import init_cached_hf_modules
from vllm.v1.attention.backends.pallas import TPU_HEAD_SIZE_ALIGNMENT
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.utils import report_usage_stats
from vllm.v1.worker.tpu_model_runner import TPUModelRunner
from vllm.v1.worker.utils import bind_kv_cache
import os
import torch
import torch.distributed
import torch.nn as nn
import torch_xla.core.xla_model as xm
import torch_xla.debug.profiler as xp
import torch_xla.runtime as xr
import tpu_info
import vllm.envs as envs
'A TPU worker class.'
logger = init_logger(__name__)

class TPUWorker:

    def __init__(self, vllm_config: VllmConfig, local_rank: int, rank: int, distributed_init_method: str, is_driver_worker: bool=False):
        self.is_driver_worker = is_driver_worker
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.use_spmd = envs.VLLM_XLA_USE_SPMD
        self.original_parallel_config = None
        if self.use_spmd:
            self.original_parallel_config = self.parallel_config
            self.parallel_config.tensor_parallel_size = 1
            self.parallel_config.pipeline_parallel_size = 1
            self.parallel_config.world_size = 1
        self.scheduler_config = vllm_config.scheduler_config
        self.device_config = vllm_config.device_config
        self.speculative_config = vllm_config.speculative_config
        self.prompt_adapter_config = vllm_config.prompt_adapter_config
        self.observability_config = vllm_config.observability_config
        self.parallel_config.rank = rank
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        if self.cache_config.cache_dtype == 'auto':
            self.cache_dtype = self.model_config.dtype
        else:
            self.cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[self.cache_config.cache_dtype]
        if self.model_config.trust_remote_code:
            init_cached_hf_modules()
        self.profiler = None
        self.profile_dir = None
        if envs.VLLM_TORCH_PROFILER_DIR and self.rank < 1:
            self.profile_dir = envs.VLLM_TORCH_PROFILER_DIR
            logger.info('Profiling enabled. Traces will be saved to: %s', self.profile_dir)
        if self.model_config.seed is None:
            self.model_config.seed = 0

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int) -> None:
        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

    def init_device(self):
        os.environ['PJRT_DEVICE'] = 'TPU'
        os.environ['LIBTPU_INIT_ARGS'] = os.environ.get('LIBTPU_INIT_ARGS', '') + ' --xla_tpu_force_1d_allreduce_at_chunk_count=1 --xla_jf_conv_input_fusion=False'
        torch.set_grad_enabled(False)
        torch.set_default_dtype(self.model_config.dtype)
        self._init_tpu_worker_distributed_environment(self.parallel_config, self.rank, self.distributed_init_method, self.local_rank)
        self.device = xm.xla_device()
        self.device_config.device = self.device
        set_random_seed(self.model_config.seed)
        if self.model_config.seed is not None:
            xm.set_rng_state(self.model_config.seed, self.device)
        torch._dynamo.config.cache_size_limit = 128
        world_size = self.parallel_config.world_size
        rank = xr.global_ordinal()
        if envs.VLLM_XLA_CACHE_PATH:
            per_rank_path = os.path.join(envs.VLLM_XLA_CACHE_PATH, f'tp{world_size}_rank{rank}')
            xr.initialize_cache(per_rank_path, readonly=False)
        self.model_runner = TPUModelRunner(self.vllm_config, self.device, self.original_parallel_config)
        if rank == 0:
            report_usage_stats(self.vllm_config)

    def determine_available_memory(self) -> int:
        kv_caches: dict[str, torch.Tensor] = {}
        kv_cache_spec = self.model_runner.get_kv_cache_spec()
        for layer_name, layer_spec in kv_cache_spec.items():
            if isinstance(layer_spec, AttentionSpec):
                dtype = layer_spec.dtype
                tpu_kv_cache = torch.tensor([], dtype=dtype).to(self.device)
                kv_caches[layer_name] = tpu_kv_cache
            else:
                raise NotImplementedError(f"Unsupported KV cache spec '{type(layer_spec)}'")
        runner_kv_caches: list[torch.Tensor] = []
        bind_kv_cache(kv_caches, self.vllm_config.compilation_config.static_forward_context, runner_kv_caches)
        with self.model_runner.maybe_setup_dummy_loras(self.lora_config):
            self.model_runner.profile_run(self.model_runner.max_num_tokens)
        xm.wait_device_ops()
        self.model_runner.reset_dynamo_cache()
        if self.use_spmd:
            chip_type, _ = tpu_info.device.get_local_chips()
            device_usage = tpu_info.metrics.get_chip_usage(chip_type)
            total_memory_size = device_usage[0].total_memory
            current_mem = device_usage[0].memory_usage
        else:
            m = xm.get_memory_info(self.device)
            total_memory_size = m['bytes_limit']
            current_mem = m['bytes_used']
        profiled = current_mem * 1.02
        usable_memory_size = int(total_memory_size * self.cache_config.gpu_memory_utilization)
        tpu_kv_cache_bytes = max(usable_memory_size - profiled, 0)
        head_size = self.model_config.get_head_size()
        if head_size > 0:
            padded_head_size = cdiv(head_size, TPU_HEAD_SIZE_ALIGNMENT) * TPU_HEAD_SIZE_ALIGNMENT
            if padded_head_size != head_size:
                logger.warning_once('head size is padded to %d', padded_head_size)
            tpu_kv_cache_bytes = tpu_kv_cache_bytes * head_size // padded_head_size
        return int(tpu_kv_cache_bytes)

    def execute_model(self, scheduler_output: 'SchedulerOutput') -> Optional[ModelRunnerOutput]:
        output = self.model_runner.execute_model(scheduler_output)
        return output if self.is_driver_worker else None

    def profile(self, is_start: bool=True):
        if self.rank < 1:
            if self.profile_dir is None:
                raise RuntimeError('Profiler is not enabled.')
            if is_start:
                if self.profiler is None:
                    self.profiler = xp.start_server(9012)
                xp.start_trace(self.profile_dir)
            else:
                xp.stop_trace()

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.model_runner.add_lora(lora_request)

    def load_model(self) -> None:
        self.model_runner.load_model()

    def update_config(self, overrides: dict[str, Any]) -> None:
        self.model_runner.update_config(overrides)

    def compile_or_warm_up_model(self) -> None:
        if not self.model_config.enforce_eager:
            self.model_runner.capture_model()
        set_random_seed(self.model_config.seed)

    def get_model(self) -> nn.Module:
        return self.model_runner.get_model()

    def get_supported_pooling_tasks(self) -> list[PoolingTask]:
        return self.model_runner.get_supported_pooling_tasks()

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        return self.model_runner.get_kv_cache_spec()

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        """Allocate GPU KV cache with the specified kv_cache_config."""
        self.model_runner.initialize_kv_cache(kv_cache_config)

    def check_health(self) -> None:
        return

    def _init_tpu_worker_distributed_environment(self, parallel_config: ParallelConfig, rank: int, distributed_init_method: Optional[str]=None, local_rank: int=-1) -> None:
        """Initialize the distributed environment."""
        if self.use_spmd:
            xr.use_spmd()
        init_distributed_environment(world_size=parallel_config.world_size, rank=rank, local_rank=local_rank, distributed_init_method=distributed_init_method, backend=current_platform.dist_backend)
        ensure_model_parallel_initialized(parallel_config.tensor_parallel_size, parallel_config.pipeline_parallel_size)
try:
    TPUWorker = TPUCommonsWorker
except ImportError:
    logger.info("tpu_commons not found, using vLLM's TPUWorker.")
    pass