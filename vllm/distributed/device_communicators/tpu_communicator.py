from .base_device_communicator import DeviceCommunicatorBase
from torch.distributed import ProcessGroup
from torch_xla._internal import pjrt
from torch_xla.distributed.xla_multiprocessing import create_optimized_replica_groups
from tpu_commons.distributed.device_communicators import TpuCommunicator as TpuCommonsCommunicator
from typing import Optional
from vllm.config import get_current_vllm_config
from vllm.executor import ray_utils
from vllm.logger import init_logger
from vllm.platforms import current_platform
import os
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
USE_RAY = parallel_config = get_current_vllm_config().parallel_config.distributed_executor_backend == 'ray'
logger = init_logger(__name__)
if current_platform.is_tpu():
    if USE_RAY:

class TpuCommunicator(DeviceCommunicatorBase):

    def __init__(self, cpu_group: ProcessGroup, device: Optional[torch.device]=None, device_group: Optional[ProcessGroup]=None, unique_name: str=''):
        super().__init__(cpu_group, device, device_group, unique_name)
        global_rank = self.global_rank
        global_world_size = self.global_world_size
        if USE_RAY:
            logger.info('TpuCommunicator initialized with RAY')
            num_nodes = ray_utils.get_num_tpu_nodes()
            num_nodes_in_pg = ray_utils.get_num_nodes_in_placement_group()
            if num_nodes_in_pg > 0:
                num_nodes = num_nodes_in_pg
            local_world_size = global_world_size // num_nodes
            local_rank = global_rank % local_world_size
        else:
            logger.info('TpuCommunicator initialized with MP')
            num_hosts = torch_xla.tpu.num_tpu_workers()
            assert num_hosts == 1
            local_world_size = torch_xla.tpu.num_available_chips()
            local_rank = global_rank % local_world_size
        os.environ['CLOUD_TPU_TASK_ID'] = str(global_rank)
        os.environ['TPU_VISIBLE_CHIPS'] = str(local_rank)
        pjrt.initialize_multiprocess(local_rank, local_world_size)
        xr._init_world_size_ordinal()
        self.groups = create_optimized_replica_groups()

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        return xm.all_reduce(xm.REDUCE_SUM, input_, groups=self.groups)

    def all_gather(self, input_: torch.Tensor, dim: int=-1) -> torch.Tensor:
        assert dim == -1, 'TPUs only support dim=-1 for all-gather.'
        return xm.all_gather(input_, dim=dim)
try:
    TpuCommunicator = TpuCommonsCommunicator
except ImportError:
    logger.info("tpu_commons not found, using vLLM's TpuCommunicator")
    pass