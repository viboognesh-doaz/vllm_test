from vllm.logger import init_logger
import os
import torch
logger = init_logger(__name__)
if os.environ.get('NCCL_CUMEM_ENABLE', '0') != '0':
    logger.warning('NCCL_CUMEM_ENABLE is set to %s, skipping override. This may increase memory overhead with cudagraph+allreduce: https://github.com/NVIDIA/nccl/issues/1234', os.environ['NCCL_CUMEM_ENABLE'])
elif not os.path.exists('/dev/nvidia-caps-imex-channels'):
    os.environ['NCCL_CUMEM_ENABLE'] = '0'
os.environ['PYTORCH_NVML_BASED_CUDA_CHECK'] = '1'
os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'
torch._inductor.config.compile_threads = 1