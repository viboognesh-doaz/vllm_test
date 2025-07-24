from unittest.mock import patch
import torch
from vllm.engine.arg_utils import EngineArgs
from vllm.utils import get_distributed_init_method, get_ip, get_open_port
from vllm.worker.cache_engine import CacheEngine
from vllm.worker.worker import Worker

def test_gpu_memory_profiling():
    engine_args = EngineArgs(model='facebook/opt-125m', dtype='half', load_format='dummy')
    engine_config = engine_args.create_engine_config()
    engine_config.cache_config.num_gpu_blocks = 1000
    engine_config.cache_config.num_cpu_blocks = 1000
    distributed_init_method = get_distributed_init_method(get_ip(), get_open_port())
    worker = Worker(vllm_config=engine_config, local_rank=0, rank=0, distributed_init_method=distributed_init_method, is_driver_worker=True)

    def mock_mem_info():
        current_usage = torch.cuda.memory_stats()['allocated_bytes.all.current']
        mock_total_bytes = 10 * 1024 ** 3
        free = mock_total_bytes - current_usage
        return (free, mock_total_bytes)
    with patch('torch.cuda.mem_get_info', side_effect=mock_mem_info):
        worker.init_device()
        worker.load_model()
        gpu_blocks, _ = worker.determine_num_available_blocks()
    block_size = CacheEngine.get_cache_block_size(engine_config.cache_config, engine_config.model_config, engine_config.parallel_config)
    expected_blocks = 8.28 * 1024 ** 3 // block_size
    assert abs(gpu_blocks - expected_blocks) < 100