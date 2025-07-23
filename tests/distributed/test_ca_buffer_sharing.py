from vllm.distributed.device_communicators.cuda_wrapper import CudaRTLibrary
from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce
import ctypes
import torch
import torch.distributed as dist
dist.init_process_group(backend='gloo')
rank = local_rank = dist.get_rank()
world_size = dist.get_world_size()
lib = CudaRTLibrary()
lib.cudaSetDevice(rank)
buffer_size_in_bytes = 1024
byte_value = 2
pointers = CustomAllreduce.create_shared_buffer(buffer_size_in_bytes)
print(f'Rank {rank} has pointers {pointers}')
dist.barrier()
torch.cuda.synchronize()
if rank == 0:
    for p in pointers:
        pointer = ctypes.c_void_p(p)
        lib.cudaMemset(pointer, byte_value, buffer_size_in_bytes)
dist.barrier()
torch.cuda.synchronize()
host_data = (ctypes.c_char * buffer_size_in_bytes)()
for p in pointers:
    pointer = ctypes.c_void_p(p)
    lib.cudaMemcpy(host_data, pointer, buffer_size_in_bytes)
    for i in range(buffer_size_in_bytes):
        assert ord(host_data[i]) == byte_value, f'Rank {rank} failed to verify buffer {p}. Expected {byte_value}, got {ord(host_data[i])}'
print(f'Rank {rank} verified all buffers')
dist.barrier()
torch.cuda.synchronize()
CustomAllreduce.free_shared_buffer(pointers)