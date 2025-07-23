from tqdm import tqdm
from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer.kv_pipe.pynccl_pipe import PyNcclPipe
import os
import time
import torch

def test_run(my_rank, pipe):
    print(f'rank {my_rank} test_run starts....')
    x = torch.tensor([1]).to(pipe.device)
    y = torch.tensor([[2.0, 3.0, 4.0, 8.0]]).to(pipe.device)
    if my_rank == 0:
        pipe.send_tensor(x)
        print(f'rank {my_rank} sent tensor x')
        pipe.send_tensor(y)
        print(f'rank {my_rank} sent tensor y')
        x2 = pipe.recv_tensor()
        print(f'rank {my_rank} received x2 = ', x2)
        y2 = pipe.recv_tensor()
        print(f'rank {my_rank} received y2 = ', y2)
    else:
        x2 = pipe.recv_tensor()
        print(f'rank {my_rank} received x2 = ', x2)
        y2 = pipe.recv_tensor()
        print(f'rank {my_rank} received y2 = ', y2)
        pipe.send_tensor(x)
        print(f'rank {my_rank} sent tensor x')
        pipe.send_tensor(y)
        print(f'rank {my_rank} sent tensor y')
    assert torch.allclose(x, x2)
    assert torch.allclose(y, y2)
    print(f'rank {my_rank} test_run passed!')

def stress_test(my_rank, pipe):
    print(f'rank {my_rank} stress_test starts....')
    tensors: list[torch.Tensor] = []
    torch.distributed.barrier()
    torch.manual_seed(0)
    for i in tqdm(range(500)):
        mean = torch.rand(1).item() * 100
        std = torch.rand(1).item() * 100
        size = torch.randint(900, 1000, (2,))
        x = torch.normal(mean * 1.0, std * 1.0, size=size.tolist()).to(pipe.device)
        if torch.rand(1).item() < 0.05:
            tensors.append(None)
            tensors.append(None)
            tensors.append(None)
        else:
            tensors.append(x)
            tensors.append(x.mean().unsqueeze(0))
            tensors.append(x.std().unsqueeze(0))
    torch.distributed.barrier()
    for i in tqdm(range(500)):
        if my_rank == int(i % 10 > 3):
            pipe.send_tensor(tensors[3 * i])
            pipe.send_tensor(tensors[3 * i + 1])
            pipe.send_tensor(tensors[3 * i + 2])
        else:
            x = pipe.recv_tensor()
            mean = pipe.recv_tensor()
            std = pipe.recv_tensor()
            if x is None:
                assert mean is None
                assert std is None
            else:
                assert torch.allclose(x, tensors[3 * i])
                assert x.mean() == mean[0]
                assert x.std() == std[0]
        torch.distributed.barrier()

def latency_test(my_rank, pipe, nelement, ntensor):
    latencies = []
    torch.distributed.barrier()
    for i in tqdm(range(500)):
        tensors = []
        if my_rank == 0:
            tensors = [torch.rand(nelement).to(pipe.device) for _ in range(ntensor)]
        torch.distributed.barrier()
        if my_rank == 0:
            t = torch.tensor([time.time()], dtype=torch.float64).to(pipe.device)
            for tensor in tensors:
                pipe.send_tensor(tensor)
            pipe.send_tensor(t)
        else:
            for _ in range(ntensor):
                pipe.recv_tensor()
            t = pipe.recv_tensor()
            latencies.append(time.time() - t.item())
    torch.distributed.barrier()
    print('Latency test passed.')
    print('Latency:', torch.tensor(latencies).mean().item() * 1000, 'ms')
if __name__ == '__main__':
    my_rank = int(os.environ['RANK'])
    torch.distributed.init_process_group(backend='gloo', init_method='tcp://localhost:12398', world_size=2, rank=my_rank)
    config = KVTransferConfig(kv_connector='PyNcclConnector', kv_buffer_device='cuda', kv_buffer_size=1000000000.0, kv_rank=my_rank, kv_role='kv_both', kv_parallel_size=2, kv_ip='127.0.0.1', kv_port=12345)
    pipe = PyNcclPipe(local_rank=my_rank, config=config)
    test_run(my_rank, pipe)
    stress_test(my_rank, pipe)