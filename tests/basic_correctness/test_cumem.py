from ..utils import create_new_process_for_each_test
from vllm import LLM, SamplingParams
from vllm.device_allocator.cumem import CuMemAllocator
from vllm.utils import GiB_bytes
import pytest
import torch

@create_new_process_for_each_test()
def test_python_error():
    """
    Test if Python error occurs when there's low-level
    error happening from the C++ side.
    """
    allocator = CuMemAllocator.get_instance()
    total_bytes = torch.cuda.mem_get_info()[1]
    alloc_bytes = int(total_bytes * 0.7)
    tensors = []
    with allocator.use_memory_pool():
        x = torch.empty(alloc_bytes, dtype=torch.uint8, device='cuda')
        tensors.append(x)
    allocator.sleep()
    y = torch.empty(alloc_bytes, dtype=torch.uint8, device='cuda')
    tensors.append(y)
    with pytest.raises(RuntimeError):
        allocator.wake_up()

@create_new_process_for_each_test()
def test_basic_cumem():
    shape = (1024, 1024)
    x = torch.empty(shape, device='cuda')
    x.zero_()
    allocator = CuMemAllocator.get_instance()
    with allocator.use_memory_pool():
        y = torch.empty(shape, device='cuda')
        y.zero_()
        y += 1
        z = torch.empty(shape, device='cuda')
        z.zero_()
        z += 2
    output = x + y + z
    assert torch.allclose(output, torch.ones_like(output) * 3)
    free_bytes = torch.cuda.mem_get_info()[0]
    allocator.sleep()
    free_bytes_after_sleep = torch.cuda.mem_get_info()[0]
    assert free_bytes_after_sleep > free_bytes
    allocator.wake_up()
    output = x + y + z
    assert torch.allclose(output, torch.ones_like(output) * 3)

@create_new_process_for_each_test()
def test_cumem_with_cudagraph():
    allocator = CuMemAllocator.get_instance()
    with allocator.use_memory_pool():
        weight = torch.eye(1024, device='cuda')
    with allocator.use_memory_pool(tag='discard'):
        cache = torch.empty(1024, 1024, device='cuda')

    def model(x):
        out = x @ weight
        cache[:out.size(0)].copy_(out)
        return out + 1
    x = torch.empty(128, 1024, device='cuda')
    model(x)
    model_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(model_graph):
        y = model(x)
    free_bytes = torch.cuda.mem_get_info()[0]
    allocator.sleep()
    free_bytes_after_sleep = torch.cuda.mem_get_info()[0]
    assert free_bytes_after_sleep > free_bytes
    allocator.wake_up()
    x.random_()
    model_graph.replay()
    assert torch.allclose(x, cache[:x.size(0)])
    assert torch.allclose(y, x + 1)

@create_new_process_for_each_test()
@pytest.mark.parametrize('model, use_v1', [('meta-llama/Llama-3.2-1B', True), ('facebook/opt-125m', False)])
def test_end_to_end(monkeypatch: pytest.MonkeyPatch, model: str, use_v1: bool):
    with monkeypatch.context() as m:
        m.setenv('VLLM_USE_V1', '1' if use_v1 else '0')
        free, total = torch.cuda.mem_get_info()
        used_bytes_baseline = total - free
        llm = LLM(model, enable_sleep_mode=True)
        prompt = 'How are you?'
        sampling_params = SamplingParams(temperature=0, max_tokens=10)
        output = llm.generate(prompt, sampling_params)
        llm.sleep(level=1)
        free_gpu_bytes_after_sleep, total = torch.cuda.mem_get_info()
        used_bytes = total - free_gpu_bytes_after_sleep - used_bytes_baseline
        if use_v1:
            assert used_bytes < 7 * GiB_bytes
        else:
            assert used_bytes < 2 * GiB_bytes
        llm.wake_up()
        output2 = llm.generate(prompt, sampling_params)
        assert output[0].outputs[0].text == output2[0].outputs[0].text
        llm.sleep(level=1)
        llm.wake_up(tags=['weights'])
        free_gpu_bytes_wake_up_w, total = torch.cuda.mem_get_info()
        used_bytes = total - free_gpu_bytes_wake_up_w - used_bytes_baseline
        if use_v1:
            assert used_bytes < 10 * GiB_bytes
        else:
            assert used_bytes < 6 * GiB_bytes
        llm.wake_up(tags=['kv_cache'])
        output3 = llm.generate(prompt, sampling_params)
        assert output[0].outputs[0].text == output3[0].outputs[0].text