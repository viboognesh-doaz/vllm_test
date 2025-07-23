from vllm.attention.ops.nki_flash_attn import reshape_and_cache
import pytest
import torch

@pytest.mark.parametrize('num_tokens, n_kv_head, d_head, num_blocks, block_size', [(32, 12, 64, 4, 128), (1, 12, 64, 4, 128), (128, 12, 64, 4, 128), (64, 16, 96, 8, 256), (256, 16, 96, 8, 256), (48, 32, 128, 16, 512), (512, 32, 128, 16, 512), (1024, 8, 32, 32, 32), (16, 64, 256, 4, 64), (2048, 24, 128, 64, 128), (4, 2, 16, 2, 16), (1, 1, 8, 1, 8)])
def test_reshape_and_cache(num_tokens, n_kv_head, d_head, num_blocks, block_size):
    torch.manual_seed(42)
    key_cpu = torch.randn(num_tokens, n_kv_head, d_head) / torch.sqrt(torch.tensor(d_head))
    value_cpu = torch.randn(num_tokens, n_kv_head, d_head) / torch.sqrt(torch.tensor(d_head))
    key_cache_cpu = torch.zeros(num_blocks, n_kv_head, block_size, d_head)
    value_cache_cpu = torch.zeros(num_blocks, n_kv_head, block_size, d_head)
    slot_mapping_cpu = torch.randperm(num_blocks * block_size)[:num_tokens]
    block_indices = torch.div(slot_mapping_cpu, block_size, rounding_mode='floor')
    block_offsets = slot_mapping_cpu % block_size
    for i in range(num_tokens):
        block_idx = block_indices[i]
        block_offset = block_offsets[i]
        key_cache_cpu[block_idx, :, block_offset, :] = key_cpu[i]
        value_cache_cpu[block_idx, :, block_offset, :] = value_cpu[i]
    device = torch.device('xla')
    key = key_cpu.to(device)
    value = value_cpu.to(device)
    key_cache = torch.zeros_like(key_cache_cpu, device=device)
    value_cache = torch.zeros_like(value_cache_cpu, device=device)
    slot_mapping = slot_mapping_cpu.to(device)
    kv_cache = torch.stack([key_cache, value_cache])
    reshape_and_cache(key, value, kv_cache, slot_mapping)
    key_cache, value_cache = torch.unbind(kv_cache, dim=0)
    key_cache_result = key_cache.cpu()
    value_cache_result = value_cache.cpu()
    torch.testing.assert_close(key_cache_result, key_cache_cpu, rtol=1e-05, atol=1e-05)
    torch.testing.assert_close(value_cache_result, value_cache_cpu, rtol=1e-05, atol=1e-05)