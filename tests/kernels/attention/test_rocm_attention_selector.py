from vllm.attention.selector import _cached_get_attn_backend, get_attn_backend
from vllm.platforms.rocm import RocmPlatform
from vllm.utils import STR_BACKEND_ENV_VAR
import pytest
import torch

@pytest.fixture(autouse=True)
def clear_cache():
    """Clear lru cache to ensure each test case runs without caching.
    """
    _cached_get_attn_backend.cache_clear()

def test_selector(monkeypatch: pytest.MonkeyPatch):
    with monkeypatch.context() as m:
        m.setenv(STR_BACKEND_ENV_VAR, 'ROCM_FLASH')
        monkeypatch.setattr('vllm.attention.selector.current_platform', RocmPlatform())
        backend = get_attn_backend(16, torch.float16, torch.float16, 16, False)
        assert backend.get_name() == 'ROCM_FLASH' or backend.get_name() == 'TRITON_ATTN_VLLM_V1'
        m.setenv(STR_BACKEND_ENV_VAR, 'TRITON_MLA')
        backend = get_attn_backend(576, torch.bfloat16, 'auto', 16, False, use_mla=True)
        assert backend.get_name() == 'TRITON_MLA' or backend.get_name() == 'TRITON_MLA_VLLM_V1'
        m.setenv(STR_BACKEND_ENV_VAR, None)
        backend = get_attn_backend(576, torch.bfloat16, 'auto', 16, False, use_mla=True)
        assert backend.get_name() == 'TRITON_MLA' or backend.get_name() == 'TRITON_MLA_VLLM_V1'
        m.setenv(STR_BACKEND_ENV_VAR, 'ROCM_AITER_MLA')
        backend = get_attn_backend(576, torch.bfloat16, 'auto', 1, False, use_mla=True)
        assert backend.get_name() == 'ROCM_AITER_MLA' or backend.get_name() == 'ROCM_AITER_MLA_VLLM_V1'
        m.setenv(STR_BACKEND_ENV_VAR, None)
        m.setenv('VLLM_ROCM_USE_AITER', '1')
        backend = get_attn_backend(576, torch.bfloat16, 'auto', 1, False, use_mla=True)
        assert backend.get_name() == 'ROCM_AITER_MLA' or backend.get_name() == 'ROCM_AITER_MLA_VLLM_V1'