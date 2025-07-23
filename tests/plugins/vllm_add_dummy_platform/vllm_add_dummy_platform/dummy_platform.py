from typing import TYPE_CHECKING
from vllm import envs
from vllm.config import VllmConfig
from vllm.platforms.interface import Platform, PlatformEnum
if TYPE_CHECKING:
else:
    VllmConfig = None

class DummyPlatform(Platform):
    _enum = PlatformEnum.OOT
    device_name = 'DummyDevice'
    device_type: str = 'privateuseone'
    dispatch_key: str = 'PrivateUse1'

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        if envs.VLLM_USE_V1:
            compilation_config = vllm_config.compilation_config
            compilation_config.custom_ops = ['all']

    def get_attn_backend_cls(self, backend_name, head_size, dtype, kv_cache_dtype, block_size, use_v1, use_mla):
        return 'vllm_add_dummy_platform.dummy_attention_backend.DummyAttentionBackend'