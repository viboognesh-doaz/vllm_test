from typing import Optional
import vllm_add_dummy_platform.dummy_custom_ops

def dummy_platform_plugin() -> Optional[str]:
    return 'vllm_add_dummy_platform.dummy_platform.DummyPlatform'

def register_ops():