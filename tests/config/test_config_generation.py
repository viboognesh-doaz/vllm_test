from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.layers.quantization.quark.utils import deep_compare
import pytest

def test_cuda_empty_vs_unset_configs(monkeypatch: pytest.MonkeyPatch):
    """Test that configs created with normal (untouched) CUDA_VISIBLE_DEVICES
    and CUDA_VISIBLE_DEVICES="" are equivalent. This ensures consistent
    behavior regardless of whether GPU visibility is disabled via empty string
    or left in its normal state.
    """

    def create_config():
        engine_args = EngineArgs(model='deepseek-ai/DeepSeek-V2-Lite', trust_remote_code=True)
        return engine_args.create_engine_config()
    normal_config = create_config()
    with monkeypatch.context() as m:
        m.setenv('CUDA_VISIBLE_DEVICES', '')
        empty_config = create_config()
    normal_config_dict = vars(normal_config)
    empty_config_dict = vars(empty_config)
    normal_config_dict.pop('instance_id', None)
    empty_config_dict.pop('instance_id', None)
    assert deep_compare(normal_config_dict, empty_config_dict), 'Configs with normal CUDA_VISIBLE_DEVICES and CUDA_VISIBLE_DEVICES="" should be equivalent'