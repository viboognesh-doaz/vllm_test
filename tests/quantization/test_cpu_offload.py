from ..utils import compare_two_settings
from tests.quantization.utils import is_quant_method_supported
import pytest

@pytest.mark.skipif(not is_quant_method_supported('fp8'), reason='fp8 is not supported on this GPU type.')
def test_cpu_offload_fp8():
    compare_two_settings('meta-llama/Llama-3.2-1B-Instruct', ['--quantization', 'fp8'], ['--quantization', 'fp8', '--cpu-offload-gb', '1'], max_wait_seconds=480)
    compare_two_settings('neuralmagic/Qwen2-1.5B-Instruct-FP8', [], ['--cpu-offload-gb', '1'], max_wait_seconds=480)

@pytest.mark.skipif(not is_quant_method_supported('gptq_marlin'), reason='gptq_marlin is not supported on this GPU type.')
def test_cpu_offload_gptq(monkeypatch):
    monkeypatch.setenv('VLLM_TEST_FORCE_LOAD_FORMAT', 'auto')
    compare_two_settings('Qwen/Qwen2-1.5B-Instruct-GPTQ-Int4', [], ['--cpu-offload-gb', '1'], max_wait_seconds=480)
    compare_two_settings('Qwen/Qwen2-1.5B-Instruct-GPTQ-Int4', ['--quantization', 'gptq'], ['--quantization', 'gptq', '--cpu-offload-gb', '1'], max_wait_seconds=480)

@pytest.mark.skipif(not is_quant_method_supported('awq_marlin'), reason='awq_marlin is not supported on this GPU type.')
def test_cpu_offload_awq(monkeypatch):
    monkeypatch.setenv('VLLM_TEST_FORCE_LOAD_FORMAT', 'auto')
    compare_two_settings('Qwen/Qwen2-1.5B-Instruct-AWQ', [], ['--cpu-offload-gb', '1'], max_wait_seconds=480)
    compare_two_settings('Qwen/Qwen2-1.5B-Instruct-AWQ', ['--quantization', 'awq'], ['--quantization', 'awq', '--cpu-offload-gb', '1'], max_wait_seconds=480)

@pytest.mark.skipif(not is_quant_method_supported('gptq_marlin'), reason='gptq_marlin is not supported on this GPU type.')
def test_cpu_offload_compressed_tensors(monkeypatch):
    monkeypatch.setenv('VLLM_TEST_FORCE_LOAD_FORMAT', 'auto')
    compare_two_settings('nm-testing/tinyllama-oneshot-w4a16-channel-v2', [], ['--cpu-offload-gb', '1'], max_wait_seconds=480)
    compare_two_settings('nm-testing/llama7b-one-shot-2_4-w4a16-marlin24-t', [], ['--cpu-offload-gb', '1'], max_wait_seconds=480)
    compare_two_settings('nm-testing/tinyllama-oneshot-w8w8-test-static-shape-change', [], ['--cpu-offload-gb', '1'], max_wait_seconds=480)