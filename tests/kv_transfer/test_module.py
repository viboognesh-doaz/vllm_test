import pytest
import subprocess
import sys
import torch

def run_python_script(script_name, timeout):
    script_name = f'kv_transfer/{script_name}'
    try:
        process0 = subprocess.Popen([sys.executable, script_name], env={'RANK': '0'}, stdout=sys.stdout, stderr=sys.stderr)
        process1 = subprocess.Popen([sys.executable, script_name], env={'RANK': '1'}, stdout=sys.stdout, stderr=sys.stderr)
        process0.wait(timeout=timeout)
        process1.wait(timeout=timeout)
        if process0.returncode != 0:
            pytest.fail(f'Test {script_name} failed for RANK=0, {process0.returncode}')
        if process1.returncode != 0:
            pytest.fail(f'Test {script_name} failed for RANK=1, {process1.returncode}')
    except subprocess.TimeoutExpired:
        process0.terminate()
        process1.terminate()
        pytest.fail(f'Test {script_name} timed out')
    except Exception as e:
        pytest.fail(f'Test {script_name} failed with error: {str(e)}')

@pytest.mark.parametrize('script_name,timeout', [('test_lookup_buffer.py', 60), ('test_send_recv.py', 120)])
def test_run_python_script(script_name, timeout):
    if torch.cuda.device_count() < 2:
        pytest.skip(f'Skipping test {script_name} because <2 GPUs are available')
    run_python_script(script_name, timeout)