from typing import Optional
from unittest.mock import patch
from vllm.v1.utils import APIServerProcessManager, wait_for_completion_or_failure
import multiprocessing
import pytest
import socket
import threading
import time
WORKER_RUNTIME_SECONDS = 0.5

def mock_run_api_server_worker(listen_address, sock, args, client_config=None):
    """Mock run_api_server_worker that runs for a specific time."""
    print(f'Mock worker started with client_config: {client_config}')
    time.sleep(WORKER_RUNTIME_SECONDS)
    print('Mock worker completed successfully')

@pytest.fixture
def api_server_args():
    """Fixture to provide arguments for APIServerProcessManager."""
    sock = socket.socket()
    return {'target_server_fn': mock_run_api_server_worker, 'listen_address': 'localhost:8000', 'sock': sock, 'args': 'test_args', 'num_servers': 3, 'input_addresses': ['tcp://127.0.0.1:5001', 'tcp://127.0.0.1:5002', 'tcp://127.0.0.1:5003'], 'output_addresses': ['tcp://127.0.0.1:6001', 'tcp://127.0.0.1:6002', 'tcp://127.0.0.1:6003'], 'stats_update_address': 'tcp://127.0.0.1:7000'}

@pytest.mark.parametrize('with_stats_update', [True, False])
def test_api_server_process_manager_init(api_server_args, with_stats_update):
    """Test initializing the APIServerProcessManager."""
    global WORKER_RUNTIME_SECONDS
    WORKER_RUNTIME_SECONDS = 0.5
    args = api_server_args.copy()
    if not with_stats_update:
        args.pop('stats_update_address')
    manager = APIServerProcessManager(**args)
    try:
        assert len(manager.processes) == 3
        for proc in manager.processes:
            assert proc.is_alive()
        print('Waiting for processes to run...')
        time.sleep(WORKER_RUNTIME_SECONDS / 2)
        for proc in manager.processes:
            assert proc.is_alive()
    finally:
        print('Cleaning up processes...')
        manager.close()
        time.sleep(0.2)
        for proc in manager.processes:
            assert not proc.is_alive()

@patch('vllm.entrypoints.cli.serve.run_api_server_worker', mock_run_api_server_worker)
def test_wait_for_completion_or_failure(api_server_args):
    """Test that wait_for_completion_or_failure works with failures."""
    global WORKER_RUNTIME_SECONDS
    WORKER_RUNTIME_SECONDS = 1.0
    manager = APIServerProcessManager(**api_server_args)
    try:
        assert len(manager.processes) == 3
        result: dict[str, Optional[Exception]] = {'exception': None}

        def run_with_exception_capture():
            try:
                wait_for_completion_or_failure(api_server_manager=manager)
            except Exception as e:
                result['exception'] = e
        wait_thread = threading.Thread(target=run_with_exception_capture, daemon=True)
        wait_thread.start()
        time.sleep(0.2)
        assert all((proc.is_alive() for proc in manager.processes))
        print('Simulating process failure...')
        manager.processes[0].terminate()
        wait_thread.join(timeout=1.0)
        assert not wait_thread.is_alive()
        assert result['exception'] is not None
        assert 'died with exit code' in str(result['exception'])
        for i, proc in enumerate(manager.processes):
            assert not proc.is_alive(), f'Process {i} should not be alive'
    finally:
        manager.close()
        time.sleep(0.2)

@pytest.mark.timeout(30)
def test_normal_completion(api_server_args):
    """Test that wait_for_completion_or_failure works in normal completion."""
    global WORKER_RUNTIME_SECONDS
    WORKER_RUNTIME_SECONDS = 0.1
    manager = APIServerProcessManager(**api_server_args)
    try:
        remaining_processes = manager.processes.copy()
        while remaining_processes:
            for proc in remaining_processes:
                if not proc.is_alive():
                    remaining_processes.remove(proc)
            time.sleep(0.1)
        for i, proc in enumerate(manager.processes):
            assert not proc.is_alive(), f'Process {i} still alive after terminate()'
        wait_for_completion_or_failure(api_server_manager=manager)
    finally:
        manager.close()
        time.sleep(0.2)

@pytest.mark.timeout(30)
def test_external_process_monitoring(api_server_args):
    """Test that wait_for_completion_or_failure handles additional processes."""
    global WORKER_RUNTIME_SECONDS
    WORKER_RUNTIME_SECONDS = 100
    spawn_context = multiprocessing.get_context('spawn')
    external_proc = spawn_context.Process(target=mock_run_api_server_worker, name='MockExternalProcess')
    external_proc.start()

    class MockCoordinator:

        def __init__(self, proc):
            self.proc = proc

        def close(self):
            if self.proc.is_alive():
                self.proc.terminate()
                self.proc.join(timeout=0.5)
    mock_coordinator = MockCoordinator(external_proc)
    manager = APIServerProcessManager(**api_server_args)
    try:
        assert len(manager.processes) == 3
        result: dict[str, Optional[Exception]] = {'exception': None}

        def run_with_exception_capture():
            try:
                wait_for_completion_or_failure(api_server_manager=manager, coordinator=mock_coordinator)
            except Exception as e:
                result['exception'] = e
        wait_thread = threading.Thread(target=run_with_exception_capture, daemon=True)
        wait_thread.start()
        time.sleep(0.2)
        external_proc.terminate()
        wait_thread.join(timeout=1.0)
        assert not wait_thread.is_alive(), 'wait_for_completion_or_failure thread still running'
        assert result['exception'] is not None, 'No exception was raised'
        error_message = str(result['exception'])
        assert 'died with exit code' in error_message, f'Unexpected error message: {error_message}'
        assert 'MockExternalProcess' in error_message, f"Error doesn't mention external process: {error_message}"
        for i, proc in enumerate(manager.processes):
            assert not proc.is_alive(), f'API server process {i} was not terminated'
    finally:
        manager.close()
        mock_coordinator.close()
        time.sleep(0.2)