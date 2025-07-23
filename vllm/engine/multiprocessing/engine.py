from contextlib import contextmanager
from ray.exceptions import RayTaskError
from typing import Iterator, List, Optional, Union
from vllm import AsyncEngineArgs, SamplingParams
from vllm.config import VllmConfig
from vllm.engine.llm_engine import LLMEngine
from vllm.engine.multiprocessing import ENGINE_DEAD_ERROR, IPC_DATA_EXT, IPC_HEALTH_EXT, IPC_INPUT_EXT, IPC_OUTPUT_EXT, REQUEST_OUTPUTS_T, VLLM_RPC_SUCCESS_STR, RPCAbortRequest, RPCAdapterLoadedResponse, RPCError, RPCIsSleepingRequest, RPCIsSleepingResponse, RPCLoadAdapterRequest, RPCProcessRequest, RPCResetMultiModalCacheRequest, RPCResetPrefixCacheRequest, RPCSleepRequest, RPCStartupRequest, RPCStartupResponse, RPCUProfileRequest, RPCWakeUpRequest
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.plugins import load_general_plugins
from vllm.transformers_utils.config import maybe_register_config_serialize_by_value
from vllm.usage.usage_lib import UsageContext
from vllm.worker.model_runner_base import InputProcessingError
import cloudpickle
import pickle
import signal
import zmq
logger = init_logger(__name__)
POLLING_TIMEOUT_MS = 10000
HEALTHY_RESPONSE = (pickle.dumps(VLLM_RPC_SUCCESS_STR),)

class MQLLMEngine:
    """A multiprocessing wrapper for
    [`LLMEngine`][vllm.engine.llm_engine.LLMEngine].

    This class is used to wrap the
    [`LLMEngine`][vllm.engine.llm_engine.LLMEngine] class to enable use
    in concurrnet manner. It runs a background loop and uses zeromq to
    receive new requests and stream outputs incrementally via ipc.

    The [`LLMEngine`][vllm.engine.llm_engine.LLMEngine] generate or encode
    process is kicked off when a new RPCProcessRequest is received by the
    input_socket.

    The self.engine_loop checks the input_socket for new requests,
    adds them to the LLMEngine if there are any, calls the internal
    [`LLMEngine.step()`][vllm.engine.llm_engine.LLMEngine.step], and sends
    the RequestOutputs back over the output_socket.

    If use_async_sockets is set, the logic associated with reading new
    requests from the socket and sending data to the socket is passed
    as a callback to the llm_engine, which calls the logic asynchronously
    such that the IPC can be overlapped with the GPU.

    Args:
        ipc_path: Base path for zeromq interprocess messaging
        use_async_sockets: Whether to make send/recv async with GPU
        log_requests: Whether to log the requests.
        *args: Arguments for [`LLMEngine`][vllm.engine.llm_engine.LLMEngine].
        **kwargs: Arguments for [`LLMEngine`][vllm.engine.llm_engine.LLMEngine].
    """

    def __init__(self, ipc_path: str, use_async_sockets: bool, *args, log_requests: bool=True, **kwargs) -> None:
        kwargs['use_cached_outputs'] = True
        self.engine = LLMEngine(*args, **kwargs)
        self.log_requests = log_requests
        self.use_async_sockets = use_async_sockets
        if self.use_async_sockets:
            self.engine.process_request_outputs_callback = self._async_socket_engine_callback
        self.ctx = zmq.Context()
        self.input_socket = self.ctx.socket(zmq.constants.PULL)
        self.input_socket.bind(f'{ipc_path}{IPC_INPUT_EXT}')
        self.output_socket = self.ctx.socket(zmq.constants.PUSH)
        self.output_socket.bind(f'{ipc_path}{IPC_OUTPUT_EXT}')
        self.heartbeat_socket = self.ctx.socket(zmq.constants.PUSH)
        self.heartbeat_socket.bind(f'{ipc_path}{IPC_HEALTH_EXT}')
        self.data_ipc_path = f'{ipc_path}{IPC_DATA_EXT}'
        self._errored_with: Optional[BaseException] = None

    @property
    def dead_error(self) -> BaseException:
        if self._errored_with is not None:
            return ENGINE_DEAD_ERROR(self._errored_with)
        else:
            return ENGINE_DEAD_ERROR()

    @classmethod
    def from_vllm_config(cls, vllm_config: VllmConfig, usage_context: UsageContext, disable_log_requests: bool, disable_log_stats: bool, ipc_path: str) -> 'MQLLMEngine':
        load_general_plugins()
        use_async_sockets = vllm_config.model_config.use_async_output_proc
        return cls(vllm_config=vllm_config, executor_class=LLMEngine._get_executor_cls(vllm_config), ipc_path=ipc_path, usage_context=usage_context, use_async_sockets=use_async_sockets, log_requests=not disable_log_requests, log_stats=not disable_log_stats)

    @staticmethod
    def from_engine_args(engine_args: AsyncEngineArgs, usage_context: UsageContext, ipc_path: str):
        """Creates an MQLLMEngine from the engine arguments."""
        vllm_config = engine_args.create_engine_config(usage_context)
        return MQLLMEngine.from_vllm_config(ipc_path=ipc_path, vllm_config=vllm_config, usage_context=usage_context, disable_log_requests=engine_args.disable_log_requests, disable_log_stats=engine_args.disable_log_stats)

    def start(self):
        try:
            try:
                logger.debug('Starting Startup Loop.')
                self.run_startup_loop()
                logger.debug('Starting Engine Loop.')
                self.run_engine_loop()
            except Exception as e:
                logger.exception(repr(e))
        except KeyboardInterrupt:
            logger.debug('Shutting down MQLLMEngine.')
        finally:
            logger.debug('MQLLMEngine is shut down.')
            self.cleanup()

    def cleanup(self):
        """Cleanup zeromq state on shutdown."""
        self.ctx.destroy(linger=0)
        del self.engine

    @contextmanager
    def make_data_socket(self) -> Iterator[zmq.Socket]:
        socket = self.ctx.socket(zmq.constants.ROUTER)
        try:
            socket.bind(self.data_ipc_path)
            yield socket
        finally:
            socket.close(linger=0)

    def run_startup_loop(self) -> None:
        """Startup loop for sending data from Engine -> Client."""
        with self.make_data_socket() as socket:
            response: Union[RPCStartupResponse, BaseException]
            try:
                identity, message = socket.recv_multipart(copy=False)
                request: RPCStartupRequest = pickle.loads(message.buffer)
                if request == RPCStartupRequest.IS_SERVER_READY:
                    tracing_enabled = self.engine.is_tracing_enabled()
                    response = RPCStartupResponse(tracing_enabled=tracing_enabled)
            except Exception as e:
                response = e
            socket.send_multipart((identity, pickle.dumps(response)), copy=False)

    def run_engine_loop(self):
        """Core busy loop of the LLMEngine."""
        while True:
            if not self.engine.has_unfinished_requests():
                while self.input_socket.poll(timeout=POLLING_TIMEOUT_MS) == 0:
                    self._health_check()
                    self.engine.do_log_stats()
                    logger.debug('Waiting for new requests in engine loop.')
            self.handle_new_input()
            request_outputs = self.engine_step()
            if not self.use_async_sockets:
                self._send_outputs(request_outputs)

    def engine_step(self) -> List[RequestOutput]:
        """Engine step wrapper with error handling."""
        try:
            return self.engine.step()
        except SystemExit:
            raise
        except InputProcessingError as e:
            rpc_err = RPCError(request_id=e.request_id, is_engine_errored=False, exception=e.__cause__)
            self._send_outputs(rpc_err)
            return []
        except BaseException as e:
            self._set_errored(e)
            rpc_err = RPCError(request_id=None, is_engine_errored=True, exception=e)
            self._send_outputs(rpc_err)
            raise e

    def handle_new_input(self):
        """Handle new input from the socket"""
        try:
            while self.input_socket.poll(timeout=0) != 0:
                frames = self.input_socket.recv_multipart(copy=False)
                request = pickle.loads(frames[0].buffer)
                if isinstance(request, RPCProcessRequest):
                    if len(frames) > 1:
                        assert isinstance(request.params, SamplingParams)
                        lprocs = cloudpickle.loads(frames[1].buffer)
                        request.params.logits_processors = lprocs
                    self._handle_process_request(request)
                elif isinstance(request, RPCAbortRequest):
                    self._handle_abort_request(request)
                elif isinstance(request, RPCUProfileRequest):
                    if request == RPCUProfileRequest.START_PROFILE:
                        self.start_profile()
                    else:
                        self.stop_profile()
                elif isinstance(request, RPCLoadAdapterRequest):
                    self._handle_load_adapter_request(request)
                elif isinstance(request, RPCResetMultiModalCacheRequest):
                    self.reset_mm_cache()
                elif isinstance(request, RPCResetPrefixCacheRequest):
                    self.reset_prefix_cache()
                elif isinstance(request, RPCSleepRequest):
                    self.sleep(request.value)
                elif isinstance(request, RPCWakeUpRequest):
                    self.wake_up(request.tags)
                elif isinstance(request, RPCIsSleepingRequest):
                    self._handle_is_sleeping_request(request)
                else:
                    raise ValueError(f'Unknown RPCRequest Type: {type(request)}')
        except Exception as e:
            self._set_errored(e)
            self._send_unhealthy(e)
            raise e from None

    def _handle_process_request(self, request: RPCProcessRequest):
        """Handle RPCProcessRequest by adding it to the LLMEngine."""
        request_id = request.request_id
        if self._errored_with is not None:
            rpc_err = RPCError(request_id=request_id, is_engine_errored=True, exception=ENGINE_DEAD_ERROR(self._errored_with))
            self._send_outputs(rpc_err)
        try:
            self.engine.add_request(request_id=request_id, prompt=request.prompt, params=request.params, lora_request=request.lora_request, trace_headers=request.trace_headers, prompt_adapter_request=request.prompt_adapter_request, priority=request.priority)
            if self.log_requests:
                logger.info('Added request %s.', request.request_id)
        except Exception as e:
            logger.debug('Failed to add request %s to engine. %s', request.request_id, e)
            is_errored = self._errored_with is not None
            rpc_err = RPCError(request_id=request_id, is_engine_errored=is_errored, exception=e)
            self._send_outputs(rpc_err)
            self.engine.abort_request(request_id)

    def _handle_abort_request(self, request: RPCAbortRequest):
        self.engine.abort_request(request.request_id)
        if self.log_requests:
            logger.info('Aborted request %s.', request.request_id)

    def _handle_load_adapter_request(self, request: RPCLoadAdapterRequest):
        try:
            self.engine.add_lora(request.lora_request)
        except BaseException as e:
            rpc_err = RPCError(request_id=request.request_id, is_engine_errored=False, exception=e)
            self._send_outputs(rpc_err)
            return
        self._send_outputs(RPCAdapterLoadedResponse(request_id=request.request_id))

    def _handle_is_sleeping_request(self, request: RPCIsSleepingRequest):
        is_sleeping = self.is_sleeping()
        self._send_outputs(RPCIsSleepingResponse(request_id=request.request_id, is_sleeping=is_sleeping))

    def _health_check(self):
        if self._errored_with is not None:
            self._send_unhealthy(self._errored_with)
        try:
            self.engine.check_health()
            self._send_healthy()
        except Exception as e:
            self._set_errored(e)
            self._send_unhealthy(e)

    def _send_outputs(self, outputs: REQUEST_OUTPUTS_T):
        """Send outputs back to the engine client. These can be:
        - Exceptions
        - A list of generation outputs
        - A response from loading a lora adapter
        """
        if outputs:
            try:
                if isinstance(outputs, RPCError) and isinstance(outputs.exception, RayTaskError):
                    outputs.exception = outputs.exception.cause
            except ImportError:
                pass
            output_bytes = pickle.dumps(outputs)
            self.output_socket.send_multipart((output_bytes,), copy=False)

    def _send_healthy(self):
        """Send HEALTHY message to RPCClient."""
        if not self.heartbeat_socket.closed:
            self.heartbeat_socket.send_multipart(HEALTHY_RESPONSE, copy=False)

    def _send_unhealthy(self, error: BaseException):
        """Send UNHEALTHY message to RPCClient."""
        if not self.heartbeat_socket.closed:
            error_bytes = pickle.dumps(error)
            self.heartbeat_socket.send_multipart((error_bytes,), copy=False)

    def _async_socket_engine_callback(self, request_outputs: REQUEST_OUTPUTS_T):
        """Callback used by engine to make socket handling async with GPU."""
        self._send_outputs(request_outputs)
        self.handle_new_input()

    def _set_errored(self, e: BaseException):
        """Log and set errored status if this is the first issue."""
        if self._errored_with is None:
            self._errored_with = e

    def start_profile(self) -> None:
        self.engine.start_profile()

    def stop_profile(self) -> None:
        self.engine.stop_profile()

    def reset_mm_cache(self) -> bool:
        return self.engine.reset_mm_cache()

    def reset_prefix_cache(self) -> bool:
        return self.engine.reset_prefix_cache()

    def sleep(self, level: int=1) -> None:
        self.engine.sleep(level)

    def wake_up(self, tags: Optional[list[str]]=None) -> None:
        self.engine.wake_up(tags)

    def is_sleeping(self) -> bool:
        return self.engine.is_sleeping()

def signal_handler(*_) -> None:
    raise KeyboardInterrupt('MQLLMEngine terminated')

def run_mp_engine(vllm_config: VllmConfig, usage_context: UsageContext, ipc_path: str, disable_log_stats: bool, disable_log_requests: bool, engine_alive):
    try:
        maybe_register_config_serialize_by_value()
        engine = MQLLMEngine.from_vllm_config(vllm_config=vllm_config, usage_context=usage_context, disable_log_stats=disable_log_stats, disable_log_requests=disable_log_requests, ipc_path=ipc_path)
        signal.signal(signal.SIGTERM, signal_handler)
        engine.start()
    except BaseException as e:
        logger.exception(e)
        engine_alive.value = False
        raise e from None