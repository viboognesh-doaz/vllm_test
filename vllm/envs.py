from typing import TYPE_CHECKING, Any, Callable, Optional
from urllib.parse import urlparse
import hashlib
import os
import sys
import tempfile
if TYPE_CHECKING:
    VLLM_HOST_IP: str = ''
    VLLM_PORT: Optional[int] = None
    VLLM_RPC_BASE_PATH: str = tempfile.gettempdir()
    VLLM_USE_MODELSCOPE: bool = False
    VLLM_RINGBUFFER_WARNING_INTERVAL: int = 60
    VLLM_NCCL_SO_PATH: Optional[str] = None
    LD_LIBRARY_PATH: Optional[str] = None
    VLLM_USE_TRITON_FLASH_ATTN: bool = True
    VLLM_V1_USE_PREFILL_DECODE_ATTENTION: bool = False
    VLLM_FLASH_ATTN_VERSION: Optional[int] = None
    LOCAL_RANK: int = 0
    CUDA_VISIBLE_DEVICES: Optional[str] = None
    VLLM_ENGINE_ITERATION_TIMEOUT_S: int = 60
    VLLM_API_KEY: Optional[str] = None
    S3_ACCESS_KEY_ID: Optional[str] = None
    S3_SECRET_ACCESS_KEY: Optional[str] = None
    S3_ENDPOINT_URL: Optional[str] = None
    VLLM_MODEL_REDIRECT_PATH: Optional[str] = None
    VLLM_CACHE_ROOT: str = os.path.expanduser('~/.cache/vllm')
    VLLM_CONFIG_ROOT: str = os.path.expanduser('~/.config/vllm')
    VLLM_USAGE_STATS_SERVER: str = 'https://stats.vllm.ai'
    VLLM_NO_USAGE_STATS: bool = False
    VLLM_DO_NOT_TRACK: bool = False
    VLLM_USAGE_SOURCE: str = ''
    VLLM_CONFIGURE_LOGGING: int = 1
    VLLM_LOGGING_LEVEL: str = 'INFO'
    VLLM_LOGGING_PREFIX: str = ''
    VLLM_LOGGING_CONFIG_PATH: Optional[str] = None
    VLLM_LOGITS_PROCESSOR_THREADS: Optional[int] = None
    VLLM_TRACE_FUNCTION: int = 0
    VLLM_ATTENTION_BACKEND: Optional[str] = None
    VLLM_USE_FLASHINFER_SAMPLER: Optional[bool] = None
    VLLM_FLASHINFER_FORCE_TENSOR_CORES: bool = False
    VLLM_PP_LAYER_PARTITION: Optional[str] = None
    VLLM_CPU_KVCACHE_SPACE: int = 0
    VLLM_CPU_OMP_THREADS_BIND: str = ''
    VLLM_CPU_NUM_OF_RESERVED_CPU: Optional[int] = None
    VLLM_CPU_MOE_PREPACK: bool = True
    VLLM_CPU_SGL_KERNEL: bool = False
    VLLM_XLA_CACHE_PATH: str = os.path.join(VLLM_CACHE_ROOT, 'xla_cache')
    VLLM_XLA_CHECK_RECOMPILATION: bool = False
    VLLM_FUSED_MOE_CHUNK_SIZE: int = 64 * 1024
    VLLM_ENABLE_FUSED_MOE_ACTIVATION_CHUNKING: bool = True
    VLLM_USE_RAY_SPMD_WORKER: bool = False
    VLLM_USE_RAY_COMPILED_DAG: bool = False
    VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE: str = 'auto'
    VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM: bool = False
    VLLM_XLA_USE_SPMD: bool = False
    VLLM_WORKER_MULTIPROC_METHOD: str = 'fork'
    VLLM_ASSETS_CACHE: str = os.path.join(VLLM_CACHE_ROOT, 'assets')
    VLLM_IMAGE_FETCH_TIMEOUT: int = 5
    VLLM_VIDEO_FETCH_TIMEOUT: int = 30
    VLLM_AUDIO_FETCH_TIMEOUT: int = 10
    VLLM_VIDEO_LOADER_BACKEND: str = 'opencv'
    VLLM_MM_INPUT_CACHE_GIB: int = 8
    VLLM_TARGET_DEVICE: str = 'cuda'
    MAX_JOBS: Optional[str] = None
    NVCC_THREADS: Optional[str] = None
    VLLM_USE_PRECOMPILED: bool = False
    VLLM_TEST_USE_PRECOMPILED_NIGHTLY_WHEEL: bool = False
    VLLM_NO_DEPRECATION_WARNING: bool = False
    VLLM_KEEP_ALIVE_ON_ENGINE_DEATH: bool = False
    CMAKE_BUILD_TYPE: Optional[str] = None
    VERBOSE: bool = False
    VLLM_ALLOW_LONG_MAX_MODEL_LEN: bool = False
    VLLM_RPC_TIMEOUT: int = 10000
    VLLM_HTTP_TIMEOUT_KEEP_ALIVE: int = 5
    VLLM_PLUGINS: Optional[list[str]] = None
    VLLM_LORA_RESOLVER_CACHE_DIR: Optional[str] = None
    VLLM_TORCH_PROFILER_DIR: Optional[str] = None
    VLLM_USE_TRITON_AWQ: bool = False
    VLLM_ALLOW_RUNTIME_LORA_UPDATING: bool = False
    VLLM_SKIP_P2P_CHECK: bool = False
    VLLM_DISABLED_KERNELS: list[str] = []
    VLLM_USE_V1: bool = True
    VLLM_ROCM_USE_AITER: bool = False
    VLLM_ROCM_USE_AITER_PAGED_ATTN: bool = False
    VLLM_ROCM_USE_AITER_LINEAR: bool = True
    VLLM_ROCM_USE_AITER_MOE: bool = True
    VLLM_ROCM_USE_AITER_RMSNORM: bool = True
    VLLM_ROCM_USE_AITER_MLA: bool = True
    VLLM_ROCM_USE_AITER_MHA: bool = True
    VLLM_ROCM_USE_SKINNY_GEMM: bool = True
    VLLM_ROCM_FP8_PADDING: bool = True
    VLLM_ROCM_MOE_PADDING: bool = True
    VLLM_ROCM_CUSTOM_PAGED_ATTN: bool = True
    VLLM_ENABLE_V1_MULTIPROCESSING: bool = True
    VLLM_LOG_BATCHSIZE_INTERVAL: float = -1
    VLLM_DISABLE_COMPILE_CACHE: bool = False
    Q_SCALE_CONSTANT: int = 200
    K_SCALE_CONSTANT: int = 200
    V_SCALE_CONSTANT: int = 100
    VLLM_SERVER_DEV_MODE: bool = False
    VLLM_V1_OUTPUT_PROC_CHUNK_SIZE: int = 128
    VLLM_MLA_DISABLE: bool = False
    VLLM_RAY_PER_WORKER_GPUS: float = 1.0
    VLLM_RAY_BUNDLE_INDICES: str = ''
    VLLM_CUDART_SO_PATH: Optional[str] = None
    VLLM_DP_RANK: int = 0
    VLLM_DP_RANK_LOCAL: int = -1
    VLLM_DP_SIZE: int = 1
    VLLM_DP_MASTER_IP: str = ''
    VLLM_DP_MASTER_PORT: int = 0
    VLLM_MOE_DP_CHUNK_SIZE: int = 256
    VLLM_RANDOMIZE_DP_DUMMY_INPUTS: bool = False
    VLLM_MARLIN_USE_ATOMIC_ADD: bool = False
    VLLM_V0_USE_OUTLINES_CACHE: bool = False
    VLLM_V1_USE_OUTLINES_CACHE: bool = False
    VLLM_TPU_BUCKET_PADDING_GAP: int = 0
    VLLM_TPU_MOST_MODEL_LEN: Optional[int] = None
    VLLM_USE_DEEP_GEMM: bool = False
    VLLM_USE_FLASHINFER_MOE_FP8: bool = False
    VLLM_USE_FLASHINFER_MOE_FP4: bool = False
    VLLM_XGRAMMAR_CACHE_MB: int = 0
    VLLM_MSGPACK_ZERO_COPY_THRESHOLD: int = 256
    VLLM_ALLOW_INSECURE_SERIALIZATION: bool = False
    VLLM_NIXL_SIDE_CHANNEL_HOST: str = 'localhost'
    VLLM_NIXL_SIDE_CHANNEL_PORT: int = 5557
    VLLM_ALL2ALL_BACKEND: str = 'naive'
    VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE: int = 163840
    VLLM_TOOL_PARSE_REGEX_TIMEOUT_SECONDS: int = 1
    VLLM_SLEEP_WHEN_IDLE: bool = False
    VLLM_MQ_MAX_CHUNK_BYTES_MB: int = 16
    VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS: int = 300
    VLLM_KV_CACHE_LAYOUT: Optional[str] = None
    VLLM_COMPUTE_NANS_IN_LOGITS: bool = False
    VLLM_USE_NVFP4_CT_EMULATIONS: bool = False
    VLLM_ROCM_QUICK_REDUCE_QUANTIZATION: str = 'NONE'
    VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16: bool = True
    VLLM_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB: Optional[int] = None
    VLLM_NIXL_ABORT_REQUEST_TIMEOUT: int = 120
    VLLM_USE_CUDNN_PREFILL: bool = False
    VLLM_LOOPBACK_IP: str = ''

def get_default_cache_root():
    return os.getenv('XDG_CACHE_HOME', os.path.join(os.path.expanduser('~'), '.cache'))

def get_default_config_root():
    return os.getenv('XDG_CONFIG_HOME', os.path.join(os.path.expanduser('~'), '.config'))

def maybe_convert_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    return int(value)

def get_vllm_port() -> Optional[int]:
    """Get the port from VLLM_PORT environment variable.

    Returns:
        The port number as an integer if VLLM_PORT is set, None otherwise.

    Raises:
        ValueError: If VLLM_PORT is a URI, suggest k8s service discovery issue.
    """
    if 'VLLM_PORT' not in os.environ:
        return None
    port = os.getenv('VLLM_PORT', '0')
    try:
        return int(port)
    except ValueError as err:
        parsed = urlparse(port)
        if parsed.scheme:
            raise ValueError(f"VLLM_PORT '{port}' appears to be a URI. This may be caused by a Kubernetes service discovery issue,check the warning in: https://docs.vllm.ai/en/stable/serving/env_vars.html") from None
        raise ValueError(f"VLLM_PORT '{port}' must be a valid integer") from err
environment_variables: dict[str, Callable[[], Any]] = {'VLLM_TARGET_DEVICE': lambda: os.getenv('VLLM_TARGET_DEVICE', 'cuda'), 'MAX_JOBS': lambda: os.getenv('MAX_JOBS', None), 'NVCC_THREADS': lambda: os.getenv('NVCC_THREADS', None), 'VLLM_USE_PRECOMPILED': lambda: bool(os.environ.get('VLLM_USE_PRECOMPILED')) or bool(os.environ.get('VLLM_PRECOMPILED_WHEEL_LOCATION')), 'VLLM_TEST_USE_PRECOMPILED_NIGHTLY_WHEEL': lambda: bool(int(os.getenv('VLLM_TEST_USE_PRECOMPILED_NIGHTLY_WHEEL', '0'))), 'CMAKE_BUILD_TYPE': lambda: os.getenv('CMAKE_BUILD_TYPE'), 'VERBOSE': lambda: bool(int(os.getenv('VERBOSE', '0'))), 'VLLM_CONFIG_ROOT': lambda: os.path.expanduser(os.getenv('VLLM_CONFIG_ROOT', os.path.join(get_default_config_root(), 'vllm'))), 'VLLM_CACHE_ROOT': lambda: os.path.expanduser(os.getenv('VLLM_CACHE_ROOT', os.path.join(get_default_cache_root(), 'vllm'))), 'VLLM_HOST_IP': lambda: os.getenv('VLLM_HOST_IP', ''), 'VLLM_PORT': get_vllm_port, 'VLLM_RPC_BASE_PATH': lambda: os.getenv('VLLM_RPC_BASE_PATH', tempfile.gettempdir()), 'VLLM_USE_MODELSCOPE': lambda: os.environ.get('VLLM_USE_MODELSCOPE', 'False').lower() == 'true', 'VLLM_RINGBUFFER_WARNING_INTERVAL': lambda: int(os.environ.get('VLLM_RINGBUFFER_WARNING_INTERVAL', '60')), 'CUDA_HOME': lambda: os.environ.get('CUDA_HOME', None), 'VLLM_NCCL_SO_PATH': lambda: os.environ.get('VLLM_NCCL_SO_PATH', None), 'LD_LIBRARY_PATH': lambda: os.environ.get('LD_LIBRARY_PATH', None), 'VLLM_USE_TRITON_FLASH_ATTN': lambda: os.environ.get('VLLM_USE_TRITON_FLASH_ATTN', 'True').lower() in ('true', '1'), 'VLLM_V1_USE_PREFILL_DECODE_ATTENTION': lambda: os.getenv('VLLM_V1_USE_PREFILL_DECODE_ATTENTION', 'False').lower() in ('true', '1'), 'VLLM_FLASH_ATTN_VERSION': lambda: maybe_convert_int(os.environ.get('VLLM_FLASH_ATTN_VERSION', None)), 'VLLM_TEST_DYNAMO_FULLGRAPH_CAPTURE': lambda: bool(os.environ.get('VLLM_TEST_DYNAMO_FULLGRAPH_CAPTURE', '1') != '0'), 'VLLM_USE_STANDALONE_COMPILE': lambda: os.environ.get('VLLM_USE_STANDALONE_COMPILE', '1') == '1', 'LOCAL_RANK': lambda: int(os.environ.get('LOCAL_RANK', '0')), 'CUDA_VISIBLE_DEVICES': lambda: os.environ.get('CUDA_VISIBLE_DEVICES', None), 'VLLM_ENGINE_ITERATION_TIMEOUT_S': lambda: int(os.environ.get('VLLM_ENGINE_ITERATION_TIMEOUT_S', '60')), 'VLLM_API_KEY': lambda: os.environ.get('VLLM_API_KEY', None), 'VLLM_DEBUG_LOG_API_SERVER_RESPONSE': lambda: os.environ.get('VLLM_DEBUG_LOG_API_SERVER_RESPONSE', 'False').lower() == 'true', 'S3_ACCESS_KEY_ID': lambda: os.environ.get('S3_ACCESS_KEY_ID', None), 'S3_SECRET_ACCESS_KEY': lambda: os.environ.get('S3_SECRET_ACCESS_KEY', None), 'S3_ENDPOINT_URL': lambda: os.environ.get('S3_ENDPOINT_URL', None), 'VLLM_USAGE_STATS_SERVER': lambda: os.environ.get('VLLM_USAGE_STATS_SERVER', 'https://stats.vllm.ai'), 'VLLM_NO_USAGE_STATS': lambda: os.environ.get('VLLM_NO_USAGE_STATS', '0') == '1', 'VLLM_DO_NOT_TRACK': lambda: (os.environ.get('VLLM_DO_NOT_TRACK', None) or os.environ.get('DO_NOT_TRACK', None) or '0') == '1', 'VLLM_USAGE_SOURCE': lambda: os.environ.get('VLLM_USAGE_SOURCE', 'production'), 'VLLM_CONFIGURE_LOGGING': lambda: int(os.getenv('VLLM_CONFIGURE_LOGGING', '1')), 'VLLM_LOGGING_CONFIG_PATH': lambda: os.getenv('VLLM_LOGGING_CONFIG_PATH'), 'VLLM_LOGGING_LEVEL': lambda: os.getenv('VLLM_LOGGING_LEVEL', 'INFO').upper(), 'VLLM_LOGGING_PREFIX': lambda: os.getenv('VLLM_LOGGING_PREFIX', ''), 'VLLM_LOGITS_PROCESSOR_THREADS': lambda: int(os.getenv('VLLM_LOGITS_PROCESSOR_THREADS', '0')) if 'VLLM_LOGITS_PROCESSOR_THREADS' in os.environ else None, 'VLLM_TRACE_FUNCTION': lambda: int(os.getenv('VLLM_TRACE_FUNCTION', '0')), 'VLLM_ATTENTION_BACKEND': lambda: os.getenv('VLLM_ATTENTION_BACKEND', None), 'VLLM_USE_FLASHINFER_SAMPLER': lambda: bool(int(os.environ['VLLM_USE_FLASHINFER_SAMPLER'])) if 'VLLM_USE_FLASHINFER_SAMPLER' in os.environ else None, 'VLLM_FLASHINFER_FORCE_TENSOR_CORES': lambda: bool(int(os.getenv('VLLM_FLASHINFER_FORCE_TENSOR_CORES', '0'))), 'VLLM_PP_LAYER_PARTITION': lambda: os.getenv('VLLM_PP_LAYER_PARTITION', None), 'VLLM_CPU_KVCACHE_SPACE': lambda: int(os.getenv('VLLM_CPU_KVCACHE_SPACE', '0')), 'VLLM_CPU_OMP_THREADS_BIND': lambda: os.getenv('VLLM_CPU_OMP_THREADS_BIND', 'auto'), 'VLLM_CPU_NUM_OF_RESERVED_CPU': lambda: int(os.getenv('VLLM_CPU_NUM_OF_RESERVED_CPU', '0')) if 'VLLM_CPU_NUM_OF_RESERVED_CPU' in os.environ else None, 'VLLM_CPU_MOE_PREPACK': lambda: bool(int(os.getenv('VLLM_CPU_MOE_PREPACK', '1'))), 'VLLM_CPU_SGL_KERNEL': lambda: bool(int(os.getenv('VLLM_CPU_SGL_KERNEL', '0'))), 'VLLM_USE_RAY_SPMD_WORKER': lambda: bool(int(os.getenv('VLLM_USE_RAY_SPMD_WORKER', '0'))), 'VLLM_USE_RAY_COMPILED_DAG': lambda: bool(int(os.getenv('VLLM_USE_RAY_COMPILED_DAG', '0'))), 'VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE': lambda: os.getenv('VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE', 'auto'), 'VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM': lambda: bool(int(os.getenv('VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM', '0'))), 'VLLM_WORKER_MULTIPROC_METHOD': lambda: os.getenv('VLLM_WORKER_MULTIPROC_METHOD', 'fork'), 'VLLM_ASSETS_CACHE': lambda: os.path.expanduser(os.getenv('VLLM_ASSETS_CACHE', os.path.join(get_default_cache_root(), 'vllm', 'assets'))), 'VLLM_IMAGE_FETCH_TIMEOUT': lambda: int(os.getenv('VLLM_IMAGE_FETCH_TIMEOUT', '5')), 'VLLM_VIDEO_FETCH_TIMEOUT': lambda: int(os.getenv('VLLM_VIDEO_FETCH_TIMEOUT', '30')), 'VLLM_AUDIO_FETCH_TIMEOUT': lambda: int(os.getenv('VLLM_AUDIO_FETCH_TIMEOUT', '10')), 'VLLM_VIDEO_LOADER_BACKEND': lambda: os.getenv('VLLM_VIDEO_LOADER_BACKEND', 'opencv'), 'VLLM_MM_INPUT_CACHE_GIB': lambda: int(os.getenv('VLLM_MM_INPUT_CACHE_GIB', '4')), 'VLLM_XLA_CACHE_PATH': lambda: os.path.expanduser(os.getenv('VLLM_XLA_CACHE_PATH', os.path.join(get_default_cache_root(), 'vllm', 'xla_cache'))), 'VLLM_XLA_CHECK_RECOMPILATION': lambda: bool(int(os.getenv('VLLM_XLA_CHECK_RECOMPILATION', '0'))), 'VLLM_XLA_USE_SPMD': lambda: bool(int(os.getenv('VLLM_XLA_USE_SPMD', '0'))), 'VLLM_FUSED_MOE_CHUNK_SIZE': lambda: int(os.getenv('VLLM_FUSED_MOE_CHUNK_SIZE', '32768')), 'VLLM_ENABLE_FUSED_MOE_ACTIVATION_CHUNKING': lambda: bool(int(os.getenv('VLLM_ENABLE_FUSED_MOE_ACTIVATION_CHUNKING', '1'))), 'VLLM_NO_DEPRECATION_WARNING': lambda: bool(int(os.getenv('VLLM_NO_DEPRECATION_WARNING', '0'))), 'VLLM_KEEP_ALIVE_ON_ENGINE_DEATH': lambda: bool(os.getenv('VLLM_KEEP_ALIVE_ON_ENGINE_DEATH', 0)), 'VLLM_ALLOW_LONG_MAX_MODEL_LEN': lambda: os.environ.get('VLLM_ALLOW_LONG_MAX_MODEL_LEN', '0').strip().lower() in ('1', 'true'), 'VLLM_TEST_FORCE_FP8_MARLIN': lambda: os.environ.get('VLLM_TEST_FORCE_FP8_MARLIN', '0').strip().lower() in ('1', 'true'), 'VLLM_TEST_FORCE_LOAD_FORMAT': lambda: os.getenv('VLLM_TEST_FORCE_LOAD_FORMAT', 'dummy'), 'VLLM_RPC_TIMEOUT': lambda: int(os.getenv('VLLM_RPC_TIMEOUT', '10000')), 'VLLM_HTTP_TIMEOUT_KEEP_ALIVE': lambda: int(os.environ.get('VLLM_HTTP_TIMEOUT_KEEP_ALIVE', '5')), 'VLLM_PLUGINS': lambda: None if 'VLLM_PLUGINS' not in os.environ else os.environ['VLLM_PLUGINS'].split(','), 'VLLM_LORA_RESOLVER_CACHE_DIR': lambda: os.getenv('VLLM_LORA_RESOLVER_CACHE_DIR', None), 'VLLM_TORCH_PROFILER_DIR': lambda: None if os.getenv('VLLM_TORCH_PROFILER_DIR', None) is None else os.path.expanduser(os.getenv('VLLM_TORCH_PROFILER_DIR', '.')), 'VLLM_USE_TRITON_AWQ': lambda: bool(int(os.getenv('VLLM_USE_TRITON_AWQ', '0'))), 'VLLM_ALLOW_RUNTIME_LORA_UPDATING': lambda: os.environ.get('VLLM_ALLOW_RUNTIME_LORA_UPDATING', '0').strip().lower() in ('1', 'true'), 'VLLM_SKIP_P2P_CHECK': lambda: os.getenv('VLLM_SKIP_P2P_CHECK', '0') == '1', 'VLLM_DISABLED_KERNELS': lambda: [] if 'VLLM_DISABLED_KERNELS' not in os.environ else os.environ['VLLM_DISABLED_KERNELS'].split(','), 'VLLM_USE_V1': lambda: bool(int(os.getenv('VLLM_USE_V1', '1'))), 'VLLM_ROCM_USE_AITER': lambda: os.getenv('VLLM_ROCM_USE_AITER', 'False').lower() in ('true', '1'), 'VLLM_ROCM_USE_AITER_PAGED_ATTN': lambda: os.getenv('VLLM_ROCM_USE_AITER_PAGED_ATTN', 'False').lower() in ('true', '1'), 'VLLM_ROCM_USE_AITER_LINEAR': lambda: os.getenv('VLLM_ROCM_USE_AITER_LINEAR', 'True').lower() in ('true', '1'), 'VLLM_ROCM_USE_AITER_MOE': lambda: os.getenv('VLLM_ROCM_USE_AITER_MOE', 'True').lower() in ('true', '1'), 'VLLM_ROCM_USE_AITER_RMSNORM': lambda: os.getenv('VLLM_ROCM_USE_AITER_RMSNORM', 'True').lower() in ('true', '1'), 'VLLM_ROCM_USE_AITER_MLA': lambda: os.getenv('VLLM_ROCM_USE_AITER_MLA', 'True').lower() in ('true', '1'), 'VLLM_ROCM_USE_AITER_MHA': lambda: os.getenv('VLLM_ROCM_USE_AITER_MHA', 'True').lower() in ('true', '1'), 'VLLM_ROCM_USE_SKINNY_GEMM': lambda: os.getenv('VLLM_ROCM_USE_SKINNY_GEMM', 'True').lower() in ('true', '1'), 'VLLM_ROCM_FP8_PADDING': lambda: bool(int(os.getenv('VLLM_ROCM_FP8_PADDING', '1'))), 'VLLM_ROCM_MOE_PADDING': lambda: bool(int(os.getenv('VLLM_ROCM_MOE_PADDING', '1'))), 'VLLM_ROCM_CUSTOM_PAGED_ATTN': lambda: os.getenv('VLLM_ROCM_CUSTOM_PAGED_ATTN', 'True').lower() in ('true', '1'), 'VLLM_ROCM_QUICK_REDUCE_QUANTIZATION': lambda: os.getenv('VLLM_ROCM_QUICK_REDUCE_QUANTIZATION', 'NONE').upper(), 'VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16': lambda: os.getenv('VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16', 'True').lower() in ('true', '1'), 'VLLM_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB': lambda: maybe_convert_int(os.environ.get('VLLM_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB', None)), 'Q_SCALE_CONSTANT': lambda: int(os.getenv('Q_SCALE_CONSTANT', '200')), 'K_SCALE_CONSTANT': lambda: int(os.getenv('K_SCALE_CONSTANT', '200')), 'V_SCALE_CONSTANT': lambda: int(os.getenv('V_SCALE_CONSTANT', '100')), 'VLLM_ENABLE_V1_MULTIPROCESSING': lambda: bool(int(os.getenv('VLLM_ENABLE_V1_MULTIPROCESSING', '1'))), 'VLLM_LOG_BATCHSIZE_INTERVAL': lambda: float(os.getenv('VLLM_LOG_BATCHSIZE_INTERVAL', '-1')), 'VLLM_DISABLE_COMPILE_CACHE': lambda: bool(int(os.getenv('VLLM_DISABLE_COMPILE_CACHE', '0'))), 'VLLM_SERVER_DEV_MODE': lambda: bool(int(os.getenv('VLLM_SERVER_DEV_MODE', '0'))), 'VLLM_V1_OUTPUT_PROC_CHUNK_SIZE': lambda: int(os.getenv('VLLM_V1_OUTPUT_PROC_CHUNK_SIZE', '128')), 'VLLM_MLA_DISABLE': lambda: bool(int(os.getenv('VLLM_MLA_DISABLE', '0'))), 'VLLM_RAY_PER_WORKER_GPUS': lambda: float(os.getenv('VLLM_RAY_PER_WORKER_GPUS', '1.0')), 'VLLM_RAY_BUNDLE_INDICES': lambda: os.getenv('VLLM_RAY_BUNDLE_INDICES', ''), 'VLLM_CUDART_SO_PATH': lambda: os.getenv('VLLM_CUDART_SO_PATH', None), 'VLLM_DP_RANK': lambda: int(os.getenv('VLLM_DP_RANK', '0')), 'VLLM_DP_RANK_LOCAL': lambda: int(os.getenv('VLLM_DP_RANK_LOCAL', sys.modules[__name__].VLLM_DP_RANK)), 'VLLM_DP_SIZE': lambda: int(os.getenv('VLLM_DP_SIZE', '1')), 'VLLM_DP_MASTER_IP': lambda: os.getenv('VLLM_DP_MASTER_IP', '127.0.0.1'), 'VLLM_DP_MASTER_PORT': lambda: int(os.getenv('VLLM_DP_MASTER_PORT', '0')), 'VLLM_MOE_DP_CHUNK_SIZE': lambda: int(os.getenv('VLLM_MOE_DP_CHUNK_SIZE', '256')), 'VLLM_RANDOMIZE_DP_DUMMY_INPUTS': lambda: os.environ.get('VLLM_RANDOMIZE_DP_DUMMY_INPUTS', '0') == '1', 'VLLM_CI_USE_S3': lambda: os.environ.get('VLLM_CI_USE_S3', '0') == '1', 'VLLM_MODEL_REDIRECT_PATH': lambda: os.environ.get('VLLM_MODEL_REDIRECT_PATH', None), 'VLLM_MARLIN_USE_ATOMIC_ADD': lambda: os.environ.get('VLLM_MARLIN_USE_ATOMIC_ADD', '0') == '1', 'VLLM_V0_USE_OUTLINES_CACHE': lambda: os.environ.get('VLLM_V0_USE_OUTLINES_CACHE', '0') == '1', 'VLLM_V1_USE_OUTLINES_CACHE': lambda: os.environ.get('VLLM_V1_USE_OUTLINES_CACHE', '0') == '1', 'VLLM_TPU_BUCKET_PADDING_GAP': lambda: int(os.environ['VLLM_TPU_BUCKET_PADDING_GAP']) if 'VLLM_TPU_BUCKET_PADDING_GAP' in os.environ else 0, 'VLLM_TPU_MOST_MODEL_LEN': lambda: maybe_convert_int(os.environ.get('VLLM_TPU_MOST_MODEL_LEN', None)), 'VLLM_USE_DEEP_GEMM': lambda: bool(int(os.getenv('VLLM_USE_DEEP_GEMM', '0'))), 'VLLM_USE_FLASHINFER_MOE_FP8': lambda: bool(int(os.getenv('VLLM_USE_FLASHINFER_MOE_FP8', '0'))), 'VLLM_USE_FLASHINFER_MOE_FP4': lambda: bool(int(os.getenv('VLLM_USE_FLASHINFER_MOE_FP4', '0'))), 'VLLM_XGRAMMAR_CACHE_MB': lambda: int(os.getenv('VLLM_XGRAMMAR_CACHE_MB', '512')), 'VLLM_MSGPACK_ZERO_COPY_THRESHOLD': lambda: int(os.getenv('VLLM_MSGPACK_ZERO_COPY_THRESHOLD', '256')), 'VLLM_ALLOW_INSECURE_SERIALIZATION': lambda: bool(int(os.getenv('VLLM_ALLOW_INSECURE_SERIALIZATION', '0'))), 'VLLM_NIXL_SIDE_CHANNEL_HOST': lambda: os.getenv('VLLM_NIXL_SIDE_CHANNEL_HOST', 'localhost'), 'VLLM_NIXL_SIDE_CHANNEL_PORT': lambda: int(os.getenv('VLLM_NIXL_SIDE_CHANNEL_PORT', '5557')), 'VLLM_ALL2ALL_BACKEND': lambda: os.getenv('VLLM_ALL2ALL_BACKEND', 'naive'), 'VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE': lambda: int(os.getenv('VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE', '163840')), 'VLLM_TOOL_PARSE_REGEX_TIMEOUT_SECONDS': lambda: int(os.getenv('VLLM_TOOL_PARSE_REGEX_TIMEOUT_SECONDS', '1')), 'VLLM_SLEEP_WHEN_IDLE': lambda: bool(int(os.getenv('VLLM_SLEEP_WHEN_IDLE', '0'))), 'VLLM_MQ_MAX_CHUNK_BYTES_MB': lambda: int(os.getenv('VLLM_MQ_MAX_CHUNK_BYTES_MB', '16')), 'VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS': lambda: int(os.getenv('VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS', '300')), 'VLLM_KV_CACHE_LAYOUT': lambda: os.getenv('VLLM_KV_CACHE_LAYOUT', None), 'VLLM_COMPUTE_NANS_IN_LOGITS': lambda: bool(int(os.getenv('VLLM_COMPUTE_NANS_IN_LOGITS', '0'))), 'VLLM_USE_NVFP4_CT_EMULATIONS': lambda: bool(int(os.getenv('VLLM_USE_NVFP4_CT_EMULATIONS', '0'))), 'VLLM_NIXL_ABORT_REQUEST_TIMEOUT': lambda: int(os.getenv('VLLM_NIXL_ABORT_REQUEST_TIMEOUT', '120')), 'VLLM_USE_CUDNN_PREFILL': lambda: bool(int(os.getenv('VLLM_USE_CUDNN_PREFILL', '0'))), 'VLLM_USE_TRTLLM_DECODE_ATTENTION': lambda: os.getenv('VLLM_USE_TRTLLM_DECODE_ATTENTION', None), 'VLLM_LOOPBACK_IP': lambda: os.getenv('VLLM_LOOPBACK_IP', '')}

def __getattr__(name: str):
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

def __dir__():
    return list(environment_variables.keys())

def is_set(name: str):
    """Check if an environment variable is explicitly set."""
    if name in environment_variables:
        return name in os.environ
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

def set_vllm_use_v1(use_v1: bool):
    if is_set('VLLM_USE_V1'):
        raise ValueError('Should not call set_vllm_use_v1() if VLLM_USE_V1 is set explicitly by the user. Please raise this as a Github Issue and explicitly set VLLM_USE_V1=0 or 1.')
    os.environ['VLLM_USE_V1'] = '1' if use_v1 else '0'

def compute_hash() -> str:
    """
    WARNING: Whenever a new key is added to this environment
    variables, ensure that it is included in the factors list if
    it affects the computation graph. For example, different values
    of VLLM_PP_LAYER_PARTITION will generate different computation
    graphs, so it is included in the factors list. The env vars that
    affect the choice of different kernels or attention backends should
    also be included in the factors list.
    """
    factors: list[Any] = []

    def factorize(name: str):
        if __getattr__(name):
            factors.append(__getattr__(name))
        else:
            factors.append('None')
    environment_variables_to_hash = ['VLLM_PP_LAYER_PARTITION', 'VLLM_MLA_DISABLE', 'VLLM_USE_TRITON_FLASH_ATTN', 'VLLM_USE_TRITON_AWQ', 'VLLM_DP_RANK', 'VLLM_DP_SIZE', 'VLLM_USE_STANDALONE_COMPILE', 'VLLM_FUSED_MOE_CHUNK_SIZE']
    for key in environment_variables_to_hash:
        if key in environment_variables:
            factorize(key)
    hash_str = hashlib.md5(str(factors).encode(), usedforsecurity=False).hexdigest()
    return hash_str