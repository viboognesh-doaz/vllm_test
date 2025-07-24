from importlib.metadata import PackageNotFoundError, version
from vllm.utils import import_pynvml
import amdsmi
import intel_extension_for_pytorch
import libtpu
import neuronx_distributed_inference
import os
import torch
import transformers_neuronx
import logging
import traceback
from itertools import chain
from typing import TYPE_CHECKING, Optional
from vllm.plugins import load_plugins_by_group
from vllm.utils import resolve_obj_by_qualname, supports_xccl
from .interface import _Backend
from .interface import CpuArchEnum, Platform, PlatformEnum
logger = logging.getLogger(__name__)

def vllm_version_matches_substr(substr: str) -> bool:
    """
    Check to see if the vLLM version matches a substring.
    """
    try:
        vllm_version = version('vllm')
    except PackageNotFoundError as e:
        logger.warning('The vLLM package was not found, so its version could not be inspected. This may cause platform detection to fail.')
        raise e
    return substr in vllm_version

def tpu_platform_plugin() -> Optional[str]:
    is_tpu = False
    logger.debug('Checking if TPU platform is available.')
    try:
        is_tpu = True
        logger.debug('Confirmed TPU platform is available.')
    except Exception as e:
        logger.debug('TPU platform is not available because: %s', str(e))
    return 'vllm.platforms.tpu.TpuPlatform' if is_tpu else None

def cuda_platform_plugin() -> Optional[str]:
    is_cuda = False
    logger.debug('Checking if CUDA platform is available.')
    try:
        pynvml = import_pynvml()
        pynvml.nvmlInit()
        try:
            is_cuda = pynvml.nvmlDeviceGetCount() > 0 and (not vllm_version_matches_substr('cpu'))
            if pynvml.nvmlDeviceGetCount() <= 0:
                logger.debug('CUDA platform is not available because no GPU is found.')
            if vllm_version_matches_substr('cpu'):
                logger.debug('CUDA platform is not available because vLLM is built with CPU.')
            if is_cuda:
                logger.debug('Confirmed CUDA platform is available.')
        finally:
            pynvml.nvmlShutdown()
    except Exception as e:
        logger.debug('Exception happens when checking CUDA platform: %s', str(e))
        if 'nvml' not in e.__class__.__name__.lower():
            raise e

        def cuda_is_jetson() -> bool:
            return os.path.isfile('/etc/nv_tegra_release') or os.path.exists('/sys/class/tegra-firmware')
        if cuda_is_jetson():
            logger.debug('Confirmed CUDA platform is available on Jetson.')
            is_cuda = True
        else:
            logger.debug('CUDA platform is not available because: %s', str(e))
    return 'vllm.platforms.cuda.CudaPlatform' if is_cuda else None

def rocm_platform_plugin() -> Optional[str]:
    is_rocm = False
    logger.debug('Checking if ROCm platform is available.')
    try:
        amdsmi.amdsmi_init()
        try:
            if len(amdsmi.amdsmi_get_processor_handles()) > 0:
                is_rocm = True
                logger.debug('Confirmed ROCm platform is available.')
            else:
                logger.debug('ROCm platform is not available because no GPU is found.')
        finally:
            amdsmi.amdsmi_shut_down()
    except Exception as e:
        logger.debug('ROCm platform is not available because: %s', str(e))
    return 'vllm.platforms.rocm.RocmPlatform' if is_rocm else None

def xpu_platform_plugin() -> Optional[str]:
    is_xpu = False
    logger.debug('Checking if XPU platform is available.')
    try:
        if supports_xccl():
            dist_backend = 'xccl'
        else:
            dist_backend = 'ccl'
            import oneccl_bindings_for_pytorch
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            is_xpu = True
            from vllm.platforms.xpu import XPUPlatform
            XPUPlatform.dist_backend = dist_backend
            logger.debug('Confirmed %s backend is available.', XPUPlatform.dist_backend)
            logger.debug('Confirmed XPU platform is available.')
    except Exception as e:
        logger.debug('XPU platform is not available because: %s', str(e))
    return 'vllm.platforms.xpu.XPUPlatform' if is_xpu else None

def cpu_platform_plugin() -> Optional[str]:
    is_cpu = False
    logger.debug('Checking if CPU platform is available.')
    try:
        is_cpu = vllm_version_matches_substr('cpu')
        if is_cpu:
            logger.debug('Confirmed CPU platform is available because vLLM is built with CPU.')
        if not is_cpu:
            import sys
            is_cpu = sys.platform.startswith('darwin')
            if is_cpu:
                logger.debug('Confirmed CPU platform is available because the machine is MacOS.')
    except Exception as e:
        logger.debug('CPU platform is not available because: %s', str(e))
    return 'vllm.platforms.cpu.CpuPlatform' if is_cpu else None

def neuron_platform_plugin() -> Optional[str]:
    tnx_installed = False
    nxd_installed = False
    logger.debug('Checking if Neuron platform is available.')
    try:
        tnx_installed = True
        logger.debug('Confirmed Neuron platform is available because transformers_neuronx is found.')
    except ImportError:
        pass
    try:
        nxd_installed = True
        logger.debug('Confirmed Neuron platform is available because neuronx_distributed_inference is found.')
    except ImportError:
        pass
    is_neuron = tnx_installed or nxd_installed
    return 'vllm.platforms.neuron.NeuronPlatform' if is_neuron else None
builtin_platform_plugins = {'tpu': tpu_platform_plugin, 'cuda': cuda_platform_plugin, 'rocm': rocm_platform_plugin, 'xpu': xpu_platform_plugin, 'cpu': cpu_platform_plugin, 'neuron': neuron_platform_plugin}

def resolve_current_platform_cls_qualname() -> str:
    platform_plugins = load_plugins_by_group('vllm.platform_plugins')
    activated_plugins = []
    for name, func in chain(builtin_platform_plugins.items(), platform_plugins.items()):
        try:
            assert callable(func)
            platform_cls_qualname = func()
            if platform_cls_qualname is not None:
                activated_plugins.append(name)
        except Exception:
            pass
    activated_builtin_plugins = list(set(activated_plugins) & set(builtin_platform_plugins.keys()))
    activated_oot_plugins = list(set(activated_plugins) & set(platform_plugins.keys()))
    if len(activated_oot_plugins) >= 2:
        raise RuntimeError(f'Only one platform plugin can be activated, but got: {activated_oot_plugins}')
    elif len(activated_oot_plugins) == 1:
        platform_cls_qualname = platform_plugins[activated_oot_plugins[0]]()
        logger.info('Platform plugin %s is activated', activated_oot_plugins[0])
    elif len(activated_builtin_plugins) >= 2:
        raise RuntimeError(f'Only one platform plugin can be activated, but got: {activated_builtin_plugins}')
    elif len(activated_builtin_plugins) == 1:
        platform_cls_qualname = builtin_platform_plugins[activated_builtin_plugins[0]]()
        logger.info('Automatically detected platform %s.', activated_builtin_plugins[0])
    else:
        platform_cls_qualname = 'vllm.platforms.interface.UnspecifiedPlatform'
        logger.info('No platform detected, vLLM is running on UnspecifiedPlatform')
    return platform_cls_qualname
_current_platform = None
_init_trace: str = ''
if TYPE_CHECKING:
    current_platform: Platform

def __getattr__(name: str):
    if name == 'current_platform':
        global _current_platform
        if _current_platform is None:
            platform_cls_qualname = resolve_current_platform_cls_qualname()
            _current_platform = resolve_obj_by_qualname(platform_cls_qualname)()
            global _init_trace
            _init_trace = ''.join(traceback.format_stack())
        return _current_platform
    elif name in globals():
        return globals()[name]
    else:
        raise AttributeError(f"No attribute named '{name}' exists in {__name__}.")
__all__ = ['Platform', 'PlatformEnum', 'current_platform', 'CpuArchEnum', '_init_trace']