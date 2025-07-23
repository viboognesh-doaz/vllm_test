from collections import namedtuple
from platform import machine
from vllm import __version__, __version_tuple__
from vllm.envs import environment_variables
import datetime
import importlib.util
import locale
import os
import platform
import platform
import regex as re
import subprocess
import sys
import torch
import torch.backends.xnnpack
try:
    TORCH_AVAILABLE = True
except (ImportError, NameError, AttributeError, OSError):
    TORCH_AVAILABLE = False
SystemEnv = namedtuple('SystemEnv', ['torch_version', 'is_debug_build', 'cuda_compiled_version', 'gcc_version', 'clang_version', 'cmake_version', 'os', 'libc_version', 'python_version', 'python_platform', 'is_cuda_available', 'cuda_runtime_version', 'cuda_module_loading', 'nvidia_driver_version', 'nvidia_gpu_models', 'cudnn_version', 'pip_version', 'pip_packages', 'conda_packages', 'hip_compiled_version', 'hip_runtime_version', 'miopen_runtime_version', 'caching_allocator_config', 'is_xnnpack_available', 'cpu_info', 'rocm_version', 'neuron_sdk_version', 'vllm_version', 'vllm_build_flags', 'gpu_topo', 'env_vars'])
DEFAULT_CONDA_PATTERNS = {'torch', 'numpy', 'cudatoolkit', 'soumith', 'mkl', 'magma', 'triton', 'optree', 'nccl', 'transformers', 'zmq', 'nvidia', 'pynvml'}
DEFAULT_PIP_PATTERNS = {'torch', 'numpy', 'mypy', 'flake8', 'triton', 'optree', 'onnx', 'nccl', 'transformers', 'zmq', 'nvidia', 'pynvml'}

def run(command):
    """Return (return-code, stdout, stderr)."""
    shell = True if type(command) is str else False
    try:
        p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=shell)
        raw_output, raw_err = p.communicate()
        rc = p.returncode
        if get_platform() == 'win32':
            enc = 'oem'
        else:
            enc = locale.getpreferredencoding()
        output = raw_output.decode(enc)
        if command == 'nvidia-smi topo -m':
            output = output.rstrip()
        else:
            output = output.strip()
        err = raw_err.decode(enc)
        return (rc, output, err.strip())
    except FileNotFoundError:
        cmd_str = command if isinstance(command, str) else command[0]
        return (127, '', f'Command not found: {cmd_str}')

def run_and_read_all(run_lambda, command):
    """Run command using run_lambda; reads and returns entire output if rc is 0."""
    rc, out, _ = run_lambda(command)
    if rc != 0:
        return None
    return out

def run_and_parse_first_match(run_lambda, command, regex):
    """Run command using run_lambda, returns the first regex match if it exists."""
    rc, out, _ = run_lambda(command)
    if rc != 0:
        return None
    match = re.search(regex, out)
    if match is None:
        return None
    return match.group(1)

def run_and_return_first_line(run_lambda, command):
    """Run command using run_lambda and returns first line if output is not empty."""
    rc, out, _ = run_lambda(command)
    if rc != 0:
        return None
    return out.split('\n')[0]

def get_conda_packages(run_lambda, patterns=None):
    if patterns is None:
        patterns = DEFAULT_CONDA_PATTERNS
    conda = os.environ.get('CONDA_EXE', 'conda')
    out = run_and_read_all(run_lambda, [conda, 'list'])
    if out is None:
        return out
    return '\n'.join((line for line in out.splitlines() if not line.startswith('#') and any((name in line for name in patterns))))

def get_gcc_version(run_lambda):
    return run_and_parse_first_match(run_lambda, 'gcc --version', 'gcc (.*)')

def get_clang_version(run_lambda):
    return run_and_parse_first_match(run_lambda, 'clang --version', 'clang version (.*)')

def get_cmake_version(run_lambda):
    return run_and_parse_first_match(run_lambda, 'cmake --version', 'cmake (.*)')

def get_nvidia_driver_version(run_lambda):
    if get_platform() == 'darwin':
        cmd = 'kextstat | grep -i cuda'
        return run_and_parse_first_match(run_lambda, cmd, 'com[.]nvidia[.]CUDA [(](.*?)[)]')
    smi = get_nvidia_smi()
    return run_and_parse_first_match(run_lambda, smi, 'Driver Version: (.*?) ')

def get_gpu_info(run_lambda):
    if get_platform() == 'darwin' or (TORCH_AVAILABLE and hasattr(torch.version, 'hip') and (torch.version.hip is not None)):
        if TORCH_AVAILABLE and torch.cuda.is_available():
            if torch.version.hip is not None:
                prop = torch.cuda.get_device_properties(0)
                if hasattr(prop, 'gcnArchName'):
                    gcnArch = ' ({})'.format(prop.gcnArchName)
                else:
                    gcnArch = 'NoGCNArchNameOnOldPyTorch'
            else:
                gcnArch = ''
            return torch.cuda.get_device_name(None) + gcnArch
        return None
    smi = get_nvidia_smi()
    uuid_regex = re.compile(' \\(UUID: .+?\\)')
    rc, out, _ = run_lambda(smi + ' -L')
    if rc != 0:
        return None
    return re.sub(uuid_regex, '', out)

def get_running_cuda_version(run_lambda):
    return run_and_parse_first_match(run_lambda, 'nvcc --version', 'release .+ V(.*)')

def get_cudnn_version(run_lambda):
    """Return a list of libcudnn.so; it's hard to tell which one is being used."""
    if get_platform() == 'win32':
        system_root = os.environ.get('SYSTEMROOT', 'C:\\Windows')
        cuda_path = os.environ.get('CUDA_PATH', '%CUDA_PATH%')
        where_cmd = os.path.join(system_root, 'System32', 'where')
        cudnn_cmd = '{} /R "{}\\bin" cudnn*.dll'.format(where_cmd, cuda_path)
    elif get_platform() == 'darwin':
        cudnn_cmd = 'ls /usr/local/cuda/lib/libcudnn*'
    else:
        cudnn_cmd = 'ldconfig -p | grep libcudnn | rev | cut -d" " -f1 | rev'
    rc, out, _ = run_lambda(cudnn_cmd)
    if len(out) == 0 or (rc != 1 and rc != 0):
        l = os.environ.get('CUDNN_LIBRARY')
        if l is not None and os.path.isfile(l):
            return os.path.realpath(l)
        return None
    files_set = set()
    for fn in out.split('\n'):
        fn = os.path.realpath(fn)
        if os.path.isfile(fn):
            files_set.add(fn)
    if not files_set:
        return None
    files = sorted(files_set)
    if len(files) == 1:
        return files[0]
    result = '\n'.join(files)
    return 'Probably one of the following:\n{}'.format(result)

def get_nvidia_smi():
    smi = 'nvidia-smi'
    if get_platform() == 'win32':
        system_root = os.environ.get('SYSTEMROOT', 'C:\\Windows')
        program_files_root = os.environ.get('PROGRAMFILES', 'C:\\Program Files')
        legacy_path = os.path.join(program_files_root, 'NVIDIA Corporation', 'NVSMI', smi)
        new_path = os.path.join(system_root, 'System32', smi)
        smis = [new_path, legacy_path]
        for candidate_smi in smis:
            if os.path.exists(candidate_smi):
                smi = '"{}"'.format(candidate_smi)
                break
    return smi

def get_rocm_version(run_lambda):
    """Returns the ROCm version if available, otherwise 'N/A'."""
    return run_and_parse_first_match(run_lambda, 'hipcc --version', 'HIP version: (\\S+)')

def get_neuron_sdk_version(run_lambda):
    try:
        result = run_lambda(['neuron-ls'])
        return result if result[0] == 0 else 'N/A'
    except Exception:
        return 'N/A'

def get_vllm_version():
    if __version__ == 'dev':
        return 'N/A (dev)'
    version_str = __version_tuple__[-1]
    if isinstance(version_str, str) and version_str.startswith('g'):
        if '.' in version_str:
            git_sha = version_str.split('.')[0][1:]
            date = version_str.split('.')[-1][1:]
            return f'{__version__} (git sha: {git_sha}, date: {date})'
        else:
            git_sha = version_str[1:]
            return f'{__version__} (git sha: {git_sha})'
    return __version__

def summarize_vllm_build_flags():
    return 'CUDA Archs: {}; ROCm: {}; Neuron: {}'.format(os.environ.get('TORCH_CUDA_ARCH_LIST', 'Not Set'), 'Enabled' if os.environ.get('ROCM_HOME') else 'Disabled', 'Enabled' if os.environ.get('NEURON_CORES') else 'Disabled')

def get_gpu_topo(run_lambda):
    output = None
    if get_platform() == 'linux':
        output = run_and_read_all(run_lambda, 'nvidia-smi topo -m')
        if output is None:
            output = run_and_read_all(run_lambda, 'rocm-smi --showtopo')
    return output

def get_cpu_info(run_lambda):
    rc, out, err = (0, '', '')
    if get_platform() == 'linux':
        rc, out, err = run_lambda('lscpu')
    elif get_platform() == 'win32':
        rc, out, err = run_lambda('wmic cpu get Name,Manufacturer,Family,Architecture,ProcessorType,DeviceID,         CurrentClockSpeed,MaxClockSpeed,L2CacheSize,L2CacheSpeed,Revision /VALUE')
    elif get_platform() == 'darwin':
        rc, out, err = run_lambda('sysctl -n machdep.cpu.brand_string')
    cpu_info = 'None'
    if rc == 0:
        cpu_info = out
    else:
        cpu_info = err
    return cpu_info

def get_platform():
    if sys.platform.startswith('linux'):
        return 'linux'
    elif sys.platform.startswith('win32'):
        return 'win32'
    elif sys.platform.startswith('cygwin'):
        return 'cygwin'
    elif sys.platform.startswith('darwin'):
        return 'darwin'
    else:
        return sys.platform

def get_mac_version(run_lambda):
    return run_and_parse_first_match(run_lambda, 'sw_vers -productVersion', '(.*)')

def get_windows_version(run_lambda):
    system_root = os.environ.get('SYSTEMROOT', 'C:\\Windows')
    wmic_cmd = os.path.join(system_root, 'System32', 'Wbem', 'wmic')
    findstr_cmd = os.path.join(system_root, 'System32', 'findstr')
    return run_and_read_all(run_lambda, '{} os get Caption | {} /v Caption'.format(wmic_cmd, findstr_cmd))

def get_lsb_version(run_lambda):
    return run_and_parse_first_match(run_lambda, 'lsb_release -a', 'Description:\\t(.*)')

def check_release_file(run_lambda):
    return run_and_parse_first_match(run_lambda, 'cat /etc/*-release', 'PRETTY_NAME="(.*)"')

def get_os(run_lambda):
    platform = get_platform()
    if platform == 'win32' or platform == 'cygwin':
        return get_windows_version(run_lambda)
    if platform == 'darwin':
        version = get_mac_version(run_lambda)
        if version is None:
            return None
        return 'macOS {} ({})'.format(version, machine())
    if platform == 'linux':
        desc = get_lsb_version(run_lambda)
        if desc is not None:
            return '{} ({})'.format(desc, machine())
        desc = check_release_file(run_lambda)
        if desc is not None:
            return '{} ({})'.format(desc, machine())
        return '{} ({})'.format(platform, machine())
    return platform

def get_python_platform():
    return platform.platform()

def get_libc_version():
    if get_platform() != 'linux':
        return 'N/A'
    return '-'.join(platform.libc_ver())

def get_pip_packages(run_lambda, patterns=None):
    """Return `pip list` output. Note: will also find conda-installed pytorch and numpy packages."""
    if patterns is None:
        patterns = DEFAULT_PIP_PATTERNS

    def run_with_pip():
        try:
            pip_spec = importlib.util.find_spec('pip')
            pip_available = pip_spec is not None
        except ImportError:
            pip_available = False
        if pip_available:
            cmd = [sys.executable, '-mpip', 'list', '--format=freeze']
        elif os.environ.get('UV') is not None:
            print('uv is set')
            cmd = ['uv', 'pip', 'list', '--format=freeze']
        else:
            raise RuntimeError('Could not collect pip list output (pip or uv module not available)')
        out = run_and_read_all(run_lambda, cmd)
        return '\n'.join((line for line in out.splitlines() if any((name in line for name in patterns))))
    pip_version = 'pip3' if sys.version[0] == '3' else 'pip'
    out = run_with_pip()
    return (pip_version, out)

def get_cachingallocator_config():
    ca_config = os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '')
    return ca_config

def get_cuda_module_loading_config():
    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.init()
        config = os.environ.get('CUDA_MODULE_LOADING', '')
        return config
    else:
        return 'N/A'

def is_xnnpack_available():
    if TORCH_AVAILABLE:
        return str(torch.backends.xnnpack.enabled)
    else:
        return 'N/A'

def get_env_vars():
    env_vars = ''
    secret_terms = ('secret', 'token', 'api', 'access', 'password')
    report_prefix = ('TORCH', 'NCCL', 'PYTORCH', 'CUDA', 'CUBLAS', 'CUDNN', 'OMP_', 'MKL_', 'NVIDIA')
    for k, v in os.environ.items():
        if any((term in k.lower() for term in secret_terms)):
            continue
        if k in environment_variables:
            env_vars = env_vars + '{}={}'.format(k, v) + '\n'
        if k.startswith(report_prefix):
            env_vars = env_vars + '{}={}'.format(k, v) + '\n'
    return env_vars

def get_env_info():
    run_lambda = run
    pip_version, pip_list_output = get_pip_packages(run_lambda)
    if TORCH_AVAILABLE:
        version_str = torch.__version__
        debug_mode_str = str(torch.version.debug)
        cuda_available_str = str(torch.cuda.is_available())
        cuda_version_str = torch.version.cuda
        if not hasattr(torch.version, 'hip') or torch.version.hip is None:
            hip_compiled_version = hip_runtime_version = miopen_runtime_version = 'N/A'
        else:

            def get_version_or_na(cfg, prefix):
                _lst = [s.rsplit(None, 1)[-1] for s in cfg if prefix in s]
                return _lst[0] if _lst else 'N/A'
            cfg = torch._C._show_config().split('\n')
            hip_runtime_version = get_version_or_na(cfg, 'HIP Runtime')
            miopen_runtime_version = get_version_or_na(cfg, 'MIOpen')
            cuda_version_str = 'N/A'
            hip_compiled_version = torch.version.hip
    else:
        version_str = debug_mode_str = cuda_available_str = cuda_version_str = 'N/A'
        hip_compiled_version = hip_runtime_version = miopen_runtime_version = 'N/A'
    sys_version = sys.version.replace('\n', ' ')
    conda_packages = get_conda_packages(run_lambda)
    rocm_version = get_rocm_version(run_lambda)
    neuron_sdk_version = get_neuron_sdk_version(run_lambda)
    vllm_version = get_vllm_version()
    vllm_build_flags = summarize_vllm_build_flags()
    gpu_topo = get_gpu_topo(run_lambda)
    return SystemEnv(torch_version=version_str, is_debug_build=debug_mode_str, python_version='{} ({}-bit runtime)'.format(sys_version, sys.maxsize.bit_length() + 1), python_platform=get_python_platform(), is_cuda_available=cuda_available_str, cuda_compiled_version=cuda_version_str, cuda_runtime_version=get_running_cuda_version(run_lambda), cuda_module_loading=get_cuda_module_loading_config(), nvidia_gpu_models=get_gpu_info(run_lambda), nvidia_driver_version=get_nvidia_driver_version(run_lambda), cudnn_version=get_cudnn_version(run_lambda), hip_compiled_version=hip_compiled_version, hip_runtime_version=hip_runtime_version, miopen_runtime_version=miopen_runtime_version, pip_version=pip_version, pip_packages=pip_list_output, conda_packages=conda_packages, os=get_os(run_lambda), libc_version=get_libc_version(), gcc_version=get_gcc_version(run_lambda), clang_version=get_clang_version(run_lambda), cmake_version=get_cmake_version(run_lambda), caching_allocator_config=get_cachingallocator_config(), is_xnnpack_available=is_xnnpack_available(), cpu_info=get_cpu_info(run_lambda), rocm_version=rocm_version, neuron_sdk_version=neuron_sdk_version, vllm_version=vllm_version, vllm_build_flags=vllm_build_flags, gpu_topo=gpu_topo, env_vars=get_env_vars())
env_info_fmt = '\n==============================\n        System Info\n==============================\nOS                           : {os}\nGCC version                  : {gcc_version}\nClang version                : {clang_version}\nCMake version                : {cmake_version}\nLibc version                 : {libc_version}\n\n==============================\n       PyTorch Info\n==============================\nPyTorch version              : {torch_version}\nIs debug build               : {is_debug_build}\nCUDA used to build PyTorch   : {cuda_compiled_version}\nROCM used to build PyTorch   : {hip_compiled_version}\n\n==============================\n      Python Environment\n==============================\nPython version               : {python_version}\nPython platform              : {python_platform}\n\n==============================\n       CUDA / GPU Info\n==============================\nIs CUDA available            : {is_cuda_available}\nCUDA runtime version         : {cuda_runtime_version}\nCUDA_MODULE_LOADING set to   : {cuda_module_loading}\nGPU models and configuration : {nvidia_gpu_models}\nNvidia driver version        : {nvidia_driver_version}\ncuDNN version                : {cudnn_version}\nHIP runtime version          : {hip_runtime_version}\nMIOpen runtime version       : {miopen_runtime_version}\nIs XNNPACK available         : {is_xnnpack_available}\n\n==============================\n          CPU Info\n==============================\n{cpu_info}\n\n==============================\nVersions of relevant libraries\n==============================\n{pip_packages}\n{conda_packages}\n'.strip()
env_info_fmt += '\n\n'
env_info_fmt += '\n==============================\n         vLLM Info\n==============================\nROCM Version                 : {rocm_version}\nNeuron SDK Version           : {neuron_sdk_version}\nvLLM Version                 : {vllm_version}\nvLLM Build Flags:\n  {vllm_build_flags}\nGPU Topology:\n  {gpu_topo}\n\n==============================\n     Environment Variables\n==============================\n{env_vars}\n'.strip()

def pretty_str(envinfo):

    def replace_nones(dct, replacement='Could not collect'):
        for key in dct.keys():
            if dct[key] is not None:
                continue
            dct[key] = replacement
        return dct

    def replace_bools(dct, true='Yes', false='No'):
        for key in dct.keys():
            if dct[key] is True:
                dct[key] = true
            elif dct[key] is False:
                dct[key] = false
        return dct

    def prepend(text, tag='[prepend]'):
        lines = text.split('\n')
        updated_lines = [tag + line for line in lines]
        return '\n'.join(updated_lines)

    def replace_if_empty(text, replacement='No relevant packages'):
        if text is not None and len(text) == 0:
            return replacement
        return text

    def maybe_start_on_next_line(string):
        if string is not None and len(string.split('\n')) > 1:
            return '\n{}\n'.format(string)
        return string
    mutable_dict = envinfo._asdict()
    mutable_dict['nvidia_gpu_models'] = maybe_start_on_next_line(envinfo.nvidia_gpu_models)
    dynamic_cuda_fields = ['cuda_runtime_version', 'nvidia_gpu_models', 'nvidia_driver_version']
    all_cuda_fields = dynamic_cuda_fields + ['cudnn_version']
    all_dynamic_cuda_fields_missing = all((mutable_dict[field] is None for field in dynamic_cuda_fields))
    if TORCH_AVAILABLE and (not torch.cuda.is_available()) and all_dynamic_cuda_fields_missing:
        for field in all_cuda_fields:
            mutable_dict[field] = 'No CUDA'
        if envinfo.cuda_compiled_version is None:
            mutable_dict['cuda_compiled_version'] = 'None'
    mutable_dict = replace_bools(mutable_dict)
    mutable_dict = replace_nones(mutable_dict)
    mutable_dict['pip_packages'] = replace_if_empty(mutable_dict['pip_packages'])
    mutable_dict['conda_packages'] = replace_if_empty(mutable_dict['conda_packages'])
    if mutable_dict['pip_packages']:
        mutable_dict['pip_packages'] = prepend(mutable_dict['pip_packages'], '[{}] '.format(envinfo.pip_version))
    if mutable_dict['conda_packages']:
        mutable_dict['conda_packages'] = prepend(mutable_dict['conda_packages'], '[conda] ')
    mutable_dict['cpu_info'] = envinfo.cpu_info
    return env_info_fmt.format(**mutable_dict)

def get_pretty_env_info():
    return pretty_str(get_env_info())

def main():
    print('Collecting environment information...')
    output = get_pretty_env_info()
    print(output)
    if TORCH_AVAILABLE and hasattr(torch, 'utils') and hasattr(torch.utils, '_crash_handler'):
        minidump_dir = torch.utils._crash_handler.DEFAULT_MINIDUMP_DIR
        if sys.platform == 'linux' and os.path.exists(minidump_dir):
            dumps = [os.path.join(minidump_dir, dump) for dump in os.listdir(minidump_dir)]
            latest = max(dumps, key=os.path.getctime)
            ctime = os.path.getctime(latest)
            creation_time = datetime.datetime.fromtimestamp(ctime).strftime('%Y-%m-%d %H:%M:%S')
            msg = '\n*** Detected a minidump at {} created on {}, '.format(latest, creation_time) + 'if this is related to your bug please include it when you file a report ***'
            print(msg, file=sys.stderr)
if __name__ == '__main__':
    main()