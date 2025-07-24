from torch._dynamo.backends.registry import list_backends
from torch.distributed import DistNetworkError
from vllm import __version__
from vllm.compilation.backends import VllmBackend
from vllm.compilation.counter import compilation_counter
from vllm.distributed.utils import get_pp_indices
from vllm.distributed.utils import stateless_init_torch_distributed_process_group
from vllm.executor.executor_base import ExecutorBase
from vllm.model_executor.models.config import MODELS_CONFIG_MAP, HybridAttentionMambaModelConfig
from vllm.platforms import current_platform
from vllm.tracing import is_otel_available, otel_import_error_traceback
import copy
import ast
import enum
import hashlib
import inspect
import json
import textwrap
import uuid
import warnings
from collections import Counter
from contextlib import contextmanager
from dataclasses import MISSING, Field, asdict, field, fields, is_dataclass, replace
from functools import cached_property
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Literal, Optional, Protocol, TypeVar, Union, cast, get_args
import regex as re
import torch
from pydantic import ConfigDict, SkipValidation, TypeAdapter, field_validator, model_validator
from pydantic.dataclasses import dataclass
from safetensors.torch import _TYPES as _SAFETENSORS_TO_TORCH_DTYPE
from torch.distributed import ProcessGroup, ReduceOp
from typing_extensions import Self, runtime_checkable
import vllm.envs as envs
from vllm import version
from vllm.compilation.inductor_pass import CallableInductorPass, InductorPass
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.transformers_utils.config import ConfigFormat, get_config, get_hf_image_processor_config, get_hf_text_config, get_pooling_config, get_sentence_transformer_tokenizer_config, is_encoder_decoder, try_get_generation_config, try_get_safetensors_metadata, try_get_tokenizer_config, uses_mrope
from vllm.transformers_utils.s3_utils import S3Model
from vllm.transformers_utils.utils import is_s3, maybe_model_redirect
from vllm.utils import DEFAULT_MAX_NUM_BATCHED_TOKENS, MULTIMODAL_MODEL_MAX_NUM_BATCHED_TOKENS, POOLING_MODEL_MAX_NUM_BATCHED_TOKENS, GiB_bytes, LayerBlockType, LazyLoader, common_broadcastable_dtype, cuda_device_count_stateless, get_cpu_memory, get_open_port, is_torch_equal_or_newer, random_uuid, resolve_obj_by_qualname
if TYPE_CHECKING:
    from _typeshed import DataclassInstance
    from ray.util.placement_group import PlacementGroup
    from transformers.configuration_utils import PretrainedConfig
    import vllm.model_executor.layers.quantization as me_quant
    import vllm.model_executor.models as me_models
    from vllm.executor.executor_base import ExecutorBase
    from vllm.model_executor.layers.quantization import QuantizationMethods
    from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
    from vllm.model_executor.model_loader import BaseModelLoader
    from vllm.model_executor.model_loader.tensorizer import TensorizerConfig
    ConfigType = type[DataclassInstance]
    HfOverrides = Union[dict, Callable[[type], type]]
else:
    DataclassInstance = Any
    PlacementGroup = Any
    PretrainedConfig = Any
    ExecutorBase = Any
    QuantizationConfig = Any
    QuantizationMethods = Any
    BaseModelLoader = Any
    TensorizerConfig = Any
    ConfigType = type
    HfOverrides = Union[dict[str, Any], Callable[[type], type]]
    me_quant = LazyLoader('model_executor', globals(), 'vllm.model_executor.layers.quantization')
    me_models = LazyLoader('model_executor', globals(), 'vllm.model_executor.models')
logger = init_logger(__name__)
DataclassInstanceT = TypeVar('DataclassInstanceT', bound=DataclassInstance)
ConfigT = TypeVar('ConfigT', bound=ConfigType)
TaskOption = Literal['auto', 'generate', 'embedding', 'embed', 'classify', 'score', 'reward', 'transcription', 'draft']
_ResolvedTask = Literal['generate', 'transcription', 'pooling', 'embed', 'classify', 'reward', 'draft']
RunnerOption = Literal['auto', 'generate', 'pooling', 'draft']
RunnerType = Literal['generate', 'pooling', 'draft']
_RUNNER_TASKS: dict[RunnerType, list[_ResolvedTask]] = {'generate': ['generate', 'transcription'], 'pooling': ['pooling', 'embed', 'classify', 'reward'], 'draft': []}

@runtime_checkable
class SupportsHash(Protocol):

    def compute_hash(self) -> str:
        ...

class SupportsMetricsInfo(Protocol):

    def metrics_info(self) -> dict[str, str]:
        ...

class ModelImpl(str, enum.Enum):
    AUTO = 'auto'
    VLLM = 'vllm'
    TRANSFORMERS = 'transformers'

def get_attr_docs(cls: type[Any]) -> dict[str, str]:
    """
    Get any docstrings placed after attribute assignments in a class body.

    https://davidism.com/mit-license/
    """

    def pairwise(iterable):
        """
        Manually implement https://docs.python.org/3/library/itertools.html#itertools.pairwise

        Can be removed when Python 3.9 support is dropped.
        """
        iterator = iter(iterable)
        a = next(iterator, None)
        for b in iterator:
            yield (a, b)
            a = b
    cls_node = ast.parse(textwrap.dedent(inspect.getsource(cls))).body[0]
    if not isinstance(cls_node, ast.ClassDef):
        raise TypeError('Given object was not a class.')
    out = {}
    for a, b in pairwise(cls_node.body):
        if not isinstance(a, (ast.Assign, ast.AnnAssign)) or not isinstance(b, ast.Expr) or (not isinstance(b.value, ast.Constant)) or (not isinstance(b.value.value, str)):
            continue
        doc = inspect.cleandoc(b.value.value)
        targets = a.targets if isinstance(a, ast.Assign) else [a.target]
        for target in targets:
            if not isinstance(target, ast.Name):
                continue
            out[target.id] = doc
    return out

def config(cls: ConfigT) -> ConfigT:
    """
    A decorator that ensures all fields in a dataclass have default values
    and that each field has a docstring.

    If a `ConfigT` is used as a CLI argument itself, the default value provided
    by `get_kwargs` will be the result parsing a JSON string as the kwargs
    (i.e. `ConfigT(**json.loads(cli_arg))`). However, if a particular `ConfigT`
    requires custom construction from CLI (i.e. `CompilationConfig`), it can
    have a `from_cli` method, which will be called instead.

    Config validation is performed by the tools/validate_config.py
    script, which is invoked during the pre-commit checks.
    """
    return cls

def get_field(cls: ConfigType, name: str) -> Field:
    """Get the default factory field of a dataclass by name. Used for getting
    default factory fields in `EngineArgs`."""
    if not is_dataclass(cls):
        raise TypeError('The given class is not a dataclass.')
    cls_fields = {f.name: f for f in fields(cls)}
    if name not in cls_fields:
        raise ValueError(f"Field '{name}' not found in {cls.__name__}.")
    named_field: Field = cls_fields[name]
    if (default_factory := named_field.default_factory) is not MISSING:
        return field(default_factory=default_factory)
    if (default := named_field.default) is not MISSING:
        return field(default=default)
    raise ValueError(f'{cls.__name__}.{name} must have a default value or default factory.')

def is_init_field(cls: ConfigType, name: str) -> bool:
    return next(f for f in fields(cls) if f.name == name).init
TokenizerMode = Literal['auto', 'slow', 'mistral', 'custom']
ModelDType = Literal['auto', 'half', 'float16', 'bfloat16', 'float', 'float32']

@config
@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ModelConfig:
    """Configuration for the model."""
    model: str = 'Qwen/Qwen3-0.6B'
    'Name or path of the Hugging Face model to use. It is also used as the\n    content for `model_name` tag in metrics output when `served_model_name` is\n    not specified.'
    runner: RunnerOption = 'auto'
    'The type of model runner to use. Each vLLM instance only supports one\n    model runner, even if the same model can be used for multiple types.'
    task: TaskOption = 'auto'
    'The task to use the model for. If the model supports more than one\n    model runner, this is used to select which model runner to run.\n\n    Note that the model may support other tasks using the same model runner.'
    tokenizer: SkipValidation[str] = None
    'Name or path of the Hugging Face tokenizer to use. If unspecified, model\n    name or path will be used.'
    tokenizer_mode: TokenizerMode = 'auto'
    'Tokenizer mode:\n\n    - "auto" will use the fast tokenizer if available.\n\n    - "slow" will always use the slow tokenizer.\n\n    - "mistral" will always use the tokenizer from `mistral_common`.\n\n    - "custom" will use --tokenizer to select the preregistered tokenizer.'
    trust_remote_code: bool = False
    'Trust remote code (e.g., from HuggingFace) when downloading the model\n    and tokenizer.'
    dtype: Union[ModelDType, torch.dtype] = 'auto'
    'Data type for model weights and activations:\n\n    - "auto" will use FP16 precision for FP32 and FP16 models, and BF16\n    precision for BF16 models.\n\n    - "half" for FP16. Recommended for AWQ quantization.\n\n    - "float16" is the same as "half".\n\n    - "bfloat16" for a balance between precision and range.\n\n    - "float" is shorthand for FP32 precision.\n\n    - "float32" for FP32 precision.'
    seed: Optional[int] = None
    'Random seed for reproducibility. Initialized to None in V0, but\n    initialized to 0 in V1.'
    hf_config_path: Optional[str] = None
    'Name or path of the Hugging Face config to use. If unspecified, model\n    name or path will be used.'
    allowed_local_media_path: str = ''
    'Allowing API requests to read local images or videos from directories\n    specified by the server file system. This is a security risk. Should only\n    be enabled in trusted environments.'
    revision: Optional[str] = None
    'The specific model version to use. It can be a branch name, a tag name,\n    or a commit id. If unspecified, will use the default version.'
    code_revision: Optional[str] = None
    'The specific revision to use for the model code on the Hugging Face Hub.\n    It can be a branch name, a tag name, or a commit id. If unspecified, will\n    use the default version.'
    rope_scaling: dict[str, Any] = field(default_factory=dict)
    'RoPE scaling configuration. For example,\n    `{"rope_type":"dynamic","factor":2.0}`.'
    rope_theta: Optional[float] = None
    'RoPE theta. Use with `rope_scaling`. In some cases, changing the RoPE\n    theta improves the performance of the scaled model.'
    tokenizer_revision: Optional[str] = None
    'The specific revision to use for the tokenizer on the Hugging Face Hub.\n    It can be a branch name, a tag name, or a commit id. If unspecified, will\n    use the default version.'
    max_model_len: SkipValidation[int] = None
    'Model context length (prompt and output). If unspecified, will be\n    automatically derived from the model config.\n\n    When passing via `--max-model-len`, supports k/m/g/K/M/G in human-readable\n    format. Examples:\n\n    - 1k -> 1000\n\n    - 1K -> 1024\n\n    - 25.6k -> 25,600'
    spec_target_max_model_len: Optional[int] = None
    'Specify the maximum length for spec decoding draft models.'
    quantization: SkipValidation[Optional[QuantizationMethods]] = None
    'Method used to quantize the weights. If `None`, we first check the\n    `quantization_config` attribute in the model config file. If that is\n    `None`, we assume the model weights are not quantized and use `dtype` to\n    determine the data type of the weights.'
    enforce_eager: bool = False
    'Whether to always use eager-mode PyTorch. If True, we will disable CUDA\n    graph and always execute the model in eager mode. If False, we will use\n    CUDA graph and eager execution in hybrid for maximal performance and\n    flexibility.'
    max_seq_len_to_capture: int = 8192
    'Maximum sequence len covered by CUDA graphs. When a sequence has context\n    length larger than this, we fall back to eager mode. Additionally for\n    encoder-decoder models, if the sequence length of the encoder input is\n    larger than this, we fall back to the eager mode.'
    max_logprobs: int = 20
    'Maximum number of log probabilities to return when `logprobs` is\n    specified in `SamplingParams`. The default value comes the default for the\n    OpenAI Chat Completions API.'
    disable_sliding_window: bool = False
    'Whether to disable sliding window. If True, we will disable the sliding\n    window functionality of the model, capping to sliding window size. If the\n    model does not support sliding window, this argument is ignored.'
    disable_cascade_attn: bool = False
    "Disable cascade attention for V1. While cascade attention does not\n    change the mathematical correctness, disabling it could be useful for\n    preventing potential numerical issues. Note that even if this is set to\n    False, cascade attention will be only used when the heuristic tells that\n    it's beneficial."
    skip_tokenizer_init: bool = False
    'Skip initialization of tokenizer and detokenizer. Expects valid\n    `prompt_token_ids` and `None` for prompt from the input. The generated\n    output will contain token ids.'
    enable_prompt_embeds: bool = False
    'If `True`, enables passing text embeddings as inputs via the\n    `prompt_embeds` key. Note that enabling this will double the time required\n    for graph compilation.'
    served_model_name: Optional[Union[str, list[str]]] = None
    'The model name(s) used in the API. If multiple names are provided, the\n    server will respond to any of the provided names. The model name in the\n    model field of a response will be the first name in this list. If not\n    specified, the model name will be the same as the `--model` argument. Noted\n    that this name(s) will also be used in `model_name` tag content of\n    prometheus metrics, if multiple names provided, metrics tag will take the\n    first one.'
    limit_mm_per_prompt: dict[str, int] = field(default_factory=dict)
    'Maximum number of data items per modality per prompt. Only applicable\n    for multimodal models.'
    interleave_mm_strings: bool = False
    'Enable fully interleaved support for multimodal prompts, while using \n    --chat-template-content-format=string. Defaults to False.'
    media_io_kwargs: dict[str, dict[str, Any]] = field(default_factory=dict)
    'Additional args passed to process media inputs, keyed by modalities. \n    For example, to set num_frames for video, set \n    `--media-io-kwargs \'{"video": {"num_frames": 40} }\'` '
    use_async_output_proc: bool = True
    'Whether to use async output processor.'
    config_format: Union[str, ConfigFormat] = ConfigFormat.AUTO.value
    'The format of the model config to load:\n\n    - "auto" will try to load the config in hf format if available else it\n    will try to load in mistral format.\n\n    - "hf" will load the config in hf format.\n\n    - "mistral" will load the config in mistral format.'
    hf_token: Optional[Union[bool, str]] = None
    'The token to use as HTTP bearer authorization for remote files . If\n    `True`, will use the token generated when running `huggingface-cli login`\n    (stored in `~/.huggingface`).'
    hf_overrides: HfOverrides = field(default_factory=dict)
    'If a dictionary, contains arguments to be forwarded to the Hugging Face\n    config. If a callable, it is called to update the HuggingFace config.'
    mm_processor_kwargs: Optional[dict[str, Any]] = None
    'Arguments to be forwarded to the model\'s processor for multi-modal data,\n    e.g., image processor. Overrides for the multi-modal processor obtained\n    from `AutoProcessor.from_pretrained`. The available overrides depend on the\n    model that is being run. For example, for Phi-3-Vision: `{"num_crops": 4}`.\n    '
    disable_mm_preprocessor_cache: bool = False
    'If `True`, disable caching of the multi-modal preprocessor/mapper (not\n    recommended).'
    override_neuron_config: dict[str, Any] = field(default_factory=dict)
    'Initialize non-default neuron config or override default neuron config\n    that are specific to Neuron devices, this argument will be used to\n    configure the neuron config that can not be gathered from the vllm\n    arguments. e.g. `{"cast_logits_dtype": "bfloat16"}`.'
    pooler_config: Optional['PoolerConfig'] = field(init=False)
    'Pooler config which controls the behaviour of output pooling in pooling\n    models.'
    override_pooler_config: Optional[Union[dict, 'PoolerConfig']] = None
    'Initialize non-default pooling config or override default pooling config\n    for the pooling model. e.g. `{"pooling_type": "mean", "normalize": false}`.\n    '
    logits_processor_pattern: Optional[str] = None
    'Optional regex pattern specifying valid logits processor qualified names\n    that can be passed with the `logits_processors` extra completion argument.\n    Defaults to `None`, which allows no processors.'
    generation_config: str = 'auto'
    'The folder path to the generation config. Defaults to `"auto"`, the\n    generation config will be loaded from model path. If set to `"vllm"`, no\n    generation config is loaded, vLLM defaults will be used. If set to a folder\n    path, the generation config will be loaded from the specified folder path.\n    If `max_new_tokens` is specified in generation config, then it sets a\n    server-wide limit on the number of output tokens for all requests.'
    override_generation_config: dict[str, Any] = field(default_factory=dict)
    'Overrides or sets generation config. e.g. `{"temperature": 0.5}`. If\n    used with `--generation-config auto`, the override parameters will be\n    merged with the default config from the model. If used with\n    `--generation-config vllm`, only the override parameters are used.'
    enable_sleep_mode: bool = False
    'Enable sleep mode for the engine (only cuda platform is supported).'
    model_impl: Union[str, ModelImpl] = ModelImpl.AUTO.value
    'Which implementation of the model to use:\n\n    - "auto" will try to use the vLLM implementation, if it exists, and fall\n    back to the Transformers implementation if no vLLM implementation is\n    available.\n\n    - "vllm" will use the vLLM model implementation.\n\n    - "transformers" will use the Transformers model implementation.'
    override_attention_dtype: Optional[str] = None
    'Override dtype for attention'

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        factors: list[Any] = []
        factors.append(self.model)
        factors.append(self.dtype)
        factors.append(self.quantization)
        factors.append(self.revision)
        factors.append(self.code_revision)
        factors.append(self.max_model_len)
        factors.append(self.max_logprobs)
        factors.append(self.disable_sliding_window)
        factors.append(self.trust_remote_code)
        factors.append(self.generation_config)
        factors.append(self.model_impl)
        factors.append(self.override_generation_config)
        factors.append(self.rope_scaling)
        factors.append(self.rope_theta)
        factors.append(self.hf_config.to_json_string())
        str_factors = str(factors)
        assert_hashable(str_factors)
        return hashlib.sha256(str(factors).encode()).hexdigest()

    def __post_init__(self) -> None:
        if envs.VLLM_USE_V1 and self.seed is None:
            self.seed = 0
            if not envs.VLLM_ENABLE_V1_MULTIPROCESSING:
                logger.warning('The global random seed is set to %d. Since VLLM_ENABLE_V1_MULTIPROCESSING is set to False, this may affect the random state of the Python process that launched vLLM.', self.seed)
        self.served_model_name = get_served_model_name(self.model, self.served_model_name)
        self.model = maybe_model_redirect(self.model)
        if self.tokenizer is None:
            self.tokenizer = self.model
        if self.tokenizer_revision is None:
            self.tokenizer_revision = self.revision
        self.tokenizer = maybe_model_redirect(self.tokenizer)
        if isinstance(self.hf_config_path, str):
            self.hf_config_path = maybe_model_redirect(self.hf_config_path)
        if callable(self.hf_overrides):
            hf_overrides_kw = {}
            hf_overrides_fn = self.hf_overrides
        else:
            hf_overrides_kw = self.hf_overrides
            hf_overrides_fn = None
        if self.rope_scaling:
            hf_override: dict[str, Any] = {'rope_scaling': self.rope_scaling}
            hf_overrides_kw.update(hf_override)
            hf_overrides_str = json.dumps(hf_overrides_kw)
            msg = f"`--rope-scaling` will be removed in a future release. 'Please instead use `--hf-overrides '{hf_overrides_str}'`"
            warnings.warn(DeprecationWarning(msg), stacklevel=2)
        if self.rope_theta is not None:
            hf_override = {'rope_theta': self.rope_theta}
            hf_overrides_kw.update(hf_override)
            hf_overrides_str = json.dumps(hf_overrides_kw)
            msg = f"`--rope-theta` will be removed in a future release. 'Please instead use `--hf-overrides '{hf_overrides_str}'`"
            warnings.warn(DeprecationWarning(msg), stacklevel=2)
        self.maybe_pull_model_tokenizer_for_s3(self.model, self.tokenizer)
        if (backend := envs.VLLM_ATTENTION_BACKEND) and backend == 'FLASHINFER' and (find_spec('flashinfer') is None):
            raise ValueError('VLLM_ATTENTION_BACKEND is set to FLASHINFER, but flashinfer module was not found. See https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile for instructions on how to install it.')
        if self.override_attention_dtype is not None and (not current_platform.is_rocm()):
            warnings.warn('override-attention-dtype is set but not using ROCm platform', stacklevel=2)
        if self.enable_sleep_mode and (not current_platform.is_sleep_mode_available()):
            raise ValueError('Sleep mode is not supported on current platform.')
        if isinstance(self.config_format, str):
            self.config_format = ConfigFormat(self.config_format)
        hf_config = get_config(self.hf_config_path or self.model, self.trust_remote_code, self.revision, self.code_revision, self.config_format, hf_overrides_kw=hf_overrides_kw, hf_overrides_fn=hf_overrides_fn)
        self.hf_config = hf_config
        self.hf_text_config = get_hf_text_config(self.hf_config)
        self.attention_chunk_size = getattr(self.hf_text_config, 'attention_chunk_size', None)
        self.encoder_config = self._get_encoder_config()
        self.hf_image_processor_config = get_hf_image_processor_config(self.model, hf_token=self.hf_token, revision=self.revision)
        if self.task == 'score':
            if self._is_classify_task(self.architectures):
                self.task = 'classify'
            else:
                self.task = 'embed'
        elif self.task == 'embedding':
            msg = "The 'embedding' task has been renamed to 'embed', please use the new name. The old name will be removed in v1.0."
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            self.task = 'embed'
        model_info, arch = self.registry.inspect_model_cls(self.architectures)
        self._model_info = model_info
        self._architecture = arch
        all_supported_tasks = self._get_supported_tasks(self.task)
        logger.debug('Tasks supported by runner type: %s', all_supported_tasks)
        supported_runner_types = self._get_supported_runner_types(all_supported_tasks)
        runner_type = self._resolve_runner(self.runner, self.task, supported_runner_types, all_supported_tasks)
        logger.debug('Selected runner type: %s', runner_type)
        if runner_type == 'pooling' and self.task == 'auto':
            selected_task = all_supported_tasks[runner_type][-1]
            assert selected_task != 'pooling'
            self.task = selected_task
        self.supported_runner_types = supported_runner_types
        self.runner_type = runner_type
        self.supported_tasks = all_supported_tasks[runner_type]
        if self.runner_type in ('draft', 'generate') and self.task != 'transcription':
            self.truncation_side = 'left'
        else:
            self.truncation_side = 'right'
        self.pooler_config = self._init_pooler_config()
        self.dtype = _get_and_verify_dtype(self.model, self.hf_config, self.dtype, is_pooling_model=self.runner_type == 'pooling', revision=self.revision)
        if self.hf_text_config.model_type == 'gemma2':
            self.hf_text_config.sliding_window_pattern = 2
        sliding_window = getattr(self.hf_text_config, 'sliding_window', None)
        sliding_window_pattern = getattr(self.hf_text_config, 'sliding_window_pattern', None)
        has_interleaved_attention = sliding_window_pattern is not None or isinstance(sliding_window, list)
        if not self.disable_sliding_window and has_interleaved_attention:
            if (backend := envs.VLLM_ATTENTION_BACKEND) in ('XFORMERS', 'FLASHINFER'):
                sliding_window_len_min = get_min_sliding_window(self.hf_text_config.sliding_window)
                logger.warning_once('%s has interleaved attention, which is currently not supported by the %s backend. Disabling sliding window and capping the max length to the sliding window size (%d).', self.hf_text_config.model_type, backend, sliding_window_len_min)
                self.disable_sliding_window = True
            else:
                self.hf_text_config.interleaved_sliding_window = sliding_window
                if hasattr(self.hf_text_config, 'sliding_window'):
                    delattr(self.hf_text_config, 'sliding_window')
                sliding_window = None
        self.original_max_model_len = self.max_model_len
        self.max_model_len = self.get_and_verify_max_len(self.max_model_len)
        self.multimodal_config = self._init_multimodal_config()
        if not self.skip_tokenizer_init:
            self._verify_tokenizer_mode()
        self.is_attention_free = self._init_attention_free()
        self.is_hybrid = self._init_is_hybrid()
        self.has_noops = self._init_has_noops()
        self.has_inner_state = self._init_has_inner_state()
        if not current_platform.is_neuron() and self.override_neuron_config:
            raise ValueError('`override_neuron_config` is only supported on Neuron.')
        self._verify_quantization()
        self._verify_cuda_graph()
        self._verify_bnb_config()

    @field_validator('quantization', mode='before')
    @classmethod
    def validate_quantization_before(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.lower()
        return value

    @model_validator(mode='after')
    def validate_model_config_after(self: 'ModelConfig') -> 'ModelConfig':
        if not isinstance(self.tokenizer, str):
            raise ValueError('tokenizer must be a string after __post_init__.')
        if not isinstance(self.max_model_len, int):
            raise ValueError('max_model_len must be an integer after __post_init__.')
        return self

    def _get_transformers_backend_cls(self) -> str:
        """Determine which Transformers backend class will be used if
        `model_impl` is set to `transformers` or `auto`."""
        if self.hf_config != self.hf_text_config:
            return 'TransformersForMultimodalLM'
        else:
            return 'TransformersForCausalLM'

    @property
    def registry(self):
        return me_models.ModelRegistry

    @property
    def architectures(self) -> list[str]:
        architectures = getattr(self.hf_config, 'architectures', [])
        transformers_backend_cls = self._get_transformers_backend_cls()
        if self.model_impl != ModelImpl.VLLM.value and all(arch != transformers_backend_cls for arch in architectures):
            architectures.append(transformers_backend_cls)
        return architectures

    @property
    def architecture(self) -> str:
        return self._architecture

    @property
    def model_info(self):
        return self._model_info

    def maybe_pull_model_tokenizer_for_s3(self, model: str, tokenizer: str) -> None:
        """Pull model/tokenizer from S3 to temporary directory when needed.

        Args:
            model: Model name or path
            tokenizer: Tokenizer name or path
        """
        if not (is_s3(model) or is_s3(tokenizer)):
            return
        if is_s3(model):
            s3_model = S3Model()
            s3_model.pull_files(model, allow_pattern=['*.model', '*.py', '*.json'])
            self.model_weights = model
            self.model = s3_model.dir
            if model == tokenizer:
                s3_model.pull_files(model, ignore_pattern=['*.pt', '*.safetensors', '*.bin', '*.tensors'])
                self.tokenizer = s3_model.dir
                return
        if is_s3(tokenizer):
            s3_tokenizer = S3Model()
            s3_tokenizer.pull_files(model, ignore_pattern=['*.pt', '*.safetensors', '*.bin', '*.tensors'])
            self.tokenizer = s3_tokenizer.dir

    def _init_multimodal_config(self) -> Optional['MultiModalConfig']:
        if self.registry.is_multimodal_model(self.architectures):
            return MultiModalConfig(limit_per_prompt=self.limit_mm_per_prompt, media_io_kwargs=self.media_io_kwargs, mm_processor_kwargs=self.mm_processor_kwargs, disable_mm_preprocessor_cache=self.disable_mm_preprocessor_cache, interleave_mm_strings=self.interleave_mm_strings)
        if self.limit_mm_per_prompt:
            raise ValueError('`limit_mm_per_prompt` is only supported for multimodal models.')
        if self.mm_processor_kwargs:
            raise ValueError('`mm_processor_kwargs` is only supported for multimodal models.')
        if self.disable_mm_preprocessor_cache:
            raise ValueError('`disable_mm_preprocessor_cache` is only supported for multimodal models.')
        if self.interleave_mm_strings:
            raise ValueError('`interleave_mm_strings` is only supported for multimodal models.')
        return None

    def _get_encoder_config(self):
        return get_sentence_transformer_tokenizer_config(self.model, self.revision)

    def _init_pooler_config(self) -> Optional['PoolerConfig']:
        if self.runner_type == 'pooling':
            if isinstance(self.override_pooler_config, dict):
                self.override_pooler_config = PoolerConfig(**self.override_pooler_config)
            pooler_config = self.override_pooler_config or PoolerConfig()
            base_config = get_pooling_config(self.model, self.revision)
            if base_config is not None:
                for k, v in base_config.items():
                    if getattr(pooler_config, k) is None:
                        setattr(pooler_config, k, v)
            if self.is_matryoshka:
                if pooler_config.normalize is None:
                    pooler_config.normalize = True
                elif not pooler_config.normalize:
                    raise ValueError('`normalize` must be enabled (set to True) for models that are compatible with Matryoshka Representation.')
            return pooler_config
        return None

    def _init_attention_free(self) -> bool:
        return self.registry.is_attention_free_model(self.architectures)

    def _init_is_hybrid(self) -> bool:
        return self.registry.is_hybrid_model(self.architectures)

    def _init_has_noops(self) -> bool:
        architectures = getattr(self.hf_config, 'architectures', [])
        return self.registry.is_noops_model(architectures)

    def _init_has_inner_state(self) -> bool:
        return self.registry.model_has_inner_state(self.architectures)

    def _verify_tokenizer_mode(self) -> None:
        tokenizer_mode = cast(TokenizerMode, self.tokenizer_mode.lower())
        if tokenizer_mode not in get_args(TokenizerMode):
            raise ValueError(f'Unknown tokenizer mode: {self.tokenizer_mode}. Must be one of {get_args(TokenizerMode)}.')
        self.tokenizer_mode = tokenizer_mode

    def _is_classify_task(self, architectures: list[str]):
        for arch in architectures:
            if arch.endswith('ForSequenceClassification'):
                return True
        return self.registry.is_cross_encoder_model(architectures)

    def _get_preferred_pooling_task(self, architectures: list[str]) -> _ResolvedTask:
        model_id = self.model
        if get_pooling_config(model_id, self.revision):
            return 'embed'
        if self.registry.is_transcription_model(architectures):
            return 'transcription'
        suffix_to_preferred_task: list[tuple[str, _ResolvedTask]] = [('EmbeddingModel', 'embed'), ('RewardModel', 'reward')]
        for suffix, pref_task in suffix_to_preferred_task:
            if self.architecture.endswith(suffix):
                return pref_task
        return 'embed'

    def _get_supported_generation_tasks(self, task_option: TaskOption) -> list[_ResolvedTask]:
        registry = self.registry
        architectures = self.architectures
        if registry.is_transcription_only_model(architectures):
            return ['transcription']
        supported_tasks = list[_ResolvedTask]()
        if registry.is_text_generation_model(architectures):
            supported_tasks.append('generate')
            if registry.is_transcription_model(architectures):
                supported_tasks.append('transcription')
        return supported_tasks

    def _get_supported_pooling_tasks(self, task_option: TaskOption) -> list[_ResolvedTask]:
        registry = self.registry
        architectures = self.architectures
        supported_tasks = list[_ResolvedTask]()
        if registry.is_pooling_model(architectures):
            supported_tasks.append('pooling')
            if task_option == 'auto':
                preferred_task = self._get_preferred_pooling_task(architectures)
                supported_tasks.append(preferred_task)
            elif task_option in _RUNNER_TASKS['pooling']:
                supported_tasks.append(cast(_ResolvedTask, task_option))
        return supported_tasks

    def _get_supported_tasks(self, task_option: TaskOption) -> dict[RunnerType, list[_ResolvedTask]]:
        if self._is_classify_task(self.architectures):
            return {'generate': [], 'pooling': ['classify'], 'draft': []}
        else:
            return {'generate': self._get_supported_generation_tasks(task_option), 'pooling': self._get_supported_pooling_tasks(task_option), 'draft': ['draft']}

    def _get_supported_runner_types(self, supported_tasks: dict[RunnerType, list[_ResolvedTask]]) -> set[RunnerType]:
        return {runner for runner, runner_tasks in supported_tasks.items() if len(runner_tasks) > 0}

    def _resolve_runner(self, runner_option: RunnerOption, task_option: TaskOption, supported_runner_types: set[RunnerType], supported_tasks: dict[RunnerType, list[_ResolvedTask]]) -> RunnerType:
        if not supported_runner_types:
            raise ValueError('This model does not support any model runners!')
        if runner_option != 'auto':
            if runner_option not in supported_runner_types:
                raise ValueError(f'This model does not support runner={runner_option!r}. Available runners: {supported_runner_types}')
            return runner_option
        if task_option != 'auto':
            for runner, runner_tasks in supported_tasks.items():
                if task_option in runner_tasks:
                    return runner
            else:
                task_runner: RunnerType = next((runner for runner, tasks in _RUNNER_TASKS.items() if task_option in tasks))
                raise ValueError(f'This model does not support task={task_option!r}. Available tasks for runner={task_runner!r}: {supported_tasks[task_runner]}')
        if 'classify' in supported_tasks.get('pooling', []):
            return 'pooling'
        suffix_to_preferred_runner: list[tuple[str, RunnerType]] = [('ForCausalLM', 'generate'), ('ForConditionalGeneration', 'generate'), ('ChatModel', 'generate'), ('LMHeadModel', 'generate'), ('EmbeddingModel', 'pooling'), ('RewardModel', 'pooling')]
        for suffix, pref_runner in suffix_to_preferred_runner:
            if self.architecture.endswith(suffix) and pref_runner in supported_runner_types:
                return pref_runner
        if 'generate' in supported_runner_types:
            return 'generate'
        if 'pooling' in supported_runner_types:
            return 'pooling'
        raise AssertionError('This line should not be reached')

    def _parse_quant_hf_config(self):
        quant_cfg = getattr(self.hf_config, 'quantization_config', None)
        if quant_cfg is None:
            quant_cfg = getattr(self.hf_config, 'compression_config', None)
        return quant_cfg

    def _verify_quantization(self) -> None:
        supported_quantization = me_quant.QUANTIZATION_METHODS
        optimized_quantization_methods = ['fp8', 'marlin', 'modelopt', 'gptq_marlin_24', 'gptq_marlin', 'awq_marlin', 'fbgemm_fp8', 'compressed-tensors', 'experts_int8', 'quark', 'modelopt_fp4', 'bitblas', 'gptq_bitblas', 'inc']
        if self.quantization is not None:
            self.quantization = cast(me_quant.QuantizationMethods, self.quantization)
        quant_cfg = self._parse_quant_hf_config()
        if quant_cfg is not None:
            quant_method = quant_cfg.get('quant_method', '').lower()
            quant_method = quant_method.replace('compressed_tensors', 'compressed-tensors')
            quant_cfg['quant_method'] = quant_method
            overrides = ['marlin', 'bitblas', 'gptq_marlin_24', 'gptq_marlin', 'gptq_bitblas', 'awq_marlin', 'ipex', 'moe_wna16']
            quantization_methods = [q for q in supported_quantization if q not in overrides]
            quantization_methods = quantization_methods + overrides
            for name in quantization_methods:
                method = me_quant.get_quantization_config(name)
                quantization_override = method.override_quantization_method(quant_cfg, self.quantization)
                if quantization_override is not None:
                    if name in get_args(me_quant.QuantizationMethods) and name not in overrides:
                        raise ValueError(f'Quantization method {name} is an override but is has not been added to the `overrides` list above. This is necessary to ensure that the overrides are checked in order of preference.')
                    quant_method = quantization_override
                    self.quantization = quantization_override
                    break
            if self.quantization is None:
                self.quantization = quant_method
            elif self.quantization != quant_method:
                raise ValueError(f'Quantization method specified in the model config ({quant_method}) does not match the quantization method specified in the `quantization` argument ({self.quantization}).')
        if self.quantization is not None:
            if self.quantization not in supported_quantization:
                raise ValueError(f'Unknown quantization method: {self.quantization}. Must be one of {supported_quantization}.')
            from vllm.platforms import current_platform
            current_platform.verify_quantization(self.quantization)
            if self.quantization not in optimized_quantization_methods:
                logger.warning('%s quantization is not fully optimized yet. The speed can be slower than non-quantized models.', self.quantization)

    def _verify_cuda_graph(self) -> None:
        self.max_seq_len_to_capture = min(self.max_seq_len_to_capture, self.max_model_len)
        ROCM_UNSUPPORTED_MODELS = ['mllama']
        unsupported_rocm = self.hf_config.model_type in ROCM_UNSUPPORTED_MODELS or self.is_encoder_decoder
        if unsupported_rocm and (not self.enforce_eager) and current_platform.is_rocm():
            logger.warning('CUDA graph is not supported for %s on ROCm yet, fallback to eager mode.', self.hf_config.model_type)
            self.enforce_eager = True

    def _verify_bnb_config(self) -> None:
        """
        The current version of bitsandbytes (0.46.1) with 8-bit models does not
        yet support CUDA graph.
        # TODO Remove this when bitsandbytes supports.
        """
        is_bitsandbytes = self.quantization == 'bitsandbytes'
        has_quantization_config = getattr(self.hf_config, 'quantization_config', None) is not None
        is_8bit = self.hf_config.quantization_config.get('load_in_8bit', False) if has_quantization_config else False
        if all([is_bitsandbytes, has_quantization_config, is_8bit, not self.enforce_eager]):
            logger.warning('CUDA graph is not supported on BitsAndBytes 8bit yet, fallback to the eager mode.')
            self.enforce_eager = True

    def _verify_with_expert_parallelism(self) -> None:
        num_expert_names = ['moe_num_experts', 'num_experts', 'n_routed_experts', 'num_local_experts']
        num_experts = 0
        for name in num_expert_names:
            num_experts = getattr(self.hf_text_config, name, 0)
            if num_experts > 0:
                break
        if num_experts < 1:
            raise ValueError('Number of experts in the model must be greater than 0 when expert parallelism is enabled.')

    def verify_dual_chunk_attention_config(self, load_config: 'LoadConfig') -> None:
        if hasattr(self.hf_config, 'dual_chunk_attention_config'):
            from vllm.model_executor.model_loader.weight_utils import get_sparse_attention_config
            sparse_attn_config = get_sparse_attention_config(self, load_config)
            if sparse_attn_config:
                self.hf_config.dual_chunk_attention_config['sparse_attention_config'] = sparse_attn_config
                if 'sparse_attention_enabled' not in self.hf_config.dual_chunk_attention_config:
                    self.hf_config.dual_chunk_attention_config['sparse_attention_enabled'] = True

    def verify_async_output_proc(self, parallel_config, speculative_config, device_config) -> None:
        if not self.use_async_output_proc:
            return
        if parallel_config.pipeline_parallel_size > 1:
            self.use_async_output_proc = False
            return
        if not current_platform.is_async_output_supported(self.enforce_eager):
            self.use_async_output_proc = False
            return
        if envs.VLLM_USE_RAY_SPMD_WORKER:
            self.use_async_output_proc = False
            return
        if self.runner_type == 'pooling':
            self.use_async_output_proc = False
        if speculative_config:
            self.use_async_output_proc = False

    def verify_with_parallel_config(self, parallel_config: 'ParallelConfig') -> None:
        if parallel_config.distributed_executor_backend == 'external_launcher':
            assert self.seed is not None, 'Seed must be set when using external launcher backend to make sure sampling results are the same across workers.'
        total_num_attention_heads = getattr(self.hf_text_config, 'num_attention_heads', 0)
        tensor_parallel_size = parallel_config.tensor_parallel_size
        if total_num_attention_heads % tensor_parallel_size != 0:
            raise ValueError(f'Total number of attention heads ({total_num_attention_heads}) must be divisible by tensor parallel size ({tensor_parallel_size}).')
        if parallel_config.enable_expert_parallel:
            self._verify_with_expert_parallelism()
        pipeline_parallel_size = parallel_config.pipeline_parallel_size
        if pipeline_parallel_size > 1:
            if not self.registry.is_pp_supported_model(self.architectures):
                raise NotImplementedError('Pipeline parallelism is not supported for this model. Supported models implement the `SupportsPP` interface.')
            if self.use_async_output_proc:
                self.use_async_output_proc = False

    def get_hf_config_sliding_window(self) -> Union[Optional[int], list[Optional[int]]]:
        """Get the sliding window size, or None if disabled."""
        if hasattr(self.hf_text_config, 'use_sliding_window') and (not self.hf_text_config.use_sliding_window):
            return None
        return getattr(self.hf_text_config, 'sliding_window', None)

    def get_sliding_window(self) -> Optional[Union[int, list[Optional[int]]]]:
        """Get the sliding window size, or None if disabled.
        """
        if self.disable_sliding_window:
            return None
        return self.get_hf_config_sliding_window()

    def get_vocab_size(self) -> int:
        return self.hf_text_config.vocab_size

    def get_hidden_size(self) -> int:
        return self.hf_text_config.hidden_size

    @property
    def is_deepseek_mla(self) -> bool:
        if not hasattr(self.hf_text_config, 'model_type'):
            return False
        elif self.hf_text_config.model_type in ('deepseek_v2', 'deepseek_v3', 'deepseek_mtp', 'kimi_k2'):
            return self.hf_text_config.kv_lora_rank is not None
        elif self.hf_text_config.model_type == 'eagle':
            return self.hf_text_config.model.model_type in ('deepseek_v2', 'deepseek_v3') and self.hf_text_config.kv_lora_rank is not None
        return False

    def get_head_size(self) -> int:
        if self.is_deepseek_mla:
            qk_rope_head_dim = getattr(self.hf_text_config, 'qk_rope_head_dim', 0)
            if self.use_mla:
                return self.hf_text_config.kv_lora_rank + qk_rope_head_dim
            else:
                qk_nope_head_dim = getattr(self.hf_text_config, 'qk_nope_head_dim', 0)
                if qk_rope_head_dim and qk_nope_head_dim:
                    return qk_rope_head_dim + qk_nope_head_dim
        if hasattr(self.hf_text_config, 'model_type') and self.hf_text_config.model_type == 'zamba2':
            return self.hf_text_config.attention_head_dim
        if self.is_attention_free:
            return 0
        if getattr(self.hf_text_config, 'head_dim', None) is not None:
            return self.hf_text_config.head_dim
        return self.hf_text_config.hidden_size // self.hf_text_config.num_attention_heads

    def get_total_num_kv_heads(self) -> int:
        """Returns the total number of KV heads."""
        falcon_model_types = ['falcon', 'RefinedWeb', 'RefinedWebModel']
        new_decoder_arch_falcon = self.hf_config.model_type in falcon_model_types and getattr(self.hf_config, 'new_decoder_architecture', False)
        if not new_decoder_arch_falcon and getattr(self.hf_text_config, 'multi_query', False):
            return 1
        if self.hf_config.model_type == 'mpt':
            if 'kv_n_heads' in self.hf_config.attn_config:
                return self.hf_config.attn_config['kv_n_heads']
            return self.hf_config.num_attention_heads
        if self.hf_config.model_type == 'dbrx':
            return getattr(self.hf_config.attn_config, 'kv_n_heads', self.hf_config.num_attention_heads)
        if self.hf_config.model_type == 'nemotron-nas':
            for block in self.hf_config.block_configs:
                if not block.attention.no_op:
                    return self.hf_config.num_attention_heads // block.attention.n_heads_in_group
            raise RuntimeError("Couldn't determine number of kv heads")
        if self.is_attention_free:
            return 0
        attributes = ['n_head_kv', 'num_kv_heads', 'num_key_value_heads', 'multi_query_group_num']
        for attr in attributes:
            num_kv_heads = getattr(self.hf_text_config, attr, None)
            if num_kv_heads is not None:
                return num_kv_heads
        return self.hf_text_config.num_attention_heads

    def get_num_kv_heads(self, parallel_config: 'ParallelConfig') -> int:
        """Returns the number of KV heads per GPU."""
        if self.use_mla:
            return 1
        total_num_kv_heads = self.get_total_num_kv_heads()
        return max(1, total_num_kv_heads // parallel_config.tensor_parallel_size)

    def get_num_attention_heads(self, parallel_config: 'ParallelConfig') -> int:
        num_heads = getattr(self.hf_text_config, 'num_attention_heads', 0)
        return num_heads // parallel_config.tensor_parallel_size

    def get_layers_start_end_indices(self, parallel_config: 'ParallelConfig') -> tuple[int, int]:
        if self.hf_text_config.model_type == 'deepseek_mtp' or self.hf_config.model_type == 'mimo_mtp' or self.hf_config.model_type == 'glm4_moe_mtp':
            total_num_hidden_layers = getattr(self.hf_text_config, 'num_nextn_predict_layers', 0)
        else:
            total_num_hidden_layers = getattr(self.hf_text_config, 'num_hidden_layers', 0)
        pp_rank = parallel_config.rank // parallel_config.tensor_parallel_size % parallel_config.pipeline_parallel_size
        pp_size = parallel_config.pipeline_parallel_size
        start, end = get_pp_indices(total_num_hidden_layers, pp_rank, pp_size)
        return (start, end)

    def get_num_layers(self, parallel_config: 'ParallelConfig') -> int:
        start, end = self.get_layers_start_end_indices(parallel_config)
        return end - start

    def get_num_layers_by_block_type(self, parallel_config: 'ParallelConfig', block_type: LayerBlockType=LayerBlockType.attention) -> int:
        attn_block_type = block_type == LayerBlockType.attention
        is_transformer = not self.is_hybrid and (not self.has_noops) and (not self.is_attention_free)
        start, end = self.get_layers_start_end_indices(parallel_config)
        if is_transformer:
            return end - start if attn_block_type else 0
        elif self.is_attention_free:
            return 0 if attn_block_type else end - start
        elif self.has_noops:
            block_configs = self.hf_config.block_configs
            return sum(not bc.attention.no_op for bc in block_configs[start:end])
        else:
            layers_block_type_value = getattr(self.hf_config, 'layers_block_type', None)
            if layers_block_type_value is not None:
                if hasattr(self.hf_text_config, 'model_type') and self.hf_text_config.model_type == 'zamba2':
                    if attn_block_type:
                        return sum(t == 'hybrid' for t in layers_block_type_value[start:end])
                    else:
                        return self.get_num_layers(parallel_config)
                return sum(t == block_type.value for t in layers_block_type_value[start:end])
            attn_type_list = getattr(self.hf_config, 'attn_type_list', None)
            if attn_type_list:
                return sum(t == 1 for t in attn_type_list[start:end])
            if layers_block_type_value is None and attn_type_list is None:
                raise ValueError(f'The model is an hybrid without alayers_block_type or an attn_type_list in the hf_config,cannot determine the num of {block_type.value} layers')
            return sum(t == 1 for t in attn_type_list[start:end])

    def get_mamba_chunk_size(self) -> Optional[int]:
        """
        Returns the mamba chunk size if it exists
        """
        chunk_size = getattr(self.hf_text_config, 'mamba_chunk_size', None)
        if chunk_size is None:
            chunk_size = getattr(self.hf_text_config, 'chunk_size', None)
        return chunk_size

    def get_multimodal_config(self) -> 'MultiModalConfig':
        """
        Get the multimodal configuration of the model.

        Raises:
            ValueError: If the model is not multimodal.
        """
        if self.multimodal_config is None:
            raise ValueError('The model is not multimodal.')
        return self.multimodal_config

    def try_get_generation_config(self) -> dict[str, Any]:
        if self.generation_config in ('auto', 'vllm'):
            config = try_get_generation_config(self.hf_config_path or self.model, trust_remote_code=self.trust_remote_code, revision=self.revision)
        else:
            config = try_get_generation_config(self.generation_config, trust_remote_code=self.trust_remote_code)
        if config is None:
            return {}
        return config.to_diff_dict()

    def get_diff_sampling_param(self) -> dict[str, Any]:
        """
        This method returns a dictionary containing the parameters
        that differ from the default sampling parameters. If
        `generation_config` is `"vllm"`, an empty dictionary is returned.

        Returns:
            dict[str, Any]: A dictionary with the differing sampling
            parameters, if `generation_config` is `"vllm"` an empty dictionary.
        """
        if self.generation_config == 'vllm':
            config = {}
        else:
            config = self.try_get_generation_config()
        config.update(self.override_generation_config)
        available_params = ['repetition_penalty', 'temperature', 'top_k', 'top_p', 'min_p', 'max_new_tokens']
        if any(p in config for p in available_params):
            diff_sampling_param = {p: config.get(p) for p in available_params if config.get(p) is not None}
            if 'max_new_tokens' in diff_sampling_param:
                diff_sampling_param['max_tokens'] = diff_sampling_param.pop('max_new_tokens')
        else:
            diff_sampling_param = {}
        if diff_sampling_param:
            logger.warning_once("Default sampling parameters have been overridden by the model's Hugging Face generation config recommended from the model creator. If this is not intended, please relaunch vLLM instance with `--generation-config vllm`.")
        return diff_sampling_param

    @property
    def is_encoder_decoder(self) -> bool:
        """Extract the HF encoder/decoder model flag."""
        "\n        For Mllama, VLLM overrides HF's is_encoder_decoder flag and sets it to\n        True to enable cross-attention\n        Neuron needs all multimodal data to be in the decoder and does not\n        need to explicitly enable cross-attention\n        "
        if current_platform.is_neuron() and self.hf_config.model_type == 'mllama':
            return False
        return is_encoder_decoder(self.hf_config)

    @property
    def uses_mrope(self) -> bool:
        return uses_mrope(self.hf_config)

    @property
    def is_multimodal_model(self) -> bool:
        return self.multimodal_config is not None

    @property
    def is_cross_encoder(self) -> bool:
        return self.task == 'classify'

    @property
    def use_mla(self) -> bool:
        return self.is_deepseek_mla and (not envs.VLLM_MLA_DISABLE)

    @property
    def is_v1_compatible(self) -> bool:
        architectures = getattr(self.hf_config, 'architectures', [])
        return me_models.ModelRegistry.is_v1_compatible(architectures)

    @property
    def is_matryoshka(self) -> bool:
        return bool(getattr(self.hf_config, 'matryoshka_dimensions', None)) or getattr(self.hf_config, 'is_matryoshka', False)

    @property
    def matryoshka_dimensions(self):
        return getattr(self.hf_config, 'matryoshka_dimensions', None)

    @property
    def use_pad_token(self) -> bool:
        return getattr(self.hf_config, 'use_pad_token', True)

    def get_and_verify_max_len(self, max_model_len: int):
        tokenizer_config = None
        if self.runner_type == 'pooling' and getattr(self.hf_config, 'position_embedding_type', '') == 'absolute':
            tokenizer_config = try_get_tokenizer_config(self.tokenizer, trust_remote_code=self.trust_remote_code, revision=self.tokenizer_revision)
        max_model_len = _get_and_verify_max_len(hf_config=self.hf_text_config, tokenizer_config=tokenizer_config, max_model_len=max_model_len, disable_sliding_window=self.disable_sliding_window, sliding_window_len=self.get_hf_config_sliding_window(), spec_target_max_model_len=self.spec_target_max_model_len, encoder_config=self.encoder_config)
        logger.info('Using max model len %s', max_model_len)
        return max_model_len
BlockSize = Literal[1, 8, 16, 32, 64, 128]
CacheDType = Literal['auto', 'fp8', 'fp8_e4m3', 'fp8_e5m2', 'fp8_inc']
PrefixCachingHashAlgo = Literal['builtin', 'sha256', 'sha256_cbor_64bit']

@config
@dataclass
class CacheConfig:
    """Configuration for the KV cache."""
    block_size: SkipValidation[BlockSize] = None
    'Size of a contiguous cache block in number of tokens. This is ignored on\n    neuron devices and set to `--max-model-len`. On CUDA devices, only block\n    sizes up to 32 are supported. On HPU devices, block size defaults to 128.\n\n    This config has no static default. If left unspecified by the user, it will\n    be set in `Platform.check_and_update_config()` based on the current\n    platform.'
    gpu_memory_utilization: float = 0.9
    'The fraction of GPU memory to be used for the model executor, which can\n    range from 0 to 1. For example, a value of 0.5 would imply 50% GPU memory\n    utilization. If unspecified, will use the default value of 0.9. This is a\n    per-instance limit, and only applies to the current vLLM instance. It does\n    not matter if you have another vLLM instance running on the same GPU. For\n    example, if you have two vLLM instances running on the same GPU, you can\n    set the GPU memory utilization to 0.5 for each instance.'
    swap_space: float = 4
    'Size of the CPU swap space per GPU (in GiB).'
    cache_dtype: CacheDType = 'auto'
    'Data type for kv cache storage. If "auto", will use model data type.\n    CUDA 11.8+ supports fp8 (=fp8_e4m3) and fp8_e5m2. ROCm (AMD GPU) supports\n    fp8 (=fp8_e4m3). Intel Gaudi (HPU) supports fp8 (using fp8_inc).'
    is_attention_free: bool = False
    'Whether the model is attention-free. This is primarily set in\n    `ModelConfig` and that value should be manually duplicated here.'
    num_gpu_blocks_override: Optional[int] = None
    'Number of GPU blocks to use. This overrides the profiled `num_gpu_blocks`\n    if specified. Does nothing if `None`. Used for testing preemption.'
    sliding_window: Optional[int] = None
    'Sliding window size for the KV cache. This is primarily set in\n    `ModelConfig` and that value should be manually duplicated here.'
    enable_prefix_caching: Optional[bool] = None
    'Whether to enable prefix caching. Disabled by default for V0. Enabled by\n    default for V1.'
    prefix_caching_hash_algo: PrefixCachingHashAlgo = 'builtin'
    'Set the hash algorithm for prefix caching:\n\n    - "builtin" is Python\'s built-in hash.\n\n    - "sha256" is collision resistant but with certain overheads.\n    This option uses Pickle for object serialization before hashing.\n\n    - "sha256_cbor_64bit" provides a reproducible, cross-language compatible \n    hash. It serializes objects using canonical CBOR and hashes them with \n    SHA-256. The resulting hash consists of the lower 64 bits of the SHA-256\n    digest.'
    cpu_offload_gb: float = 0
    'The space in GiB to offload to CPU, per GPU. Default is 0, which means\n    no offloading. Intuitively, this argument can be seen as a virtual way to\n    increase the GPU memory size. For example, if you have one 24 GB GPU and\n    set this to 10, virtually you can think of it as a 34 GB GPU. Then you can\n    load a 13B model with BF16 weight, which requires at least 26GB GPU memory.\n    Note that this requires fast CPU-GPU interconnect, as part of the model is\n    loaded from CPU memory to GPU memory on the fly in each model forward pass.\n    '
    calculate_kv_scales: bool = False
    'This enables dynamic calculation of `k_scale` and `v_scale` when\n    kv_cache_dtype is fp8. If `False`, the scales will be loaded from the model\n    checkpoint if available. Otherwise, the scales will default to 1.0.'
    cpu_kvcache_space_bytes: Optional[int] = None
    '(CPU backend only) CPU key-value cache space.'
    mamba_page_size_padded: Optional[int] = None
    ' Optional override for mamba page size; used by hybrid mamba/attention\n    models to ensure exact alignment with attention page size.'
    num_gpu_blocks: Optional[int] = field(default=None, init=False)
    'The number of blocks to allocate for GPU memory.'
    num_cpu_blocks: Optional[int] = field(default=None, init=False)
    'The number of blocks to allocate for CPU memory.'

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        factors: list[Any] = []
        factors.append(self.cache_dtype)
        hash_str = hashlib.md5(str(factors).encode(), usedforsecurity=False).hexdigest()
        return hash_str

    def __post_init__(self) -> None:
        self.swap_space_bytes = self.swap_space * GiB_bytes
        self._verify_cache_dtype()
        self._verify_prefix_caching()

    def metrics_info(self):
        return {key: str(value) for key, value in self.__dict__.items()}

    @model_validator(mode='after')
    def _verify_args(self) -> Self:
        if self.cpu_offload_gb < 0:
            raise ValueError(f'CPU offload space must be non-negative, but got {self.cpu_offload_gb}')
        if self.gpu_memory_utilization > 1.0:
            raise ValueError(f'GPU memory utilization must be less than 1.0. Got {self.gpu_memory_utilization}.')
        return self

    def _verify_cache_dtype(self) -> None:
        if self.cache_dtype == 'auto':
            pass
        elif self.cache_dtype in get_args(CacheDType):
            logger.info('Using fp8 data type to store kv cache. It reduces the GPU memory footprint and boosts the performance. Meanwhile, it may cause accuracy drop without a proper scaling factor.')
        else:
            raise ValueError(f'Unknown kv cache dtype: {self.cache_dtype}')

    def _verify_prefix_caching(self) -> None:
        if not self.enable_prefix_caching:
            return
        if self.sliding_window is not None and (not envs.VLLM_USE_V1):
            raise NotImplementedError('Prefix caching is not supported with sliding window. Run with --disable-sliding-window to use prefix caching.')
        if self.enable_prefix_caching and self.prefix_caching_hash_algo not in get_args(PrefixCachingHashAlgo):
            raise ValueError(f'Unknown prefix caching hash algorithm: {self.prefix_caching_hash_algo}. Must be one of {get_args(PrefixCachingHashAlgo)}.')

    def verify_with_parallel_config(self, parallel_config: 'ParallelConfig') -> None:
        total_cpu_memory = get_cpu_memory()
        num_gpus_per_node = parallel_config.tensor_parallel_size
        cpu_memory_usage = self.swap_space_bytes * num_gpus_per_node
        msg = f'{cpu_memory_usage / GiB_bytes:.2f} GiB out of the {total_cpu_memory / GiB_bytes:.2f} GiB total CPU memory is allocated for the swap space.'
        if cpu_memory_usage > 0.7 * total_cpu_memory:
            raise ValueError('Too large swap space. ' + msg)
        elif cpu_memory_usage > 0.4 * total_cpu_memory:
            logger.warning('Possibly too large swap space. %s', msg)

class LoadFormat(str, enum.Enum):
    AUTO = 'auto'
    PT = 'pt'
    SAFETENSORS = 'safetensors'
    NPCACHE = 'npcache'
    DUMMY = 'dummy'
    TENSORIZER = 'tensorizer'
    SHARDED_STATE = 'sharded_state'
    GGUF = 'gguf'
    BITSANDBYTES = 'bitsandbytes'
    MISTRAL = 'mistral'
    RUNAI_STREAMER = 'runai_streamer'
    RUNAI_STREAMER_SHARDED = 'runai_streamer_sharded'
    FASTSAFETENSORS = 'fastsafetensors'

@config
@dataclass
class LoadConfig:
    """Configuration for loading the model weights."""
    load_format: Union[str, LoadFormat, 'BaseModelLoader'] = LoadFormat.AUTO.value
    'The format of the model weights to load:\n\n    - "auto" will try to load the weights in the safetensors format and fall\n    back to the pytorch bin format if safetensors format is not available.\n\n    - "pt" will load the weights in the pytorch bin format.\n\n    - "safetensors" will load the weights in the safetensors format.\n\n    - "npcache" will load the weights in pytorch format and store a numpy cache\n    to speed up the loading.\n\n    - "dummy" will initialize the weights with random values, which is mainly\n    for profiling.\n\n    - "tensorizer" will use CoreWeave\'s tensorizer library for fast weight\n    loading. See the Tensorize vLLM Model script in the Examples section for\n    more information.\n\n    - "runai_streamer" will load the Safetensors weights using Run:ai Model\n    Streamer.\n\n    - "bitsandbytes" will load the weights using bitsandbytes quantization.\n\n    - "sharded_state" will load weights from pre-sharded checkpoint files,\n    supporting efficient loading of tensor-parallel models.\n\n    - "gguf" will load weights from GGUF format files (details specified in\n    https://github.com/ggml-org/ggml/blob/master/docs/gguf.md).\n\n    - "mistral" will load weights from consolidated safetensors files used by\n    Mistral models.'
    download_dir: Optional[str] = None
    'Directory to download and load the weights, default to the default\n    cache directory of Hugging Face.'
    model_loader_extra_config: Union[dict, TensorizerConfig] = field(default_factory=dict)
    'Extra config for model loader. This will be passed to the model loader\n    corresponding to the chosen load_format.'
    device: Optional[str] = None
    'Device to which model weights will be loaded, default to\n    device_config.device'
    ignore_patterns: Optional[Union[list[str], str]] = None
    'The list of patterns to ignore when loading the model. Default to\n    "original/**/*" to avoid repeated loading of llama\'s checkpoints.'
    use_tqdm_on_load: bool = True
    'Whether to enable tqdm for showing progress bar when loading model\n    weights.'
    pt_load_map_location: Union[str, dict[str, str]] = 'cpu'
    '\n    pt_load_map_location: the map location for loading pytorch checkpoint, to\n    support loading checkpoints can only be loaded on certain devices like\n    "cuda", this is equivalent to {"": "cuda"}. Another supported format is\n    mapping from different devices like from GPU 1 to GPU 0:\n    {"cuda:1": "cuda:0"}. Note that when passed from command line, the strings\n    in dictionary needs to be double quoted for json parsing. For more details,\n    see original doc for `map_location` in https://pytorch.org/docs/stable/generated/torch.load.html\n    '

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        factors: list[Any] = []
        hash_str = hashlib.md5(str(factors).encode(), usedforsecurity=False).hexdigest()
        return hash_str

    def __post_init__(self):
        if isinstance(self.load_format, str):
            load_format = self.load_format.lower()
            self.load_format = LoadFormat(load_format)
        if self.ignore_patterns is not None and len(self.ignore_patterns) > 0:
            logger.info('Ignoring the following patterns when downloading weights: %s', self.ignore_patterns)
        else:
            self.ignore_patterns = ['original/**/*']
DistributedExecutorBackend = Literal['ray', 'mp', 'uni', 'external_launcher']

@config
@dataclass
class ParallelConfig:
    """Configuration for the distributed execution."""
    pipeline_parallel_size: int = 1
    'Number of pipeline parallel groups.'
    tensor_parallel_size: int = 1
    'Number of tensor parallel groups.'
    data_parallel_size: int = 1
    'Number of data parallel groups. MoE layers will be sharded according to\n    the product of the tensor parallel size and data parallel size.'
    data_parallel_size_local: int = 1
    'Number of local data parallel groups.'
    data_parallel_rank: int = 0
    'Rank of the data parallel group.'
    data_parallel_rank_local: Optional[int] = None
    'Local rank of the data parallel group,\n    set only in SPMD mode.'
    data_parallel_master_ip: str = '127.0.0.1'
    'IP of the data parallel master.'
    data_parallel_rpc_port: int = 29550
    'Port for data parallel messaging.'
    data_parallel_master_port: int = 29500
    'Port of the data parallel master.'
    data_parallel_backend: str = 'mp'
    'Backend to use for data parallel, either "mp" or "ray".'
    data_parallel_external_lb: bool = False
    'Whether to use "external" DP LB mode. Applies only to online serving\n    and when data_parallel_size > 0. Set implicitly when\n    data_parallel_rank is provided explicitly to vllm serve.'
    enable_expert_parallel: bool = False
    'Use expert parallelism instead of tensor parallelism for MoE layers.'
    enable_eplb: bool = False
    'Enable expert parallelism load balancing for MoE layers.'
    num_redundant_experts: int = 0
    'Number of redundant experts to use for expert parallelism.'
    eplb_window_size: int = 1000
    'Window size for expert load recording.'
    eplb_step_interval: int = 3000
    '\n    Interval for rearranging experts in expert parallelism.\n\n    Note that if this is greater than the EPLB window size, only the metrics\n    of the last `eplb_window_size` steps will be used for rearranging experts.\n    '
    eplb_log_balancedness: bool = False
    '\n    Log the balancedness each step of expert parallelism.\n    This is turned off by default since it will cause communication overhead.\n    '
    max_parallel_loading_workers: Optional[int] = None
    'Maximum number of parallel loading workers when loading model\n    sequentially in multiple batches. To avoid RAM OOM when using tensor\n    parallel and large models.'
    disable_custom_all_reduce: bool = False
    'Disable the custom all-reduce kernel and fall back to NCCL.'
    ray_workers_use_nsight: bool = False
    'Whether to profile Ray workers with nsight, see https://docs.ray.io/en/latest/ray-observability/user-guides/profiling.html#profiling-nsight-profiler.'
    placement_group: Optional['PlacementGroup'] = None
    'ray distributed model workers placement group.'
    distributed_executor_backend: Optional[Union[DistributedExecutorBackend, type['ExecutorBase']]] = None
    'Backend to use for distributed model\n    workers, either "ray" or "mp" (multiprocessing). If the product\n    of pipeline_parallel_size and tensor_parallel_size is less than\n    or equal to the number of GPUs available, "mp" will be used to\n    keep processing on a single host. Otherwise, this will default\n    to "ray" if Ray is installed and fail otherwise. Note that tpu\n    only support Ray for distributed inference.'
    worker_cls: str = 'auto'
    'The full name of the worker class to use. If "auto", the worker class\n    will be determined based on the platform.'
    sd_worker_cls: str = 'auto'
    'The full name of the worker class to use for speculative decoding.\n    If "auto", the worker class will be determined based on the platform.'
    worker_extension_cls: str = ''
    'The full name of the worker extension class to use. The worker extension\n    class is dynamically inherited by the worker class. This is used to inject\n    new attributes and methods to the worker class for use in collective_rpc\n    calls.'
    world_size: int = field(init=False)
    'world_size is TPxPP, it affects the number of workers we create.'
    rank: int = 0
    'Global rank in distributed setup.'
    enable_multimodal_encoder_data_parallel: bool = False
    ' Use data parallelism instead of tensor parallelism for vision encoder.\n    Only support LLama4 for now'

    @property
    def world_size_across_dp(self) -> int:
        """world_size_across_dp is TPxPPxDP, it is the size of the world
        including data parallelism."""
        return self.world_size * self.data_parallel_size

    def get_next_dp_init_port(self) -> int:
        """
        We might need to initialize process groups in multiple
        processes that is related to data parallelism,
        e.g. both in the worker and in the engine, which
        can live in different processes. To avoid port conflicts, we
        increment the port number each time we need to initialize a
        new process group related to data parallelism.
        """
        answer = self.data_parallel_master_port
        self.data_parallel_master_port += 1
        return answer

    def stateless_init_dp_group(self) -> 'ProcessGroup':
        max_retries = 5
        last_exc: Optional[Exception] = None
        for _ in range(max_retries):
            try:
                return stateless_init_torch_distributed_process_group(self.data_parallel_master_ip, self.get_next_dp_init_port(), self.data_parallel_rank, self.data_parallel_size, backend='gloo')
            except DistNetworkError as e:
                if 'EADDRINUSE' in str(e):
                    logger.warning('Address already in use. Retrying with a new port.')
                    last_exc = e
                    continue
                raise e
        assert last_exc is not None
        raise last_exc

    @staticmethod
    def has_unfinished_dp(dp_group: 'ProcessGroup', has_unfinished: bool) -> bool:
        tensor = torch.tensor([has_unfinished], dtype=torch.int32, device='cpu')
        torch.distributed.all_reduce(tensor, op=ReduceOp.MAX, group=dp_group)
        aggregated_has_unfinished = bool(tensor.item())
        return aggregated_has_unfinished

    @staticmethod
    def sync_kv_cache_memory_size(dp_group: 'ProcessGroup', kv_cache_memory: int) -> int:
        if kv_cache_memory == -1:
            kv_cache_memory = torch.iinfo(torch.int64).max
        tensor = torch.tensor([kv_cache_memory], dtype=torch.int64, device='cpu')
        torch.distributed.all_reduce(tensor, op=ReduceOp.MIN, group=dp_group)
        return tensor.item()

    def compute_hash(self):
        """
        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        factors: list[Any] = []
        factors.append(self.pipeline_parallel_size)
        factors.append(self.tensor_parallel_size)
        factors.append(self.enable_expert_parallel)
        factors.append(self.data_parallel_size)
        factors.append(envs.VLLM_ALL2ALL_BACKEND)
        return hashlib.sha256(str(factors).encode()).hexdigest()

    def __post_init__(self) -> None:
        self.world_size = self.pipeline_parallel_size * self.tensor_parallel_size
        if self.data_parallel_size_local > self.data_parallel_size:
            raise ValueError(f'data_parallel_size_local ({self.data_parallel_size_local}) must be <= data_parallel_size ({self.data_parallel_size})')
        if self.data_parallel_size > 1 or self.data_parallel_size_local == 0:
            self.data_parallel_master_port = get_open_port()
            if not 0 <= self.data_parallel_rank < self.data_parallel_size:
                raise ValueError(f'data_parallel_rank ({self.data_parallel_rank}) must be in the range [0, {self.data_parallel_size})')
        else:
            self.data_parallel_size = envs.VLLM_DP_SIZE
            self.data_parallel_rank = envs.VLLM_DP_RANK
            self.data_parallel_rank_local = envs.VLLM_DP_RANK_LOCAL
            self.data_parallel_master_ip = envs.VLLM_DP_MASTER_IP
            self.data_parallel_master_port = envs.VLLM_DP_MASTER_PORT
            if self.data_parallel_external_lb:
                raise ValueError('data_parallel_external_lb can only be set when data_parallel_size > 1')
        if self.distributed_executor_backend == 'external_launcher':
            import os
            os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
            logger.info('Disabling V1 multiprocessing for external launcher.')
        if self.enable_eplb:
            if not current_platform.is_cuda():
                raise ValueError('Expert parallelism load balancing is only supported on CUDA devices now.')
            if self.num_redundant_experts < 0:
                raise ValueError(f'num_redundant_experts must be non-negative, but got {self.num_redundant_experts}.')
        elif self.num_redundant_experts != 0:
            raise ValueError(f'num_redundant_experts should be used with EPLB.{self.num_redundant_experts}.')
        if self.distributed_executor_backend is None and self.world_size > 1:
            from vllm.executor import ray_utils
            backend: DistributedExecutorBackend = 'mp'
            ray_found = ray_utils.ray_is_available()
            if current_platform.is_neuron():
                backend = 'uni'
            elif current_platform.is_tpu() and envs.VLLM_XLA_USE_SPMD:
                backend = 'uni'
            elif current_platform.is_cuda() and cuda_device_count_stateless() < self.world_size:
                if not ray_found:
                    raise ValueError('Unable to load Ray which is required for multi-node inference, please install Ray with `pip install ray`.') from ray_utils.ray_import_err
                backend = 'ray'
            elif self.data_parallel_backend == 'ray':
                logger.info('Using ray distributed inference because data_parallel_backend is ray')
                backend = 'ray'
            elif ray_found:
                if self.placement_group:
                    backend = 'ray'
                else:
                    from ray import is_initialized as ray_is_initialized
                    if ray_is_initialized():
                        from ray.util import get_current_placement_group
                        if get_current_placement_group():
                            backend = 'ray'
            self.distributed_executor_backend = backend
            logger.debug('Defaulting to use %s for distributed inference', backend)
        if self.distributed_executor_backend is None and self.world_size == 1:
            self.distributed_executor_backend = 'uni'

    @property
    def use_ray(self) -> bool:
        return self.distributed_executor_backend == 'ray' or (isinstance(self.distributed_executor_backend, type) and self.distributed_executor_backend.uses_ray)

    @model_validator(mode='after')
    def _verify_args(self) -> Self:
        if self.distributed_executor_backend not in ('ray', 'mp', 'uni', 'external_launcher', None) and (not (isinstance(self.distributed_executor_backend, type) and issubclass(self.distributed_executor_backend, ExecutorBase))):
            raise ValueError(f"Unrecognized distributed executor backend {self.distributed_executor_backend}. Supported values are 'ray', 'mp' 'uni', 'external_launcher' or custom ExecutorBase subclass.")
        if self.use_ray:
            from vllm.executor import ray_utils
            ray_utils.assert_ray_available()
        if not current_platform.use_custom_allreduce():
            self.disable_custom_all_reduce = True
            logger.debug('Disabled the custom all-reduce kernel because it is not supported on current platform.')
        if self.ray_workers_use_nsight and (not self.use_ray):
            raise ValueError('Unable to use nsight profiling unless workers run with Ray.')
        return self
PreemptionMode = Literal['swap', 'recompute']
SchedulerPolicy = Literal['fcfs', 'priority']

@config
@dataclass
class SchedulerConfig:
    """Scheduler configuration."""
    runner_type: RunnerType = 'generate'
    'The runner type to launch for the model.'
    max_num_batched_tokens: SkipValidation[int] = None
    'Maximum number of tokens to be processed in a single iteration.\n\n    This config has no static default. If left unspecified by the user, it will\n    be set in `EngineArgs.create_engine_config` based on the usage context.'
    max_num_seqs: SkipValidation[int] = None
    'Maximum number of sequences to be processed in a single iteration.\n\n    This config has no static default. If left unspecified by the user, it will\n    be set in `EngineArgs.create_engine_config` based on the usage context.'
    max_model_len: SkipValidation[int] = None
    'Maximum length of a sequence (including prompt and generated text). This\n    is primarily set in `ModelConfig` and that value should be manually\n    duplicated here.'
    max_num_partial_prefills: int = 1
    'For chunked prefill, the maximum number of sequences that can be\n    partially prefilled concurrently.'
    max_long_partial_prefills: int = 1
    'For chunked prefill, the maximum number of prompts longer than\n    long_prefill_token_threshold that will be prefilled concurrently. Setting\n    this less than max_num_partial_prefills will allow shorter prompts to jump\n    the queue in front of longer prompts in some cases, improving latency.'
    long_prefill_token_threshold: int = 0
    'For chunked prefill, a request is considered long if the prompt is\n    longer than this number of tokens.'
    num_lookahead_slots: int = 0
    'The number of slots to allocate per sequence per\n    step, beyond the known token ids. This is used in speculative\n    decoding to store KV activations of tokens which may or may not be\n    accepted.\n\n    NOTE: This will be replaced by speculative config in the future; it is\n    present to enable correctness tests until then.'
    cuda_graph_sizes: list[int] = field(default_factory=list)
    'Cuda graph capture sizes\n    1. if none provided, then default set to [min(max_num_seqs * 2, 512)]\n    2. if one value is provided, then the capture list would follow the\n    pattern: [1, 2, 4] + [i for i in range(8, cuda_graph_sizes + 1, 8)]\n    3. more than one value (e.g. 1 2 128) is provided, then the capture list\n    will follow the provided list.'
    delay_factor: float = 0.0
    'Apply a delay (of delay factor multiplied by previous\n    prompt latency) before scheduling next prompt.'
    enable_chunked_prefill: SkipValidation[bool] = None
    'If True, prefill requests can be chunked based\n    on the remaining max_num_batched_tokens.'
    is_multimodal_model: bool = False
    'True if the model is multimodal.'
    max_num_encoder_input_tokens: int = field(init=False)
    'Multimodal encoder compute budget, only used in V1.\n\n    NOTE: This is not currently configurable. It will be overridden by\n    max_num_batched_tokens in case max multimodal embedding size is larger.'
    encoder_cache_size: int = field(init=False)
    'Multimodal encoder cache size, only used in V1.\n\n    NOTE: This is not currently configurable. It will be overridden by\n    max_num_batched_tokens in case max multimodal embedding size is larger.'
    preemption_mode: Optional[PreemptionMode] = None
    'Whether to perform preemption by swapping or\n    recomputation. If not specified, we determine the mode as follows:\n    We use recomputation by default since it incurs lower overhead than\n    swapping. However, when the sequence group has multiple sequences\n    (e.g., beam search), recomputation is not currently supported. In\n    such a case, we use swapping instead.'
    num_scheduler_steps: int = 1
    'Maximum number of forward steps per scheduler call.'
    multi_step_stream_outputs: bool = True
    'If False, then multi-step will stream outputs at the end of all steps'
    send_delta_data: bool = False
    'Private API. If used, scheduler sends delta data to\n    workers instead of an entire data. It should be enabled only\n    when SPMD worker architecture is enabled. I.e.,\n    VLLM_USE_RAY_SPMD_WORKER=1'
    policy: SchedulerPolicy = 'fcfs'
    'The scheduling policy to use:\n\n    - "fcfs" means first come first served, i.e. requests are handled in order\n    of arrival.\n\n    - "priority" means requests are handled based on given priority (lower\n    value means earlier handling) and time of arrival deciding any ties).'
    chunked_prefill_enabled: bool = field(init=False)
    'True if chunked prefill is enabled.'
    disable_chunked_mm_input: bool = False
    'If set to true and chunked prefill is enabled, we do not want to\n    partially schedule a multimodal item. Only used in V1\n    This ensures that if a request has a mixed prompt\n    (like text tokens TTTT followed by image tokens IIIIIIIIII) where only\n    some image tokens can be scheduled (like TTTTIIIII, leaving IIIII),\n    it will be scheduled as TTTT in one step and IIIIIIIIII in the next.'
    scheduler_cls: Union[str, type[object]] = 'vllm.core.scheduler.Scheduler'
    'The scheduler class to use. "vllm.core.scheduler.Scheduler" is the\n    default scheduler. Can be a class directly or the path to a class of form\n    "mod.custom_class".'
    disable_hybrid_kv_cache_manager: bool = False
    'If set to True, KV cache manager will allocate the same size of KV cache\n    for all attention layers even if there are multiple type of attention layers\n    like full attention and sliding window attention.\n    '
    async_scheduling: bool = False
    'EXPERIMENTAL: If set to True, perform async scheduling. This may help\n    reduce the CPU overheads, leading to better latency and throughput. However,\n    async scheduling is currently not supported with some features such as\n    structured outputs, speculative decoding, and pipeline parallelism.\n    '

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        factors: list[Any] = []
        hash_str = hashlib.md5(str(factors).encode(), usedforsecurity=False).hexdigest()
        return hash_str

    def __post_init__(self) -> None:
        if self.max_model_len is None:
            self.max_model_len = 8192
        if self.max_num_seqs is None:
            self.max_num_seqs = 128
        if self.max_num_batched_tokens is None:
            if self.enable_chunked_prefill:
                if self.num_scheduler_steps > 1:
                    self.max_num_batched_tokens = max(self.max_model_len, DEFAULT_MAX_NUM_BATCHED_TOKENS)
                else:
                    self.max_num_batched_tokens = DEFAULT_MAX_NUM_BATCHED_TOKENS
            else:
                self.max_num_batched_tokens = max(self.max_model_len, DEFAULT_MAX_NUM_BATCHED_TOKENS)
            if self.runner_type == 'pooling':
                self.max_num_batched_tokens = max(self.max_num_batched_tokens, POOLING_MODEL_MAX_NUM_BATCHED_TOKENS)
            if self.is_multimodal_model:
                self.max_num_batched_tokens = max(self.max_num_batched_tokens, MULTIMODAL_MODEL_MAX_NUM_BATCHED_TOKENS)
            self.max_num_batched_tokens = min(self.max_num_seqs * self.max_model_len, self.max_num_batched_tokens)
        self.max_num_encoder_input_tokens = self.max_num_batched_tokens
        self.encoder_cache_size = self.max_num_batched_tokens
        if self.enable_chunked_prefill:
            logger.info('Chunked prefill is enabled with max_num_batched_tokens=%d.', self.max_num_batched_tokens)
        self.chunked_prefill_enabled = self.enable_chunked_prefill
        if self.max_num_partial_prefills > 1:
            if self.long_prefill_token_threshold == 0:
                self.long_prefill_token_threshold = int(self.max_model_len * 0.04)
            logger.info('Concurrent partial prefills enabled with max_num_partial_prefills=%d, max_long_partial_prefills=%d, long_prefill_token_threshold=%d', self.max_num_partial_prefills, self.max_long_partial_prefills, self.long_prefill_token_threshold)
        if not self.cuda_graph_sizes:
            self.cuda_graph_sizes = [min(self.max_num_seqs * 2, 512)]
        if self.async_scheduling:
            self.scheduler_cls = 'vllm.v1.core.sched.async_scheduler.AsyncScheduler'

    @model_validator(mode='after')
    def _verify_args(self) -> Self:
        if self.max_num_batched_tokens < self.max_model_len and (not self.chunked_prefill_enabled):
            raise ValueError(f'max_num_batched_tokens ({self.max_num_batched_tokens}) is smaller than max_model_len ({self.max_model_len}). This effectively limits the maximum sequence length to max_num_batched_tokens and makes vLLM reject longer sequences. Please increase max_num_batched_tokens or decrease max_model_len.')
        if self.max_num_batched_tokens < self.max_num_seqs:
            raise ValueError(f'max_num_batched_tokens ({self.max_num_batched_tokens}) must be greater than or equal to max_num_seqs ({self.max_num_seqs}).')
        if self.max_num_batched_tokens > self.max_num_seqs * self.max_model_len:
            logger.warning('max_num_batched_tokens (%d) exceeds max_num_seqs * max_model_len (%d). This may lead to unexpected behavior.', self.max_num_batched_tokens, self.max_num_seqs * self.max_model_len)
        if self.num_lookahead_slots < 0:
            raise ValueError(f'num_lookahead_slots ({self.num_lookahead_slots}) must be greater than or equal to 0.')
        if self.num_scheduler_steps < 1:
            raise ValueError(f'num_scheduler_steps ({self.num_scheduler_steps}) must be greater than or equal to 1.')
        if self.max_num_partial_prefills < 1:
            raise ValueError(f'max_num_partial_prefills ({self.max_num_partial_prefills}) must be greater than or equal to 1.')
        elif self.max_num_partial_prefills > 1:
            if not self.chunked_prefill_enabled:
                raise ValueError('Chunked prefill must be enabled to set max_num_partial_prefills > 1.')
            if self.long_prefill_token_threshold > self.max_model_len:
                raise ValueError(f'long_prefill_token_threshold ({self.long_prefill_token_threshold}) cannot be greater than the max_model_len ({self.max_model_len}).')
        if self.max_long_partial_prefills < 1 or self.max_long_partial_prefills > self.max_num_partial_prefills:
            raise ValueError(f'max_long_partial_prefills ({self.max_long_partial_prefills}) must be greater than or equal to 1 and less than or equal to max_num_partial_prefills ({self.max_num_partial_prefills}).')
        return self

    @property
    def is_multi_step(self) -> bool:
        return self.num_scheduler_steps > 1
Device = Literal['auto', 'cuda', 'neuron', 'cpu', 'tpu', 'xpu']

@config
@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class DeviceConfig:
    """Configuration for the device to use for vLLM execution."""
    device: SkipValidation[Optional[Union[Device, torch.device]]] = 'auto'
    'Device type for vLLM execution.\n    This parameter is deprecated and will be\n    removed in a future release.\n    It will now be set automatically based\n    on the current platform.'
    device_type: str = field(init=False)
    'Device type from the current platform. This is set in\n    `__post_init__`.'

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        factors: list[Any] = []
        hash_str = hashlib.md5(str(factors).encode(), usedforsecurity=False).hexdigest()
        return hash_str

    def __post_init__(self):
        if self.device == 'auto':
            from vllm.platforms import current_platform
            self.device_type = current_platform.device_type
            if not self.device_type:
                raise RuntimeError('Failed to infer device type, please set the environment variable `VLLM_LOGGING_LEVEL=DEBUG` to turn on verbose logging to help debug the issue.')
        elif isinstance(self.device, str):
            self.device_type = self.device
        elif isinstance(self.device, torch.device):
            self.device_type = self.device.type
        if self.device_type in ['neuron']:
            self.device = torch.device('cpu')
        elif self.device_type in ['tpu']:
            self.device = None
        else:
            self.device = torch.device(self.device_type)
SpeculativeMethod = Literal['ngram', 'eagle', 'eagle3', 'medusa', 'mlp_speculator', 'draft_model', 'deepseek_mtp']

@config
@dataclass
class SpeculativeConfig:
    """Configuration for speculative decoding."""
    num_speculative_tokens: SkipValidation[int] = None
    'The number of speculative tokens, if provided. It will default to the\n    number in the draft model config if present, otherwise, it is required.'
    model: Optional[str] = None
    'The name of the draft model, eagle head, or additional weights, if\n    provided.'
    method: Optional[SpeculativeMethod] = None
    'The name of the speculative method to use. If users provide and set the\n    `model` param, the speculative method type will be detected automatically\n    if possible, if `model` param is not provided, the method name must be\n    provided.\n\n    If using `ngram` method, the related configuration `prompt_lookup_max` and\n    `prompt_lookup_min` should be considered.'
    draft_tensor_parallel_size: Optional[int] = None
    "The degree of the tensor parallelism for the draft model. Can only be 1\n    or the same as the target model's tensor parallel size."
    disable_logprobs: bool = True
    'If set to True, token log probabilities are not returned during\n    speculative decoding. If set to False, token log probabilities are returned\n    according to the log probability settings in SamplingParams.'
    quantization: Optional[me_quant.QuantizationMethods] = None
    'Quantization method that was used to quantize the draft model weights.\n    If `None`, we assume the model weights are not quantized. Note that it only\n    takes effect when using the draft model-based speculative method.'
    max_model_len: Optional[int] = None
    'The maximum model length of the draft model. Used when testing the\n    ability to skip speculation for some sequences.'
    revision: Optional[str] = None
    'The specific model version to use for the draft model. It can be a\n    branch name, a tag name, or a commit id. If unspecified, will use the\n    default version.'
    code_revision: Optional[str] = None
    'The specific revision to use for the draft model code on Hugging Face\n    Hub. It can be a branch name, a tag name, or a commit id. If unspecified,\n    will use the default version.'
    disable_by_batch_size: Optional[int] = None
    'Disable speculative decoding for new incoming requests when the number\n    of enqueued requests is larger than this value, if provided.'
    prompt_lookup_max: Optional[int] = None
    'Maximum size of ngram token window when using Ngram proposer, required\n    when method is set to ngram.'
    prompt_lookup_min: Optional[int] = None
    'Minimum size of ngram token window when using Ngram proposer, if\n    provided. Defaults to 1.'
    speculative_token_tree: Optional[str] = None
    'Specifies the tree structure for speculative token generation.\n    '
    target_model_config: SkipValidation[ModelConfig] = None
    'The configuration of the target model.'
    target_parallel_config: SkipValidation[ParallelConfig] = None
    'The parallel configuration for the target model.'
    enable_chunked_prefill: SkipValidation[bool] = None
    "Whether vLLM is configured to use chunked prefill or not. Used for\n    raising an error since it's not yet compatible with speculative decode."
    disable_log_stats: SkipValidation[bool] = None
    'Whether to disable the periodic printing of stage times in speculative\n    decoding.'
    draft_model_config: SkipValidation[ModelConfig] = None
    'The configuration of the draft model initialized internal.'
    draft_parallel_config: SkipValidation[ParallelConfig] = None
    'The parallel configuration for the draft model initialized internal.'

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        factors: list[Any] = []
        factors.append(self.method == 'eagle3')
        hash_str = hashlib.md5(str(factors).encode(), usedforsecurity=False).hexdigest()
        return hash_str

    @classmethod
    def from_dict(cls, dict_value: dict) -> 'SpeculativeConfig':
        """Parse the CLI value for the speculative config."""
        return cls(**dict_value)

    @staticmethod
    def hf_config_override(hf_config: PretrainedConfig) -> PretrainedConfig:
        if hf_config.model_type == 'deepseek_v3':
            hf_config.model_type = 'deepseek_mtp'
        if hf_config.model_type == 'deepseek_mtp':
            n_predict = getattr(hf_config, 'num_nextn_predict_layers', None)
            hf_config.update({'n_predict': n_predict, 'architectures': ['DeepSeekMTPModel']})
        if hf_config.architectures[0] == 'MiMoForCausalLM':
            hf_config.model_type = 'mimo_mtp'
            n_predict = getattr(hf_config, 'num_nextn_predict_layers', None)
            hf_config.update({'num_hidden_layers': 0, 'n_predict': n_predict, 'architectures': ['MiMoMTPModel']})
        if hf_config.architectures[0] == 'Glm4MoeForCausalLM':
            hf_config.model_type = 'glm4_moe_mtp'
            n_predict = getattr(hf_config, 'num_nextn_predict_layers', None)
            hf_config.update({'num_hidden_layers': 0, 'n_predict': n_predict, 'architectures': ['Glm4MoeMTPModel']})
        return hf_config

    def __post_init__(self):
        if self.model is None and self.num_speculative_tokens is not None:
            if self.target_model_config and (self.target_model_config.hf_text_config.model_type == 'deepseek_v3' or self.target_model_config.hf_text_config.model_type == 'mimo'):
                self.model = self.target_model_config.model
            elif self.method in ('ngram', '[ngram]'):
                self.model = 'ngram'
            else:
                raise ValueError('num_speculative_tokens was provided without speculative model.')
        if self.method is None and (self.model is not None and self.model in ('ngram', '[ngram]')):
            self.method = 'ngram'
        if self.method in ('ngram', '[ngram]'):
            self.method = 'ngram'
            if self.prompt_lookup_min is None and self.prompt_lookup_max is None:
                self.prompt_lookup_min = 5
                self.prompt_lookup_max = 5
            elif self.prompt_lookup_min is None:
                assert self.prompt_lookup_max is not None
                self.prompt_lookup_min = self.prompt_lookup_max
            elif self.prompt_lookup_max is None:
                assert self.prompt_lookup_min is not None
                self.prompt_lookup_max = self.prompt_lookup_min
            if self.prompt_lookup_min < 1:
                raise ValueError(f'prompt_lookup_min={self.prompt_lookup_min} must be > 0')
            if self.prompt_lookup_max < 1:
                raise ValueError(f'prompt_lookup_max={self.prompt_lookup_max} must be > 0')
            if self.prompt_lookup_min > self.prompt_lookup_max:
                raise ValueError(f'prompt_lookup_min={self.prompt_lookup_min} must be <= prompt_lookup_max={self.prompt_lookup_max}')
            self.draft_model_config = self.target_model_config
            self.draft_parallel_config = self.target_parallel_config
        else:
            self.prompt_lookup_max = 0
            self.prompt_lookup_min = 0
            if self.model is not None:
                self.draft_model_config = ModelConfig(model=self.model, runner='draft', tokenizer=self.target_model_config.tokenizer, tokenizer_mode=self.target_model_config.tokenizer_mode, trust_remote_code=self.target_model_config.trust_remote_code, allowed_local_media_path=self.target_model_config.allowed_local_media_path, dtype=self.target_model_config.dtype, seed=self.target_model_config.seed, revision=self.revision, code_revision=self.code_revision, tokenizer_revision=self.target_model_config.tokenizer_revision, spec_target_max_model_len=self.target_model_config.max_model_len, quantization=self.quantization, enforce_eager=self.target_model_config.enforce_eager, max_seq_len_to_capture=self.target_model_config.max_seq_len_to_capture, max_logprobs=self.target_model_config.max_logprobs, hf_overrides=SpeculativeConfig.hf_config_override)
                if self.method in ('eagle', 'eagle3'):
                    pass
                elif 'eagle-' in self.draft_model_config.model.lower() or 'eagle3-' in self.draft_model_config.model.lower():
                    self.method = 'eagle'
                elif self.draft_model_config.hf_config.model_type == 'medusa':
                    self.method = 'medusa'
                elif self.draft_model_config.hf_config.model_type == 'mlp_speculator':
                    self.method = 'mlp_speculator'
                elif self.draft_model_config.hf_config.model_type in ('deepseek_mtp', 'mimo_mtp', 'glm4_moe_mtp'):
                    self.method = 'deepseek_mtp'
                    if self.num_speculative_tokens > 1:
                        logger.warning('All Deepseek MTP models only have one layer. Might need some code changes to support multiple layers.')
                else:
                    self.method = 'draft_model'
                    raise NotImplementedError('Speculative decoding with draft model is not supported yet. Please consider using other speculative decoding methods such as ngram, medusa, eagle, or deepseek_mtp.')
                if self.method in ('eagle', 'eagle3'):
                    if self.enable_chunked_prefill and (not envs.VLLM_USE_V1):
                        raise ValueError('Chunked prefill and EAGLE are not compatible when using V0.')
                    from vllm.transformers_utils.configs.eagle import EAGLEConfig
                    if isinstance(self.draft_model_config.hf_config, EAGLEConfig):
                        pass
                    else:
                        eagle_config = EAGLEConfig(self.draft_model_config.hf_config, method=self.method, model_type='eagle')
                        self.draft_model_config.hf_config = eagle_config
                if self.num_speculative_tokens is not None and hasattr(self.draft_model_config.hf_config, 'num_lookahead_tokens'):
                    self.draft_model_config.hf_config.num_lookahead_tokens = self.num_speculative_tokens
                n_predict = getattr(self.draft_model_config.hf_config, 'n_predict', None)
                if n_predict is not None:
                    if self.num_speculative_tokens is None:
                        self.num_speculative_tokens = n_predict
                    elif self.num_speculative_tokens > n_predict and self.num_speculative_tokens % n_predict != 0:
                        raise ValueError(f'num_speculative_tokens:{self.num_speculative_tokens} must be divisible by n_predict={n_predict!r}')
                self.draft_tensor_parallel_size = SpeculativeConfig._verify_and_get_draft_tp(self.target_parallel_config, self.draft_tensor_parallel_size, self.draft_model_config.hf_config)
                self.draft_model_config.max_model_len = SpeculativeConfig._maybe_override_draft_max_model_len(self.max_model_len, self.draft_model_config.max_model_len, self.target_model_config.max_model_len)
                self.draft_parallel_config = SpeculativeConfig.create_draft_parallel_config(self.target_parallel_config, self.draft_tensor_parallel_size)

    @staticmethod
    def _maybe_override_draft_max_model_len(speculative_max_model_len: Optional[int], draft_max_model_len: int, target_max_model_len: int) -> int:
        """Determine the max sequence len for the draft model. This is usually
        the draft_max_model_len, but may be the target_max_model_len if it is
        less than the draft_max_model_len, or may be speculative_max_model_len
        if it is specified.

        This is necessary so that sequences do not exceed the capacity of the
        draft model or the target model.

        speculative_max_model_len is mainly used for testing that sequences can
        skip speculation.
        """
        if speculative_max_model_len is not None:
            if speculative_max_model_len > draft_max_model_len:
                raise ValueError(f'speculative_max_model_len={speculative_max_model_len!r} cannot be larger than draft_max_model_len={draft_max_model_len!r}')
            if speculative_max_model_len > target_max_model_len:
                raise ValueError(f'speculative_max_model_len={speculative_max_model_len!r} cannot be larger than target_max_model_len={target_max_model_len!r}')
            return speculative_max_model_len
        return min(draft_max_model_len, target_max_model_len)

    @staticmethod
    def _verify_and_get_draft_tp(target_parallel_config: ParallelConfig, speculative_draft_tensor_parallel_size: Optional[int], draft_hf_config: PretrainedConfig) -> int:
        """
        Verifies and adjusts the tensor parallel size for a draft model
        specified using speculative_draft_tensor_parallel_size.
        """
        if speculative_draft_tensor_parallel_size is None:
            if draft_hf_config.model_type == 'mlp_speculator':
                speculative_draft_tensor_parallel_size = 1
                if target_parallel_config.tensor_parallel_size > 1:
                    logger.warning('%s cannot currently be run with tp>1; setting speculative_draft_tensor_parallel_size=1', draft_hf_config.model_type)
            else:
                speculative_draft_tensor_parallel_size = target_parallel_config.tensor_parallel_size
        elif speculative_draft_tensor_parallel_size not in (1, target_parallel_config.tensor_parallel_size):
            raise ValueError(f'speculative_draft_tensor_parallel_size={speculative_draft_tensor_parallel_size!r} cannot be other value than 1 or target model tensor_parallel_size')
        return speculative_draft_tensor_parallel_size

    @staticmethod
    def create_draft_parallel_config(target_parallel_config: ParallelConfig, speculative_draft_tensor_parallel_size: int) -> ParallelConfig:
        """Create a parallel config for use by the draft worker.

        This is mostly a copy of the target parallel config, except the tp_size.
        """
        draft_parallel_config = ParallelConfig(pipeline_parallel_size=target_parallel_config.pipeline_parallel_size, tensor_parallel_size=speculative_draft_tensor_parallel_size, distributed_executor_backend=target_parallel_config.distributed_executor_backend, max_parallel_loading_workers=target_parallel_config.max_parallel_loading_workers, disable_custom_all_reduce=target_parallel_config.disable_custom_all_reduce, ray_workers_use_nsight=target_parallel_config.ray_workers_use_nsight, placement_group=target_parallel_config.placement_group)
        return draft_parallel_config

    @model_validator(mode='after')
    def _verify_args(self) -> Self:
        if self.num_speculative_tokens is None:
            raise ValueError('num_speculative_tokens must be provided with speculative model unless the draft model config contains an n_predict parameter.')
        if self.num_speculative_tokens <= 0:
            raise ValueError(f'Expected num_speculative_tokens to be greater than zero ({self.num_speculative_tokens}).')
        if self.draft_model_config:
            self.draft_model_config.verify_with_parallel_config(self.draft_parallel_config)
        if self.disable_by_batch_size is not None and self.disable_by_batch_size < 2:
            raise ValueError(f'Expect the batch size threshold of disabling speculative decoding is > 1, but got self.disable_by_batch_size={self.disable_by_batch_size!r}')
        if self.method == 'eagle3' and self.target_model_config and ('llama' not in self.target_model_config.hf_text_config.model_type):
            raise ValueError(f'Eagle3 is only supported for Llama models. Got self.target_model_config.hf_text_config.model_type={self.target_model_config.hf_text_config.model_type!r}')
        return self

    @property
    def num_lookahead_slots(self) -> int:
        """The number of additional slots the scheduler should allocate per
        step, in addition to the slots allocated for each known token.

        This is equal to the number of speculative tokens, as each speculative
        token must be scored.
        """
        return self.num_speculative_tokens

    def use_eagle(self) -> bool:
        return self.method in ('eagle', 'eagle3', 'deepseek_mtp')

    def __repr__(self) -> str:
        method = self.method
        model = None if method == 'ngram' else self.draft_model_config.model
        num_spec_tokens = self.num_speculative_tokens
        return f'SpeculativeConfig(method={method!r}, model={model!r}, num_spec_tokens={num_spec_tokens!r})'
LoRADType = Literal['auto', 'float16', 'bfloat16']

@config
@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class LoRAConfig:
    """Configuration for LoRA."""
    max_lora_rank: int = 16
    'Max LoRA rank.'
    max_loras: int = 1
    'Max number of LoRAs in a single batch.'
    fully_sharded_loras: bool = False
    'By default, only half of the LoRA computation is sharded with tensor\n    parallelism. Enabling this will use the fully sharded layers. At high\n    sequence length, max rank or tensor parallel size, this is likely faster.\n    '
    max_cpu_loras: Optional[int] = None
    'Maximum number of LoRAs to store in CPU memory. Must be >= than\n    `max_loras`.'
    lora_dtype: Union[torch.dtype, LoRADType] = 'auto'
    'Data type for LoRA. If auto, will default to base model dtype.'
    lora_extra_vocab_size: int = 256
    'Maximum size of extra vocabulary that can be present in a LoRA adapter\n    (added to the base model vocabulary).'
    lora_vocab_padding_size: ClassVar[int] = current_platform.get_lora_vocab_padding_size()
    default_mm_loras: Optional[dict[str, str]] = None
    'Dictionary mapping specific modalities to LoRA model paths; this field\n    is only applicable to multimodal models and should be leveraged when a\n    model always expects a LoRA to be active when a given modality is present.\n    Note that currently, if a request provides multiple additional\n    modalities, each of which have their own LoRA, we do NOT apply\n    default_mm_loras because we currently only support one lora adapter\n    per prompt. When run in offline mode, the lora IDs for n modalities\n    will be automatically assigned to 1-n with the names of the modalities\n    in alphabetic order.'
    bias_enabled: bool = False
    'Enable bias for LoRA adapters.'

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        factors: list[Any] = []
        factors.append(self.max_lora_rank)
        factors.append(self.max_loras)
        factors.append(self.fully_sharded_loras)
        factors.append(self.lora_dtype)
        factors.append(self.lora_extra_vocab_size)
        factors.append(self.lora_vocab_padding_size)
        factors.append(self.bias_enabled)
        hash_str = hashlib.md5(str(factors).encode(), usedforsecurity=False).hexdigest()
        return hash_str

    def __post_init__(self):
        possible_max_ranks = (8, 16, 32, 64, 128, 256, 320, 512)
        possible_lora_extra_vocab_size = (256, 512)
        if self.max_lora_rank not in possible_max_ranks:
            raise ValueError(f'max_lora_rank ({self.max_lora_rank}) must be one of {possible_max_ranks}.')
        if self.lora_extra_vocab_size not in possible_lora_extra_vocab_size:
            raise ValueError(f'lora_extra_vocab_size ({self.lora_extra_vocab_size}) must be one of {possible_lora_extra_vocab_size}.')
        if self.max_loras < 1:
            raise ValueError(f'max_loras ({self.max_loras}) must be >= 1.')
        if self.max_cpu_loras is None:
            self.max_cpu_loras = self.max_loras
        elif self.max_cpu_loras < self.max_loras:
            raise ValueError(f'max_cpu_loras ({self.max_cpu_loras}) must be >= max_loras ({self.max_loras})')

    def verify_with_cache_config(self, cache_config: CacheConfig):
        if cache_config.cpu_offload_gb > 0 and (not envs.VLLM_USE_V1):
            raise ValueError('V0 LoRA does not support CPU offload, please use V1.')

    def verify_with_model_config(self, model_config: ModelConfig):
        if self.lora_dtype in (None, 'auto'):
            self.lora_dtype = model_config.dtype
        elif isinstance(self.lora_dtype, str):
            self.lora_dtype = getattr(torch, self.lora_dtype)

@config
@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class PromptAdapterConfig:
    """Configuration for PromptAdapters."""
    max_prompt_adapters: int = 1
    'Max number of PromptAdapters in a batch.'
    max_prompt_adapter_token: int = 0
    'Max number of PromptAdapters tokens.'
    max_cpu_prompt_adapters: Optional[int] = None
    'Maximum number of PromptAdapters to store in CPU memory. Must be >= than\n    `max_prompt_adapters`.'
    prompt_adapter_dtype: Union[torch.dtype, str] = 'auto'
    'Data type for PromptAdapter. If auto, will default to base model dtype.\n    '

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        factors: list[Any] = []
        hash_str = hashlib.md5(str(factors).encode(), usedforsecurity=False).hexdigest()
        return hash_str

    def __post_init__(self):
        if self.max_prompt_adapters < 1:
            raise ValueError(f'max_prompt_adapters ({self.max_prompt_adapters}) must be >= 1.')
        if self.max_prompt_adapter_token == 0:
            raise ValueError('max_prompt_adapter_token must be set.')
        if self.max_cpu_prompt_adapters is None:
            self.max_cpu_prompt_adapters = self.max_prompt_adapters

    def verify_with_model_config(self, model_config: ModelConfig):
        if self.prompt_adapter_dtype == 'auto':
            self.prompt_adapter_dtype = model_config.dtype
        elif isinstance(self.prompt_adapter_dtype, str):
            self.prompt_adapter_dtype = getattr(torch, self.prompt_adapter_dtype)

@config
@dataclass
class MultiModalConfig:
    """Controls the behavior of multimodal models."""
    limit_per_prompt: dict[str, int] = cast(dict[str, int], get_field(ModelConfig, 'limit_mm_per_prompt'))
    '\n    The maximum number of input items allowed per prompt for each modality.\n    Defaults to 1 (V0) or 999 (V1) for each modality.\n\n    For example, to allow up to 16 images and 2 videos per prompt:\n    `{"images": 16, "videos": 2}`\n    '
    media_io_kwargs: dict[str, dict[str, Any]] = field(default_factory=dict)
    'Additional args passed to process media inputs, keyed by modalities. \n    For example, to set num_frames for video, set \n    `--media-io-kwargs \'{"video": {"num_frames": 40} }\'` '
    mm_processor_kwargs: Optional[dict[str, object]] = None
    '\n    Overrides for the multi-modal processor obtained from\n    `transformers.AutoProcessor.from_pretrained`.\n\n    The available overrides depend on the model that is being run.\n\n    For example, for Phi-3-Vision:\n    `{"num_crops": 4}`.\n    '
    disable_mm_preprocessor_cache: bool = False
    '\n    If `True`, disable caching of the processed multi-modal inputs.\n    '
    interleave_mm_strings: bool = False
    '\n    Enable fully interleaved support for multimodal prompts.\n    '

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        factors: list[Any] = []
        hash_str = hashlib.md5(str(factors).encode(), usedforsecurity=False).hexdigest()
        return hash_str

    def get_limit_per_prompt(self, modality: str) -> int:
        """
        Get the maximum number of input items allowed per prompt
        for the given modality.
        """
        return self.limit_per_prompt.get(modality, 999 if envs.VLLM_USE_V1 else 1)

@config
@dataclass
class PoolerConfig:
    """Controls the behavior of output pooling in pooling models."""
    pooling_type: Optional[str] = None
    '\n    The pooling method of the pooling model. This should be a key in\n    [`vllm.model_executor.layers.pooler.PoolingType`][].\n    '
    normalize: Optional[bool] = None
    '\n    Whether to normalize the pooled outputs. Usually, this should be set to\n    ``True`` for embedding outputs.\n    '
    softmax: Optional[bool] = None
    '\n    Whether to apply softmax to the pooled outputs. Usually, this should be set\n    to ``True`` for classification outputs.\n    '
    step_tag_id: Optional[int] = None
    '\n    If set, only the score corresponding to the ``step_tag_id`` in the\n    generated sentence should be returned. Otherwise, the scores for all tokens\n    are returned.\n    '
    returned_token_ids: Optional[list[int]] = None
    '\n    A list of indices for the vocabulary dimensions to be extracted,\n    such as the token IDs of ``good_token`` and ``bad_token`` in the\n    ``math-shepherd-mistral-7b-prm`` model.\n    '

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        factors: list[Any] = []
        hash_str = hashlib.md5(str(factors).encode(), usedforsecurity=False).hexdigest()
        return hash_str
_STR_DTYPE_TO_TORCH_DTYPE = {'half': torch.float16, 'float16': torch.float16, 'float': torch.float32, 'float32': torch.float32, 'bfloat16': torch.bfloat16}
_FLOAT16_NOT_SUPPORTED_MODELS = {'gemma2': 'Numerical instability. Please use bfloat16 or float32 instead.', 'gemma3': 'Numerical instability. Please use bfloat16 or float32 instead.', 'plamo2': 'Numerical instability. Please use bfloat16 or float32 instead.', 'glm4': 'Numerical instability. Please use bfloat16 or float32 instead.'}

def _is_valid_dtype(model_type: str, dtype: torch.dtype):
    if model_type in _FLOAT16_NOT_SUPPORTED_MODELS and dtype == torch.float16:
        return False
    return True

def _check_valid_dtype(model_type: str, dtype: torch.dtype):
    if model_type in _FLOAT16_NOT_SUPPORTED_MODELS and dtype == torch.float16:
        reason = _FLOAT16_NOT_SUPPORTED_MODELS[model_type]
        raise ValueError(f'The model type {model_type!r} does not support float16. Reason: {reason}')
    return True

def _find_dtype(model_id: str, config: PretrainedConfig, *, revision: Optional[str]):
    config_dtype = getattr(config, 'torch_dtype', None)
    if config_dtype is None:
        config_dtype = getattr(config.get_text_config(), 'torch_dtype', None)
    if config_dtype is None and hasattr(config, 'vision_config'):
        config_dtype = getattr(config.vision_config, 'torch_dtype', None)
    if config_dtype is None and hasattr(config, 'encoder_config'):
        config_dtype = getattr(config.encoder_config, 'torch_dtype', None)
    if config_dtype is None:
        repo_mt = try_get_safetensors_metadata(model_id, revision=revision)
        if repo_mt and (files_mt := repo_mt.files_metadata):
            param_dtypes: set[torch.dtype] = {_SAFETENSORS_TO_TORCH_DTYPE[dtype_str] for file_mt in files_mt.values() for dtype_str in file_mt.parameter_count if dtype_str in _SAFETENSORS_TO_TORCH_DTYPE}
            if param_dtypes:
                return common_broadcastable_dtype(param_dtypes)
    if config_dtype is None:
        config_dtype = torch.float32
    return config_dtype

def _resolve_auto_dtype(model_type: str, config_dtype: torch.dtype, *, is_pooling_model: bool):
    supported_dtypes = [dtype for dtype in current_platform.supported_dtypes if _is_valid_dtype(model_type, dtype)]
    if is_pooling_model and torch.float16 in supported_dtypes:
        preferred_dtype = torch.float16
    else:
        preferred_dtype = supported_dtypes[0]
    if config_dtype == torch.float32:
        config_dtype = preferred_dtype
    if config_dtype in supported_dtypes:
        return config_dtype
    device_name = current_platform.get_device_name()
    device_capability = current_platform.get_device_capability()
    if device_capability is None:
        device_str = f'{device_name!r}'
    else:
        version_str = device_capability.as_version_str()
        device_str = f'{device_name!r} (with compute capability {version_str})'
    logger.warning("Your device %s doesn't support %s. Falling back to %s for compatibility.", device_str, config_dtype, preferred_dtype)
    return preferred_dtype

def _get_and_verify_dtype(model_id: str, config: PretrainedConfig, dtype: Union[str, torch.dtype], *, is_pooling_model: bool, revision: Optional[str]=None) -> torch.dtype:
    config_dtype = _find_dtype(model_id, config, revision=revision)
    model_type = config.model_type
    if isinstance(dtype, str):
        dtype = dtype.lower()
        if dtype == 'auto':
            torch_dtype = _resolve_auto_dtype(model_type, config_dtype, is_pooling_model=is_pooling_model)
        else:
            if dtype not in _STR_DTYPE_TO_TORCH_DTYPE:
                raise ValueError(f'Unknown dtype: {dtype!r}')
            torch_dtype = _STR_DTYPE_TO_TORCH_DTYPE[dtype]
    elif isinstance(dtype, torch.dtype):
        torch_dtype = dtype
    else:
        raise ValueError(f'Unknown dtype: {dtype}')
    _check_valid_dtype(model_type, torch_dtype)
    if torch_dtype != config_dtype:
        if torch_dtype == torch.float32:
            logger.info('Upcasting %s to %s.', config_dtype, torch_dtype)
        elif config_dtype == torch.float32:
            logger.info('Downcasting %s to %s.', config_dtype, torch_dtype)
        else:
            logger.warning('Casting %s to %s.', config_dtype, torch_dtype)
    return torch_dtype

def _get_and_verify_max_len(hf_config: PretrainedConfig, tokenizer_config: Optional[dict], max_model_len: Optional[int], disable_sliding_window: bool, sliding_window_len: Optional[Union[int, list[Optional[int]]]], spec_target_max_model_len: Optional[int]=None, encoder_config: Optional[Any]=None) -> int:
    """Get and verify the model's maximum length."""
    derived_max_model_len = float('inf')
    possible_keys = ['max_position_embeddings', 'n_positions', 'max_seq_len', 'seq_length', 'model_max_length', 'max_target_positions', 'max_sequence_length', 'max_seq_length', 'seq_len']
    max_len_key = None
    for key in possible_keys:
        max_len = getattr(hf_config, key, None)
        if max_len is not None:
            max_len_key = key if max_len < derived_max_model_len else max_len_key
            derived_max_model_len = min(derived_max_model_len, max_len)
    if (tmp_max_len := getattr(hf_config, 'model_max_length', None)):
        max_len_key = 'model_max_length'
        derived_max_model_len = tmp_max_len
    if disable_sliding_window and sliding_window_len is not None:
        sliding_window_len_min = get_min_sliding_window(sliding_window_len)
        max_len_key = 'sliding_window' if sliding_window_len_min < derived_max_model_len else max_len_key
        derived_max_model_len = min(derived_max_model_len, sliding_window_len_min)
    if tokenizer_config:
        tokenizer_model_max_length = tokenizer_config.get('model_max_length', derived_max_model_len)
        derived_max_model_len = min(derived_max_model_len, tokenizer_model_max_length)
    if derived_max_model_len == float('inf'):
        if max_model_len is not None:
            return max_model_len
        if spec_target_max_model_len is not None:
            return spec_target_max_model_len
        default_max_len = 2048
        logger.warning("The model's config.json does not contain any of the following keys to determine the original maximum length of the model: %s. Assuming the model's maximum length is %d.", possible_keys, default_max_len)
        derived_max_model_len = default_max_len
    rope_scaling = getattr(hf_config, 'rope_scaling', None)
    if rope_scaling is not None and 'gemma3' not in hf_config.model_type:
        rope_type = rope_scaling['rope_type']
        if rope_type not in ('su', 'longrope', 'llama3'):
            if disable_sliding_window:
                raise NotImplementedError('Disabling sliding window is not supported for models with rope_scaling. Please raise an issue so we can investigate.')
            scaling_factor = rope_scaling.get('factor', 1.0)
            if rope_type == 'yarn':
                derived_max_model_len = rope_scaling['original_max_position_embeddings']
            derived_max_model_len *= scaling_factor
    if encoder_config and 'max_seq_length' in encoder_config:
        derived_max_model_len = encoder_config['max_seq_length']
    if max_model_len is None:
        max_model_len = int(derived_max_model_len)
        if current_platform.is_tpu():
            logger.warning("--max-model-len is not specified, it's currently using model's default length %s, which might be too large.Please input with --max-model-len based on your request input length and output length, to avoid unnecessary degradation.", max_model_len)
    elif max_model_len > derived_max_model_len:
        model_max_length = getattr(hf_config, 'model_max_length', None)
        if model_max_length is not None and max_model_len <= model_max_length:
            if disable_sliding_window:
                raise NotImplementedError('Disabling sliding window is not supported for models model_max_length in the config. Please raise an issue so we can investigate.')
        else:
            msg = f"User-specified max_model_len ({max_model_len}) is greater than the derived max_model_len ({max_len_key}={derived_max_model_len} or model_max_length={model_max_length} in model's config.json). This may lead to incorrect model outputs or CUDA errors."
            if envs.VLLM_ALLOW_LONG_MAX_MODEL_LEN:
                logger.warning('%s Make sure the value is correct and within the model context size.', msg)
            else:
                raise ValueError(f'{msg} To allow overriding this maximum, set the env var VLLM_ALLOW_LONG_MAX_MODEL_LEN=1')
    return int(max_model_len)

def get_min_sliding_window(sliding_window: Union[int, list[Optional[int]]]) -> int:
    if isinstance(sliding_window, list):
        return min(s for s in sliding_window if s is not None)
    return sliding_window

def get_served_model_name(model: str, served_model_name: Optional[Union[str, list[str]]]):
    """
    If the input is a non-empty list, the first model_name in
    `served_model_name` is taken.
    If the input is a non-empty string, it is used directly.
    For cases where the input is either an empty string or an
    empty list, the fallback is to use `self.model`.
    """
    if not served_model_name:
        return model
    if isinstance(served_model_name, list):
        return served_model_name[0]
    return served_model_name
GuidedDecodingBackendV0 = Literal['auto', 'outlines', 'lm-format-enforcer', 'xgrammar', 'guidance']
GuidedDecodingBackendV1 = Literal['auto', 'xgrammar', 'guidance', 'outlines']
GuidedDecodingBackend = Literal[GuidedDecodingBackendV0, GuidedDecodingBackendV1]

@config
@dataclass
class DecodingConfig:
    """Dataclass which contains the decoding strategy of the engine."""
    backend: GuidedDecodingBackend = 'auto' if envs.VLLM_USE_V1 else 'xgrammar'
    'Which engine will be used for guided decoding (JSON schema / regex etc)\n    by default. With "auto", we will make opinionated choices based on request\n    contents and what the backend libraries currently support, so the behavior\n    is subject to change in each release.'
    disable_fallback: bool = False
    'If `True`, vLLM will not fallback to a different backend on error.'
    disable_any_whitespace: bool = False
    'If `True`, the model will not generate any whitespace during guided\n    decoding. This is only supported for xgrammar and guidance backends.'
    disable_additional_properties: bool = False
    'If `True`, the `guidance` backend will not use `additionalProperties`\n    in the JSON schema. This is only supported for the `guidance` backend and\n    is used to better align its behaviour with `outlines` and `xgrammar`.'
    reasoning_backend: str = ''
    "Select the reasoning parser depending on the model that you're using.\n    This is used to parse the reasoning content into OpenAI API format."

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        factors: list[Any] = []
        hash_str = hashlib.md5(str(factors).encode(), usedforsecurity=False).hexdigest()
        return hash_str

    def __post_init__(self):
        if envs.VLLM_USE_V1:
            valid_guided_backends = get_args(GuidedDecodingBackendV1)
        else:
            valid_guided_backends = get_args(GuidedDecodingBackendV0)
        if self.backend not in valid_guided_backends:
            raise ValueError(f"Invalid backend '{self.backend}', must be one of {valid_guided_backends}")
        if self.disable_any_whitespace and self.backend not in ('xgrammar', 'guidance'):
            raise ValueError('disable_any_whitespace is only supported for xgrammar and guidance backends.')
        if self.disable_additional_properties and self.backend != 'guidance':
            raise ValueError('disable_additional_properties is only supported for the guidance backend.')
DetailedTraceModules = Literal['model', 'worker', 'all']

@config
@dataclass
class ObservabilityConfig:
    """Configuration for observability - metrics and tracing."""
    show_hidden_metrics_for_version: Optional[str] = None
    'Enable deprecated Prometheus metrics that have been hidden since the\n    specified version. For example, if a previously deprecated metric has been\n    hidden since the v0.7.0 release, you use\n    `--show-hidden-metrics-for-version=0.7` as a temporary escape hatch while\n    you migrate to new metrics. The metric is likely to be removed completely\n    in an upcoming release.'

    @cached_property
    def show_hidden_metrics(self) -> bool:
        """Check if the hidden metrics should be shown."""
        if self.show_hidden_metrics_for_version is None:
            return False
        return version._prev_minor_version_was(self.show_hidden_metrics_for_version)
    otlp_traces_endpoint: Optional[str] = None
    'Target URL to which OpenTelemetry traces will be sent.'
    collect_detailed_traces: Optional[list[DetailedTraceModules]] = None
    'It makes sense to set this only if `--otlp-traces-endpoint` is set. If\n    set, it will collect detailed traces for the specified modules. This\n    involves use of possibly costly and or blocking operations and hence might\n    have a performance impact.\n\n    Note that collecting detailed timing information for each request can be\n    expensive.'

    @cached_property
    def collect_model_forward_time(self) -> bool:
        """Whether to collect model forward time for the request."""
        return self.collect_detailed_traces is not None and ('model' in self.collect_detailed_traces or 'all' in self.collect_detailed_traces)

    @cached_property
    def collect_model_execute_time(self) -> bool:
        """Whether to collect model execute time for the request."""
        return self.collect_detailed_traces is not None and ('worker' in self.collect_detailed_traces or 'all' in self.collect_detailed_traces)

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        factors: list[Any] = []
        hash_str = hashlib.md5(str(factors).encode(), usedforsecurity=False).hexdigest()
        return hash_str

    def __post_init__(self):
        if self.collect_detailed_traces is not None and len(self.collect_detailed_traces) == 1 and (',' in self.collect_detailed_traces[0]):
            self._parse_collect_detailed_traces()
        if not is_otel_available() and self.otlp_traces_endpoint is not None:
            raise ValueError(f"OpenTelemetry is not available. Unable to configure 'otlp_traces_endpoint'. Ensure OpenTelemetry packages are installed. Original error:\n{otel_import_error_traceback}")

    def _parse_collect_detailed_traces(self):
        assert isinstance(self.collect_detailed_traces, list)
        self.collect_detailed_traces = cast(list[DetailedTraceModules], self.collect_detailed_traces[0].split(','))
KVProducer = Literal['kv_producer', 'kv_both']
KVConsumer = Literal['kv_consumer', 'kv_both']
KVRole = Literal[KVProducer, KVConsumer]

@config
@dataclass
class KVTransferConfig:
    """Configuration for distributed KV cache transfer."""
    kv_connector: Optional[str] = None
    'The KV connector for vLLM to transmit KV caches between vLLM instances.\n    '
    engine_id: Optional[str] = None
    'The engine id for KV transfers.'
    kv_buffer_device: Optional[str] = 'cuda'
    "The device used by kv connector to buffer the KV cache.\n    Currently only support 'cuda'."
    kv_buffer_size: float = 1000000000.0
    'The buffer size for TorchDistributedConnector. Measured in number of\n    bytes. Recommended value: 1e9 (about 1GB).'
    kv_role: Optional[KVRole] = None
    "Whether this vLLM instance produces, consumes KV cache, or both. Choices\n    are 'kv_producer', 'kv_consumer', and 'kv_both'."
    kv_rank: Optional[int] = None
    'The rank of this vLLM instance in the KV cache transfer. Typical value:\n    0 for prefill instance, 1 for decode instance.\n    Currently only 1P1D is supported.'
    kv_parallel_size: int = 1
    'The number of parallel instances for KV cache transfer. For\n    PyNcclConnector, this should be 2.'
    kv_ip: str = '127.0.0.1'
    'The KV connector ip, used to build distributed connection.'
    kv_port: int = 14579
    'The KV connector port, used to build distributed connection.'
    kv_connector_extra_config: dict[str, Any] = field(default_factory=dict)
    'any extra config that the connector may need.'
    kv_connector_module_path: Optional[str] = None
    'The Python module path to dynamically load the KV connector from.\n    Only supported in V1.'

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        factors: list[Any] = []
        hash_str = hashlib.md5(str(factors).encode(), usedforsecurity=False).hexdigest()
        return hash_str

    def __post_init__(self) -> None:
        if self.engine_id is None:
            self.engine_id = str(uuid.uuid4())
        if self.kv_role is not None and self.kv_role not in get_args(KVRole):
            raise ValueError(f'Unsupported kv_role: {self.kv_role}. Supported roles are {get_args(KVRole)}')
        if self.kv_connector is not None and self.kv_role is None:
            raise ValueError(f'Please specify kv_disagg_role when kv_connector is set, supported roles are {get_args(KVRole)}')

    @property
    def is_kv_transfer_instance(self) -> bool:
        return self.kv_connector is not None and self.kv_role in get_args(KVRole)

    @property
    def is_kv_producer(self) -> bool:
        return self.kv_connector is not None and self.kv_role in get_args(KVProducer)

    @property
    def is_kv_consumer(self) -> bool:
        return self.kv_connector is not None and self.kv_role in get_args(KVConsumer)

    def get_from_extra_config(self, key, default) -> Any:
        return self.kv_connector_extra_config.get(key, default)

@config
@dataclass
class KVEventsConfig:
    """Configuration for KV event publishing."""
    enable_kv_cache_events: bool = False
    'If True, enable KV cache events for tracking block storage and removal.\n    Events can be published externally by zmq using the event publisher config.\n    '
    publisher: str = 'null'
    'The publisher to use for publishing kv events. Can be "null", "zmq".\n    '
    endpoint: str = 'tcp://*:5557'
    'The zmq endpoint to use for publishing kv events.\n    '
    replay_endpoint: Optional[str] = None
    'The zmq endpoint to use for replaying kv events.\n    '
    buffer_steps: int = 10000
    'The number of steps to cache for replay endpoint. Will only save\n    events from the last N steps for the replay endpoint.\n    '
    hwm: int = 100000
    'The zmq high water mark for the event publisher. After queueing N events,\n    events will start dropping if the consumer is not keeping up.\n    '
    max_queue_size: int = 100000
    'The maximum number of events to queue while waiting for publishing.\n    '
    topic: str = ''
    'The topic to use for the event publisher. Consumers can subscribe to\n    this topic to receive events.\n    '

class CompilationLevel:
    NO_COMPILATION = 0
    DYNAMO_AS_IS = 1
    DYNAMO_ONCE = 2
    PIECEWISE = 3

@config
@dataclass
class PassConfig:
    """Configuration for custom Inductor passes.

    This is separate from general `CompilationConfig` so that inductor passes
    don't all have access to full configuration - that would create a cycle as
    the `PassManager` is set as a property of config."""
    enable_fusion: bool = field(default_factory=lambda: not envs.VLLM_USE_V1)
    'Whether to enable the custom fusion (RMSNorm/SiluMul+quant) pass.'
    enable_attn_fusion: bool = False
    'Whether to enable the custom attention+quant fusion pass.'
    enable_noop: bool = field(default_factory=lambda: not envs.VLLM_USE_V1)
    'Whether to enable the custom no-op elimination pass.'
    enable_sequence_parallelism: bool = False
    'Whether to enable sequence parallelism.'
    enable_async_tp: bool = False
    'Whether to enable async TP.'
    enable_fi_allreduce_fusion: bool = False
    'Whether to enable flashinfer allreduce fusion.'
    fi_allreduce_fusion_max_token_num: int = 1024
    'Max number of tokens to used in flashinfer allreduce fusion.'

    def uuid(self):
        """
        Produces a hash unique to the pass configuration.
        Any new fields that affect compilation should be added to the hash.
        Any future fields that don't affect compilation should be excluded.
        """
        return InductorPass.hash_dict(asdict(self))

    def __post_init__(self) -> None:
        if not self.enable_noop:
            if self.enable_fusion:
                logger.warning_once('Fusion enabled but reshape elimination disabled. RMSNorm/SiluMul + quant (fp8) fusion might not work')
            if self.enable_attn_fusion:
                logger.warning_once('Fusion enabled but reshape elimination disabled. Attention + quant (fp8) fusion might not work')

@config
@dataclass
class CompilationConfig:
    """Configuration for compilation. It has three parts:

    - Top-level Compilation control:
        - [`level`][vllm.config.CompilationConfig.level]
        - [`debug_dump_path`][vllm.config.CompilationConfig.debug_dump_path]
        - [`cache_dir`][vllm.config.CompilationConfig.cache_dir]
        - [`backend`][vllm.config.CompilationConfig.backend]
        - [`custom_ops`][vllm.config.CompilationConfig.custom_ops]
        - [`splitting_ops`][vllm.config.CompilationConfig.splitting_ops]
    - CudaGraph capture:
        - [`use_cudagraph`][vllm.config.CompilationConfig.use_cudagraph]
        - [`cudagraph_capture_sizes`]
        [vllm.config.CompilationConfig.cudagraph_capture_sizes]
        - [`cudagraph_num_of_warmups`]
        [vllm.config.CompilationConfig.cudagraph_num_of_warmups]
        - [`cudagraph_copy_inputs`]
        [vllm.config.CompilationConfig.cudagraph_copy_inputs]
        - [`full_cuda_graph`][vllm.config.CompilationConfig.full_cuda_graph]
    - Inductor compilation:
        - [`use_inductor`][vllm.config.CompilationConfig.use_inductor]
        - [`compile_sizes`][vllm.config.CompilationConfig.compile_sizes]
        - [`inductor_compile_config`]
        [vllm.config.CompilationConfig.inductor_compile_config]
        - [`inductor_passes`][vllm.config.CompilationConfig.inductor_passes]
        - custom inductor passes

    Why we have different sizes for cudagraph and inductor:
    - cudagraph: a cudagraph captured for a specific size can only be used
        for the same size. We need to capture all the sizes we want to use.
    - inductor: a graph compiled by inductor for a general shape can be used
        for different sizes. Inductor can also compile for specific sizes,
        where it can have more information to optimize the graph with fully
        static shapes. However, we find the general shape compilation is
        sufficient for most cases. It might be beneficial to compile for
        certain small batchsizes, where inductor is good at optimizing.
    """
    level: int = 0
    'The level of compilation:\n\n    - 0: no compilation.\n    - 1: dynamo as is.\n    - 2: dynamo once.\n    - 3: piecewise compilation.'
    debug_dump_path: str = ''
    'The path to dump the debug information.'
    cache_dir: str = ''
    'The directory to store the compiled graph, to accelerate Inductor\n    compilation. By default, it will use model-related information to generate\n    a cache directory.'
    backend: str = ''
    'The backend for compilation. It needs to be a string:\n\n    - "" (empty string): use the default backend.\n    - "eager"/"openxla"/...: use the specified backend registered in PyTorch.\n    - "full.module.name": a qualified name which can be used to import the\n\n    backend function.\n    We use string to avoid serialization issues when using compilation in a\n    distributed setting. When the compilation level is 1 or 2, the backend is\n    used for the compilation directly (it sees the whole graph). When the\n    compilation level is 3, the backend is used for the piecewise compilation\n    (it sees a part of the graph).'
    custom_ops: list[str] = field(default_factory=list)
    "Fine-grained control over which custom ops to enable/disable. Use 'all'\n    to enable all, 'none' to disable all. Also specify a list of custom op\n    names to enable (prefixed with a '+'), or disable (prefixed with a '-').\n    Examples:\n\n    - 'all,-op1' to enable all except op1\n    - 'none,+op1,+op2' to enable only op1 and op2\n\n    By default, all custom ops are enabled when running without Inductor and\n    disabled when running with Inductor: level>=PIECEWISE and use_inductor=True.\n    Inductor generates (fused) Triton kernels for disabled custom ops."
    splitting_ops: list[str] = field(default_factory=list)
    'A list of ops to split the full graph into subgraphs, used in piecewise\n    compilation.'
    use_inductor: bool = True
    'Whether to use inductor compilation:\n\n    - False: inductor compilation is not used. graph runs in eager\n        (custom_ops enabled by default).\n    - True: inductor compilation is used (custom_ops disabled by default).\n        One graph for symbolic shape and one graph per size in compile_sizes\n        are compiled using configurations in inductor_compile_config.\n        \n    This setting is ignored if level<PIECEWISE.'
    compile_sizes: Optional[list[Union[int, str]]] = None
    'Sizes to compile for inductor. In addition\n    to integers, it also supports "cudagraph_capture_sizes" to\n    specify the sizes for cudagraph capture.'
    inductor_compile_config: dict = field(default_factory=dict)
    'Additional configurations for inductor.\n    - None: use default configurations.'
    inductor_passes: dict[str, str] = field(default_factory=dict)
    'Additional passes for inductor. It is a dictionary\n    from pass name to pass function qualified name. We use function\n    name because the config uses JSON format. If we pass the config\n    from Python, functions can also be passed directly via Python object\n    constructor, e.g. `CompilationConfig(inductor_passes={"a": func})`.'
    use_cudagraph: bool = field(default_factory=lambda: envs.VLLM_USE_V1)
    'Whether to use cudagraph inside compilation.\n    - False: cudagraph inside compilation is not used.\n    - True: cudagraph inside compilation is used. It requires\n        that all input buffers have fixed addresses, and all\n        splitting ops write their outputs to input buffers.\n    In the vLLM V1 Engine, this flag only applies for\n    CompilationLevel.PIECEWISE (aka -O3).\n    Note that this is orthogonal to the cudagraph capture logic\n    outside of compilation.\n    TODO: move outside cudagraph logic into compilation.\n    torch.compile will handle cudagraph capture logic in the future.'
    cudagraph_num_of_warmups: int = 0
    'Number of warmup runs for cudagraph.\n    It means the first several runs will be treated as warmup runs.\n    Only after that, the execution will be recorded, and the recorded\n    cudagraph will be used for subsequent runs.'
    cudagraph_capture_sizes: Optional[list[int]] = None
    'Sizes to capture cudagraph.\n    - None (default): capture sizes are inferred from vllm config.\n    - list[int]: capture sizes are specified as given.'
    cudagraph_copy_inputs: bool = False
    'Whether to copy input tensors for\n    cudagraph. If the caller can guarantee that the same input buffers\n    are always used, it can set this to False. Otherwise, it should\n    set this to True, and the compiler will copy the input to an\n    internally managed buffer. Default is False.'
    full_cuda_graph: bool = False
    'whether to use a full cuda graph for the entire forward pass rather than\n    splitting certain operations such as attention into subgraphs. Thus this\n    flag cannot be used together with splitting_ops. This may provide\n    performance benefits for smaller models.'
    pass_config: PassConfig = field(default_factory=PassConfig)
    'Custom inductor passes, see PassConfig for more details'
    max_capture_size: int = field(default=None, init=False)
    'not configurable, computed after init'
    local_cache_dir: str = field(default=None, init=False)
    'local cache dir for each rank'
    bs_to_padded_graph_size: list[int] = field(default=None, init=False)
    'optimization:\n    Intuitively, bs_to_padded_graph_size should be dict[int, int].\n    since we know all keys are in a range [0, max_capture_size],\n    we can optimize it to list[int] for better lookup performance.'
    enabled_custom_ops: Counter[str] = field(default_factory=Counter, init=False)
    'custom ops that are enabled'
    disabled_custom_ops: Counter[str] = field(default_factory=Counter, init=False)
    'custom ops that are disabled'
    traced_files: set[str] = field(default_factory=set, init=False)
    'files that are traced for compilation'
    compilation_time: float = field(default=0.0, init=False)
    'time taken for compilation'
    static_forward_context: dict[str, Any] = field(default_factory=dict, init=False)
    'Per-model forward context\n    Map from layer name to layer objects that need to be accessed outside\n    model code, e.g., Attention, FusedMOE when dp_size>1.'

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        factors: list[Any] = []
        factors.append(self.level)
        factors.append(self.backend)
        factors.append(self.custom_ops)
        factors.append(self.splitting_ops)
        factors.append(self.use_inductor)
        factors.append(self.inductor_compile_config)
        factors.append(self.inductor_passes)
        factors.append(self.pass_config.uuid())
        return hashlib.sha256(str(factors).encode()).hexdigest()

    def __repr__(self) -> str:
        exclude = {'static_forward_context': True, 'enabled_custom_ops': True, 'disabled_custom_ops': True, 'compilation_time': True, 'bs_to_padded_graph_size': True, 'pass_config': True, 'traced_files': True, 'inductor_compile_config': {'post_grad_custom_post_pass': True}}
        return str(TypeAdapter(CompilationConfig).dump_json(self, exclude=exclude, exclude_unset=True).decode())
    __str__ = __repr__

    @classmethod
    def from_cli(cls, cli_value: str) -> 'CompilationConfig':
        """Parse the CLI value for the compilation config.
        -O1, -O2, -O3, etc. is handled in FlexibleArgumentParser.
        """
        return TypeAdapter(CompilationConfig).validate_json(cli_value)

    def __post_init__(self) -> None:
        count_none = self.custom_ops.count('none')
        count_all = self.custom_ops.count('all')
        assert count_none + count_all <= 1, "Can only specify 'none' or 'all'"
        if is_torch_equal_or_newer('2.6'):
            KEY = 'enable_auto_functionalized_v2'
            if KEY not in self.inductor_compile_config:
                self.inductor_compile_config[KEY] = False
        for k, v in self.inductor_passes.items():
            if not isinstance(v, str):
                assert callable(v), f'pass {k} should be callable or a qualified name'
                self.inductor_compile_config[k] = v if isinstance(v, InductorPass) else CallableInductorPass(v)
                continue
            names = v.split('.')
            module = '.'.join(names[:-1])
            func_name = names[-1]
            func = __import__(module).__dict__[func_name]
            self.inductor_compile_config[k] = func if isinstance(func, InductorPass) else CallableInductorPass(func)
        if isinstance(self.pass_config, dict):
            self.pass_config = PassConfig(**self.pass_config)

    def init_backend(self, vllm_config: 'VllmConfig') -> Union[str, Callable]:
        if self.level == CompilationLevel.NO_COMPILATION:
            raise ValueError('No compilation level is set.')
        torch_backends = list_backends(exclude_tags=tuple())
        if self.level in [CompilationLevel.DYNAMO_AS_IS, CompilationLevel.DYNAMO_ONCE]:
            if self.backend == '':
                return 'eager'
            if self.backend in torch_backends:
                return self.backend
            return resolve_obj_by_qualname(self.backend)
        assert self.level == CompilationLevel.PIECEWISE
        return VllmBackend(vllm_config)

    def init_with_cudagraph_sizes(self, cudagraph_capture_sizes: list[int]) -> None:
        """To complete the initialization of config,
        we need to know the cudagraph sizes."""
        if self.cudagraph_capture_sizes is None:
            self.cudagraph_capture_sizes = cudagraph_capture_sizes
        else:
            dedup_sizes = list(set(self.cudagraph_capture_sizes))
            if len(dedup_sizes) < len(self.cudagraph_capture_sizes):
                logger.info('cudagraph sizes specified by model runner %s is overridden by config %s', cudagraph_capture_sizes, dedup_sizes)
            self.cudagraph_capture_sizes = dedup_sizes
        computed_compile_sizes = []
        if self.compile_sizes is not None:
            self.compile_sizes = list(set(self.compile_sizes))
            for x in self.compile_sizes:
                if isinstance(x, str):
                    assert x == 'cudagraph_capture_sizes', f"Unrecognized size type in compile_sizes, expect 'cudagraph_capture_sizes', got {x}"
                    computed_compile_sizes.extend(self.cudagraph_capture_sizes)
                else:
                    assert isinstance(x, int)
                    computed_compile_sizes.append(x)
        self.compile_sizes = computed_compile_sizes
        self.cudagraph_capture_sizes.sort(reverse=True)
        self.max_capture_size = self.cudagraph_capture_sizes[0] if self.cudagraph_capture_sizes else 0
        self.bs_to_padded_graph_size = [0 for i in range(self.max_capture_size + 1)]
        for end, start in zip(self.cudagraph_capture_sizes, self.cudagraph_capture_sizes[1:] + [0]):
            for bs in range(start, end):
                if bs == start:
                    self.bs_to_padded_graph_size[bs] = start
                else:
                    self.bs_to_padded_graph_size[bs] = end
        self.bs_to_padded_graph_size[self.max_capture_size] = self.max_capture_size

    def set_splitting_ops_for_v1(self):
        if self.splitting_ops and self.full_cuda_graph:
            raise ValueError(f'full_cuda_graph cannot be used together with splitting_ops, as Full CUDA graph will override the splitting_ops: {self.splitting_ops}')
        if not self.splitting_ops:
            self.splitting_ops = [] if self.full_cuda_graph else ['vllm.unified_attention', 'vllm.unified_attention_with_output', 'vllm.mamba_mixer2']

@config
@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class VllmConfig:
    """Dataclass which contains all vllm-related configuration. This
    simplifies passing around the distinct configurations in the codebase.
    """
    model_config: ModelConfig = None
    'Model configuration.'
    cache_config: CacheConfig = field(default_factory=CacheConfig)
    'Cache configuration.'
    parallel_config: ParallelConfig = field(default_factory=ParallelConfig)
    'Parallel configuration.'
    scheduler_config: SchedulerConfig = field(default_factory=SchedulerConfig)
    'Scheduler configuration.'
    device_config: DeviceConfig = field(default_factory=DeviceConfig)
    'Device configuration.'
    load_config: LoadConfig = field(default_factory=LoadConfig)
    'Load configuration.'
    lora_config: Optional[LoRAConfig] = None
    'LoRA configuration.'
    speculative_config: Optional[SpeculativeConfig] = None
    'Speculative decoding configuration.'
    decoding_config: DecodingConfig = field(default_factory=DecodingConfig)
    'Decoding configuration.'
    observability_config: Optional[ObservabilityConfig] = None
    'Observability configuration.'
    prompt_adapter_config: Optional[PromptAdapterConfig] = None
    'Prompt adapter configuration.'
    quant_config: Optional[QuantizationConfig] = None
    'Quantization configuration.'
    compilation_config: CompilationConfig = field(default_factory=CompilationConfig)
    '`torch.compile` and cudagraph capture configuration for the model.\n\n    As a shorthand, `-O<n>` can be used to directly specify the compilation\n    level `n`: `-O3` is equivalent to `-O.level=3` (same as `-O=\'{"level":3}\'`).\n    Currently, -O <n> and -O=<n> are supported as well but this will likely be \n    removed in favor of clearer -O<n> syntax in the future.\n\n    NOTE: level 0 is the default level without any optimization. level 1 and 2\n    are for internal testing only. level 3 is the recommended level for\n    production, also default in V1.\n\n    You can specify the full compilation config like so:\n    `{"level": 3, "cudagraph_capture_sizes": [1, 2, 4, 8]}`\n    '
    kv_transfer_config: Optional[KVTransferConfig] = None
    'The configurations for distributed KV cache transfer.'
    kv_events_config: Optional[KVEventsConfig] = None
    'The configurations for event publishing.'
    additional_config: Union[dict, SupportsHash] = field(default_factory=dict)
    'Additional config for specified platform. Different platforms may\n    support different configs. Make sure the configs are valid for the platform\n    you are using. Contents must be hashable.'
    instance_id: str = ''
    'The ID of the vLLM instance.'

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        factors: list[Any] = []
        vllm_factors: list[Any] = []
        vllm_factors.append(__version__)
        vllm_factors.append(envs.VLLM_USE_V1)
        if self.model_config:
            vllm_factors.append(self.model_config.compute_hash())
        else:
            vllm_factors.append('None')
        if self.cache_config:
            vllm_factors.append(self.cache_config.compute_hash())
        else:
            vllm_factors.append('None')
        if self.parallel_config:
            vllm_factors.append(self.parallel_config.compute_hash())
        else:
            vllm_factors.append('None')
        if self.scheduler_config:
            vllm_factors.append(self.scheduler_config.compute_hash())
        else:
            vllm_factors.append('None')
        if self.device_config:
            vllm_factors.append(self.device_config.compute_hash())
        else:
            vllm_factors.append('None')
        if self.load_config:
            vllm_factors.append(self.load_config.compute_hash())
        else:
            vllm_factors.append('None')
        if self.lora_config:
            vllm_factors.append(self.lora_config.compute_hash())
            vllm_factors.append(str(self.scheduler_config.max_num_batched_tokens))
        else:
            vllm_factors.append('None')
        if self.speculative_config:
            vllm_factors.append(self.speculative_config.compute_hash())
        else:
            vllm_factors.append('None')
        if self.decoding_config:
            vllm_factors.append(self.decoding_config.compute_hash())
        else:
            vllm_factors.append('None')
        if self.observability_config:
            vllm_factors.append(self.observability_config.compute_hash())
        else:
            vllm_factors.append('None')
        if self.prompt_adapter_config:
            vllm_factors.append(self.prompt_adapter_config.compute_hash())
        else:
            vllm_factors.append('None')
        if self.quant_config:
            pass
        if self.compilation_config:
            vllm_factors.append(self.compilation_config.compute_hash())
        else:
            vllm_factors.append('None')
        if self.kv_transfer_config:
            vllm_factors.append(self.kv_transfer_config.compute_hash())
        else:
            vllm_factors.append('None')
        if self.additional_config:
            if isinstance((additional_config := self.additional_config), dict):
                additional_config_hash = hashlib.md5(json.dumps(additional_config, sort_keys=True).encode(), usedforsecurity=False).hexdigest()
            else:
                additional_config_hash = additional_config.compute_hash()
            vllm_factors.append(additional_config_hash)
        else:
            vllm_factors.append('None')
        factors.append(vllm_factors)
        hash_str = hashlib.md5(str(factors).encode(), usedforsecurity=False).hexdigest()[:10]
        return hash_str

    def pad_for_cudagraph(self, batch_size: int) -> int:
        return self.compilation_config.bs_to_padded_graph_size[batch_size]

    @staticmethod
    def _get_quantization_config(model_config: ModelConfig, load_config: LoadConfig) -> Optional[QuantizationConfig]:
        """Get the quantization config."""
        if model_config.quantization is not None:
            from vllm.model_executor.model_loader.weight_utils import get_quant_config
            quant_config = get_quant_config(model_config, load_config)
            capability_tuple = current_platform.get_device_capability()
            if capability_tuple is not None:
                capability = capability_tuple.to_int()
                if capability < quant_config.get_min_capability():
                    raise ValueError(f'The quantization method {model_config.quantization} is not supported for the current GPU. Minimum capability: {quant_config.get_min_capability()}. Current capability: {capability}.')
            supported_dtypes = quant_config.get_supported_act_dtypes()
            if model_config.dtype not in supported_dtypes:
                raise ValueError(f'{model_config.dtype} is not supported for quantization method {model_config.quantization}. Supported dtypes: {supported_dtypes}')
            return quant_config
        return None

    @staticmethod
    def get_quantization_config(model_config: ModelConfig, load_config: LoadConfig) -> Optional[QuantizationConfig]:
        return VllmConfig._get_quantization_config(copy.deepcopy(model_config), load_config)

    def with_hf_config(self, hf_config: PretrainedConfig, architectures: Optional[list[str]]=None) -> 'VllmConfig':
        if architectures is not None:
            hf_config = copy.deepcopy(hf_config)
            hf_config.architectures = architectures
        model_config = copy.deepcopy(self.model_config)
        model_config.hf_config = hf_config
        return replace(self, model_config=model_config)

    def __post_init__(self):
        """Verify configs are valid & consistent with each other.
        """
        self.try_verify_and_update_config()
        if self.model_config is not None:
            self.model_config.verify_async_output_proc(self.parallel_config, self.speculative_config, self.device_config)
            self.model_config.verify_with_parallel_config(self.parallel_config)
            self.model_config.verify_dual_chunk_attention_config(self.load_config)
        self.cache_config.verify_with_parallel_config(self.parallel_config)
        if self.lora_config is not None:
            self.lora_config.verify_with_cache_config(self.cache_config)
            self.lora_config.verify_with_model_config(self.model_config)
        if self.prompt_adapter_config is not None:
            self.prompt_adapter_config.verify_with_model_config(self.model_config)
        if self.quant_config is None and self.model_config is not None:
            self.quant_config = VllmConfig._get_quantization_config(self.model_config, self.load_config)
        if self.model_config is not None and self.scheduler_config.chunked_prefill_enabled and (self.model_config.dtype == torch.float32) and (current_platform.get_device_capability() == (7, 5)):
            logger.warning_once("Turing devices tensor cores do not support float32 matmul. To workaround this limitation, vLLM will set 'ieee' input precision for chunked prefill triton kernels.")
        if self.compilation_config.pass_config.enable_async_tp:
            self.compilation_config.pass_config.enable_sequence_parallelism = True
        if self.compilation_config.pass_config.enable_sequence_parallelism:
            self.compilation_config.custom_ops.append('+rms_norm')
        if envs.VLLM_USE_V1 and self.model_config is not None and (not self.model_config.enforce_eager):
            self.compilation_config.cudagraph_num_of_warmups = 1
            self.compilation_config.level = CompilationLevel.PIECEWISE
            self.compilation_config.set_splitting_ops_for_v1()
        self._set_cudagraph_sizes()
        if self.cache_config.cpu_offload_gb > 0 and self.compilation_config.level != CompilationLevel.NO_COMPILATION and (not envs.VLLM_USE_V1):
            logger.warning('CPU offload is not supported with `torch.compile` in v0 yet. Disabling `torch.compile`.')
            self.compilation_config.level = CompilationLevel.NO_COMPILATION
        if not envs.VLLM_USE_V1 and self.lora_config is not None and (self.compilation_config.level != CompilationLevel.NO_COMPILATION):
            logger.warning('LoRA for V0 is not supported with `torch.compile` yet. Disabling `torch.compile`.')
            self.compilation_config.level = CompilationLevel.NO_COMPILATION
        if self.compilation_config.full_cuda_graph and (not self.model_config.disable_cascade_attn):
            logger.info('full_cuda_graph is not supported with cascade attention. Disabling cascade attention.')
            self.model_config.disable_cascade_attn = True
        disable_chunked_prefill_reasons: list[str] = []
        if self.model_config and self.model_config.pooler_config:
            pooling_type = self.model_config.pooler_config.pooling_type
            if pooling_type is None or pooling_type.lower() != 'last':
                disable_chunked_prefill_reasons.append('Only "last" pooling supports chunked prefill and prefix caching; disabling both.')
        if disable_chunked_prefill_reasons:
            for reason in disable_chunked_prefill_reasons:
                logger.info(reason)
            self.scheduler_config.chunked_prefill_enabled = False
            self.scheduler_config.long_prefill_token_threshold = 0
            self.scheduler_config.max_num_batched_tokens = max(self.scheduler_config.max_model_len, DEFAULT_MAX_NUM_BATCHED_TOKENS)
            if self.cache_config is not None:
                self.cache_config.enable_prefix_caching = False
        if self.kv_events_config is not None and self.kv_events_config.enable_kv_cache_events and (not self.cache_config.enable_prefix_caching):
            logger.warning('KV cache events are on, but prefix caching is not enabled.Use --enable-prefix-caching to enable.')
        if self.kv_events_config is not None and self.kv_events_config.publisher != 'null' and (not self.kv_events_config.enable_kv_cache_events):
            logger.warning('KV cache events are disabled,but the scheduler is configured to publish them.Modify KVEventsConfig.enable_kv_cache_eventsto True to enable.')
        current_platform.check_and_update_config(self)
        if not self.instance_id:
            self.instance_id = random_uuid()[:5]
        if envs.VLLM_USE_V1 and (not self.scheduler_config.disable_hybrid_kv_cache_manager):
            if not (current_platform.is_cuda() or current_platform.is_rocm()):
                self.scheduler_config.disable_hybrid_kv_cache_manager = True
            if self.kv_transfer_config is not None:
                self.scheduler_config.disable_hybrid_kv_cache_manager = True
            if self.kv_events_config is not None:
                self.scheduler_config.disable_hybrid_kv_cache_manager = True
            if self.model_config is not None and self.model_config.attention_chunk_size is not None and (self.speculative_config is not None) and self.speculative_config.use_eagle():
                self.scheduler_config.disable_hybrid_kv_cache_manager = True

    def update_sizes_for_sequence_parallelism(self, possible_sizes: list) -> list:
        removed_sizes = [size for size in possible_sizes if size % self.parallel_config.tensor_parallel_size != 0]
        if removed_sizes:
            logger.warning('Batch sizes %s are removed because they are not multiple of tp_size %d when sequence parallelism is enabled', removed_sizes, self.parallel_config.tensor_parallel_size)
        return [size for size in possible_sizes if size % self.parallel_config.tensor_parallel_size == 0]

    def _set_cudagraph_sizes(self):
        """
        cudagraph batchsize padding logic:

        `[1, 2, 4] + [8 * i for i in range(1, 1025)]` is a list of all possible
        batch sizes that cudagraph will capture.

        Depending on the engine's configuration of `max_num_seqs`, the
        candidate batch sizes to capture cudagraph will shrink to the subset
        which just cover the range of `[1, max_num_seqs]`. In the common case,
        `max_num_seqs` is 256, and the cudagraph batch sizes will be
        `[1, 2, 4, 8, 16, 24, 32, 40, ..., 256]`.

        However, if users specify the cudagraph capture sizes through
        compilation config, we will use the specified sizes instead.

        In the end, `vllm_config.compilation_config.cudagraph_capture_sizes`
        will be the final sizes to capture cudagraph (in descending order).

        During runtime, if batchsize is larger than
        `vllm_config.compilation_config.cudagraph_capture_sizes`,
        no cudagraph will be used.
        If the batch size is no larger than
        `vllm_config.compilation_config.cudagraph_capture_sizes`,
        we can quickly find the padded graph size for a given batch size by
        looking up `vllm_config.compilation_config.bs_to_padded_graph_size`.
        """
        if not envs.VLLM_USE_V1:
            batch_size_capture_list = []
            if self.scheduler_config is not None and self.model_config is not None and (not self.model_config.enforce_eager):
                possible_sizes = [1, 2, 4] + [8 * i for i in range(1, 1025)]
                if self.parallel_config.tensor_parallel_size > 1 and self.compilation_config.pass_config.enable_sequence_parallelism:
                    possible_sizes = self.update_sizes_for_sequence_parallelism(possible_sizes)
                larger_sizes = [x for x in possible_sizes if x >= self.scheduler_config.max_num_seqs]
                if larger_sizes:
                    max_batchsize_to_capture = larger_sizes[0]
                else:
                    max_batchsize_to_capture = possible_sizes[-1]
                batch_size_capture_list = [size for size in possible_sizes if size <= max_batchsize_to_capture]
        else:
            batch_size_capture_list = []
            if self.model_config is not None and (not self.model_config.enforce_eager):
                cuda_graph_sizes = self.scheduler_config.cuda_graph_sizes
                if len(cuda_graph_sizes) == 1:
                    batch_size_capture_list = [1, 2, 4] + [i for i in range(8, cuda_graph_sizes[0] + 1, 8)]
                elif len(cuda_graph_sizes) > 1:
                    batch_size_capture_list = sorted(cuda_graph_sizes)
                else:
                    raise TypeError(f'Invalid value for cuda_graph_sizes={cuda_graph_sizes!r}.')
                if self.parallel_config.tensor_parallel_size > 1 and self.compilation_config.pass_config.enable_sequence_parallelism:
                    batch_size_capture_list = self.update_sizes_for_sequence_parallelism(batch_size_capture_list)
                max_num_tokens = self.scheduler_config.max_num_batched_tokens
                batch_size_capture_list = [size for size in batch_size_capture_list if size <= max_num_tokens]
        self.compilation_config.init_with_cudagraph_sizes(batch_size_capture_list)

    def recalculate_max_model_len(self, max_model_len: int):
        model_config = self.model_config
        max_model_len = model_config.get_and_verify_max_len(max_model_len)
        self.model_config.max_model_len = max_model_len
        self.scheduler_config.max_model_len = max_model_len

    def try_verify_and_update_config(self):
        architecture = getattr(self.model_config, 'architecture', None)
        if architecture is None:
            return
        cls = MODELS_CONFIG_MAP.get(architecture, None)
        if cls is not None:
            cls.verify_and_update_config(self)
        if self.model_config.is_hybrid:
            HybridAttentionMambaModelConfig.verify_and_update_config(self)
        if self.model_config.task == 'classify':
            from vllm.model_executor.models.adapters import SequenceClassificationConfig
            SequenceClassificationConfig.verify_and_update_config(self)

    def __str__(self):
        return f'model={self.model_config.model!r}, speculative_config={self.speculative_config!r}, tokenizer={self.model_config.tokenizer!r}, skip_tokenizer_init={self.model_config.skip_tokenizer_init}, tokenizer_mode={self.model_config.tokenizer_mode}, revision={self.model_config.revision}, override_neuron_config={self.model_config.override_neuron_config}, tokenizer_revision={self.model_config.tokenizer_revision}, trust_remote_code={self.model_config.trust_remote_code}, dtype={self.model_config.dtype}, max_seq_len={self.model_config.max_model_len}, download_dir={self.load_config.download_dir!r}, load_format={self.load_config.load_format}, tensor_parallel_size={self.parallel_config.tensor_parallel_size}, pipeline_parallel_size={self.parallel_config.pipeline_parallel_size}, disable_custom_all_reduce={self.parallel_config.disable_custom_all_reduce}, quantization={self.model_config.quantization}, enforce_eager={self.model_config.enforce_eager}, kv_cache_dtype={self.cache_config.cache_dtype},  device_config={self.device_config.device}, decoding_config={self.decoding_config!r}, observability_config={self.observability_config!r}, seed={self.model_config.seed}, served_model_name={self.model_config.served_model_name}, num_scheduler_steps={self.scheduler_config.num_scheduler_steps}, multi_step_stream_outputs={self.scheduler_config.multi_step_stream_outputs}, enable_prefix_caching={self.cache_config.enable_prefix_caching}, chunked_prefill_enabled={self.scheduler_config.chunked_prefill_enabled}, use_async_output_proc={self.model_config.use_async_output_proc}, pooler_config={self.model_config.pooler_config!r}, compilation_config={self.compilation_config!r}'
_current_vllm_config: Optional[VllmConfig] = None
_current_prefix: Optional[str] = None

@contextmanager
def set_current_vllm_config(vllm_config: VllmConfig, check_compile=False, prefix: Optional[str]=None):
    """
    Temporarily set the current vLLM config.
    Used during model initialization.
    We save the current vLLM config in a global variable,
    so that all modules can access it, e.g. custom ops
    can access the vLLM config to determine how to dispatch.
    """
    global _current_vllm_config, _current_prefix
    old_vllm_config = _current_vllm_config
    old_prefix = _current_prefix
    num_models_seen = compilation_counter.num_models_seen
    try:
        _current_vllm_config = vllm_config
        _current_prefix = prefix
        yield
    except Exception:
        raise
    else:
        logger.debug('enabled custom ops: %s', vllm_config.compilation_config.enabled_custom_ops)
        logger.debug('disabled custom ops: %s', vllm_config.compilation_config.disabled_custom_ops)
        if check_compile and vllm_config.compilation_config.level == CompilationLevel.PIECEWISE and (compilation_counter.num_models_seen == num_models_seen):
            logger.warning('`torch.compile` is turned on, but the model %s does not support it. Please open an issue on GitHub if you want it to be supported.', vllm_config.model_config.model)
    finally:
        _current_vllm_config = old_vllm_config
        _current_prefix = old_prefix

def get_current_vllm_config() -> VllmConfig:
    if _current_vllm_config is None:
        logger.warning('Current vLLM config is not set.')
        from vllm.config import VllmConfig
        return VllmConfig()
    return _current_vllm_config

def get_current_model_prefix() -> str:
    """
    Get the prefix of the model that's currently being initialized.
    """
    assert _current_prefix is not None, 'Current model prefix is not set. '
    return _current_prefix

def contains_object_print(text):
    """
    Check if the text looks like a printed Python object, e.g.
    contains any substring matching the pattern: "at 0xFFFFFFF>"
    We match against 0x followed by 2-16 hex chars (there's
    a max of 16 on a 64 bit system).

    Args:
        text (str): The text to check

    Returns:
        result (bool): `True` if a match is found, `False` otherwise.
    """
    pattern = 'at 0x[a-fA-F0-9]{2,16}>'
    match = re.search(pattern, text)
    return match is not None

def assert_hashable(text):
    if not contains_object_print(text):
        return True
    raise AssertionError(f'vLLM tried to hash some configs that may have Python objects ids in them. This is a bug, please file an issue. Text being hashed: {text}')
T = TypeVar('T')

def get_layers_from_vllm_config(vllm_config: VllmConfig, layer_type: type[T]) -> dict[str, T]:
    return {layer_name: layer for layer_name, layer in vllm_config.compilation_config.static_forward_context.items() if isinstance(layer, layer_type)}

@config
@dataclass
class SpeechToTextConfig:
    """Configuration for speech-to-text models."""
    sample_rate: float = 16000
    'Sample rate (Hz) to resample input audio to. Most speech models expect\n    16kHz audio input. The input audio will be automatically resampled to this\n    rate before processing.'
    max_audio_clip_s: int = 30
    'Maximum duration in seconds for a single audio clip without chunking.\n    Audio longer than this will be split into smaller chunks if\n    `allow_audio_chunking` evaluates to True, otherwise it will be rejected.'
    overlap_chunk_second: int = 1
    'Overlap duration in seconds between consecutive audio chunks when\n    splitting long audio. This helps maintain context across chunk boundaries\n    and improves transcription quality at split points.'
    min_energy_split_window_size: Optional[int] = 1600
    'Window size in samples for finding low-energy (quiet) regions to split\n    audio chunks. The algorithm looks for the quietest moment within this\n    window to minimize cutting through speech. Default 1600 samples ≈ 100ms\n    at 16kHz. If None, no chunking will be done.'

    @property
    def allow_audio_chunking(self) -> bool:
        return self.min_energy_split_window_size is not None

def update_config(config: DataclassInstanceT, overrides: dict[str, Any]) -> DataclassInstanceT:
    processed_overrides = {}
    for field_name, value in overrides.items():
        assert hasattr(config, field_name), f'{type(config)} has no field `{field_name}`'
        current_value = getattr(config, field_name)
        if is_dataclass(current_value) and (not is_dataclass(value)):
            assert isinstance(value, dict), f'Overrides to {type(config)}.{field_name} must be a dict  or {type(current_value)}, but got {type(value)}'
            value = update_config(current_value, value)
        processed_overrides[field_name] = value
    return replace(config, **processed_overrides)