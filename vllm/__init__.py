from .version import __version__, __version_tuple__
from importlib import import_module
from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.llm_engine import LLMEngine
from vllm.entrypoints.llm import LLM
from vllm.executor.ray_utils import initialize_ray_cluster
from vllm.inputs import PromptType, TextPrompt, TokensPrompt
from vllm.model_executor.models import ModelRegistry
from vllm.outputs import ClassificationOutput, ClassificationRequestOutput, CompletionOutput, EmbeddingOutput, EmbeddingRequestOutput, PoolingOutput, PoolingRequestOutput, RequestOutput, ScoringOutput, ScoringRequestOutput
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
import typing
import vllm.env_override
'vLLM: a high-throughput and memory-efficient inference engine for LLMs'
MODULE_ATTRS = {'AsyncEngineArgs': '.engine.arg_utils:AsyncEngineArgs', 'EngineArgs': '.engine.arg_utils:EngineArgs', 'AsyncLLMEngine': '.engine.async_llm_engine:AsyncLLMEngine', 'LLMEngine': '.engine.llm_engine:LLMEngine', 'LLM': '.entrypoints.llm:LLM', 'initialize_ray_cluster': '.executor.ray_utils:initialize_ray_cluster', 'PromptType': '.inputs:PromptType', 'TextPrompt': '.inputs:TextPrompt', 'TokensPrompt': '.inputs:TokensPrompt', 'ModelRegistry': '.model_executor.models:ModelRegistry', 'SamplingParams': '.sampling_params:SamplingParams', 'PoolingParams': '.pooling_params:PoolingParams', 'ClassificationOutput': '.outputs:ClassificationOutput', 'ClassificationRequestOutput': '.outputs:ClassificationRequestOutput', 'CompletionOutput': '.outputs:CompletionOutput', 'EmbeddingOutput': '.outputs:EmbeddingOutput', 'EmbeddingRequestOutput': '.outputs:EmbeddingRequestOutput', 'PoolingOutput': '.outputs:PoolingOutput', 'PoolingRequestOutput': '.outputs:PoolingRequestOutput', 'RequestOutput': '.outputs:RequestOutput', 'ScoringOutput': '.outputs:ScoringOutput', 'ScoringRequestOutput': '.outputs:ScoringRequestOutput'}
if typing.TYPE_CHECKING:
else:

    def __getattr__(name: str) -> typing.Any:
        if name in MODULE_ATTRS:
            module_name, attr_name = MODULE_ATTRS[name].split(':')
            module = import_module(module_name, __package__)
            return getattr(module, attr_name)
        else:
            raise AttributeError(f'module {__package__} has no attribute {name}')
__all__ = ['__version__', '__version_tuple__', 'LLM', 'ModelRegistry', 'PromptType', 'TextPrompt', 'TokensPrompt', 'SamplingParams', 'RequestOutput', 'CompletionOutput', 'PoolingOutput', 'PoolingRequestOutput', 'EmbeddingOutput', 'EmbeddingRequestOutput', 'ClassificationOutput', 'ClassificationRequestOutput', 'ScoringOutput', 'ScoringRequestOutput', 'LLMEngine', 'EngineArgs', 'AsyncLLMEngine', 'AsyncEngineArgs', 'initialize_ray_cluster', 'PoolingParams']