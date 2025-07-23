from .monitor import start_monitoring_torch_compile
from torch._dynamo.symbolic_convert import InliningInstructionTranslator
from typing import Callable, Optional, TypeVar, Union, overload
from unittest.mock import patch
from vllm.compilation.counter import compilation_counter
from vllm.compilation.wrapper import TorchCompileWrapperWithCustomDispatcher
from vllm.config import CompilationLevel, VllmConfig
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors
from vllm.utils import supports_dynamo
import inspect
import torch
import torch.nn as nn
logger = init_logger(__name__)
_T = TypeVar('_T', bound=type[nn.Module])

@overload
def support_torch_compile(*, dynamic_arg_dims: Optional[dict[str, Union[int, list[int]]]]) -> Callable[[_T], _T]:
    ...

@overload
def support_torch_compile(cls: _T) -> _T:
    ...

def support_torch_compile(cls: Optional[_T]=None, *, dynamic_arg_dims: Optional[dict[str, Union[int, list[int]]]]=None) -> Union[Callable[[_T], _T], _T]:
    """
    A decorator to add support for compiling the forward method of a class.

    Usage 1: use directly as a decorator without arguments:

    ```python
    @support_torch_compile
    class MyModel(nn.Module):
        def forward(self, x: torch.Tensor, y: Optional[torch.Tensor]):
            ...
    ```

    Usage 2: use as a decorator with arguments:

    ```python
    @support_torch_compile(dynamic_arg_dims={"x": 0, "y": 0})
    class MyModel(nn.Module):
        def forward(self, x: torch.Tensor, y: Optional[torch.Tensor]):
            ...
    ```

    `dynamic_arg_dims` is a dictionary that maps argument names to the dynamic
    dimensions of the argument. The dynamic dimensions can be either a single
    integer or a list of integers.

    if `dynamic_arg_dims` is `None`, it is inferred from the type annotation
    of the `forward` method, based on the following default rules:

    - if the argument is annotated as `torch.Tensor` or
        `Optional[torch.Tensor]`, the first dimension will be
        marked as dynamic.
    - if the argument is annotated as `IntermediateTensors`, the first
        dimension of all the tensors in the intermediate tensors
        will be marked as dynamic.

    During runtime, when we actually mark dimensions of tensors,
     it depends on the value of arguments:

    - if it is a single integer (can be negative), the corresponding dimension 
        of the argument will be marked as dynamic.
    - if it is `None`, ignored.
    - if it is `IntermediateTensors`, all the tensors in the intermediate
        tensors will be marked as dynamic.
    - otherwise, it will raise an error.

    NOTE: if an argument is `None`, it should always be passed as `None` during
    the lifetime of the model, otherwise, it cannot be captured as a single
    computation graph.
    """

    def cls_decorator_helper(cls: _T) -> _T:
        if not hasattr(cls, 'forward'):
            raise TypeError('decorated class should have a forward method.')
        sig = inspect.signature(cls.forward)
        inferred_dynamic_arg_dims = dynamic_arg_dims
        if inferred_dynamic_arg_dims is None:
            inferred_dynamic_arg_dims = {}
            for k, v in sig.parameters.items():
                if v.annotation in [torch.Tensor, Optional[torch.Tensor], IntermediateTensors, Optional[IntermediateTensors]]:
                    inferred_dynamic_arg_dims[k] = 0
            logger.debug('Inferred dynamic dimensions for forward method of %s: %s', cls, list(inferred_dynamic_arg_dims.keys()))
        if len(inferred_dynamic_arg_dims) == 0:
            raise ValueError(f'No dynamic dimensions found in the forward method of {cls}. Please provide dynamic_arg_dims explicitly.')
        for k in inferred_dynamic_arg_dims:
            if k not in sig.parameters:
                raise ValueError(f'Argument {k} not found in the forward method of {cls}')
        return _support_torch_compile(cls, inferred_dynamic_arg_dims)
    if cls is not None:
        assert isinstance(cls, type)
        return cls_decorator_helper(cls)
    return cls_decorator_helper

def _support_torch_compile(cls: _T, dynamic_arg_dims: dict[str, Union[int, list[int]]]) -> _T:
    """
    A decorator to add support for compiling the forward method of a class.
    """
    if TorchCompileWrapperWithCustomDispatcher in cls.__bases__:
        return cls
    cls.__bases__ = cls.__bases__ + (TorchCompileWrapperWithCustomDispatcher,)
    old_init = cls.__init__

    def __init__(self, *, vllm_config: VllmConfig, prefix: str='', **kwargs):
        old_init(self, vllm_config=vllm_config, prefix=prefix, **kwargs)
        self.vllm_config = vllm_config
        self.do_not_compile = vllm_config.compilation_config.level in [CompilationLevel.NO_COMPILATION, CompilationLevel.DYNAMO_AS_IS] or not supports_dynamo()
        if self.do_not_compile:
            return
        compilation_counter.num_models_seen += 1
        TorchCompileWrapperWithCustomDispatcher.__init__(self, compilation_level=vllm_config.compilation_config.level)
    cls.__init__ = __init__

    def __call__(self, *args, **kwargs):
        if self.do_not_compile or torch.compiler.is_compiling():
            return self.forward(*args, **kwargs)
        if len(self.compiled_codes) < 1:
            sig = inspect.signature(self.__class__.forward)
            bound_args = sig.bind(self, *args, **kwargs)
            bound_args.apply_defaults()
            for k, dims in dynamic_arg_dims.items():
                arg = bound_args.arguments.get(k)
                if arg is not None:
                    dims = [dims] if isinstance(dims, int) else dims
                    if isinstance(arg, torch.Tensor):
                        dims = [arg.ndim + dim if dim < 0 else dim for dim in dims]
                        torch._dynamo.mark_dynamic(arg, dims)
                    elif isinstance(arg, IntermediateTensors):
                        for tensor in arg.tensors.values():
                            dims = [tensor.ndim + dim if dim < 0 else dim for dim in dims]
                            torch._dynamo.mark_dynamic(tensor, dims)
                    else:
                        raise ValueError(f'Unsupported dynamic dimensions {dims} for argument {k} with type {type(arg)}.')
            start_monitoring_torch_compile(self.vllm_config)
            logger.debug('Start compiling function %s', self.original_code_object)
        if len(self.compiled_codes) < 1 or not self.use_custom_dispatcher:
            torch._dynamo.eval_frame.remove_from_cache(self.original_code_object)
            self.vllm_config.compilation_config.traced_files.add(self.original_code_object.co_filename)
            inline_call = InliningInstructionTranslator.inline_call

            def patched_inline_call(parent, func, args, kwargs):
                code = func.get_code()
                self.vllm_config.compilation_config.traced_files.add(code.co_filename)
                return inline_call(parent, func, args, kwargs)
            with patch.object(InliningInstructionTranslator, 'inline_call', patched_inline_call):
                output = self.compiled_callable(*args, **kwargs)
            return output
        with self.dispatch_to_code(0):
            model_output = self.forward(*args, **kwargs)
            return model_output
    cls.__call__ = __call__
    return cls