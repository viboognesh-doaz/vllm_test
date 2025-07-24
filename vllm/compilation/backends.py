from .monitor import torch_compile_start_time
from torch._dynamo.utils import lazy_format_graph_code
from torch._guards import detect_fake_mode
from torch.fx.experimental.symbolic_shapes import is_symbolic
import ast
import dataclasses
import os
import pprint
import time
from collections.abc import Sequence
from contextlib import contextmanager
from typing import Any, Callable, Optional
import torch
import torch.fx as fx
from torch._dispatch.python import enable_python_dispatcher
import vllm.envs as envs
from vllm.config import CompilationConfig, VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils import is_torch_equal_or_newer, resolve_obj_by_qualname
from .compiler_interface import CompilerInterface, EagerAdaptor, InductorAdaptor, InductorStandaloneAdaptor
from .counter import compilation_counter
from .inductor_pass import InductorPass
from .pass_manager import PostGradPassManager
logger = init_logger(__name__)

def make_compiler(compilation_config: CompilationConfig) -> CompilerInterface:
    if compilation_config.use_inductor:
        if envs.VLLM_USE_STANDALONE_COMPILE and is_torch_equal_or_newer('2.8.0.dev'):
            logger.debug('Using InductorStandaloneAdaptor')
            return InductorStandaloneAdaptor()
        else:
            logger.debug('Using InductorAdaptor')
            return InductorAdaptor()
    else:
        logger.debug('Using EagerAdaptor')
        return EagerAdaptor()

class CompilerManager:
    """
    A manager to manage the compilation process, including
    caching the compiled graph, loading the compiled graph,
    and compiling the graph.

    The cache is a dict mapping
    `(runtime_shape, graph_index, backend_name)`
    to `any_data` returned from the compiler.

    When serializing the cache, we save it to a Python file
    for readability. We don't use json here because json doesn't
    support int as key.
    """

    def __init__(self, compilation_config: CompilationConfig):
        self.cache: dict[tuple[Optional[int], int, str], Any] = dict()
        self.is_cache_updated = False
        self.compilation_config = compilation_config
        self.compiler = make_compiler(compilation_config)

    def compute_hash(self, vllm_config: VllmConfig) -> str:
        return self.compiler.compute_hash(vllm_config)

    def initialize_cache(self, cache_dir: str, disable_cache: bool=False, prefix: str=''):
        """
        Initialize the cache directory for the compiler.

        The organization of the cache directory is as follows:
        cache_dir=/path/to/hash_str/rank_i_j/prefix/
        inside cache_dir, there will be:
        - vllm_compile_cache.py
        - computation_graph.py
        - transformed_code.py

        for multiple prefixes, they can share the same
        base cache dir of /path/to/hash_str/rank_i_j/ ,
        to store some common compilation artifacts.
        """
        self.disable_cache = disable_cache
        self.cache_dir = cache_dir
        self.cache_file_path = os.path.join(cache_dir, 'vllm_compile_cache.py')
        if not disable_cache and os.path.exists(self.cache_file_path):
            with open(self.cache_file_path) as f:
                self.cache = ast.literal_eval(f.read())
        self.compiler.initialize_cache(cache_dir=cache_dir, disable_cache=disable_cache, prefix=prefix)

    def save_to_file(self):
        if self.disable_cache or not self.is_cache_updated:
            return
        printer = pprint.PrettyPrinter(indent=4)
        data = printer.pformat(self.cache)
        with open(self.cache_file_path, 'w') as f:
            f.write(data)

    def load(self, graph: fx.GraphModule, example_inputs: list[Any], graph_index: int, runtime_shape: Optional[int]=None) -> Optional[Callable]:
        if (runtime_shape, graph_index, self.compiler.name) not in self.cache:
            return None
        handle = self.cache[runtime_shape, graph_index, self.compiler.name]
        compiled_graph = self.compiler.load(handle, graph, example_inputs, graph_index, runtime_shape)
        if runtime_shape is None:
            logger.debug('Directly load the %s-th graph for dynamic shape from %s via handle %s', graph_index, self.compiler.name, handle)
        else:
            logger.debug('Directly load the %s-th graph for shape %s from %s via handle %s', graph_index, str(runtime_shape), self.compiler.name, handle)
        return compiled_graph

    def compile(self, graph: fx.GraphModule, example_inputs, additional_inductor_config, compilation_config: CompilationConfig, graph_index: int=0, num_graphs: int=1, runtime_shape: Optional[int]=None) -> Any:
        if graph_index == 0:
            global compilation_start_time
            compilation_start_time = time.time()
        compilation_counter.num_backend_compilations += 1
        compiled_graph = None
        compiled_graph = self.load(graph, example_inputs, graph_index, runtime_shape)
        if compiled_graph is not None:
            if graph_index == num_graphs - 1:
                now = time.time()
                elapsed = now - compilation_start_time
                if runtime_shape is None:
                    logger.info('Directly load the compiled graph(s) for dynamic shape from the cache, took %.3f s', elapsed)
                else:
                    logger.info('Directly load the compiled graph(s) for shape %s from the cache, took %.3f s', str(runtime_shape), elapsed)
            return compiled_graph
        if isinstance(self.compiler, InductorAdaptor):
            maybe_key = None
        else:
            maybe_key = f'artifact_shape_{runtime_shape}_subgraph_{graph_index}'
        compiled_graph, handle = self.compiler.compile(graph, example_inputs, additional_inductor_config, runtime_shape, maybe_key)
        assert compiled_graph is not None, 'Failed to compile the graph'
        if not envs.VLLM_DISABLE_COMPILE_CACHE and handle is not None:
            self.cache[runtime_shape, graph_index, self.compiler.name] = handle
            compilation_counter.num_cache_entries_updated += 1
            self.is_cache_updated = True
            if graph_index == 0:
                if runtime_shape is None:
                    logger.info('Cache the graph for dynamic shape for later use')
                else:
                    logger.info('Cache the graph of shape %s for later use', str(runtime_shape))
            if runtime_shape is None:
                logger.debug('Store the %s-th graph for dynamic shape from %s via handle %s', graph_index, self.compiler.name, handle)
            else:
                logger.debug('Store the %s-th graph for shape %s from %s via handle %s', graph_index, str(runtime_shape), self.compiler.name, handle)
        if graph_index == num_graphs - 1:
            now = time.time()
            elapsed = now - compilation_start_time
            compilation_config.compilation_time += elapsed
            if runtime_shape is None:
                logger.info('Compiling a graph for dynamic shape takes %.2f s', elapsed)
            else:
                logger.info('Compiling a graph for shape %s takes %.2f s', runtime_shape, elapsed)
        return compiled_graph

@dataclasses.dataclass
class SplitItem:
    submod_name: str
    graph_id: int
    is_splitting_graph: bool
    graph: fx.GraphModule

def split_graph(graph: fx.GraphModule, ops: list[str]) -> tuple[fx.GraphModule, list[SplitItem]]:
    subgraph_id = 0
    node_to_subgraph_id = {}
    split_op_graphs = []
    for node in graph.graph.nodes:
        if node.op in ('output', 'placeholder'):
            continue
        if node.op == 'call_function' and str(node.target) in ops:
            subgraph_id += 1
            node_to_subgraph_id[node] = subgraph_id
            split_op_graphs.append(subgraph_id)
            subgraph_id += 1
        else:
            node_to_subgraph_id[node] = subgraph_id
    split_gm = torch.fx.passes.split_module.split_module(graph, None, lambda node: node_to_subgraph_id[node], keep_original_order=True)
    outputs = []
    names = [name for name, module in split_gm.named_modules()]
    for name in names:
        if '.' in name or name == '':
            continue
        module = getattr(split_gm, name)
        graph_id = int(name.replace('submod_', ''))
        outputs.append(SplitItem(name, graph_id, graph_id in split_op_graphs, module))
    outputs.sort(key=lambda x: x.graph_id)
    return (split_gm, outputs)
global_graph_pool = None
compilation_start_time = 0.0

class PiecewiseCompileInterpreter(torch.fx.Interpreter):
    """Code adapted from `torch.fx.passes.shape_prop.ShapeProp`.
    It runs the given graph with fake inputs, and compile some
    submodules specified by `compile_submod_names` with the given
    compilation configs.

    NOTE: the order in `compile_submod_names` matters, because
    it will be used to determine the order of the compiled piecewise
    graphs. The first graph will handle logging, and the last graph
    has some special cudagraph output handling.
    """

    def __init__(self, module: torch.fx.GraphModule, compile_submod_names: list[str], vllm_config: VllmConfig, graph_pool, vllm_backend: 'VllmBackend'):
        super().__init__(module)
        self.fake_mode = detect_fake_mode()
        self.compile_submod_names = compile_submod_names
        self.compilation_config = vllm_config.compilation_config
        self.graph_pool = graph_pool
        self.vllm_config = vllm_config
        self.vllm_backend = vllm_backend
        self.extra_traceback = False

    def run(self, *args):
        fake_args = [self.fake_mode.from_tensor(t) if isinstance(t, torch.Tensor) else t for t in args]
        with self.fake_mode, enable_python_dispatcher():
            return super().run(*fake_args)

    def call_module(self, target: torch.fx.node.Target, args: tuple[torch.fx.node.Argument, ...], kwargs: dict[str, Any]) -> Any:
        assert isinstance(target, str)
        output = super().call_module(target, args, kwargs)
        if target in self.compile_submod_names:
            index = self.compile_submod_names.index(target)
            submod = self.fetch_attr(target)
            sym_shape_indices = [i for i, x in enumerate(args) if isinstance(x, torch.SymInt)]
            global compilation_start_time
            compiled_graph_for_dynamic_shape = self.vllm_backend.compiler_manager.compile(submod, args, self.compilation_config.inductor_compile_config, self.compilation_config, graph_index=index, num_graphs=len(self.compile_submod_names), runtime_shape=None)
            piecewise_backend = resolve_obj_by_qualname(current_platform.get_piecewise_backend_cls())
            self.module.__dict__[target] = piecewise_backend(submod, self.vllm_config, self.graph_pool, index, len(self.compile_submod_names), sym_shape_indices, compiled_graph_for_dynamic_shape, self.vllm_backend)
            compilation_counter.num_piecewise_capturable_graphs_seen += 1
        return output
model_tag: str = 'backbone'

@contextmanager
def set_model_tag(tag: str):
    """Context manager to set the model tag."""
    global model_tag
    assert tag != model_tag, f'Model tag {tag} is the same as the current tag {model_tag}.'
    old_tag = model_tag
    model_tag = tag
    try:
        yield
    finally:
        model_tag = old_tag

class VllmBackend:
    """The compilation backend for `torch.compile` with vLLM.
    It is used for compilation level of `CompilationLevel.PIECEWISE`,
    where we customize the compilation.

    The major work of this backend is to split the graph into
    piecewise graphs, and pass them to the piecewise backend.

    This backend also adds the PostGradPassManager to Inductor config,
    which handles the post-grad passes.
    """
    vllm_config: VllmConfig
    compilation_config: CompilationConfig
    graph_pool: Any
    _called: bool = False
    graph: fx.GraphModule
    split_gm: fx.GraphModule
    piecewise_graphs: list[SplitItem]
    returned_callable: Callable
    post_grad_passes: Sequence[Callable]
    sym_tensor_indices: list[int]
    input_buffers: list[torch.Tensor]
    compiler_manager: CompilerManager

    def __init__(self, vllm_config: VllmConfig, prefix: str=''):
        self.prefix = prefix or model_tag
        global global_graph_pool
        if global_graph_pool is None:
            global_graph_pool = current_platform.graph_pool_handle()
        self.graph_pool = global_graph_pool
        self.post_grad_pass_manager = PostGradPassManager()
        self.sym_tensor_indices = []
        self.input_buffers = []
        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config
        self.compiler_manager: CompilerManager = CompilerManager(self.compilation_config)

    def configure_post_pass(self):
        config = self.compilation_config
        self.post_grad_pass_manager.configure(self.vllm_config)
        inductor_config = config.inductor_compile_config
        PASS_KEY = 'post_grad_custom_post_pass'
        if PASS_KEY in inductor_config:
            if isinstance(inductor_config[PASS_KEY], PostGradPassManager):
                assert inductor_config[PASS_KEY].uuid() == self.post_grad_pass_manager.uuid()
            else:
                assert isinstance(inductor_config[PASS_KEY], InductorPass)
                self.post_grad_pass_manager.add(inductor_config[PASS_KEY])
        inductor_config[PASS_KEY] = self.post_grad_pass_manager

    def __call__(self, graph: fx.GraphModule, example_inputs) -> Callable:
        vllm_config = self.vllm_config
        if not self.compilation_config.cache_dir:
            factors = []
            env_hash = envs.compute_hash()
            factors.append(env_hash)
            config_hash = vllm_config.compute_hash()
            factors.append(config_hash)
            forward_code_files = list(sorted(self.compilation_config.traced_files))
            self.compilation_config.traced_files.clear()
            logger.debug('Traced files (to be considered for compilation cache):\n%s', '\n'.join(forward_code_files))
            hash_content = []
            for filepath in forward_code_files:
                hash_content.append(filepath)
                if filepath == '<string>':
                    continue
                with open(filepath) as f:
                    hash_content.append(f.read())
            import hashlib
            code_hash = hashlib.md5('\n'.join(hash_content).encode(), usedforsecurity=False).hexdigest()
            factors.append(code_hash)
            compiler_hash = self.compiler_manager.compute_hash(vllm_config)
            factors.append(compiler_hash)
            hash_key = hashlib.md5(str(factors).encode(), usedforsecurity=False).hexdigest()[:10]
            cache_dir = os.path.join(envs.VLLM_CACHE_ROOT, 'torch_compile_cache', hash_key)
            self.compilation_config.cache_dir = cache_dir
        cache_dir = self.compilation_config.cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.compilation_config.cache_dir = cache_dir
        rank = vllm_config.parallel_config.rank
        dp_rank = vllm_config.parallel_config.data_parallel_rank
        local_cache_dir = os.path.join(cache_dir, f'rank_{rank}_{dp_rank}', self.prefix)
        os.makedirs(local_cache_dir, exist_ok=True)
        self.compilation_config.local_cache_dir = local_cache_dir
        disable_cache = envs.VLLM_DISABLE_COMPILE_CACHE
        if disable_cache:
            logger.info("vLLM's torch.compile cache is disabled.")
        else:
            logger.info("Using cache directory: %s for vLLM's torch.compile", local_cache_dir)
        self.compiler_manager.initialize_cache(local_cache_dir, disable_cache, self.prefix)
        compilation_counter.num_graphs_seen += 1
        dynamo_time = time.time() - torch_compile_start_time
        logger.info('Dynamo bytecode transform time: %.2f s', dynamo_time)
        self.compilation_config.compilation_time += dynamo_time
        assert not self._called, 'VllmBackend can only be called once'
        self.graph = graph
        self.configure_post_pass()
        self.split_gm, self.piecewise_graphs = split_graph(graph, self.compilation_config.splitting_ops)
        lazy_format_graph_code('before split', self.graph)
        lazy_format_graph_code('after split', self.split_gm)
        compilation_counter.num_piecewise_graphs_seen += len(self.piecewise_graphs)
        submod_names_to_compile = [item.submod_name for item in self.piecewise_graphs if not item.is_splitting_graph]
        PiecewiseCompileInterpreter(self.split_gm, submod_names_to_compile, self.vllm_config, self.graph_pool, self).run(*example_inputs)
        graph_path = os.path.join(local_cache_dir, 'computation_graph.py')
        if not os.path.exists(graph_path):
            src = 'from __future__ import annotations\nimport torch\n' + self.split_gm.print_readable(print_output=False)
            src = src.replace('<lambda>', 'GraphModule')
            with open(graph_path, 'w') as f:
                f.write(src)
            logger.debug('Computation graph saved to %s', graph_path)
        self._called = True
        if not self.compilation_config.use_cudagraph or not self.compilation_config.cudagraph_copy_inputs:
            return self.split_gm
        fake_mode = detect_fake_mode()
        fake_args = [fake_mode.from_tensor(t) if isinstance(t, torch.Tensor) else t for t in example_inputs]
        self.sym_tensor_indices = [i for i, x in enumerate(fake_args) if isinstance(x, torch._subclasses.fake_tensor.FakeTensor) and any(is_symbolic(d) for d in x.size())]
        self.input_buffers = [example_inputs[x].clone() for x in self.sym_tensor_indices]

        def copy_and_call(*args):
            list_args = list(args)
            for i, index in enumerate(self.sym_tensor_indices):
                runtime_tensor = list_args[index]
                runtime_shape = runtime_tensor.shape[0]
                static_tensor = self.input_buffers[i][:runtime_shape]
                static_tensor.copy_(runtime_tensor)
                list_args[index] = static_tensor
            return self.split_gm(*list_args)
        return copy_and_call