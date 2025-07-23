from contextlib import ExitStack
from typing import Any, Callable, Optional
from unittest.mock import patch
from vllm.compilation.backends import VllmBackend
from vllm.compilation.counter import compilation_counter
from vllm.compilation.monitor import end_monitoring_torch_compile
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.utils import weak_ref_tensors
import dataclasses
import torch
import torch.fx as fx
import vllm.envs as envs
logger = init_logger(__name__)

@dataclasses.dataclass
class ConcreteSizeEntry:
    runtime_shape: int
    need_to_compile: bool
    use_cudagraph: bool
    compiled: bool = False
    runnable: Callable = None
    num_finished_warmup: int = 0
    cudagraph: Optional[torch.cuda.CUDAGraph] = None
    output: Optional[Any] = None
    input_addresses: Optional[list[int]] = None

class CUDAPiecewiseBackend:

    def __init__(self, graph: fx.GraphModule, vllm_config: VllmConfig, graph_pool: Any, piecewise_compile_index: int, total_piecewise_compiles: int, sym_shape_indices: list[int], compiled_graph_for_general_shape: Callable, vllm_backend: VllmBackend):
        """
        The backend for piecewise compilation.
        It mainly handles the compilation and cudagraph capturing.

        We will compile `self.graph` once for the general shape,
        and then compile for different shapes specified in
        `compilation_config.compile_sizes`.

        Independently, we will capture cudagraph for different shapes.

        If a shape needs both compilation and cudagraph, we will
        compile it first, and then capture cudagraph.
        """
        self.graph = graph
        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config
        self.graph_pool = graph_pool
        self.piecewise_compile_index = piecewise_compile_index
        self.total_piecewise_compiles = total_piecewise_compiles
        self.vllm_backend = vllm_backend
        self.is_first_graph = piecewise_compile_index == 0
        self.is_last_graph = piecewise_compile_index == total_piecewise_compiles - 1
        self.compile_sizes: set[int] = set(self.compilation_config.compile_sizes)
        self.cudagraph_capture_sizes: set[int] = set(self.compilation_config.cudagraph_capture_sizes) if self.compilation_config.use_cudagraph else set()
        self.first_run_finished = False
        self.compiled_graph_for_general_shape = compiled_graph_for_general_shape
        self.sym_shape_indices = sym_shape_indices
        self.is_debugging_mode = envs.VLLM_LOGGING_LEVEL == 'DEBUG'
        self.concrete_size_entries: dict[int, ConcreteSizeEntry] = {}
        self.to_be_compiled_sizes: set[int] = self.compile_sizes.copy()
        for shape in self.compile_sizes.union(self.cudagraph_capture_sizes):
            self.concrete_size_entries[shape] = ConcreteSizeEntry(runtime_shape=shape, need_to_compile=shape in self.compile_sizes, use_cudagraph=shape in self.cudagraph_capture_sizes)

    def check_for_ending_compilation(self):
        if self.is_last_graph and (not self.to_be_compiled_sizes):
            self.vllm_backend.compiler_manager.save_to_file()
            end_monitoring_torch_compile(self.vllm_config)

    def __call__(self, *args) -> Any:
        if not self.first_run_finished:
            self.first_run_finished = True
            self.check_for_ending_compilation()
            return self.compiled_graph_for_general_shape(*args)
        runtime_shape = args[self.sym_shape_indices[0]]
        if runtime_shape not in self.concrete_size_entries:
            return self.compiled_graph_for_general_shape(*args)
        entry = self.concrete_size_entries[runtime_shape]
        if entry.runnable is None:
            entry.runnable = self.compiled_graph_for_general_shape
        if entry.need_to_compile and (not entry.compiled):
            entry.compiled = True
            self.to_be_compiled_sizes.remove(runtime_shape)
            entry.runnable = self.vllm_backend.compiler_manager.compile(self.graph, args, self.compilation_config.inductor_compile_config, self.compilation_config, graph_index=self.piecewise_compile_index, num_graphs=self.total_piecewise_compiles, runtime_shape=runtime_shape)
            if self.is_last_graph and (not self.to_be_compiled_sizes):
                self.check_for_ending_compilation()
        skip_cuda_graphs = get_forward_context().skip_cuda_graphs
        if not entry.use_cudagraph or skip_cuda_graphs:
            return entry.runnable(*args)
        if entry.cudagraph is None:
            if entry.num_finished_warmup < self.compilation_config.cudagraph_num_of_warmups:
                entry.num_finished_warmup += 1
                if self.is_first_graph:
                    logger.debug('Warming up %s/%s for shape %s', entry.num_finished_warmup, self.compilation_config.cudagraph_num_of_warmups, runtime_shape)
                return entry.runnable(*args)
            if self.is_first_graph:
                logger.debug('Capturing a cudagraph for shape %s', runtime_shape)
            input_addresses = [x.data_ptr() for x in args if isinstance(x, torch.Tensor)]
            entry.input_addresses = input_addresses
            cudagraph = torch.cuda.CUDAGraph()
            with ExitStack() as stack:
                if not self.is_first_graph:
                    stack.enter_context(patch('gc.collect', lambda: None))
                    stack.enter_context(patch('torch.cuda.empty_cache', lambda: None))
                with torch.cuda.graph(cudagraph, pool=self.graph_pool):
                    output = entry.runnable(*args)
                    if self.is_last_graph:
                        output = weak_ref_tensors(output)
            entry.output = weak_ref_tensors(output)
            entry.cudagraph = cudagraph
            compilation_counter.num_cudagraph_captured += 1
            return output
        if self.is_debugging_mode:
            new_input_addresses = [x.data_ptr() for x in args if isinstance(x, torch.Tensor)]
            assert new_input_addresses == entry.input_addresses, f'Input addresses for cudagraphs are different during replay. Expected {entry.input_addresses}, got {new_input_addresses}'
        entry.cudagraph.replay()
        return entry.output