from contextlib import contextmanager
import copy
import dataclasses

@dataclasses.dataclass
class CompilationCounter:
    num_models_seen: int = 0
    num_graphs_seen: int = 0
    num_piecewise_graphs_seen: int = 0
    num_piecewise_capturable_graphs_seen: int = 0
    num_backend_compilations: int = 0
    num_gpu_runner_capture_triggers: int = 0
    num_cudagraph_captured: int = 0
    num_inductor_compiles: int = 0
    num_eager_compiles: int = 0
    num_cache_entries_updated: int = 0
    num_compiled_artifacts_saved: int = 0

    def clone(self) -> 'CompilationCounter':
        return copy.deepcopy(self)

    @contextmanager
    def expect(self, **kwargs):
        old = self.clone()
        yield
        for k, v in kwargs.items():
            assert getattr(self, k) - getattr(old, k) == v, f'{k} not as expected, before it is {getattr(old, k)}, after it is {getattr(self, k)}, expected diff is {v}'
compilation_counter = CompilationCounter()