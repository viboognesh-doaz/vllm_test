from typing import Optional
from vllm.compilation.wrapper import TorchCompileWrapperWithCustomDispatcher
from vllm.config import CompilationLevel
import torch

class MyMod(torch.nn.Module):

    def forward(self, x: torch.Tensor, cache: Optional[torch.Tensor]=None):
        if cache is not None:
            return x + cache
        return x * 2

class MyWrapper(TorchCompileWrapperWithCustomDispatcher):

    def __init__(self, model):
        self.model = model
        compiled_callable = torch.compile(self.forward, backend='eager')
        super().__init__(compiled_callable, compilation_level=CompilationLevel.DYNAMO_ONCE)

    def forward(self, x: torch.Tensor, cache: Optional[torch.Tensor]=None):
        return self.model(x, cache)

    def __call__(self, x: torch.Tensor, cache: Optional[torch.Tensor]=None):
        if len(self.compiled_codes) == 2:
            dispatch_id = 0 if cache is None else 1
            with self.dispatch_to_code(dispatch_id):
                return self.forward(x, cache)
        else:
            return self.compiled_callable(x, cache)

def test_torch_compile_wrapper():
    mod = MyMod()
    wrappers = []
    for i in range(3):
        torch._dynamo.reset()
        wrapper = MyWrapper(mod)
        wrappers.append(wrapper)
        x = torch.tensor([1])
        wrapper(x, None)
        cache = torch.tensor([2])
        wrapper(x, cache)
        new_x = torch.tensor([3])
        assert wrapper(new_x, None).item() == 6
        assert wrapper(new_x, cache).item() == 5
    for wrapper in wrappers:
        assert len(wrapper.compiled_codes) == 2