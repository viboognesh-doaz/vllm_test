from vllm.compilation.inductor_pass import CallableInductorPass, InductorPass
from vllm.compilation.pass_manager import PostGradPassManager
from vllm.config import VllmConfig
import copy
import pytest
import torch

def simple_callable(graph: torch.fx.Graph):
    pass

def test_bad_callable():
    config = VllmConfig()
    pass_manager = PostGradPassManager()
    pass_manager.configure(config)
    with pytest.raises(AssertionError):
        pass_manager.add(simple_callable)

class ProperPass(InductorPass):

    def __call__(self, graph: torch.fx.graph.Graph) -> None:
        pass

@pytest.mark.parametrize('callable', [ProperPass(), CallableInductorPass(simple_callable), CallableInductorPass(simple_callable, InductorPass.hash_source(__file__))])
def test_pass_manager_uuid(callable):
    config = VllmConfig()
    pass_manager = PostGradPassManager()
    pass_manager.configure(config)
    pass_manager.add(callable)
    uuid1 = pass_manager.uuid()
    pass_manager.add(callable)
    uuid2 = pass_manager.uuid()
    assert uuid1 != uuid2
    pass_manager2 = PostGradPassManager()
    pass_manager2.configure(config)
    pass_manager2.add(callable)
    assert uuid1 == pass_manager2.uuid()
    config2 = copy.deepcopy(config)
    config2.compilation_config.pass_config.enable_fusion = not config2.compilation_config.pass_config.enable_fusion
    pass_manager3 = PostGradPassManager()
    pass_manager3.configure(config2)
    pass_manager3.add(callable)
    assert uuid1 != pass_manager3.uuid()