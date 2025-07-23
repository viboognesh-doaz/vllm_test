from ..utils import compare_two_settings
from vllm.config import CompilationLevel
import pytest

def test_custom_dispatcher(monkeypatch: pytest.MonkeyPatch):
    with monkeypatch.context() as m:
        m.setenv('VLLM_RPC_TIMEOUT', '30000')
        compare_two_settings('Qwen/Qwen2.5-1.5B-Instruct', arg1=['--max-model-len=256', '--max-num-seqs=32', '--enforce-eager', f'-O{CompilationLevel.DYNAMO_ONCE}'], arg2=['--max-model-len=256', '--max-num-seqs=32', '--enforce-eager', f'-O{CompilationLevel.DYNAMO_AS_IS}'], env1={}, env2={})