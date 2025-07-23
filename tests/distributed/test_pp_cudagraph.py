from __future__ import annotations
from ..utils import compare_two_settings, create_new_process_for_each_test
from typing import TYPE_CHECKING
from typing_extensions import LiteralString
import pytest
if TYPE_CHECKING:

@pytest.mark.parametrize('PP_SIZE, MODEL_NAME', [(2, 'JackFram/llama-160m')])
@pytest.mark.parametrize('ATTN_BACKEND', ['FLASH_ATTN', 'FLASHINFER'])
@create_new_process_for_each_test()
def test_pp_cudagraph(monkeypatch: pytest.MonkeyPatch, PP_SIZE: int, MODEL_NAME: str, ATTN_BACKEND: LiteralString):
    with monkeypatch.context() as m:
        cudagraph_args = ['--dtype', 'float16', '--pipeline-parallel-size', str(PP_SIZE), '--distributed-executor-backend', 'mp']
        m.setenv('VLLM_ATTENTION_BACKEND', ATTN_BACKEND)
        eager_args = cudagraph_args + ['--enforce-eager']
        compare_two_settings(MODEL_NAME, eager_args, cudagraph_args)