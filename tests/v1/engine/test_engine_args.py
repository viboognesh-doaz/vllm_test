from vllm.platforms import current_platform
from argparse import ArgumentError
import pytest
from vllm import envs
from vllm.config import VllmConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.usage.usage_lib import UsageContext
from vllm.utils import FlexibleArgumentParser
if not envs.VLLM_USE_V1:
    pytest.skip('Skipping V1 tests. Rerun with `VLLM_USE_V1=1` to test.', allow_module_level=True)

def test_prefix_caching_from_cli():
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    args = parser.parse_args([])
    vllm_config = EngineArgs.from_cli_args(args=args).create_engine_config()
    assert vllm_config.cache_config.enable_prefix_caching, 'V1 turns on prefix caching by default.'
    args = parser.parse_args(['--no-enable-prefix-caching'])
    vllm_config = EngineArgs.from_cli_args(args=args).create_engine_config()
    assert not vllm_config.cache_config.enable_prefix_caching
    args = parser.parse_args(['--enable-prefix-caching'])
    vllm_config = EngineArgs.from_cli_args(args=args).create_engine_config()
    assert vllm_config.cache_config.enable_prefix_caching
    assert vllm_config.cache_config.prefix_caching_hash_algo == 'builtin'
    args = parser.parse_args(['--prefix-caching-hash-algo', 'sha256'])
    vllm_config = EngineArgs.from_cli_args(args=args).create_engine_config()
    assert vllm_config.cache_config.prefix_caching_hash_algo == 'sha256'
    args = parser.parse_args(['--prefix-caching-hash-algo', 'builtin'])
    vllm_config = EngineArgs.from_cli_args(args=args).create_engine_config()
    assert vllm_config.cache_config.prefix_caching_hash_algo == 'builtin'
    parser.exit_on_error = False
    with pytest.raises(ArgumentError):
        args = parser.parse_args(['--prefix-caching-hash-algo', 'invalid'])

def test_defaults_with_usage_context():
    engine_args = EngineArgs(model='facebook/opt-125m')
    vllm_config: VllmConfig = engine_args.create_engine_config(UsageContext.LLM_CLASS)
    device_name = current_platform.get_device_name().lower()
    if 'h100' in device_name or 'h200' in device_name:
        default_llm_tokens = 16384
        default_server_tokens = 8192
        default_max_num_seqs = 1024
    else:
        default_llm_tokens = 8192
        default_server_tokens = 2048
        default_max_num_seqs = 256
    assert vllm_config.scheduler_config.max_num_seqs == default_max_num_seqs
    assert vllm_config.scheduler_config.max_num_batched_tokens == default_llm_tokens
    engine_args = EngineArgs(model='facebook/opt-125m')
    vllm_config = engine_args.create_engine_config(UsageContext.OPENAI_API_SERVER)
    assert vllm_config.scheduler_config.max_num_seqs == default_max_num_seqs
    assert vllm_config.scheduler_config.max_num_batched_tokens == default_server_tokens