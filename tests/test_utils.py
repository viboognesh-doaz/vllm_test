from .utils import create_new_process_for_each_test, error_on_warning
from collections.abc import AsyncIterator
from transformers import AutoTokenizer
from unittest.mock import patch
from vllm.attention import Attention
from vllm.attention import Attention
from vllm.attention import Attention
from vllm.attention import Attention
from vllm.attention import Attention, AttentionType
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed.device_communicators.cuda_wrapper import CudaRTLibrary
from vllm.transformers_utils.detokenizer_utils import convert_ids_list_to_tokens
from vllm.utils import CacheInfo, FlexibleArgumentParser, LRUCache, MemorySnapshot, PlaceholderModule, StoreBoolean, bind_kv_cache, common_broadcastable_dtype, deprecate_kwargs, get_open_port, get_tcp_uri, is_lossless_cast, join_host_port, make_zmq_path, make_zmq_socket, memory_profiling, merge_async_iterators, sha256, split_host_port, split_zmq_path, supports_kw, swap_dict_values
from vllm_test_utils.monitor import monitor
import asyncio
import hashlib
import json
import logging
import pickle
import pytest
import socket
import torch
import zmq

@pytest.mark.asyncio
async def test_merge_async_iterators():

    async def mock_async_iterator(idx: int):
        try:
            while True:
                yield f'item from iterator {idx}'
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            print(f'iterator {idx} cancelled')
    iterators = [mock_async_iterator(i) for i in range(3)]
    merged_iterator = merge_async_iterators(*iterators)

    async def stream_output(generator: AsyncIterator[tuple[int, str]]):
        async for idx, output in generator:
            print(f'idx: {idx}, output: {output}')
    task = asyncio.create_task(stream_output(merged_iterator))
    await asyncio.sleep(0.5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    for iterator in iterators:
        try:
            await asyncio.wait_for(iterator.__anext__(), 1)
        except StopAsyncIteration:
            print('Iterator was cancelled normally')
        except (Exception, asyncio.CancelledError) as e:
            raise AssertionError() from e

def test_deprecate_kwargs_always():

    @deprecate_kwargs('old_arg', is_deprecated=True)
    def dummy(*, old_arg: object=None, new_arg: object=None):
        pass
    with pytest.warns(DeprecationWarning, match="'old_arg'"):
        dummy(old_arg=1)
    with error_on_warning(DeprecationWarning):
        dummy(new_arg=1)

def test_deprecate_kwargs_never():

    @deprecate_kwargs('old_arg', is_deprecated=False)
    def dummy(*, old_arg: object=None, new_arg: object=None):
        pass
    with error_on_warning(DeprecationWarning):
        dummy(old_arg=1)
    with error_on_warning(DeprecationWarning):
        dummy(new_arg=1)

def test_deprecate_kwargs_dynamic():
    is_deprecated = True

    @deprecate_kwargs('old_arg', is_deprecated=lambda: is_deprecated)
    def dummy(*, old_arg: object=None, new_arg: object=None):
        pass
    with pytest.warns(DeprecationWarning, match="'old_arg'"):
        dummy(old_arg=1)
    with error_on_warning(DeprecationWarning):
        dummy(new_arg=1)
    is_deprecated = False
    with error_on_warning(DeprecationWarning):
        dummy(old_arg=1)
    with error_on_warning(DeprecationWarning):
        dummy(new_arg=1)

def test_deprecate_kwargs_additional_message():

    @deprecate_kwargs('old_arg', is_deprecated=True, additional_message='abcd')
    def dummy(*, old_arg: object=None, new_arg: object=None):
        pass
    with pytest.warns(DeprecationWarning, match='abcd'):
        dummy(old_arg=1)

def test_get_open_port(monkeypatch: pytest.MonkeyPatch):
    with monkeypatch.context() as m:
        m.setenv('VLLM_PORT', '5678')
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s1:
            s1.bind(('localhost', get_open_port()))
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s2:
                s2.bind(('localhost', get_open_port()))
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s3:
                    s3.bind(('localhost', get_open_port()))

@pytest.fixture
def parser():
    parser = FlexibleArgumentParser()
    parser.add_argument('--image-input-type', choices=['pixel_values', 'image_features'])
    parser.add_argument('--model-name')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--enable-feature', action='store_true')
    parser.add_argument('--hf-overrides', type=json.loads)
    parser.add_argument('-O', '--compilation-config', type=json.loads)
    return parser

@pytest.fixture
def parser_with_config():
    parser = FlexibleArgumentParser()
    parser.add_argument('serve')
    parser.add_argument('model_tag', nargs='?')
    parser.add_argument('--model', type=str)
    parser.add_argument('--served-model-name', type=str)
    parser.add_argument('--config', type=str)
    parser.add_argument('--port', type=int)
    parser.add_argument('--tensor-parallel-size', type=int)
    parser.add_argument('--trust-remote-code', action='store_true')
    parser.add_argument('--multi-step-stream-outputs', action=StoreBoolean)
    return parser

def test_underscore_to_dash(parser):
    args = parser.parse_args(['--image_input_type', 'pixel_values'])
    assert args.image_input_type == 'pixel_values'

def test_mixed_usage(parser):
    args = parser.parse_args(['--image_input_type', 'image_features', '--model-name', 'facebook/opt-125m'])
    assert args.image_input_type == 'image_features'
    assert args.model_name == 'facebook/opt-125m'

def test_with_equals_sign(parser):
    args = parser.parse_args(['--image_input_type=pixel_values', '--model-name=facebook/opt-125m'])
    assert args.image_input_type == 'pixel_values'
    assert args.model_name == 'facebook/opt-125m'

def test_with_int_value(parser):
    args = parser.parse_args(['--batch_size', '32'])
    assert args.batch_size == 32
    args = parser.parse_args(['--batch-size', '32'])
    assert args.batch_size == 32

def test_with_bool_flag(parser):
    args = parser.parse_args(['--enable_feature'])
    assert args.enable_feature is True
    args = parser.parse_args(['--enable-feature'])
    assert args.enable_feature is True

def test_invalid_choice(parser):
    with pytest.raises(SystemExit):
        parser.parse_args(['--image_input_type', 'invalid_choice'])

def test_missing_required_argument(parser):
    parser.add_argument('--required-arg', required=True)
    with pytest.raises(SystemExit):
        parser.parse_args([])

def test_cli_override_to_config(parser_with_config, cli_config_file):
    args = parser_with_config.parse_args(['serve', 'mymodel', '--config', cli_config_file, '--tensor-parallel-size', '3'])
    assert args.tensor_parallel_size == 3
    args = parser_with_config.parse_args(['serve', 'mymodel', '--tensor-parallel-size', '3', '--config', cli_config_file])
    assert args.tensor_parallel_size == 3
    assert args.port == 12312
    args = parser_with_config.parse_args(['serve', 'mymodel', '--tensor-parallel-size', '3', '--config', cli_config_file, '--port', '666'])
    assert args.tensor_parallel_size == 3
    assert args.port == 666

def test_config_args(parser_with_config, cli_config_file):
    args = parser_with_config.parse_args(['serve', 'mymodel', '--config', cli_config_file])
    assert args.tensor_parallel_size == 2
    assert args.trust_remote_code
    assert not args.multi_step_stream_outputs

def test_config_file(parser_with_config):
    with pytest.raises(FileNotFoundError):
        parser_with_config.parse_args(['serve', 'mymodel', '--config', 'test_config.yml'])
    with pytest.raises(ValueError):
        parser_with_config.parse_args(['serve', 'mymodel', '--config', './data/test_config.json'])
    with pytest.raises(ValueError):
        parser_with_config.parse_args(['serve', 'mymodel', '--tensor-parallel-size', '3', '--config', '--batch-size', '32'])

def test_no_model_tag(parser_with_config, cli_config_file):
    with pytest.raises(ValueError):
        parser_with_config.parse_args(['serve', '--config', cli_config_file])

def test_dict_args(parser):
    args = ['--model-name=something.something', '--hf-overrides.key1', 'val1', '--hf-overrides.key2.key3', 'val2', '--hf-overrides.key2.key4', 'val3', '-O.use_inductor=true', '-O.backend', 'custom', '-O1', '--hf-overrides.key5=val4', '--hf_overrides.key_6', 'val5', '--hf_overrides.key-7.key_8', 'val6', '--hf_overrides.key9', '100', '--hf_overrides.key10', '100.0', '--hf_overrides.key11', 'true', '--hf_overrides.key12.key13', 'null', '--hf_overrides.key14.key15', '-minus.and.dot', '-O.custom_ops+', '-quant_fp8', '-O.custom_ops+=+silu_mul,-rms_norm']
    parsed_args = parser.parse_args(args)
    assert parsed_args.model_name == 'something.something'
    assert parsed_args.hf_overrides == {'key1': 'val1', 'key2': {'key3': 'val2', 'key4': 'val3'}, 'key5': 'val4', 'key_6': 'val5', 'key-7': {'key_8': 'val6'}, 'key9': 100, 'key10': 100.0, 'key11': True, 'key12': {'key13': None}, 'key14': {'key15': '-minus.and.dot'}}
    assert parsed_args.compilation_config == {'level': 1, 'use_inductor': True, 'backend': 'custom', 'custom_ops': ['-quant_fp8', '+silu_mul', '-rms_norm']}

def test_duplicate_dict_args(caplog_vllm, parser):
    args = ['--model-name=something.something', '--hf-overrides.key1', 'val1', '--hf-overrides.key1', 'val2', '-O1', '-O.level', '2', '-O3']
    parsed_args = parser.parse_args(args)
    assert parsed_args.hf_overrides == {'key1': 'val2'}
    assert parsed_args.compilation_config == {'level': 3}
    assert len(caplog_vllm.records) == 1
    assert 'duplicate' in caplog_vllm.text
    assert '--hf-overrides.key1' in caplog_vllm.text
    assert '-O.level' in caplog_vllm.text

@pytest.mark.parametrize('callable,kw_name,requires_kw_only,allow_var_kwargs,is_supported', [(lambda foo: None, 'foo', True, True, False), (lambda foo: None, 'foo', False, True, True), (lambda foo=100: None, 'foo', True, True, False), (lambda *, foo: None, 'foo', False, True, True), (lambda *args: None, 'args', False, True, False), (lambda **kwargs: None, 'kwargs', False, True, False), (lambda foo: None, 'something_else', False, True, False), (lambda foo, **kwargs: None, 'something_else', False, True, True), (lambda foo, **kwargs: None, 'kwargs', True, True, False), (lambda foo, **kwargs: None, 'foo', True, True, False)])
def test_supports_kw(callable, kw_name, requires_kw_only, allow_var_kwargs, is_supported):
    assert supports_kw(callable=callable, kw_name=kw_name, requires_kw_only=requires_kw_only, allow_var_kwargs=allow_var_kwargs) == is_supported

@create_new_process_for_each_test()
def test_memory_profiling():
    lib = CudaRTLibrary()
    handle1 = lib.cudaMalloc(512 * 1024 * 1024)
    baseline_snapshot = MemorySnapshot()
    weights = torch.randn(128, 1024, 1024, device='cuda', dtype=torch.float32)
    weights_memory = 128 * 1024 * 1024 * 4

    def measure_current_non_torch():
        free, total = torch.cuda.mem_get_info()
        current_used = total - free
        current_torch = torch.cuda.memory_reserved()
        current_non_torch = current_used - current_torch
        return current_non_torch
    with memory_profiling(baseline_snapshot=baseline_snapshot, weights_memory=weights_memory) as result, monitor(measure_current_non_torch) as monitored_values:
        spike = torch.randn(256, 1024, 1024, device='cuda', dtype=torch.float32)
        del spike
        handle2 = lib.cudaMalloc(256 * 1024 * 1024)
    measured_diff = monitored_values.values[-1] - monitored_values.values[0]
    assert measured_diff == 256 * 1024 * 1024
    non_torch_ratio = result.non_torch_increase / (256 * 1024 * 1024)
    assert abs(non_torch_ratio - 1) <= 0.05
    assert result.torch_peak_increase == 1024 * 1024 * 1024
    del weights
    lib.cudaFree(handle1)
    lib.cudaFree(handle2)

def test_bind_kv_cache():
    ctx = {'layers.0.self_attn': Attention(32, 128, 0.1), 'layers.1.self_attn': Attention(32, 128, 0.1), 'layers.2.self_attn': Attention(32, 128, 0.1), 'layers.3.self_attn': Attention(32, 128, 0.1)}
    kv_cache = [torch.zeros((1,)), torch.zeros((1,)), torch.zeros((1,)), torch.zeros((1,))]
    bind_kv_cache(ctx, [kv_cache])
    assert ctx['layers.0.self_attn'].kv_cache[0] is kv_cache[0]
    assert ctx['layers.1.self_attn'].kv_cache[0] is kv_cache[1]
    assert ctx['layers.2.self_attn'].kv_cache[0] is kv_cache[2]
    assert ctx['layers.3.self_attn'].kv_cache[0] is kv_cache[3]

def test_bind_kv_cache_kv_sharing():
    ctx = {'layers.0.self_attn': Attention(32, 128, 0.1), 'layers.1.self_attn': Attention(32, 128, 0.1), 'layers.2.self_attn': Attention(32, 128, 0.1), 'layers.3.self_attn': Attention(32, 128, 0.1)}
    kv_cache = [torch.zeros((1,)), torch.zeros((1,)), torch.zeros((1,)), torch.zeros((1,))]
    shared_kv_cache_layers = {'layers.2.self_attn': 'layers.1.self_attn', 'layers.3.self_attn': 'layers.0.self_attn'}
    bind_kv_cache(ctx, [kv_cache], shared_kv_cache_layers)
    assert ctx['layers.0.self_attn'].kv_cache[0] is kv_cache[0]
    assert ctx['layers.1.self_attn'].kv_cache[0] is kv_cache[1]
    assert ctx['layers.2.self_attn'].kv_cache[0] is kv_cache[1]
    assert ctx['layers.3.self_attn'].kv_cache[0] is kv_cache[0]

def test_bind_kv_cache_non_attention():
    ctx = {'model.layers.20.attn': Attention(32, 128, 0.1), 'model.layers.28.attn': Attention(32, 128, 0.1)}
    kv_cache = [torch.zeros((1,)), torch.zeros((1,))]
    bind_kv_cache(ctx, [kv_cache])
    assert ctx['model.layers.20.attn'].kv_cache[0] is kv_cache[0]
    assert ctx['model.layers.28.attn'].kv_cache[0] is kv_cache[1]

def test_bind_kv_cache_encoder_decoder(monkeypatch: pytest.MonkeyPatch):
    with monkeypatch.context() as m:
        m.setenv('VLLM_USE_V1', '0')
        ctx = {'encoder.layers.0.self_attn.attn': Attention(32, 128, 0.1, attn_type=AttentionType.ENCODER), 'decoder.layers.0.encoder_attn.attn': Attention(32, 128, 0.1, attn_type=AttentionType.ENCODER_DECODER), 'decoder.layers.0.self_attn.attn': Attention(32, 128, 0.1, attn_type=AttentionType.DECODER)}
        kv_cache = [torch.zeros((1,))]
        encoder_kv_cache = ctx['encoder.layers.0.self_attn.attn'].kv_cache
        bind_kv_cache(ctx, [kv_cache])
        assert ctx['encoder.layers.0.self_attn.attn'].kv_cache is encoder_kv_cache
        assert ctx['decoder.layers.0.encoder_attn.attn'].kv_cache[0] is kv_cache[0]
        assert ctx['decoder.layers.0.self_attn.attn'].kv_cache[0] is kv_cache[0]

def test_bind_kv_cache_pp():
    with patch('vllm.utils.cuda_device_count_stateless', lambda: 2):
        cfg = VllmConfig(parallel_config=ParallelConfig(pipeline_parallel_size=2))
    with set_current_vllm_config(cfg):
        ctx = {'layers.0.self_attn': Attention(32, 128, 0.1)}
        kv_cache = [[torch.zeros((1,))], [torch.zeros((1,))]]
        bind_kv_cache(ctx, kv_cache)
        assert ctx['layers.0.self_attn'].kv_cache[0] is kv_cache[0][0]
        assert ctx['layers.0.self_attn'].kv_cache[1] is kv_cache[1][0]

class TestLRUCache(LRUCache):

    def _on_remove(self, key, value):
        if not hasattr(self, '_remove_counter'):
            self._remove_counter = 0
        self._remove_counter += 1

def test_lru_cache():
    cache = TestLRUCache(3)
    assert cache.stat() == CacheInfo(hits=0, total=0)
    assert cache.stat(delta=True) == CacheInfo(hits=0, total=0)
    cache.put(1, 1)
    assert len(cache) == 1
    cache.put(1, 1)
    assert len(cache) == 1
    cache.put(2, 2)
    assert len(cache) == 2
    cache.put(3, 3)
    assert len(cache) == 3
    assert set(cache.cache) == {1, 2, 3}
    cache.put(4, 4)
    assert len(cache) == 3
    assert set(cache.cache) == {2, 3, 4}
    assert cache._remove_counter == 1
    assert cache.get(2) == 2
    assert cache.stat() == CacheInfo(hits=1, total=1)
    assert cache.stat(delta=True) == CacheInfo(hits=1, total=1)
    assert cache[2] == 2
    assert cache.stat() == CacheInfo(hits=2, total=2)
    assert cache.stat(delta=True) == CacheInfo(hits=1, total=1)
    cache.put(5, 5)
    assert set(cache.cache) == {2, 4, 5}
    assert cache._remove_counter == 2
    assert cache.pop(5) == 5
    assert len(cache) == 2
    assert set(cache.cache) == {2, 4}
    assert cache._remove_counter == 3
    assert cache.get(-1) is None
    assert cache.stat() == CacheInfo(hits=2, total=3)
    assert cache.stat(delta=True) == CacheInfo(hits=0, total=1)
    cache.pop(10)
    assert len(cache) == 2
    assert set(cache.cache) == {2, 4}
    assert cache._remove_counter == 3
    cache.get(10)
    assert len(cache) == 2
    assert set(cache.cache) == {2, 4}
    assert cache._remove_counter == 3
    cache.put(6, 6)
    assert len(cache) == 3
    assert set(cache.cache) == {2, 4, 6}
    assert 2 in cache
    assert 4 in cache
    assert 6 in cache
    cache.remove_oldest()
    assert len(cache) == 2
    assert set(cache.cache) == {2, 6}
    assert cache._remove_counter == 4
    cache.clear()
    assert len(cache) == 0
    assert cache._remove_counter == 6
    assert cache.stat() == CacheInfo(hits=0, total=0)
    assert cache.stat(delta=True) == CacheInfo(hits=0, total=0)
    cache._remove_counter = 0
    cache[1] = 1
    assert len(cache) == 1
    cache[1] = 1
    assert len(cache) == 1
    cache[2] = 2
    assert len(cache) == 2
    cache[3] = 3
    assert len(cache) == 3
    assert set(cache.cache) == {1, 2, 3}
    cache[4] = 4
    assert len(cache) == 3
    assert set(cache.cache) == {2, 3, 4}
    assert cache._remove_counter == 1
    assert cache[2] == 2
    cache[5] = 5
    assert set(cache.cache) == {2, 4, 5}
    assert cache._remove_counter == 2
    del cache[5]
    assert len(cache) == 2
    assert set(cache.cache) == {2, 4}
    assert cache._remove_counter == 3
    cache.pop(10)
    assert len(cache) == 2
    assert set(cache.cache) == {2, 4}
    assert cache._remove_counter == 3
    cache[6] = 6
    assert len(cache) == 3
    assert set(cache.cache) == {2, 4, 6}
    assert 2 in cache
    assert 4 in cache
    assert 6 in cache

@pytest.mark.parametrize(('src_dtype', 'tgt_dtype', 'expected_result'), [(torch.bool, torch.int8, True), (torch.bool, torch.float16, True), (torch.bool, torch.complex32, True), (torch.int64, torch.bool, False), (torch.int64, torch.float16, True), (torch.int64, torch.complex32, True), (torch.float64, torch.bool, False), (torch.float64, torch.int8, False), (torch.float64, torch.complex32, True), (torch.complex128, torch.bool, False), (torch.complex128, torch.int8, False), (torch.complex128, torch.float16, False), (torch.bool, torch.bool, True), (torch.int8, torch.int16, True), (torch.int16, torch.int8, False), (torch.uint8, torch.int8, False), (torch.int8, torch.uint8, False), (torch.float16, torch.float32, True), (torch.float32, torch.float16, False), (torch.bfloat16, torch.float32, True), (torch.float32, torch.bfloat16, False), (torch.complex32, torch.complex64, True), (torch.complex64, torch.complex32, False)])
def test_is_lossless_cast(src_dtype, tgt_dtype, expected_result):
    assert is_lossless_cast(src_dtype, tgt_dtype) == expected_result

@pytest.mark.parametrize(('dtypes', 'expected_result'), [([torch.bool], torch.bool), ([torch.bool, torch.int8], torch.int8), ([torch.bool, torch.int8, torch.float16], torch.float16), ([torch.bool, torch.int8, torch.float16, torch.complex32], torch.complex32)])
def test_common_broadcastable_dtype(dtypes, expected_result):
    assert common_broadcastable_dtype(dtypes) == expected_result

def test_placeholder_module_error_handling():
    placeholder = PlaceholderModule('placeholder_1234')

    def build_ctx():
        return pytest.raises(ModuleNotFoundError, match='No module named')
    with build_ctx():
        int(placeholder)
    with build_ctx():
        placeholder()
    with build_ctx():
        _ = placeholder.some_attr
    with build_ctx():
        _ = placeholder.name
    _ = repr(placeholder)
    _ = str(placeholder)
    placeholder_attr = placeholder.placeholder_attr('attr')
    with build_ctx():
        int(placeholder_attr)
    with build_ctx():
        placeholder_attr()
    with build_ctx():
        _ = placeholder_attr.some_attr
    with build_ctx():
        _ = placeholder_attr.module

@pytest.mark.parametrize('obj,key1,key2', [({1: 'a', 2: 'b'}, 1, 2), ({1: 'a', 2: 'b'}, 1, 3), ({1: 'a', 2: 'b'}, 3, 4)])
def test_swap_dict_values(obj, key1, key2):
    original_obj = obj.copy()
    swap_dict_values(obj, key1, key2)
    if key1 in original_obj:
        assert obj[key2] == original_obj[key1]
    else:
        assert key2 not in obj
    if key2 in original_obj:
        assert obj[key1] == original_obj[key2]
    else:
        assert key1 not in obj

def test_model_specification(parser_with_config, cli_config_file, cli_config_file_with_model):
    args = parser_with_config.parse_args(['serve', 'cli-model', '--config', cli_config_file_with_model])
    assert args.model_tag == 'cli-model'
    assert args.served_model_name == 'mymodel'
    args = parser_with_config.parse_args(['serve', '--config', cli_config_file_with_model])
    assert args.model == 'config-model'
    assert args.served_model_name == 'mymodel'
    with pytest.raises(ValueError, match='No model specified!'):
        parser_with_config.parse_args(['serve', '--config', cli_config_file])
    with pytest.raises(ValueError, match='With `vllm serve`, you should provide the model as a positional argument or in a config file instead of via the `--model` option.'):
        parser_with_config.parse_args(['serve', '--model', 'my-model'])
    args = parser_with_config.parse_args(['serve', 'cli-model', '--config', cli_config_file_with_model])
    assert args.tensor_parallel_size == 2
    assert args.trust_remote_code is True
    assert args.multi_step_stream_outputs is False
    assert args.port == 12312

@pytest.mark.parametrize('input', [(), ('abc',), (None,), (None, bool, [1, 2, 3])])
@pytest.mark.parametrize('output', [0, 1, 2])
def test_sha256(input: tuple, output: int):
    hash = sha256(input)
    assert hash is not None
    assert isinstance(hash, int)
    assert hash != 0
    bytes = pickle.dumps(input, protocol=pickle.HIGHEST_PROTOCOL)
    assert hash == int.from_bytes(hashlib.sha256(bytes).digest(), byteorder='big')
    assert hash == sha256(input)
    assert hash != sha256(input + (1,))

@pytest.mark.parametrize('path,expected', [('ipc://some_path', ('ipc', 'some_path', '')), ('tcp://127.0.0.1:5555', ('tcp', '127.0.0.1', '5555')), ('tcp://[::1]:5555', ('tcp', '::1', '5555')), ('inproc://some_identifier', ('inproc', 'some_identifier', ''))])
def test_split_zmq_path(path, expected):
    assert split_zmq_path(path) == expected

@pytest.mark.parametrize('invalid_path', ['invalid_path', 'tcp://127.0.0.1', 'tcp://[::1]', 'tcp://:5555'])
def test_split_zmq_path_invalid(invalid_path):
    with pytest.raises(ValueError):
        split_zmq_path(invalid_path)

def test_make_zmq_socket_ipv6():
    try:
        sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        sock.close()
    except socket.error:
        pytest.skip('IPv6 is not supported on this system')
    ctx = zmq.Context()
    ipv6_path = 'tcp://[::]:5555'
    socket_type = zmq.REP
    zsock: zmq.Socket = make_zmq_socket(ctx, ipv6_path, socket_type)
    assert zsock.getsockopt(zmq.IPV6) == 1, 'IPV6 option should be enabled for IPv6 addresses'
    zsock.close()
    ctx.term()

def test_make_zmq_path():
    assert make_zmq_path('tcp', '127.0.0.1', '5555') == 'tcp://127.0.0.1:5555'
    assert make_zmq_path('tcp', '::1', '5555') == 'tcp://[::1]:5555'

def test_get_tcp_uri():
    assert get_tcp_uri('127.0.0.1', 5555) == 'tcp://127.0.0.1:5555'
    assert get_tcp_uri('::1', 5555) == 'tcp://[::1]:5555'

def test_split_host_port():
    assert split_host_port('127.0.0.1:5555') == ('127.0.0.1', 5555)
    with pytest.raises(ValueError):
        assert split_host_port('127.0.0.1::5555')
    with pytest.raises(ValueError):
        assert split_host_port('127.0.0.1:5555:')
    with pytest.raises(ValueError):
        assert split_host_port('127.0.0.15555')
    with pytest.raises(ValueError):
        assert split_host_port('127.0.0.1:5555a')
    assert split_host_port('[::1]:5555') == ('::1', 5555)
    with pytest.raises(ValueError):
        assert split_host_port('[::1]::5555')
    with pytest.raises(IndexError):
        assert split_host_port('[::1]5555')
    with pytest.raises(ValueError):
        assert split_host_port('[::1]:5555a')

def test_join_host_port():
    assert join_host_port('127.0.0.1', 5555) == '127.0.0.1:5555'
    assert join_host_port('::1', 5555) == '[::1]:5555'

def test_convert_ids_list_to_tokens():
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
    token_ids = tokenizer.encode('Hello, world!')
    assert tokenizer.convert_ids_to_tokens(token_ids) == ['Hello', ',', 'Ġworld', '!']
    tokens = convert_ids_list_to_tokens(tokenizer, token_ids)
    assert tokens == ['Hello', ',', ' world', '!']