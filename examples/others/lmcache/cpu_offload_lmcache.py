from dataclasses import asdict
from lmcache.integration.vllm.utils import ENGINE_NAME
from lmcache.v1.cache_engine import LMCacheEngineBuilder
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.engine.arg_utils import EngineArgs
import argparse
import contextlib
import os
import time
'\nThis file demonstrates the example usage of cpu offloading\nwith LMCache in vLLM v1 or v0.\n\nUsage:\n\n    Specify vLLM version\n\n    -v v0 : Use LMCacheConnector\n            model = mistralai/Mistral-7B-Instruct-v0.2\n            (Includes enable_chunked_prefill = True)\n\n    -v v1 : Use LMCacheConnectorV1 (default)\n            model = meta-llama/Meta-Llama-3.1-8B-Instruct\n            (Without enable_chunked_prefill)\n\nNote that `lmcache` is needed to run this example.\nRequirements:\nhttps://docs.lmcache.ai/getting_started/installation.html#prerequisites\nLearn more about LMCache environment setup, please refer to:\nhttps://docs.lmcache.ai/getting_started/installation.html\n'

def setup_environment_variables(vllm_version: str):
    os.environ['LMCACHE_USE_EXPERIMENTAL'] = 'True'
    os.environ['LMCACHE_CHUNK_SIZE'] = '256'
    os.environ['LMCACHE_LOCAL_CPU'] = 'True'
    os.environ['LMCACHE_MAX_LOCAL_CPU_SIZE'] = '5.0'
    if vllm_version == 'v0':
        os.environ['VLLM_USE_V1'] = '0'

@contextlib.contextmanager
def build_llm_with_lmcache(lmcache_connector: str, model: str, vllm_version: str):
    ktc = KVTransferConfig(kv_connector=lmcache_connector, kv_role='kv_both')
    if vllm_version == 'v0':
        llm_args = EngineArgs(model=model, kv_transfer_config=ktc, max_model_len=8000, gpu_memory_utilization=0.8, enable_chunked_prefill=True)
    else:
        llm_args = EngineArgs(model=model, kv_transfer_config=ktc, max_model_len=8000, gpu_memory_utilization=0.8)
    llm = LLM(**asdict(llm_args))
    try:
        yield llm
    finally:
        LMCacheEngineBuilder.destroy(ENGINE_NAME)

def print_output(llm: LLM, prompt: list[str], sampling_params: SamplingParams, req_str: str):
    start = time.time()
    outputs = llm.generate(prompt, sampling_params)
    print('-' * 50)
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f'Generated text: {generated_text!r}')
    print(f'Generation took {time.time() - start:.2f} seconds, {req_str} request done.')
    print('-' * 50)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--version', choices=['v0', 'v1'], default='v1', help='Specify vLLM version (default: v1)')
    return parser.parse_args()

def main():
    args = parse_args()
    if args.version == 'v0':
        lmcache_connector = 'LMCacheConnector'
        model = 'mistralai/Mistral-7B-Instruct-v0.2'
    else:
        lmcache_connector = 'LMCacheConnectorV1'
        model = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
    setup_environment_variables(args.version)
    with build_llm_with_lmcache(lmcache_connector, model, args.version) as llm:
        shared_prompt = 'Hello, how are you?' * 1000
        first_prompt = [shared_prompt + 'Hello, my name is']
        second_prompt = [shared_prompt + 'Tell me a very long story']
        sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=10)
        print_output(llm, first_prompt, sampling_params, 'first')
        time.sleep(1)
        print_output(llm, second_prompt, sampling_params, 'second')
if __name__ == '__main__':
    main()