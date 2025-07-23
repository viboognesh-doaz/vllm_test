from lmcache.experimental.cache_engine import LMCacheEngineBuilder
from lmcache.integration.vllm.utils import ENGINE_NAME
from multiprocessing import Event, Process
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
import os
import subprocess
import time
'\nThis file demonstrates the example usage of disaggregated prefilling\nwith LMCache.\nWe will launch 2 vllm instances (GPU 0 for prefill and GPU 1 for decode),\nand launch an additional LMCache server.\nKV cache is transferred in the following manner:\nvLLM prefill node -> LMCache server -> vLLM decode node.\n\nNote that `pip install lmcache` is needed to run this example.\nLearn more about LMCache in https://github.com/LMCache/LMCache.\n'
port = 8100
os.environ['LMCACHE_USE_EXPERIMENTAL'] = 'True'
os.environ['LMCACHE_CHUNK_SIZE'] = '256'
os.environ['LMCACHE_LOCAL_CPU'] = 'False'
os.environ['LMCACHE_MAX_LOCAL_CPU_SIZE'] = '5.0'
os.environ['LMCACHE_REMOTE_URL'] = f'lm://localhost:{port}'
os.environ['LMCACHE_REMOTE_SERDE'] = 'naive'
prompts = ['Hello, how are you?' * 1000]

def run_prefill(prefill_done, prompts):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=1)
    ktc = KVTransferConfig(kv_connector='LMCacheConnector', kv_role='kv_producer', kv_rank=0, kv_parallel_size=2)
    llm = LLM(model='mistralai/Mistral-7B-Instruct-v0.2', kv_transfer_config=ktc, max_model_len=8000, gpu_memory_utilization=0.8, enforce_eager=True)
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f'Generated text: {generated_text!r}')
    print('Prefill node is finished.')
    prefill_done.set()
    LMCacheEngineBuilder.destroy(ENGINE_NAME)

def run_decode(prefill_done, prompts, timeout=1):
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=10)
    ktc = KVTransferConfig(kv_connector='LMCacheConnector', kv_role='kv_consumer', kv_rank=1, kv_parallel_size=2)
    llm = LLM(model='mistralai/Mistral-7B-Instruct-v0.2', kv_transfer_config=ktc, max_model_len=8000, gpu_memory_utilization=0.8, enforce_eager=True)
    print('Waiting for prefill node to finish...')
    prefill_done.wait()
    time.sleep(timeout)
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f'Generated text: {generated_text!r}')
    LMCacheEngineBuilder.destroy(ENGINE_NAME)

def run_lmcache_server(port):
    server_proc = subprocess.Popen(['python', '-m', 'lmcache.experimental.server', 'localhost', str(port)])
    return server_proc

def main():
    prefill_done = Event()
    prefill_process = Process(target=run_prefill, args=(prefill_done, prompts))
    decode_process = Process(target=run_decode, args=(prefill_done, prompts))
    lmcache_server_process = run_lmcache_server(port)
    prefill_process.start()
    decode_process.start()
    decode_process.join()
    prefill_process.terminate()
    lmcache_server_process.terminate()
    lmcache_server_process.wait()
if __name__ == '__main__':
    main()