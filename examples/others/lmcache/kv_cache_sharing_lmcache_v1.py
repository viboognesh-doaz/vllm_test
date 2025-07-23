from lmcache.integration.vllm.utils import ENGINE_NAME
from lmcache.v1.cache_engine import LMCacheEngineBuilder
from multiprocessing import Event, Process
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
import os
import subprocess
import time
'\nThis file demonstrates the example usage of remote KV cache sharing\nwith LMCache.\nWe will launch 2 vllm instances, and launch an additional LMCache server.\nKV cache is transferred in the following manner:\n(1) vLLM instance 1 -> LMCache server (KV cache store).\n(2) LMCache server -> vLLM instance 2 (KV cache reuse/retrieve).\n\nNote that lmcache needs to be installed to run this example.\nLearn more about LMCache in https://github.com/LMCache/LMCache.\n'
port = 8100
os.environ['LMCACHE_USE_EXPERIMENTAL'] = 'True'
os.environ['LMCACHE_CHUNK_SIZE'] = '256'
os.environ['LMCACHE_LOCAL_CPU'] = 'False'
os.environ['LMCACHE_MAX_LOCAL_CPU_SIZE'] = '5.0'
os.environ['LMCACHE_REMOTE_URL'] = f'lm://localhost:{port}'
os.environ['LMCACHE_REMOTE_SERDE'] = 'naive'
prompts = ['Hello, how are you?' * 1000]

def run_store(store_done, prompts):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=10)
    ktc = KVTransferConfig(kv_connector='LMCacheConnectorV1', kv_role='kv_both')
    llm = LLM(model='mistralai/Mistral-7B-Instruct-v0.2', kv_transfer_config=ktc, max_model_len=8000, gpu_memory_utilization=0.8, enforce_eager=True)
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f'Generated text: {generated_text!r}')
    print('KV cache store is finished.')
    store_done.set()
    LMCacheEngineBuilder.destroy(ENGINE_NAME)

def run_retrieve(store_done, prompts, timeout=1):
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=10)
    ktc = KVTransferConfig(kv_connector='LMCacheConnectorV1', kv_role='kv_both')
    llm = LLM(model='mistralai/Mistral-7B-Instruct-v0.2', kv_transfer_config=ktc, max_model_len=8000, gpu_memory_utilization=0.8, enforce_eager=True)
    print('Waiting for KV cache store to finish...')
    store_done.wait()
    time.sleep(timeout)
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f'Generated text: {generated_text!r}')
    LMCacheEngineBuilder.destroy(ENGINE_NAME)

def run_lmcache_server(port):
    server_proc = subprocess.Popen(['python', '-m', 'lmcache.v1.server', 'localhost', str(port)])
    return server_proc

def main():
    store_done = Event()
    store_process = Process(target=run_store, args=(store_done, prompts))
    retrieve_process = Process(target=run_retrieve, args=(store_done, prompts))
    lmcache_server_process = run_lmcache_server(port)
    store_process.start()
    retrieve_process.start()
    store_process.join()
    retrieve_process.terminate()
    lmcache_server_process.terminate()
    lmcache_server_process.wait()
if __name__ == '__main__':
    main()