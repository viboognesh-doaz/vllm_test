from multiprocessing import Event, Process
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
import os
import time
'\nThis file demonstrates the example usage of disaggregated prefilling\nWe will launch 2 vllm instances (GPU 0 for prefill and GPU 1 for decode),\nand then transfer the KV cache between them.\n'

def run_prefill(prefill_done):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    prompts = ['Hello, my name is', 'Hi, your name is', 'Tell me a very long story']
    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=1)
    ktc = KVTransferConfig(kv_connector='PyNcclConnector', kv_role='kv_producer', kv_rank=0, kv_parallel_size=2)
    llm = LLM(model='meta-llama/Meta-Llama-3.1-8B-Instruct', kv_transfer_config=ktc, max_model_len=2000, gpu_memory_utilization=0.8)
    llm.generate(prompts, sampling_params)
    print('Prefill node is finished.')
    prefill_done.set()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print('Script stopped by user.')

def run_decode(prefill_done):
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    prompts = ['Hello, my name is', 'Hi, your name is', 'Tell me a very long story']
    sampling_params = SamplingParams(temperature=0, top_p=0.95)
    ktc = KVTransferConfig(kv_connector='PyNcclConnector', kv_role='kv_consumer', kv_rank=1, kv_parallel_size=2)
    llm = LLM(model='meta-llama/Meta-Llama-3.1-8B-Instruct', kv_transfer_config=ktc, max_model_len=2000, gpu_memory_utilization=0.8)
    print('Waiting for prefill node to finish...')
    prefill_done.wait()
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f'Prompt: {prompt!r}, Generated text: {generated_text!r}')

def main():
    prefill_done = Event()
    prefill_process = Process(target=run_prefill, args=(prefill_done,))
    decode_process = Process(target=run_decode, args=(prefill_done,))
    prefill_process.start()
    decode_process.start()
    decode_process.join()
    prefill_process.terminate()
if __name__ == '__main__':
    main()