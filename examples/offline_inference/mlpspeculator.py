from vllm import LLM, SamplingParams
import gc
import time
'\nThis file demonstrates the usage of text generation with an LLM model,\ncomparing the performance with and without speculative decoding.\n\nNote that still not support `v1`:\nVLLM_USE_V1=0 python examples/offline_inference/mlpspeculator.py\n'

def time_generation(llm: LLM, prompts: list[str], sampling_params: SamplingParams, title: str):
    llm.generate(prompts, sampling_params)
    llm.generate(prompts, sampling_params)
    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    end = time.time()
    print('-' * 50)
    print(title)
    print('time: ', (end - start) / sum((len(o.outputs[0].token_ids) for o in outputs)))
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f'text: {generated_text!r}')
        print('-' * 50)

def main():
    template = 'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Response:\n'
    prompts = ['Write about the president of the United States.']
    prompts = [template.format(prompt) for prompt in prompts]
    sampling_params = SamplingParams(temperature=0.0, max_tokens=200)
    llm = LLM(model='meta-llama/Llama-2-13b-chat-hf')
    time_generation(llm, prompts, sampling_params, 'Without speculation')
    del llm
    gc.collect()
    llm = LLM(model='meta-llama/Llama-2-13b-chat-hf', speculative_config={'model': 'ibm-ai-platform/llama-13b-accelerator'})
    time_generation(llm, prompts, sampling_params, 'With speculation')
if __name__ == '__main__':
    main()