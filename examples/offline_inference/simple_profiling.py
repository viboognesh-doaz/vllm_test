from vllm import LLM, SamplingParams
import os
import time
os.environ['VLLM_TORCH_PROFILER_DIR'] = './vllm_profile'
prompts = ['Hello, my name is', 'The president of the United States is', 'The capital of France is', 'The future of AI is']
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

def main():
    llm = LLM(model='facebook/opt-125m', tensor_parallel_size=1)
    llm.start_profile()
    outputs = llm.generate(prompts, sampling_params)
    llm.stop_profile()
    print('-' * 50)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f'Prompt: {prompt!r}\nGenerated text: {generated_text!r}')
        print('-' * 50)
    time.sleep(10)
if __name__ == '__main__':
    main()