from vllm import LLM, SamplingParams
import os
import random
'\nDemonstrates how to achieve reproducibility in vLLM.\n\nMain article: https://docs.vllm.ai/en/latest/usage/reproducibility.html\n'
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
SEED = 42
prompts = ['Hello, my name is', 'The president of the United States is', 'The capital of France is', 'The future of AI is']
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

def main():
    llm = LLM(model='facebook/opt-125m', seed=SEED)
    outputs = llm.generate(prompts, sampling_params)
    print('-' * 50)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f'Prompt: {prompt!r}\nGenerated text: {generated_text!r}')
        print('-' * 50)
    print(random.randint(0, 100))
if __name__ == '__main__':
    main()