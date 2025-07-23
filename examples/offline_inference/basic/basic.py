from vllm import LLM, SamplingParams
prompts = ['Hello, my name is', 'The president of the United States is', 'The capital of France is', 'The future of AI is']
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

def main():
    llm = LLM(model='facebook/opt-125m')
    outputs = llm.generate(prompts, sampling_params)
    print('\nGenerated Outputs:\n' + '-' * 60)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f'Prompt:    {prompt!r}')
        print(f'Output:    {generated_text!r}')
        print('-' * 60)
if __name__ == '__main__':
    main()