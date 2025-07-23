from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory
prefix = "You are an expert school principal, skilled in effectively managing faculty and staff. Draft 10-15 questions for a potential first grade Head Teacher for my K-12, all-girls', independent school that emphasizes community, joyful discovery, and life-long learning. The candidate is coming in for a first-round panel interview for a 8th grade Math teaching role. They have 5 years of previous teaching experience as an assistant teacher at a co-ed, public school with experience in middle school math teaching. Based on these information, fulfill the following paragraph: "
prompts = ['Hello, my name is', 'The president of the United States is', 'The capital of France is', 'The future of AI is']
generating_prompts = [prefix + prompt for prompt in prompts]
sampling_params = SamplingParams(temperature=0.0)

def main():
    regular_llm = LLM(model='facebook/opt-125m', gpu_memory_utilization=0.4)
    print('Results without `enable_prefix_caching`')
    outputs = regular_llm.generate(generating_prompts, sampling_params)
    regular_generated_texts = []
    print('-' * 50)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        regular_generated_texts.append(generated_text)
        print(f'Prompt: {prompt!r}\nGenerated text: {generated_text!r}')
        print('-' * 50)
    del regular_llm
    cleanup_dist_env_and_memory()
    prefix_cached_llm = LLM(model='facebook/opt-125m', enable_prefix_caching=True, gpu_memory_utilization=0.4)
    prefix_cached_llm.generate(generating_prompts[0], sampling_params)
    outputs = prefix_cached_llm.generate(generating_prompts, sampling_params)
    print('Results with `enable_prefix_caching`')
    cached_generated_texts = []
    print('-' * 50)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        cached_generated_texts.append(generated_text)
        print(f'Prompt: {prompt!r}\nGenerated text: {generated_text!r}')
        print('-' * 50)
    generated_same = all([regular_generated_texts[i] == cached_generated_texts[i] for i in range(len(prompts))])
    print(f'Generated answers are the same: {generated_same}')
if __name__ == '__main__':
    main()