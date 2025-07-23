from vllm import LLM, SamplingParams
from vllm.v1.metrics.reader import Counter, Gauge, Histogram, Vector
prompts = ['Hello, my name is', 'The president of the United States is', 'The capital of France is', 'The future of AI is']
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

def main():
    llm = LLM(model='facebook/opt-125m', disable_log_stats=False)
    outputs = llm.generate(prompts, sampling_params)
    print('-' * 50)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f'Prompt: {prompt!r}\nGenerated text: {generated_text!r}')
        print('-' * 50)
    for metric in llm.get_metrics():
        if isinstance(metric, Gauge):
            print(f'{metric.name} (gauge) = {metric.value}')
        elif isinstance(metric, Counter):
            print(f'{metric.name} (counter) = {metric.value}')
        elif isinstance(metric, Vector):
            print(f'{metric.name} (vector) = {metric.values}')
        elif isinstance(metric, Histogram):
            print(f'{metric.name} (histogram)')
            print(f'    sum = {metric.sum}')
            print(f'    count = {metric.count}')
            for bucket_le, value in metric.buckets.items():
                print(f'    {bucket_le} = {value}')
if __name__ == '__main__':
    main()