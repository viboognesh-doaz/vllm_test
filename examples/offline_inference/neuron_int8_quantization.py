from vllm import LLM, SamplingParams
import os
os.environ['NEURON_CONTEXT_LENGTH_BUCKETS'] = '128,512,1024,2048'
os.environ['NEURON_TOKEN_GEN_BUCKETS'] = '128,512,1024,2048'
os.environ['NEURON_QUANT_DTYPE'] = 's8'
prompts = ['Hello, my name is', 'The president of the United States is', 'The capital of France is', 'The future of AI is']
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

def main():
    llm = LLM(model='TinyLlama/TinyLlama-1.1B-Chat-v1.0', max_num_seqs=8, max_model_len=2048, block_size=2048, device='neuron', quantization='neuron_quant', override_neuron_config={'cast_logits_dtype': 'bfloat16'}, tensor_parallel_size=2)
    outputs = llm.generate(prompts, sampling_params)
    print('-' * 50)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f'Prompt: {prompt!r}\nGenerated text: {generated_text!r}')
        print('-' * 50)
if __name__ == '__main__':
    main()