from vllm import LLM, SamplingParams
'\nThis example shows how to run offline inference with an EAGLE speculative\ndecoding model on neuron. To use EAGLE speculative decoding, you must use\na draft model that is specifically fine-tuned for EAGLE speculation.\nAdditionally, to use EAGLE with NxD Inference, the draft model must include\nthe LM head weights from the target model. These weights are shared between\nthe draft and target model.\n'
prompts = ['What is annapurna labs?']

def main():
    sampling_params = SamplingParams(top_k=1, max_tokens=500, ignore_eos=True)
    llm = LLM(model='/home/ubuntu/model_hf/Meta-Llama-3.1-70B-Instruct', speculative_config={'model': '/home/ubuntu/model_hf/Llama-3.1-70B-Instruct-EAGLE-Draft', 'num_speculative_tokens': 5, 'max_model_len': 2048}, max_num_seqs=4, max_model_len=2048, block_size=2048, device='neuron', tensor_parallel_size=32, override_neuron_config={'enable_eagle_speculation': True, 'enable_fused_speculation': True})
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f'Prompt: {prompt!r}, \n\n\n Generated text: {generated_text!r}')
if __name__ == '__main__':
    main()