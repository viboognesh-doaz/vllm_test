from PIL import Image
from neuronx_distributed_inference.models.mllama.utils import add_instruct
from vllm import LLM, SamplingParams, TextPrompt
import requests
import torch

def get_image(image_url):
    image = Image.open(requests.get(image_url, stream=True).raw)
    return image
PROMPTS = ['What is in this image? Tell me a story', 'What is the recipe of mayonnaise in two sentences?', 'Describe this image', 'What is the capital of Italy famous for?']
IMAGES = [get_image('https://images.pexels.com/photos/1108099/pexels-photo-1108099.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500'), None, get_image('https://images.pexels.com/photos/1108099/pexels-photo-1108099.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500'), None]
SAMPLING_PARAMS = [dict(top_k=1, temperature=1.0, top_p=1.0, max_tokens=16) for _ in range(len(PROMPTS))]

def get_VLLM_mllama_model_inputs(prompt, single_image, sampling_params):
    input_image = single_image
    has_image = torch.tensor([1])
    if isinstance(single_image, torch.Tensor) and single_image.numel() == 0:
        has_image = torch.tensor([0])
    instruct_prompt = add_instruct(prompt, has_image)
    inputs = TextPrompt(prompt=instruct_prompt)
    if input_image is not None:
        inputs['multi_modal_data'] = {'image': input_image}
    sampling_params = SamplingParams(**sampling_params)
    return (inputs, sampling_params)

def print_outputs(outputs):
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f'Prompt: {prompt!r}, Generated text: {generated_text!r}')

def main():
    assert len(PROMPTS) == len(IMAGES) == len(SAMPLING_PARAMS), f'Text, image prompts and sampling parameters should have the \n            same batch size; but got {len(PROMPTS)}, {len(IMAGES)}, \n            and {len(SAMPLING_PARAMS)}'
    llm = LLM(model='meta-llama/Llama-3.2-11B-Vision-Instruct', max_num_seqs=1, max_model_len=4096, block_size=4096, device='neuron', tensor_parallel_size=32, override_neuron_config={'sequence_parallel_enabled': False, 'skip_warmup': True, 'save_sharded_checkpoint': True, 'on_device_sampling_config': {'global_topk': 1, 'dynamic': False, 'deterministic': False}})
    batched_inputs = []
    batched_sample_params = []
    for pmpt, img, params in zip(PROMPTS, IMAGES, SAMPLING_PARAMS):
        inputs, sampling_params = get_VLLM_mllama_model_inputs(pmpt, img, params)
        outputs = llm.generate(inputs, sampling_params)
        print_outputs(outputs)
        batched_inputs.append(inputs)
        batched_sample_params.append(sampling_params)
    outputs = llm.generate(batched_inputs, batched_sample_params)
    print_outputs(outputs)
if __name__ == '__main__':
    main()