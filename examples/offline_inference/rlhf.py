from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from rlhf_utils import stateless_init_process_group
from transformers import AutoModelForCausalLM
from vllm import LLM, SamplingParams
from vllm.utils import get_ip, get_open_port
import os
import ray
import torch
"\nDemonstrates reinforcement learning from human feedback (RLHF) using vLLM and Ray.\n\nThe script separates training and inference workloads onto distinct GPUs\nso that Ray can manage process placement and inter-process communication.\nA Hugging Face Transformer model occupies GPU 0 for training, whereas a\ntensor-parallel vLLM inference engine occupies GPU 1–2.\n\nThe example performs the following steps:\n\n* Load the training model on GPU 0.\n* Split the inference model across GPUs 1–2 using vLLM's tensor parallelism\n  and Ray placement groups.\n* Generate text from a list of prompts using the inference engine.\n* Update the weights of the training model and broadcast the updated weights\n  to the inference engine by using a Ray collective RPC group. Note that\n  for demonstration purposes we simply zero out the weights.\n\nFor a production-ready implementation that supports multiple training and\ninference replicas, see the OpenRLHF framework:\nhttps://github.com/OpenRLHF/OpenRLHF\n\nThis example assumes a single-node cluster with three GPUs, but Ray\nsupports multi-node clusters. vLLM expects the GPUs are only used for vLLM\nworkloads. Residual GPU activity interferes with vLLM memory profiling and\ncauses unexpected behavior.\n"

class MyLLM(LLM):
    """Configure the vLLM worker for Ray placement group execution."""

    def __init__(self, *args, **kwargs):
        os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        super().__init__(*args, **kwargs)
train_model = AutoModelForCausalLM.from_pretrained('facebook/opt-125m')
train_model.to('cuda:0')
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
ray.init()
pg_inference = placement_group([{'GPU': 1, 'CPU': 0}] * 2)
ray.get(pg_inference.ready())
scheduling_inference = PlacementGroupSchedulingStrategy(placement_group=pg_inference, placement_group_capture_child_tasks=True, placement_group_bundle_index=0)
llm = ray.remote(num_cpus=0, num_gpus=0, scheduling_strategy=scheduling_inference)(MyLLM).remote(model='facebook/opt-125m', enforce_eager=True, worker_extension_cls='rlhf_utils.WorkerExtension', tensor_parallel_size=2, distributed_executor_backend='ray')
prompts = ['Hello, my name is', 'The president of the United States is', 'The capital of France is', 'The future of AI is']
sampling_params = SamplingParams(temperature=0)
outputs = ray.get(llm.generate.remote(prompts, sampling_params))
print('-' * 50)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f'Prompt: {prompt!r}\nGenerated text: {generated_text!r}')
    print('-' * 50)
master_address = get_ip()
master_port = get_open_port()
handle = llm.collective_rpc.remote('init_weight_update_group', args=(master_address, master_port, 1, 3))
model_update_group = stateless_init_process_group(master_address, master_port, 0, 3, torch.device('cuda:0'))
ray.get(handle)
for name, p in train_model.named_parameters():
    p.data.zero_()
for name, p in train_model.named_parameters():
    handle = llm.collective_rpc.remote('update_weight', args=(name, p.dtype, p.shape))
    model_update_group.broadcast(p, src=0, stream=torch.cuda.current_stream())
    ray.get(handle)
assert all(ray.get(llm.collective_rpc.remote('check_weights_changed')))
outputs_updated = ray.get(llm.generate.remote(prompts, sampling_params))
print('-' * 50)
for output in outputs_updated:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f'Prompt: {prompt!r}\nGenerated text: {generated_text!r}')
    print('-' * 50)