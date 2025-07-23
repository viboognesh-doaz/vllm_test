from vllm import LLM, SamplingParams
import torch.distributed as dist
'\nexperimental support for tensor-parallel inference with torchrun,\nsee https://github.com/vllm-project/vllm/issues/11400 for\nthe motivation and use case for this example.\nrun the script with `torchrun --nproc-per-node=2 torchrun_example.py`,\nthe argument 2 should match the `tensor_parallel_size` below.\nsee `tests/distributed/test_torchrun_example.py` for the unit test.\n'
prompts = ['Hello, my name is', 'The president of the United States is', 'The capital of France is', 'The future of AI is']
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
llm = LLM(model='meta-llama/Llama-3.1-8B', tensor_parallel_size=2, pipeline_parallel_size=2, distributed_executor_backend='external_launcher', max_model_len=32768, seed=1)
outputs = llm.generate(prompts, sampling_params)
if dist.get_rank() == 0:
    print('-' * 50)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f'Prompt: {prompt!r}\nGenerated text: {generated_text!r}\n')
        print('-' * 50)
    "\nFurther tips:\n\n1. to communicate control messages across all ranks, use the cpu group,\na PyTorch ProcessGroup with GLOO backend.\n\n```python\nfrom vllm.distributed.parallel_state import get_world_group\ncpu_group = get_world_group().cpu_group\ntorch_rank = dist.get_rank(group=cpu_group)\nif torch_rank == 0:\n    # do something for rank 0, e.g. saving the results to disk.\n```\n\n2. to communicate data across all ranks, use the model's device group,\na PyTorch ProcessGroup with NCCL backend.\n```python\nfrom vllm.distributed.parallel_state import get_world_group\ndevice_group = get_world_group().device_group\n```\n\n3. to access the model directly in every rank, use the following code:\n```python\nllm.llm_engine.model_executor.driver_worker.worker.model_runner.model\n```\n"