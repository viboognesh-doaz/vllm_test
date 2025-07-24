from torch.multiprocessing.reductions import reduce_tensor
from transformers import AutoModelForCausalLM
from vllm.platforms import current_platform
'\nDemonstrates how to co-locate a vLLM inference worker and training\nactors on the same set of GPUs for reinforcement learning from human feedback\n(RLHF) workloads.\n\nRay serves as the distributed execution framework in this example. Ray\nplacement groups allocate both training actors and vLLM workers to the\nsame GPU bundles, enabling fast, in-GPU communication between the two\ncomponents.\n\nThe script shows how to do the following:\n\n* Configure environment variables (`VLLM_RAY_PER_WORKER_GPUS` and\n  `VLLM_RAY_BUNDLE_INDICES`) so that vLLM workers land on the desired\n  devices.\n* Exchange tensors between processes by means of CUDA inter-process\n  communication (IPC). CUDA IPC sidesteps NCCL limitations that occur\n  when multiple processes share a single GPU.\n\nNote that this example assumes a single-node cluster with four GPUs, but Ray\nsupports multi-node clusters. vLLM expects exclusive use of the GPUs during\nits initialization for memory profiling. Residual GPU activity interferes\nwith vLLM memory profiling and causes unexpected behavior.\n\nLearn more about Ray placement groups:\nhttps://docs.ray.io/en/latest/placement-groups.html\n'
import os
import ray
import torch
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from vllm import LLM

class MyLLM(LLM):
    """Configure the vLLM worker for Ray placement group execution.

    The constructor sets environment variables that allow multiple vLLM
    workers to share a single physical GPU and that encode the bundle
    indices assigned by the placement group.

    Args:
        *args: Positional arguments forwarded to `vllm.LLM`.
        bundle_indices (list[int]): Placement-group bundle indices
            assigned to this worker.
        **kwargs: Keyword arguments forwarded to `vllm.LLM`.
    """

    def __init__(self, *args, bundle_indices: list[int], **kwargs):
        os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        os.environ['VLLM_RAY_PER_WORKER_GPUS'] = '0.4'
        os.environ['VLLM_RAY_BUNDLE_INDICES'] = ','.join(map(str, bundle_indices))
        print(f'creating LLM with bundle_indices={bundle_indices}')
        super().__init__(*args, **kwargs)

class RayTrainingActor:
    """Training actor that hosts a Facebook OPT-125M model from Hugging Face.

    The model is loaded onto the first GPU assigned to this actor, and expose
    the CUDA IPC handles so that colocated vLLM workers can map tensors
    directly.
    """

    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained('facebook/opt-125m')
        self.model.to('cuda:0')
        for name, p in self.model.named_parameters():
            p.data.zero_()
        torch.cuda.synchronize()
        self.device_uuid = current_platform.get_device_uuid(0)

    def report_device_id(self) -> str:
        return self.device_uuid

    def get_weight_ipc_handles(self):
        data = {}
        for name, p in self.model.named_parameters():
            data[name] = reduce_tensor(p.detach())
        return {self.device_uuid: data}
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
ray.init()
pg = placement_group([{'GPU': 1, 'CPU': 0}] * 4)
ray.get(pg.ready())
print(f'placement group has bundles pg.bundle_specs={pg.bundle_specs!r}')
training_actors = []
training_actor_device_ids = []
inference_engines = []
inference_engine_device_ids = []
for bundle_index in [0, 1, 2, 3]:
    training_actor = ray.remote(num_cpus=0, num_gpus=0.4, scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg, placement_group_capture_child_tasks=True, placement_group_bundle_index=bundle_index))(RayTrainingActor).remote()
    training_actors.append(training_actor)
for bundle_index, training_actor in enumerate(training_actors):
    device_id = ray.get(training_actor.report_device_id.remote())
    print(f'training actor {bundle_index} is on {device_id}')
    training_actor_device_ids.append(device_id)
for i, bundle_indices in enumerate([[0, 1], [2, 3]]):
    llm = ray.remote(num_cpus=0, num_gpus=0, scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg, placement_group_capture_child_tasks=True))(MyLLM).remote(model='facebook/opt-125m', enforce_eager=True, worker_extension_cls='rlhf_utils.ColocateWorkerExtension', tensor_parallel_size=2, distributed_executor_backend='ray', gpu_memory_utilization=0.4, bundle_indices=bundle_indices)
    inference_engines.append(llm)
for i, llm in enumerate(inference_engines):
    inference_engine_device_ids.append(ray.get(llm.collective_rpc.remote('report_device_id', args=tuple())))
    print(f'inference engine {i} is on {inference_engine_device_ids[-1]}')
assert training_actor_device_ids[:2] == inference_engine_device_ids[0]
assert training_actor_device_ids[2:] == inference_engine_device_ids[1]
print('Gather all the IPC handles from the training actors.')
ipc_handles = {}
for actor in training_actors:
    ipc_handles.update(ray.get(actor.get_weight_ipc_handles.remote()))
print('Update the weights of the inference engines.')
for llm in inference_engines:
    ray.get(llm.collective_rpc.remote('update_weights_from_ipc_handles', args=(ipc_handles,)))
print('Check if the weights are updated.')
for llm in inference_engines:
    assert ray.get(llm.collective_rpc.remote('check_weights_changed', args=tuple()))