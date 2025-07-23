from .cli_args import make_config, make_config_arg_parser
from .common import Config, RankTensors, WeightTensors, make_modular_kernel
from .parallel_utils import ProcessGroupInfo, parallel_launch_with_config
from itertools import product
from typing import Any, Callable
from vllm import envs
from vllm.config import VllmConfig
from vllm.platforms import current_platform
import copy
import torch

def do_profile(fn: Callable, fn_kwargs: dict[Any, Any], pgi: ProcessGroupInfo, config: Config, num_warmups: int=5):
    for _ in range(num_warmups):
        fn(**fn_kwargs)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], with_stack=True, record_shapes=True) as tprof:
        fn(**fn_kwargs)
        torch.cuda.synchronize(torch.cuda.current_device())
    tprof.export_chrome_trace(f'{config.torch_trace_dir_path}/m{config.M}_{pgi.rank}_trace.json')

def profile_modular_kernel(pgi: ProcessGroupInfo, vllm_config: VllmConfig, config: Config, weights: WeightTensors, rank_tensors: RankTensors) -> None:
    assert isinstance(config.Ms, int)
    assert isinstance(config.topks, int)
    rank_weights = weights.slice_weights(pgi.rank, config.num_local_experts)
    mk = make_modular_kernel(config, vllm_config)
    mk_kwargs = {'hidden_states': rank_tensors.hidden_states, 'w1': rank_weights.w1, 'w2': rank_weights.w2, 'topk_weights': rank_tensors.topk_weights, 'topk_ids': rank_tensors.topk_ids, 'expert_map': rank_tensors.expert_map, 'w1_scale': rank_weights.w1_scale, 'w2_scale': rank_weights.w2_scale, 'a1_scale': rank_tensors.hidden_states_scale, 'global_num_experts': config.E, 'apply_router_weight_on_input': config.topk == 1}
    do_profile(mk.forward, mk_kwargs, pgi, config)

def rank_worker(pgi: ProcessGroupInfo, vllm_config: VllmConfig, cpu_group, config: Config, weights: WeightTensors):
    current_platform.seed_everything(pgi.rank)
    if config.fused_moe_chunk_size is not None:
        assert config.fused_moe_chunk_size == envs.VLLM_FUSED_MOE_CHUNK_SIZE
    weights.to_current_device()
    Ms = config.Ms
    assert isinstance(Ms, list)
    TOPKs = config.topks
    assert isinstance(TOPKs, list)
    for m, topk in product(Ms, TOPKs):
        print(f'Running m={m}, topk={topk} ...')
        cfgx = copy.deepcopy(config)
        cfgx.Ms = m
        cfgx.topks = topk
        rank_tensors = RankTensors.make(cfgx, pgi)
        profile_modular_kernel(pgi, vllm_config, cfgx, weights, rank_tensors)

def run(config: Config):
    weights: WeightTensors = WeightTensors.make(config)
    vllm_config, env_dict = config.make_env_data()
    parallel_launch_with_config(config.world_size, rank_worker, vllm_config, env_dict, config, weights)
if __name__ == '__main__':
    parser = make_config_arg_parser(description='Run single prepare-finalize & fused-experts combination testExample : python3 -m tests.kernels.moe.modular_kernel_tools.profile_modular_kernel --pf-type PplxPrepareAndFinalize --experts-type BatchedTritonExperts')
    args = parser.parse_args()
    assert args.torch_trace_dir_path is not None, 'Please pass in a directory to store torch traces'
    config = make_config(args)
    run(config)