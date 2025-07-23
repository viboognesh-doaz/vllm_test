from vllm.distributed.eplb.rebalance_execute import rearrange_expert_weights_inplace
from vllm.distributed.parallel_state import ensure_model_parallel_initialized, get_tp_group, init_distributed_environment
from vllm.utils import update_environment_variables
import multiprocessing
import os
import pytest
import random
import torch
import torch.distributed

def distributed_run(fn, world_size):
    number_of_processes = world_size
    processes: list[multiprocessing.Process] = []
    for i in range(number_of_processes):
        env: dict[str, str] = {}
        env['RANK'] = str(i)
        env['LOCAL_RANK'] = str(i)
        env['WORLD_SIZE'] = str(number_of_processes)
        env['LOCAL_WORLD_SIZE'] = str(number_of_processes)
        env['MASTER_ADDR'] = 'localhost'
        env['MASTER_PORT'] = '12345'
        p = multiprocessing.Process(target=fn, args=(env,))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
    for p in processes:
        assert p.exitcode == 0

def worker_fn_wrapper(fn):

    def wrapped_fn(env):
        update_environment_variables(env)
        local_rank = os.environ['LOCAL_RANK']
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
        init_distributed_environment()
        random.seed(42)
        torch.manual_seed(42)
        fn()
    return wrapped_fn

def create_expert_indices_with_redundancy(num_layers: int, num_logical_experts: int, total_physical_experts: int, redundancy_config: list[int]) -> torch.Tensor:
    """
    Create expert indices with redundancy.
    
    Args:
        num_layers: number of layers
        num_logical_experts: number of logical experts
        total_physical_experts: total number of physical experts
        redundancy_config: redundancy for each logical expert
    
    Returns:
        indices: Shape (num_layers, total_physical_experts)
    """
    assert sum(redundancy_config) == total_physical_experts
    assert len(redundancy_config) == num_logical_experts
    indices = torch.zeros(num_layers, total_physical_experts, dtype=torch.long)
    for layer in range(num_layers):
        physical_pos = 0
        for logical_expert_id, redundancy in enumerate(redundancy_config):
            for _ in range(redundancy):
                indices[layer, physical_pos] = logical_expert_id
                physical_pos += 1
    for layer in range(num_layers):
        indices[layer] = indices[layer][torch.randperm(indices.shape[1])]
    return indices

def create_expert_weights(num_layers: int, num_local_experts: int, hidden_sizes: list[int], rank: int, device: torch.device, physical_to_logical_mapping: torch.Tensor) -> list[list[torch.Tensor]]:
    """
    Create fake expert weights tensor for testing.
    
    Use `arange` to generate predictable weights values, based on logical
    expert ID.
    All replicas of the same logical expert should have the same weights.
    
    Args:
        physical_to_logical_mapping: Shape (num_layers, num_local_experts)
            mapping[layer, physical_pos] = logical_expert_id
    """
    expert_weights = []
    for layer in range(num_layers):
        layer_weights = []
        for weight_idx, hidden_size in enumerate(hidden_sizes):
            weight_tensor = torch.zeros(num_local_experts, hidden_size, device=device, dtype=torch.float32)
            for local_expert in range(num_local_experts):
                global_pos = rank * num_local_experts + local_expert
                logical_expert_id = physical_to_logical_mapping[layer, global_pos].item()
                base_value = logical_expert_id * 1000 + layer * 100 + weight_idx * 10
                weight_tensor[local_expert] = torch.arange(base_value, base_value + hidden_size, device=device, dtype=torch.float32)
            layer_weights.append(weight_tensor)
        expert_weights.append(layer_weights)
    return expert_weights

def create_redundancy_config(num_logical_experts: int, num_physical_experts: int) -> list[int]:
    """Create a redundancy configuration."""
    redundancy_config = [1] * num_logical_experts
    remaining = num_physical_experts - num_logical_experts
    for _ in range(remaining):
        redundancy_config[random.choice(range(num_logical_experts))] += 1
    return redundancy_config

def verify_expert_weights_after_shuffle(expert_weights: list[list[torch.Tensor]], new_indices: torch.Tensor, hidden_sizes: list[int], ep_rank: int, num_local_experts: int):
    """Verify the weights after shuffling are correct."""
    num_layers = len(expert_weights)
    for layer in range(num_layers):
        for weight_idx, hidden_size in enumerate(hidden_sizes):
            weight_tensor = expert_weights[layer][weight_idx]
            for local_expert in range(num_local_experts):
                global_pos = ep_rank * num_local_experts + local_expert
                expected_logical_expert = new_indices[layer, global_pos].item()
                actual_weights = weight_tensor[local_expert]
                expected_base = expected_logical_expert * 1000 + layer * 100 + weight_idx * 10
                expected_weights = torch.arange(expected_base, expected_base + hidden_size, device=actual_weights.device, dtype=actual_weights.dtype)
                torch.testing.assert_close(actual_weights, expected_weights, msg=f'Layer {layer}, weight {weight_idx},local expert {local_expert}: weights do not match. Expected logical expert {expected_logical_expert}')

def verify_redundant_experts_have_same_weights(expert_weights: list[list[torch.Tensor]], indices: torch.Tensor, hidden_sizes: list[int], world_size: int, num_local_experts: int):
    """
    Verify that all replicas of the same logical expert have the same weights.
    """
    num_layers = len(expert_weights)
    total_physical_experts = world_size * num_local_experts
    for layer in range(num_layers):
        all_weights: list[torch.Tensor] = []
        for weight_idx, hidden_size in enumerate(hidden_sizes):
            gathered_weights = torch.zeros(total_physical_experts, hidden_size, device=expert_weights[layer][weight_idx].device, dtype=expert_weights[layer][weight_idx].dtype)
            local_weights = expert_weights[layer][weight_idx]
            gathered_weights_list = torch.chunk(gathered_weights, world_size, dim=0)
            torch.distributed.all_gather(list(gathered_weights_list), local_weights)
            all_weights.append(gathered_weights)
        logical_expert_weights: dict[int, dict[int, torch.Tensor]] = {}
        for physical_pos in range(total_physical_experts):
            logical_expert_id = int(indices[layer, physical_pos].item())
            if logical_expert_id not in logical_expert_weights:
                logical_expert_weights[logical_expert_id] = {weight_idx: all_weights[weight_idx][physical_pos] for weight_idx in range(len(hidden_sizes))}
            else:
                for weight_idx in range(len(hidden_sizes)):
                    torch.testing.assert_close(all_weights[weight_idx][physical_pos], logical_expert_weights[logical_expert_id][weight_idx], msg=f'Layer {layer}, weight {weight_idx},logical expert {logical_expert_id}: Physical expert {physical_pos} has different weightsthan expected')

@pytest.mark.parametrize('world_size,num_layers,num_local_experts,num_logical_experts', [(2, 1, 2, 3), (2, 2, 3, 4), (2, 4, 8, 16), (4, 1, 2, 6), (4, 2, 2, 5), (4, 8, 8, 16)])
def test_rearrange_expert_weights_with_redundancy(world_size, num_layers, num_local_experts, num_logical_experts):
    """Test the functionality of rearranging expert weights with redundancy."""
    if torch.cuda.device_count() < world_size:
        pytest.skip(f'Need at least {world_size} GPUs to run the test')

    @worker_fn_wrapper
    def worker_fn():
        ensure_model_parallel_initialized(tensor_model_parallel_size=world_size, pipeline_model_parallel_size=1)
        ep_group = get_tp_group().cpu_group
        ep_rank = torch.distributed.get_rank()
        device = torch.device(f'cuda:{ep_rank}')
        total_physical_experts = world_size * num_local_experts
        hidden_sizes = [32, 64]
        redundancy_config = create_redundancy_config(num_logical_experts, total_physical_experts)
        old_indices = create_expert_indices_with_redundancy(num_layers, num_logical_experts, total_physical_experts, redundancy_config)
        new_redundancy_config = create_redundancy_config(num_logical_experts, total_physical_experts)
        new_indices = create_expert_indices_with_redundancy(num_layers, num_logical_experts, total_physical_experts, new_redundancy_config)
        expert_weights = create_expert_weights(num_layers, num_local_experts, hidden_sizes, ep_rank, device, old_indices)
        rearrange_expert_weights_inplace(old_indices, new_indices, expert_weights, ep_group, is_profile=False)
        verify_expert_weights_after_shuffle(expert_weights, new_indices, hidden_sizes, ep_rank, num_local_experts)
        verify_redundant_experts_have_same_weights(expert_weights, new_indices, hidden_sizes, world_size, num_local_experts)
    distributed_run(worker_fn, world_size)

@pytest.mark.parametrize('world_size', [2, 4])
def test_rearrange_expert_weights_no_change(world_size):
    """
    Test that when the indices do not change, the weights should remain
    unchanged.
    """
    if torch.cuda.device_count() < world_size:
        pytest.skip(f'Need at least {world_size} GPUs to run the test')

    @worker_fn_wrapper
    def worker_fn():
        ensure_model_parallel_initialized(tensor_model_parallel_size=world_size, pipeline_model_parallel_size=1)
        ep_group = get_tp_group().cpu_group
        ep_rank = torch.distributed.get_rank()
        device = torch.device(f'cuda:{ep_rank}')
        num_layers = 2
        num_local_experts = 2
        total_physical_experts = world_size * num_local_experts
        num_logical_experts = total_physical_experts // 2
        hidden_sizes = [32, 64]
        redundancy_config = [2] * num_logical_experts
        indices = create_expert_indices_with_redundancy(num_layers, num_logical_experts, total_physical_experts, redundancy_config)
        expert_weights = create_expert_weights(num_layers, num_local_experts, hidden_sizes, ep_rank, device, indices)
        original_weights = []
        for layer_weights in expert_weights:
            layer_copy = []
            for weight in layer_weights:
                layer_copy.append(weight.clone())
            original_weights.append(layer_copy)
        rearrange_expert_weights_inplace(indices, indices, expert_weights, ep_group, is_profile=False)
        for layer in range(num_layers):
            for weight_idx in range(len(hidden_sizes)):
                torch.testing.assert_close(expert_weights[layer][weight_idx], original_weights[layer][weight_idx], msg=f'Layer {layer}, weight {weight_idx} should remain unchanged')
    distributed_run(worker_fn, world_size)

@pytest.mark.parametrize('world_size', [2, 4])
def test_rearrange_expert_weights_profile_mode(world_size):
    """Test profile mode (should not copy actual weights)"""
    if torch.cuda.device_count() < world_size:
        pytest.skip(f'Need at least {world_size} GPUs to run the test')

    @worker_fn_wrapper
    def worker_fn():
        ensure_model_parallel_initialized(tensor_model_parallel_size=world_size, pipeline_model_parallel_size=1)
        ep_group = get_tp_group().cpu_group
        ep_rank = torch.distributed.get_rank()
        device = torch.device(f'cuda:{ep_rank}')
        num_layers = 1
        num_local_experts = 2
        total_physical_experts = world_size * num_local_experts
        num_logical_experts = total_physical_experts // 2
        hidden_sizes = [32]
        old_redundancy = create_redundancy_config(num_logical_experts, total_physical_experts)
        new_redundancy = create_redundancy_config(num_logical_experts, total_physical_experts)
        old_indices = create_expert_indices_with_redundancy(num_layers, num_logical_experts, total_physical_experts, old_redundancy)
        new_indices = create_expert_indices_with_redundancy(num_layers, num_logical_experts, total_physical_experts, new_redundancy)
        expert_weights = create_expert_weights(num_layers, num_local_experts, hidden_sizes, ep_rank, device, old_indices)
        original_weights = []
        for layer_weights in expert_weights:
            layer_copy = []
            for weight in layer_weights:
                layer_copy.append(weight.clone())
            original_weights.append(layer_copy)
        rearrange_expert_weights_inplace(old_indices, new_indices, expert_weights, ep_group, is_profile=True)
        for layer in range(num_layers):
            for weight_idx in range(len(hidden_sizes)):
                torch.testing.assert_close(expert_weights[layer][weight_idx], original_weights[layer][weight_idx], msg='In profile mode, the weights should remain unchanged')
    distributed_run(worker_fn, world_size)