from math import prod
from typing import Optional
from vllm import _custom_ops as ops
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe.cutlass_moe import cutlass_moe_fp8, run_cutlass_moe_fp8
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts, fused_topk
from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input
from vllm.platforms import current_platform
import dataclasses
import pytest
import torch
NUM_EXPERTS = [40, 64]
TOP_KS = [6, 8]
MNK_FACTORS = [(2, 1024, 1024), (2, 1024, 1536), (2, 3072, 1024), (2, 3072, 1536), (7, 3072, 1536), (64, 1024, 1024), (64, 1024, 1536), (64, 3072, 1024), (64, 3072, 1536), (224, 1024, 1024), (224, 1024, 1536), (224, 3072, 1024), (224, 3072, 1536), (32768, 1024, 1024)]
vllm_config = VllmConfig(parallel_config=ParallelConfig(pipeline_parallel_size=1))
vllm_config.scheduler_config.max_num_seqs = 128
vllm_config.scheduler_config.max_model_len = 8192

@dataclasses.dataclass
class MOETensors:
    a: torch.Tensor
    w1: torch.Tensor
    w2: torch.Tensor
    ab_strides1: torch.Tensor
    c_strides1: torch.Tensor
    ab_strides2: torch.Tensor
    c_strides2: torch.Tensor

    @staticmethod
    def make_moe_tensors(m: int, k: int, n: int, e: int, dtype: torch.dtype) -> 'MOETensors':
        a = torch.randn((m, k), device='cuda', dtype=dtype) / 10
        w1 = torch.randn((e, 2 * n, k), device='cuda', dtype=dtype) / 10
        w2 = torch.randn((e, k, n), device='cuda', dtype=dtype) / 10
        ab_strides1 = torch.full((e,), k, device='cuda', dtype=torch.int64)
        c_strides1 = torch.full((e,), 2 * n, device='cuda', dtype=torch.int64)
        ab_strides2 = torch.full((e,), n, device='cuda', dtype=torch.int64)
        c_strides2 = torch.full((e,), k, device='cuda', dtype=torch.int64)
        return MOETensors(a=a, w1=w1, w2=w2, ab_strides1=ab_strides1, c_strides1=c_strides1, ab_strides2=ab_strides2, c_strides2=c_strides2)

@dataclasses.dataclass
class MOETensors8Bit(MOETensors):
    a_q: Optional[torch.Tensor] = None
    w1_q: Optional[torch.Tensor] = None
    w2_q: Optional[torch.Tensor] = None
    a_scale: Optional[torch.Tensor] = None
    w1_scale: Optional[torch.Tensor] = None
    w2_scale: Optional[torch.Tensor] = None
    a_d: Optional[torch.Tensor] = None
    w1_d: Optional[torch.Tensor] = None
    w2_d: Optional[torch.Tensor] = None

    @staticmethod
    def make_moe_tensors_8bit(m: int, k: int, n: int, e: int, per_act_token: bool, per_out_channel: bool) -> 'MOETensors8Bit':
        dtype = torch.half
        q_dtype = torch.float8_e4m3fn
        moe_tensors_fp16 = MOETensors.make_moe_tensors(m, k, n, e, dtype)
        n_b_scales = 2 * n if per_out_channel else 1
        k_b_scales = k if per_out_channel else 1
        a_q, a_scale = ops.scaled_fp8_quant(moe_tensors_fp16.a, None, use_per_token_if_dynamic=per_act_token)
        w1_q = torch.empty((e, 2 * n, k), device='cuda', dtype=q_dtype)
        w2_q = torch.empty((e, k, n), device='cuda', dtype=q_dtype)
        w1_scale = torch.empty((e, n_b_scales, 1), device='cuda', dtype=torch.float32)
        w2_scale = torch.empty((e, k_b_scales, 1), device='cuda', dtype=torch.float32)
        for expert in range(e):
            w1_q[expert], w1_scale[expert] = ops.scaled_fp8_quant(moe_tensors_fp16.w1[expert], use_per_token_if_dynamic=per_out_channel)
            w2_q[expert], w2_scale[expert] = ops.scaled_fp8_quant(moe_tensors_fp16.w2[expert], use_per_token_if_dynamic=per_out_channel)
        a_d = a_q.float().mul(a_scale).to(dtype)
        w1_d = torch.empty_like(moe_tensors_fp16.w1)
        w2_d = torch.empty_like(moe_tensors_fp16.w2)
        for expert in range(e):
            w1_d[expert] = (w1_q[expert].float() * w1_scale[expert]).half()
            w2_d[expert] = (w2_q[expert].float() * w2_scale[expert]).half()
        return MOETensors8Bit(a=moe_tensors_fp16.a, w1=moe_tensors_fp16.w1, w2=moe_tensors_fp16.w2, ab_strides1=moe_tensors_fp16.ab_strides1, c_strides1=moe_tensors_fp16.c_strides1, ab_strides2=moe_tensors_fp16.ab_strides2, c_strides2=moe_tensors_fp16.c_strides2, a_q=a_q, w1_q=w1_q, w2_q=w2_q, a_scale=a_scale, w1_scale=w1_scale, w2_scale=w2_scale, a_d=a_d, w1_d=w1_d, w2_d=w2_d)

def run_with_expert_maps(num_experts: int, num_local_experts: int, **cutlass_moe_kwargs):

    def slice_experts():
        slice_params = ['w1_q', 'w2_q', 'ab_strides1', 'ab_strides2', 'c_strides1', 'c_strides2', 'w1_scale', 'w2_scale']
        full_tensors = {k: v for k, v in cutlass_moe_kwargs.items() if k in slice_params and k in cutlass_moe_kwargs}
        for i in range(0, num_experts, num_local_experts):
            s, e = (i, i + num_local_experts)
            expert_map = [-1] * num_experts
            expert_map[s:e] = list(range(num_local_experts))
            expert_map = torch.tensor(expert_map, dtype=torch.int32, device='cuda')
            cutlass_moe_kwargs['expert_map'] = expert_map
            for k, t in full_tensors.items():
                cutlass_moe_kwargs[k] = t[s:e]
            yield cutlass_moe_kwargs
    out_tensor = torch.zeros_like(cutlass_moe_kwargs['a'])
    for kwargs in slice_experts():
        out_tensor = out_tensor + cutlass_moe_fp8(**kwargs)
    return out_tensor

def run_8_bit(moe_tensors: MOETensors8Bit, topk_weights: torch.Tensor, topk_ids: torch.Tensor, per_act_token: bool, num_local_experts: Optional[int]=None) -> torch.Tensor:
    assert not any([t is None for t in [moe_tensors.w1_q, moe_tensors.w2_q, moe_tensors.w1_scale, moe_tensors.w2_scale, moe_tensors.a_scale]])
    kwargs = {'a': moe_tensors.a, 'w1_q': moe_tensors.w1_q, 'w2_q': moe_tensors.w2_q, 'topk_weights': topk_weights, 'topk_ids': topk_ids, 'w1_scale': moe_tensors.w1_scale, 'w2_scale': moe_tensors.w2_scale, 'ab_strides1': moe_tensors.ab_strides1, 'ab_strides2': moe_tensors.ab_strides2, 'c_strides1': moe_tensors.c_strides1, 'c_strides2': moe_tensors.c_strides2, 'per_act_token': per_act_token, 'a1_scale': None}
    num_experts = moe_tensors.w1.size(0)
    with_ep = num_local_experts is not None or num_local_experts == num_experts
    if not with_ep:
        return cutlass_moe_fp8(**kwargs)
    assert num_local_experts is not None
    return run_with_expert_maps(num_experts, num_local_experts, **kwargs)

@pytest.mark.parametrize('m,n,k', MNK_FACTORS)
@pytest.mark.parametrize('e', NUM_EXPERTS)
@pytest.mark.parametrize('topk', TOP_KS)
@pytest.mark.parametrize('per_act_token', [True, False])
@pytest.mark.parametrize('per_out_ch', [True, False])
@pytest.mark.skipif((lambda x: x is None or not ops.cutlass_group_gemm_supported(x.to_int()))(current_platform.get_device_capability()), reason='Grouped gemm is not supported on this GPU type.')
def test_cutlass_moe_8_bit_no_graph(m: int, n: int, k: int, e: int, topk: int, per_act_token: bool, per_out_ch: bool, monkeypatch, ep_size: Optional[int]=None):
    current_platform.seed_everything(7)
    monkeypatch.setenv('VLLM_FUSED_MOE_CHUNK_SIZE', '8192')
    with set_current_vllm_config(vllm_config):
        mt = MOETensors8Bit.make_moe_tensors_8bit(m, k, n, e, per_act_token, per_out_ch)
        score = torch.randn((m, e), device='cuda', dtype=torch.half)
        topk_weights, topk_ids, _ = fused_topk(mt.a, score, topk, renormalize=False)
        triton_output = fused_experts(mt.a_d, mt.w1_d, mt.w2_d, topk_weights, topk_ids)
        if ep_size is not None:
            assert e % ep_size == 0, 'Cannot distribute experts evenly'
            number_local_experts = e // ep_size
        else:
            number_local_experts = None
        cutlass_output = run_8_bit(mt, topk_weights, topk_ids, per_act_token, number_local_experts)
        torch.testing.assert_close(triton_output, cutlass_output, atol=0.055, rtol=0.01)

@pytest.mark.parametrize('m,n,k', MNK_FACTORS)
@pytest.mark.parametrize('e', NUM_EXPERTS)
@pytest.mark.parametrize('topk', TOP_KS)
@pytest.mark.parametrize('per_act_token', [True, False])
@pytest.mark.parametrize('per_out_ch', [True, False])
@pytest.mark.skipif((lambda x: x is None or not ops.cutlass_group_gemm_supported(x.to_int()))(current_platform.get_device_capability()), reason='Grouped gemm is not supported on this GPU type.')
def test_cutlass_moe_8_bit_cuda_graph(m: int, n: int, k: int, e: int, topk: int, per_act_token: bool, per_out_ch: bool, monkeypatch):
    current_platform.seed_everything(7)
    monkeypatch.setenv('VLLM_FUSED_MOE_CHUNK_SIZE', '8192')
    with set_current_vllm_config(vllm_config):
        dtype = torch.half
        mt = MOETensors8Bit.make_moe_tensors_8bit(m, k, n, e, per_act_token, per_out_ch)
        score = torch.randn((m, e), device='cuda', dtype=dtype)
        topk_weights, topk_ids, _ = fused_topk(mt.a, score, topk, renormalize=False)
        triton_output = fused_experts(mt.a_d, mt.w1_d, mt.w2_d, topk_weights, topk_ids)
        stream = torch.cuda.Stream()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            cutlass_output = run_8_bit(mt, topk_weights, topk_ids, per_act_token)
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(triton_output, cutlass_output, atol=0.09, rtol=0.01)

@pytest.mark.parametrize('m', [64])
@pytest.mark.parametrize('n', [1024])
@pytest.mark.parametrize('k', [4096])
@pytest.mark.parametrize('e', [16])
@pytest.mark.parametrize('topk', [1, 8])
@pytest.mark.parametrize('per_act_token', [True])
@pytest.mark.parametrize('per_out_channel', [True])
@pytest.mark.parametrize('ep_size', [1, 2, 4, 8, 16])
@pytest.mark.skipif((lambda x: x is None or not ops.cutlass_group_gemm_supported(x.to_int()))(current_platform.get_device_capability()), reason='Grouped gemm is not supported on this GPU type.')
def test_cutlass_moe_8_bit_EP(m: int, n: int, k: int, e: int, topk: int, per_act_token: bool, per_out_channel: bool, ep_size: int, monkeypatch):
    test_cutlass_moe_8_bit_no_graph(m, n, k, e, topk, per_act_token, per_out_channel, monkeypatch, ep_size)
LARGE_MNK_FACTORS = [(1, 8192, 5120, 31), (32768, 1024, 1024, 16), (65536, 512, 1024, 16)]

@pytest.mark.parametrize('m,n,k,topk', LARGE_MNK_FACTORS)
@pytest.mark.parametrize('e', [128])
@pytest.mark.parametrize('per_act_token', [False])
@pytest.mark.parametrize('per_out_channel', [True])
@pytest.mark.parametrize('ep_size', [8])
@pytest.mark.skipif((lambda x: x is None or not ops.cutlass_group_gemm_supported(x.to_int()))(current_platform.get_device_capability()), reason='Grouped gemm is not supported on this GPU type.')
def test_cutlass_moe_8_bit_EP_large(m: int, n: int, k: int, e: int, topk: int, per_act_token: bool, per_out_channel: bool, ep_size: int, monkeypatch):
    test_cutlass_moe_8_bit_no_graph(m, n, k, e, topk, per_act_token, per_out_channel, monkeypatch, ep_size)

@pytest.mark.parametrize('m,n,k,topk', [(1, 8192, 5120, 31)])
@pytest.mark.parametrize('e', [128])
@pytest.mark.parametrize('per_act_token', [False])
@pytest.mark.parametrize('per_out_channel', [True])
@pytest.mark.parametrize('ep_size', [8])
@pytest.mark.skipif((lambda x: x is None or not ops.cutlass_group_gemm_supported(x.to_int()))(current_platform.get_device_capability()), reason='Grouped gemm is not supported on this GPU type.')
def test_run_cutlass_moe_fp8(m: int, n: int, k: int, e: int, topk: int, per_act_token: bool, per_out_channel: bool, ep_size: int):
    current_platform.seed_everything(7)
    with set_current_vllm_config(vllm_config):
        mt = MOETensors8Bit.make_moe_tensors_8bit(m, k, n, e, per_act_token, per_out_channel)
        score = torch.randn((m, e), device='cuda', dtype=torch.half)
        topk_weights, topk_ids, _ = fused_topk(mt.a, score, topk, renormalize=False)
        topk_ids[0][0] = -1
        topk_ids[0][1] = 1
        workspace13_shape = (m * topk, max(2 * n, k))
        workspace2_shape = (m * topk, n)
        output_shape = (m * topk, k)
        workspace13 = torch.empty(prod(workspace13_shape), device='cuda', dtype=mt.a.dtype)
        workspace2 = torch.empty(prod(workspace2_shape), device='cuda', dtype=mt.a.dtype)
        num_local_experts = e // ep_size
        start, end = (0, num_local_experts)
        expert_map = [-1] * e
        expert_map[start:end] = list(range(num_local_experts))
        expert_map = torch.tensor(expert_map, dtype=torch.int32, device='cuda')
        ab_strides1 = torch.full((e,), k, device='cuda', dtype=torch.int64)
        ab_strides2 = torch.full((e,), n, device='cuda', dtype=torch.int64)
        c_strides1 = torch.full((e,), 2 * n, device='cuda', dtype=torch.int64)
        c_strides2 = torch.full((e,), k, device='cuda', dtype=torch.int64)
        activation = lambda o, i: torch.ops._C.silu_and_mul(o, i)
        a1q, a1q_scale = moe_kernel_quantize_input(mt.a, mt.a_scale, torch.float8_e4m3fn, per_act_token)
        global_num_experts = -1 if mt.w1_q is None else mt.w1_q.size(0)
        func = lambda output: run_cutlass_moe_fp8(output, a1q, mt.w1_q, mt.w2_q, topk_ids, activation, global_num_experts, expert_map, mt.w1_scale, mt.w2_scale, a1q_scale, None, ab_strides1, ab_strides2, c_strides1, c_strides2, workspace13, workspace2, None, mt.a.dtype, per_act_token, per_out_channel, False)
        workspace13.random_()
        output_random_workspace = torch.empty(output_shape, device='cuda', dtype=mt.a.dtype)
        func(output_random_workspace)
        workspace13.fill_(0)
        output_zero_workspace = torch.zeros(output_shape, device='cuda', dtype=mt.a.dtype)
        func(output_zero_workspace)
        torch.testing.assert_close(output_random_workspace, output_zero_workspace, atol=0.005, rtol=0.001)