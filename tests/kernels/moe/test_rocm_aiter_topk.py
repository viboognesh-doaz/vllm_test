from vllm.platforms import current_platform
import importlib.util
import pytest
import torch
import vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe
aiter_available = importlib.util.find_spec('aiter') is not None
pytestmark = pytest.mark.skipif(not (current_platform.is_rocm() and aiter_available), reason='AITER ops are only available on ROCm with aiter package installed')

def test_rocm_aiter_biased_grouped_topk_custom_op_registration():
    """Test that the custom op is correctly registered."""
    assert hasattr(torch.ops.vllm, 'rocm_aiter_biased_grouped_topk')
    assert callable(torch.ops.vllm.rocm_aiter_biased_grouped_topk)

def test_rocm_aiter_grouped_topk_custom_op_registration():
    """Test that the custom op is correctly registered."""
    assert hasattr(torch.ops.vllm, 'rocm_aiter_grouped_topk')
    assert callable(torch.ops.vllm.rocm_aiter_grouped_topk)

def test_rocm_aiter_biased_grouped_topk_torch_compile_compatibility():
    """Test that the op can be used with torch.compile."""
    token = 64
    expert = 256
    num_expert_group = 8
    topk = 8
    topk_group = 4
    renormalize = True
    scale_factor = 1.0
    gating_output = torch.randn((token, expert), dtype=torch.bfloat16, device='cuda')
    e_score_correction_bias = torch.randn((expert,), dtype=torch.bfloat16, device='cuda')
    device = gating_output.device
    topk_ids = torch.empty((token, topk), dtype=torch.int32, device=device)
    topk_weights = torch.empty((token, topk), dtype=torch.float32, device=device)

    def biased_grouped_topk_fn(gating_output, e_score_correction_bias, topk_weights, topk_ids):
        return torch.ops.vllm.rocm_aiter_biased_grouped_topk(gating_output, e_score_correction_bias, topk_weights, topk_ids, num_expert_group, topk_group, renormalize, scale_factor)
    torch.library.opcheck(torch.ops.vllm.rocm_aiter_biased_grouped_topk, (gating_output, e_score_correction_bias, topk_weights, topk_ids), kwargs={'num_expert_group': num_expert_group, 'topk_group': topk_group, 'need_renorm': renormalize, 'routed_scaling_factor': scale_factor}, test_utils='test_faketensor')
    compiled_fn = torch.compile(biased_grouped_topk_fn, fullgraph=True, backend='inductor', mode='reduce-overhead', dynamic=False)
    topk_weights_original = torch.empty((token, topk), dtype=torch.float32, device=device)
    topk_ids_original = torch.empty((token, topk), dtype=torch.int32, device=device)
    topk_weights_compiled = torch.empty((token, topk), dtype=torch.float32, device=device)
    topk_ids_compiled = torch.empty((token, topk), dtype=torch.int32, device=device)
    biased_grouped_topk_fn(gating_output, e_score_correction_bias, topk_weights_original, topk_ids_original)
    compiled_fn(gating_output, e_score_correction_bias, topk_weights_compiled, topk_ids_compiled)
    topk_ids_original, indices_original = torch.sort(topk_ids_original)
    topk_weights_original = torch.gather(topk_weights_original, 1, indices_original)
    topk_ids_compiled, indices_compiled = torch.sort(topk_ids_compiled)
    topk_weights_compiled = torch.gather(topk_weights_compiled, 1, indices_compiled)
    assert torch.allclose(topk_weights_original, topk_weights_compiled, rtol=0.01, atol=0.01)
    assert torch.allclose(topk_ids_original, topk_ids_compiled)

def test_rocm_aiter_grouped_topk_torch_compile_compatibility():
    """Test that the op can be used with torch.compile."""
    token = 64
    expert = 256
    num_expert_group = 8
    topk = 8
    topk_group = 4
    renormalize = True
    scoring_func = 'softmax'
    scale_factor = 1.0
    gating_output = torch.randn((token, expert), dtype=torch.bfloat16, device='cuda')
    device = gating_output.device
    topk_ids = torch.empty((token, topk), dtype=torch.int32, device=device)
    topk_weights = torch.empty((token, topk), dtype=torch.float32, device=device)

    def grouped_topk_fn(gating_output, topk_weights, topk_ids, scoring_func):
        return torch.ops.vllm.rocm_aiter_grouped_topk(gating_output, topk_weights, topk_ids, num_expert_group, topk_group, renormalize, scoring_func, scale_factor)
    torch.library.opcheck(torch.ops.vllm.rocm_aiter_grouped_topk, (gating_output, topk_weights, topk_ids), kwargs={'num_expert_group': num_expert_group, 'topk_group': topk_group, 'need_renorm': renormalize, 'scoring_func': scoring_func, 'routed_scaling_factor': scale_factor}, test_utils='test_faketensor')
    compiled_fn = torch.compile(grouped_topk_fn, fullgraph=True, backend='inductor', mode='reduce-overhead', dynamic=False)
    topk_weights_original = torch.empty((token, topk), dtype=torch.float32, device=device)
    topk_ids_original = torch.empty((token, topk), dtype=torch.int32, device=device)
    topk_weights_compiled = torch.empty((token, topk), dtype=torch.float32, device=device)
    topk_ids_compiled = torch.empty((token, topk), dtype=torch.int32, device=device)
    grouped_topk_fn(gating_output, topk_weights_original, topk_ids_original, scoring_func)
    compiled_fn(gating_output, topk_weights_compiled, topk_ids_compiled, scoring_func)
    topk_ids_original, indices_original = torch.sort(topk_ids_original)
    topk_weights_original = torch.gather(topk_weights_original, 1, indices_original)
    topk_ids_compiled, indices_compiled = torch.sort(topk_ids_compiled)
    topk_weights_compiled = torch.gather(topk_weights_compiled, 1, indices_compiled)
    assert torch.allclose(topk_weights_original, topk_weights_compiled, rtol=0.01, atol=0.01)
    assert torch.allclose(topk_ids_original, topk_ids_compiled)