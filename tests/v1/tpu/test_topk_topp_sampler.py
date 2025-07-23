from vllm.platforms import current_platform
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p, apply_top_k_top_p_tpu
import math
import pytest
import torch
import torch_xla.core.xla_model as xm
if not current_platform.is_tpu():
    pytest.skip('This test needs a TPU.', allow_module_level=True)
BATCH_SIZE = 1024
VOCAB_SIZE = 128 * 1024
TOLERANCE = 1e-06

def test_topk_equivalence_to_native_impl():
    with torch.device(xm.xla_device()):
        xm.set_rng_state(seed=33)
        logits = torch.rand((BATCH_SIZE, VOCAB_SIZE))
        k = torch.randint(1, 10, (BATCH_SIZE,))
        k.masked_fill_(torch.randint(0, 2, (BATCH_SIZE,), dtype=bool), VOCAB_SIZE)
        result_tpu = apply_top_k_top_p_tpu(logits=logits.clone(), k=k, p=None)
        result_native = apply_top_k_top_p(logits=logits.clone(), k=k, p=None)
        assert torch.allclose(result_native, result_tpu)

def test_topp_result_sums_past_p():
    with torch.device(xm.xla_device()):
        xm.set_rng_state(seed=33)
        logits = torch.rand((BATCH_SIZE, VOCAB_SIZE))
        probs = logits.softmax(dim=-1)
        p = torch.rand((BATCH_SIZE,))
        p.masked_fill_(torch.randint(0, 2, (BATCH_SIZE,), dtype=bool), 1)
        no_op_k = torch.tensor([VOCAB_SIZE])
        logits_masked = apply_top_k_top_p_tpu(logits=logits.clone(), k=no_op_k, p=p)
        probs.masked_fill_(logits_masked.isinf(), 0)
        masked_prob_sum = probs.sum(dim=-1)
        xm.mark_step()
    assert torch.all(torch.ge(masked_prob_sum.cpu() + TOLERANCE, p.cpu()))

def test_topp_basic():
    with torch.device(xm.xla_device()):
        logits = torch.tensor([[math.log(0.2), math.log(0.3), math.log(0.5)], [math.log(0.5), math.log(0.1), math.log(0.4)]])
        result = apply_top_k_top_p_tpu(logits=logits.clone(), k=torch.tensor([3, 3]), p=torch.tensor([0.79, 0.79]))
        xm.mark_step()
    expected_result = logits.clone().cpu()
    expected_result[0, 0] = float('-inf')
    expected_result[1, 1] = float('-inf')
    assert torch.allclose(expected_result, result.cpu())

def test_topp_select_all():
    with torch.device(xm.xla_device()):
        logits = torch.tensor([[math.log(0.2), math.log(0.3), math.log(0.5)], [math.log(0.5), math.log(0.1), math.log(0.4)]])
        result = apply_top_k_top_p_tpu(logits=logits.clone(), k=torch.tensor([3, 3]), p=torch.tensor([1.0, 1.0]))
        xm.mark_step()
    assert torch.allclose(logits.cpu(), result.cpu())

def test_topp_with_ties():
    with torch.device(xm.xla_device()):
        logits = torch.tensor([[math.log(0.3), math.log(0.3), math.log(0.3), math.log(0.1)]])
        result = apply_top_k_top_p_tpu(logits=logits.clone(), k=torch.tensor([4]), p=torch.tensor([0.2]))
        xm.mark_step()
    expected_result = logits.clone().cpu()
    expected_result[0, 3] = float('-inf')
    assert torch.allclose(expected_result, result.cpu())

def test_both_topk_topp():
    with torch.device(xm.xla_device()):
        logits = torch.tensor([[math.log(0.2), math.log(0.3), math.log(0.5)], [math.log(0.5), math.log(0.1), math.log(0.4)]])
        result = apply_top_k_top_p_tpu(logits=logits.clone(), k=torch.tensor([1, 3]), p=torch.tensor([0.79, 0.79]))
        xm.mark_step()
    expected_result = logits.clone().cpu()
    expected_result[0, 0] = float('-inf')
    expected_result[0, 1] = float('-inf')
    expected_result[1, 1] = float('-inf')
    assert torch.allclose(expected_result, result.cpu())