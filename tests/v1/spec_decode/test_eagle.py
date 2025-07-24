from vllm.model_executor.models import SupportsMultiModal
from unittest import mock
import pytest
import torch
from tests.v1.attention.utils import BatchSpec, _Backend, create_common_attn_metadata, create_standard_kv_cache_spec, get_attention_backend
from vllm.config import CacheConfig, DeviceConfig, LoadConfig, ModelConfig, ParallelConfig, SchedulerConfig, SpeculativeConfig, VllmConfig
from vllm.model_executor.models.llama import LlamaForCausalLM
from vllm.platforms import current_platform
from vllm.v1.spec_decode.eagle import EagleProposer
model_dir = 'meta-llama/Llama-3.1-8B-Instruct'
eagle_dir = 'yuhuili/EAGLE-LLaMA3.1-Instruct-8B'
eagle3_dir = 'yuhuili/EAGLE3-LLaMA3.1-Instruct-8B'

def _create_proposer(method: str, k: int) -> EagleProposer:
    model_config = ModelConfig(model=model_dir, task='generate', max_model_len=100, tokenizer=model_dir, tokenizer_mode='auto', dtype='auto', seed=None, trust_remote_code=False)
    draft_model_dir = eagle_dir if method == 'eagle' else eagle3_dir
    speculative_config = SpeculativeConfig(target_model_config=model_config, target_parallel_config=ParallelConfig(), model=draft_model_dir, method=method, num_speculative_tokens=k)
    vllm_config = VllmConfig(model_config=model_config, cache_config=CacheConfig(), speculative_config=speculative_config, device_config=DeviceConfig(device=current_platform.device_type), parallel_config=ParallelConfig(), load_config=LoadConfig(), scheduler_config=SchedulerConfig())
    return EagleProposer(vllm_config=vllm_config, device=current_platform.device_type)

def test_prepare_inputs():
    """
    cu_target_query_lens: [0, a, a + b, a + b + c]
    num_rejected_tokens: [n1, n2, n3]
    num_tokens_per_req: [a - n1, b - n2, c - n3]
    cu_num_tokens: [0, a - n1, a + b - n1 - n2, a + b + c - n1 - n2 - n3]
    token_indices: [0, 1, ..., a - n1 - 1,
                    a, a + 1, ..., a + b - n2 - 1,
                    a + b, a + b + 1, ..., a + b + c - n3 - 1]
    """
    device = torch.device(current_platform.device_type)
    batch_spec = BatchSpec(seq_lens=[4, 7, 5], query_lens=[4, 7, 5])
    common_attn_metadata = create_common_attn_metadata(batch_spec, block_size=16, device=device)
    num_rejected_tokens = torch.tensor([1, 3, 2], dtype=torch.int32, device=device)
    expected_cu_num_tokens = torch.tensor([0, 3, 7, 10], dtype=torch.int32, device=device)
    expected_token_indices = torch.tensor([0, 1, 2, 4, 5, 6, 7, 11, 12, 13], dtype=torch.int32, device=device)
    proposer = _create_proposer('eagle', 1)
    updated_metadata, token_indices = proposer.prepare_inputs(common_attn_metadata, num_rejected_tokens.cpu())
    assert torch.equal(updated_metadata.query_start_loc, expected_cu_num_tokens)
    assert token_indices.shape[0] == expected_cu_num_tokens[-1].item()
    assert torch.equal(token_indices, expected_token_indices)

@pytest.mark.parametrize('method,proposer_helper', [('eagle', lambda k: _create_proposer('eagle', k)), ('eagle3', lambda k: _create_proposer('eagle3', k))])
@pytest.mark.parametrize('pp_size', [1, 2])
@pytest.mark.parametrize('use_distinct_embed_tokens', [True, False])
@mock.patch('vllm.v1.spec_decode.eagle.get_pp_group')
@mock.patch('vllm.v1.spec_decode.eagle.get_layers_from_vllm_config')
@mock.patch('vllm.v1.spec_decode.eagle.get_model')
def test_load_model(mock_get_model, mock_get_layers, mock_get_pp_group, method, proposer_helper, pp_size, use_distinct_embed_tokens):
    mock_model = mock.MagicMock()
    if use_distinct_embed_tokens:
        mock_model.model.embed_tokens.weight.shape = (131072, 2048)
    else:
        mock_model.model.embed_tokens.weight.shape = (131072, 4096)
    mock_get_model.return_value = mock_model
    target_attn_layers = {'target_attn_1': mock.MagicMock(), 'target_attn_2': mock.MagicMock()}
    all_attn_layers = {**target_attn_layers, 'draft_extra_attn': mock.MagicMock()}
    mock_get_layers.side_effect = [target_attn_layers, all_attn_layers]
    mock_pp_group = mock.MagicMock()
    mock_pp_group.world_size = pp_size
    mock_get_pp_group.return_value = mock_pp_group

    class _TargetModelStub(LlamaForCausalLM):
        model: mock.MagicMock
        lm_head: mock.MagicMock
    target_model = mock.create_autospec(_TargetModelStub, instance=True)
    target_model.model = mock.MagicMock()
    target_model.model.embed_tokens.weight.shape = (131072, 4096)
    assert not isinstance(target_model, SupportsMultiModal)
    if method == 'eagle':
        target_model.lm_head = mock.MagicMock()
    proposer = proposer_helper(k=8)
    proposer.load_model(target_model)
    mock_get_model.assert_called_once()
    if method == 'eagle':
        assert proposer.model.lm_head == target_model.lm_head
    if pp_size > 1 or use_distinct_embed_tokens:
        assert proposer.model.model.embed_tokens != target_model.model.embed_tokens
    else:
        assert proposer.model.model.embed_tokens == target_model.model.embed_tokens

@pytest.mark.parametrize('num_speculative_tokens', [1, 3, 8])
def test_propose(num_speculative_tokens):
    device = torch.device(current_platform.device_type)
    batch_size = 2
    seq_len_1 = 5
    seq_len_2 = 3
    total_tokens = seq_len_1 + seq_len_2
    vocab_size = 100
    seq_lens = [seq_len_1, seq_len_2]
    proposer = _create_proposer('eagle', num_speculative_tokens)
    hidden_size = proposer.hidden_size

    def create_deterministic_logits(token_ids):
        logits = torch.full((batch_size, vocab_size), -100.0, device=device)
        for i, token_id in enumerate(token_ids):
            logits[i, token_id] = 100.0
        return logits
    base_token_ids = [42, 60]
    model_mock = mock.MagicMock()
    forward_returns = []
    for i in range(num_speculative_tokens):
        if i == 0:
            h_logits = torch.zeros(total_tokens, hidden_size, device=device)
            h_states = torch.zeros(total_tokens, hidden_size, device=device)
        else:
            h_logits = torch.zeros(batch_size, hidden_size, device=device)
            h_states = torch.zeros(batch_size, hidden_size, device=device)
        forward_returns.append((h_logits, h_states))
    if num_speculative_tokens == 1:
        model_mock.return_value = forward_returns[0]
    else:
        model_mock.side_effect = forward_returns
    logits_returns = []
    for i in range(num_speculative_tokens):
        current_tokens = [base_id + i for base_id in base_token_ids]
        logits_returns.append(create_deterministic_logits(current_tokens))
    if num_speculative_tokens == 1:
        model_mock.compute_logits.return_value = logits_returns[0]
    else:
        model_mock.compute_logits.side_effect = logits_returns
    proposer.model = model_mock
    proposer.attn_layer_names = ['layer.0']
    batch_spec = BatchSpec(seq_lens=seq_lens, query_lens=seq_lens)
    common_attn_metadata = create_common_attn_metadata(batch_spec, block_size=16, device=device)
    target_token_ids = torch.randint(0, vocab_size, (total_tokens,), device=device)
    target_positions = torch.cat([torch.arange(seq_len_1, device=device), torch.arange(seq_len_2, device=device)])
    target_hidden_states = torch.randn(total_tokens, hidden_size, device=device)
    next_token_ids = torch.randint(0, vocab_size, (batch_size,), dtype=torch.int32, device=device)
    sampling_metadata = mock.MagicMock()
    attn_metadata_builder_cls, _ = get_attention_backend(_Backend.FLASH_ATTN_VLLM_V1)
    attn_metadata_builder = attn_metadata_builder_cls(kv_cache_spec=create_standard_kv_cache_spec(proposer.vllm_config), vllm_config=proposer.vllm_config, device=device)
    proposer.runner = mock.MagicMock()
    proposer.runner.attn_metadata_builders = [attn_metadata_builder]
    result = proposer.propose(target_token_ids=target_token_ids, target_positions=target_positions, target_hidden_states=target_hidden_states, next_token_ids=next_token_ids, common_attn_metadata=common_attn_metadata, sampling_metadata=sampling_metadata)
    assert result.shape == (batch_size, num_speculative_tokens)
    if num_speculative_tokens == 1:
        expected_tokens = torch.tensor([[base_token_ids[0]], [base_token_ids[1]]], device=device)
    else:
        expected_tokens = torch.zeros((batch_size, num_speculative_tokens), dtype=torch.int64, device=device)
        for i in range(batch_size):
            for j in range(num_speculative_tokens):
                expected_tokens[i, j] = base_token_ids[i] + j
    assert torch.equal(result, expected_tokens)