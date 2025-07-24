from vllm.compilation.backends import set_model_tag
import numpy as np
import torch
import torch.nn as nn
from vllm.attention.layer import Attention
from vllm.config import CompilationLevel, VllmConfig, get_layers_from_vllm_config
from vllm.distributed.parallel_state import get_pp_group
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.models import supports_multimodal
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM
from vllm.utils import is_pin_memory_available
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.sample.metadata import SamplingMetadata
logger = init_logger(__name__)
PADDING_SLOT_ID = -1

class EagleProposer:

    def __init__(self, vllm_config: VllmConfig, device: torch.device, runner=None):
        self.vllm_config = vllm_config
        self.speculative_config = vllm_config.speculative_config
        self.draft_model_config = self.speculative_config.draft_model_config
        self.method = self.speculative_config.method
        self.runner = runner
        self.dtype = vllm_config.model_config.dtype
        self.max_model_len = vllm_config.model_config.max_model_len
        self.block_size = vllm_config.cache_config.block_size
        self.num_speculative_tokens = self.speculative_config.num_speculative_tokens
        self.max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.token_arange_np = np.arange(self.max_num_tokens)
        self.hidden_size = self.draft_model_config.get_hidden_size()
        self.use_cuda_graph = self.vllm_config.compilation_config.level == CompilationLevel.PIECEWISE and (not self.vllm_config.model_config.enforce_eager)
        self.cudagraph_batch_sizes = list(reversed(self.vllm_config.compilation_config.cudagraph_capture_sizes))
        self.input_ids = torch.zeros(self.max_num_tokens, dtype=torch.int32, device=device)
        self.positions = torch.zeros(self.max_num_tokens, dtype=torch.int64, device=device)
        self.hidden_states = torch.zeros((self.max_num_tokens, self.hidden_size), dtype=self.dtype, device=device)
        self.arange = torch.arange(vllm_config.scheduler_config.max_num_seqs + 1, device=device, dtype=torch.int32)

    def propose(self, target_token_ids: torch.Tensor, target_positions: torch.Tensor, target_hidden_states: torch.Tensor, next_token_ids: torch.Tensor, common_attn_metadata: CommonAttentionMetadata, sampling_metadata: SamplingMetadata) -> torch.Tensor:
        num_tokens = target_token_ids.shape[0]
        batch_size = next_token_ids.shape[0]
        last_token_indices = common_attn_metadata.query_start_loc[1:] - 1
        if self.method == 'eagle3':
            assert isinstance(self.model, Eagle3LlamaForCausalLM)
            target_hidden_states = self.model.combine_hidden_states(target_hidden_states)
            assert target_hidden_states.shape[-1] == self.hidden_size
        self.input_ids[:num_tokens - 1] = target_token_ids[1:]
        self.input_ids[last_token_indices] = next_token_ids
        assert self.runner is not None
        attn_metadata = self.runner.attn_metadata_builders[0].build(common_prefix_len=0, common_attn_metadata=common_attn_metadata, fast_build=True)
        per_layer_attn_metadata = {}
        for layer_name in self.attn_layer_names:
            per_layer_attn_metadata[layer_name] = attn_metadata
        if self.use_cuda_graph and num_tokens <= self.cudagraph_batch_sizes[-1]:
            num_input_tokens = self.vllm_config.pad_for_cudagraph(num_tokens)
        else:
            num_input_tokens = num_tokens
        self.positions[:num_tokens] = target_positions
        self.hidden_states[:num_tokens] = target_hidden_states
        with set_forward_context(per_layer_attn_metadata, self.vllm_config, num_tokens=num_input_tokens):
            ret_hidden_states = self.model(self.input_ids[:num_input_tokens], self.positions[:num_input_tokens], self.hidden_states[:num_input_tokens])
            if self.method == 'deepseek_mtp':
                last_hidden_states = ret_hidden_states
            else:
                last_hidden_states, hidden_states = ret_hidden_states
        sample_hidden_states = last_hidden_states[last_token_indices]
        logits = self.model.compute_logits(sample_hidden_states, None)
        draft_token_ids = logits.argmax(dim=-1)
        if self.num_speculative_tokens == 1:
            return draft_token_ids.view(-1, 1)
        assert isinstance(attn_metadata, FlashAttentionMetadata)
        draft_token_ids_list = [draft_token_ids]
        positions = target_positions[last_token_indices]
        hidden_states = hidden_states[last_token_indices]
        if self.use_cuda_graph and batch_size <= self.cudagraph_batch_sizes[-1]:
            input_batch_size = self.vllm_config.pad_for_cudagraph(batch_size)
        else:
            input_batch_size = batch_size
        attn_metadata.num_actual_tokens = batch_size
        attn_metadata.max_query_len = 1
        attn_metadata.query_start_loc = self.arange[:batch_size + 1]
        for _ in range(self.num_speculative_tokens - 1):
            input_ids = draft_token_ids_list[-1].int()
            positions += 1
            exceeds_max_model_len = positions >= self.max_model_len
            clamped_positions = torch.where(exceeds_max_model_len, 0, positions)
            attn_metadata.max_seq_len += 1
            attn_metadata.seq_lens += 1
            attn_metadata.max_seq_len = min(attn_metadata.max_seq_len, self.max_model_len)
            attn_metadata.seq_lens.masked_fill_(exceeds_max_model_len, 1)
            block_numbers = clamped_positions // self.block_size
            block_ids = attn_metadata.block_table.gather(dim=1, index=block_numbers.view(-1, 1))
            block_ids = block_ids.view(-1)
            attn_metadata.slot_mapping = block_ids * self.block_size + clamped_positions % self.block_size
            attn_metadata.slot_mapping.masked_fill_(exceeds_max_model_len, PADDING_SLOT_ID)
            self.input_ids[:batch_size] = input_ids
            self.positions[:batch_size] = clamped_positions
            self.hidden_states[:batch_size] = hidden_states
            with set_forward_context(per_layer_attn_metadata, self.vllm_config, num_tokens=input_batch_size):
                last_hidden_states, hidden_states = self.model(self.input_ids[:input_batch_size], self.positions[:input_batch_size], self.hidden_states[:input_batch_size])
            hidden_states = hidden_states[:batch_size]
            logits = self.model.compute_logits(last_hidden_states[:batch_size], None)
            draft_token_ids = logits.argmax(dim=-1)
            draft_token_ids_list.append(draft_token_ids)
        draft_token_ids = torch.stack(draft_token_ids_list, dim=1)
        return draft_token_ids

    def prepare_inputs(self, common_attn_metadata: CommonAttentionMetadata, num_rejected_tokens: torch.Tensor) -> tuple[CommonAttentionMetadata, torch.Tensor]:
        """
        This function is used to prepare the inputs for the spec decode.
        It updates to the common_attn_metadata to account for the rejected
        tokens (and newly sampled tokens). It also returns the token indices
        of the tokens that should be fed to the speculator.
        """
        device = common_attn_metadata.query_start_loc.device
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        new_seq_lens_cpu = common_attn_metadata.seq_lens_cpu - num_rejected_tokens
        new_query_len_per_req = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        new_num_tokens_per_req = new_query_len_per_req - num_rejected_tokens
        new_num_tokens_per_req_np = new_num_tokens_per_req.numpy()
        new_query_start_loc_cpu = torch.zeros(query_start_loc_cpu.shape, dtype=torch.int32, pin_memory=is_pin_memory_available())
        new_query_start_loc_np = new_query_start_loc_cpu.numpy()
        np.cumsum(new_num_tokens_per_req_np, out=new_query_start_loc_np[1:])
        total_num_tokens = new_query_start_loc_np[-1]
        new_query_start_locs_expanded = np.repeat(new_query_start_loc_np[:-1], new_num_tokens_per_req_np)
        token_offests = self.token_arange_np[:total_num_tokens] - new_query_start_locs_expanded
        old_query_start_locs_expanded = np.repeat(query_start_loc_cpu[:-1].numpy(), new_num_tokens_per_req_np)
        token_indices_np = token_offests + old_query_start_locs_expanded
        token_indices = torch.from_numpy(token_indices_np).to(device, non_blocking=True)
        spec_common_attn_metadata = CommonAttentionMetadata(query_start_loc=new_query_start_loc_cpu.to(device, non_blocking=True), seq_lens=new_seq_lens_cpu.to(device, non_blocking=True), query_start_loc_cpu=new_query_start_loc_cpu, seq_lens_cpu=new_seq_lens_cpu, num_computed_tokens_cpu=common_attn_metadata.num_computed_tokens_cpu, num_reqs=common_attn_metadata.num_reqs, num_actual_tokens=total_num_tokens, max_query_len=new_query_len_per_req.max().item(), block_table_tensor=common_attn_metadata.block_table_tensor, slot_mapping=common_attn_metadata.slot_mapping[token_indices])
        return (spec_common_attn_metadata, token_indices)

    def load_model(self, target_model: nn.Module) -> None:
        draft_model_config = self.vllm_config.speculative_config.draft_model_config
        target_attn_layer_names = set(get_layers_from_vllm_config(self.vllm_config, Attention).keys())
        with set_model_tag('eagle_head'):
            self.model = get_model(vllm_config=self.vllm_config, model_config=draft_model_config)
        draft_attn_layer_names = get_layers_from_vllm_config(self.vllm_config, Attention).keys() - target_attn_layer_names
        self.attn_layer_names = list(draft_attn_layer_names)
        if supports_multimodal(target_model):
            self.model.config.image_token_index = target_model.config.image_token_index
            target_language_model = target_model.get_language_model()
        else:
            target_language_model = target_model
        if get_pp_group().world_size == 1 and self.model.model.embed_tokens.weight.shape == target_language_model.model.embed_tokens.weight.shape:
            logger.info('Assuming the EAGLE head shares the same vocab embedding with the target model.')
            del self.model.model.embed_tokens
            self.model.model.embed_tokens = target_language_model.model.embed_tokens
        else:
            logger.info("The EAGLE head's vocab embedding will be loaded separately from the target model.")
        if self.vllm_config.speculative_config.method != 'eagle3' and hasattr(target_language_model, 'lm_head'):
            logger.info('Loading EAGLE LM head weights from the target model.')
            self.model.lm_head = target_language_model.lm_head

    @torch.inference_mode()
    def dummy_run(self, num_tokens: int) -> None:
        with set_forward_context(None, self.vllm_config, num_tokens=num_tokens):
            self.model(self.input_ids[:num_tokens], self.positions[:num_tokens], self.hidden_states[:num_tokens])

    def validate_same_kv_cache_group(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Validate that all eagle layers belong to the same KVCacheGroup.
        Need this assumption to ensure all eagle layers can use the
        same AttentionMetadata.
        May extend to multiple AttentionMetadata in the future.
        """
        kv_cache_groups: dict[str, int] = {}
        for id, kv_cache_group in enumerate(kv_cache_config.kv_cache_groups):
            for layer_name in kv_cache_group.layer_names:
                kv_cache_groups[layer_name] = id
        assert len(set([kv_cache_groups[layer_name] for layer_name in self.attn_layer_names])) == 1, 'All eagle layers should belong to the same kv cache group'

def compute_probs_and_sample_next_token(logits: torch.Tensor, sampling_metadata: SamplingMetadata) -> tuple[torch.Tensor, torch.Tensor]:
    if sampling_metadata.all_greedy:
        probs = logits
        next_token_ids = logits.argmax(dim=-1)
        return (next_token_ids, probs)
    is_greedy = sampling_metadata.temperature == -1
    temperature = torch.where(is_greedy, 1.0, sampling_metadata.temperature)
    logits.div_(temperature.view(-1, 1))
    probs = logits.softmax(dim=-1, dtype=torch.float32)
    q = torch.empty_like(probs)
    q.exponential_()
    next_token_ids = probs.div(q).argmax(dim=-1).view(-1)
    if not sampling_metadata.all_random:
        greedy_token_ids = probs.argmax(dim=-1)
        next_token_ids = torch.where(is_greedy, greedy_token_ids, next_token_ids)
    return (next_token_ids, probs)