from abc import abstractmethod
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from flash_attn import flash_attn_varlen_func
from itertools import accumulate
from typing import TYPE_CHECKING, Any, Dict, Generic, List, Optional, Tuple, Type, TypeVar
from vllm import _custom_ops as ops
from vllm import envs
from vllm.attention.backends.abstract import AttentionBackend, AttentionLayer, AttentionMetadata, AttentionMetadataBuilder, AttentionState, MLAAttentionImpl
from vllm.attention.backends.utils import PAD_SLOT_ID, compute_slot_mapping, compute_slot_mapping_start_idx, is_block_tables_empty
from vllm.attention.ops.merge_attn_states import merge_attn_states
from vllm.attention.ops.triton_flash_attention import triton_attention
from vllm.attention.utils.fa_utils import get_flash_attn_version
from vllm.model_executor.layers.linear import ColumnParallelLinear, LinearBase, UnquantizedLinearMethod
from vllm.multimodal import MultiModalPlaceholderMap
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON
from vllm.utils import async_tensor_h2d, cdiv, make_tensor_with_pad, round_down
from vllm.vllm_flash_attn import flash_attn_varlen_func
from vllm.worker.model_runner import ModelInputForGPUBuilder, ModelInputForGPUWithSamplingMetadata
import functools
import torch
'\n# MLA Common Components\n\nThis file implements common components for MLA implementations.\n\nFirst we define:\n\nSq      as Q sequence length\nSkv     as KV sequence length\n\nMLA has two possible ways of computing, a data-movement friendly approach and a\ncompute friendly approach, we generally want to use the compute friendly\napproach for "prefill" (i.e. the ratio Sq / Skv is "small", is near 1)\nand the data-movement friendly approach for "decode" (i.e. the ratio\nSq / Skv is "large").\n\nNOTE what we deem small and large is currently determined by if its labelled\nprefill or decode by the scheduler, but this is something we should probably\ntune.\n\nMain reference: DeepseekV2 paper, and FlashInfer Implementation\n(https://arxiv.org/abs/2405.04434 and https://github.com/flashinfer-ai/flashinfer/pull/551).\n\nDeepseek\'s MLA attention works the following way:\n* Use a single latent vector to represent the per-token entry of the KV cache.\n* For decode (i.e. the memory friendly approach) the attention "simulates" a\nmulti-head attention, while the compute is similar to multi-query attention.\n\nBelow is example of both paths assuming batchsize = 1\n\n## More Extent Definitions:\n\nC           Context length, `Skv - Sq`\nH           hidden size\nN           number of attention heads\nLq          latent dimension for Q              1536 in DSV3\nLkv         latent dimension for K/V            512 in DSV3\nP           nope dimension, no rope.            128 in DSV3\nR           rope dimension, goes through rope.  64 in DSV3\nV           V head dim.                         128 in DSV3\n\n## Vector/Matrix Definitions\n\nh_t         hidden states (input to attention)  shape [Sq, H]\nq_c         latent/compressed Q                 shape [Sq, Lq]\nq_nope      uncompressed Q (no-rope)            shape [Sq, N, P]\nq_pe        uncompressed Q (rope)               shape [Sq, N, R]\nkv_c        latent/compressed KV                shape [Skv, Lkv]\nk_pe        decoupled k position embeddings     shape [Skv, R]\nnew_kv_c    new kv_c from current iter          shape [Sq, Lkv]\nnew_k_pe    new k_pe from current iter          shape [Sq, R]\ncache_kv_c  cached k_c from previous iters      shape [C, Lkv]\ncache_k_pe  cached k_pe from previous iters     shape [C, R]\nW_DQ        project h_t to q_c                  shape [H, Lq]\nW_UQ        project q_c to q_nope               shape [Lq, N * P]\nW_QR        project q_c to q_pe                 shape [Lq, N * R]\nW_DKV       project h_t to kv_c                 shape [H, Lkv]\nW_UK        project kv_c to k_nope              shape [Lkv, N, P]\nW_KR        project h_t to k_pe                 shape [H, R]\nW_UV        project kv_c to v                   shape [Lkv, N, V]\nW_O         project v to h_t                    shape [N * V, H]\n\n\n## Compute Friendly Approach (i.e. "_forward_prefill"):\n\nq_c      = h_t @ W_DQ\nq_nope   = (q_c @ W_UQ).view(Sq, N, P)\nq_pe     = RoPE(q_c @ W_QR).view(Sq, N, R)\nnew_kv_c = h_t @ W_DKV\nnew_k_pe = RoPE(h_t @ W_KR)\nkv_c     = torch.cat([new_kv_c, cache_kv_c], dim=0)\nk_pe     = torch.cat([new_k_pe, cache_k_pe], dim=0)\nk_nope   = (kv_c @ W_UK.view(Lkv, N * P)).view(Skv, N, P)\nv        = (kv_c @ W_UV.view(Lkv, N * V)).view(Skv, N, V)\n\n// MHA with QK headdim = P + R\n//           V headdim = V\n//      spda_o shape [Sq, N, V]\nspda_o = scaled_dot_product_attention(\n    torch.cat([q_nope, q_pe], dim=-1),\n    torch.cat([k_nope, k_pe.unsqueeze(1).expand(-1, N, -1)], dim=-1),\n    v\n) \nreturn spda_o @ W_O\n\nNOTE: in the actual code, \n    `kv_b_proj` is [W_UK; W_UV] concatenated per head\n    `q_b_proj` is [W_UQ; W_QR] concatenated per head\n    `out_proj` is W_O\n\n\n## Data-Movement Friendly Approach (i.e. "_forward_decode"):\n\nRuntime\nq_c      = h_t @ W_DQ\nq_nope   = (q_c @ W_UQ).view(-1, N, P)\nql_nope  = einsum("snh,lnh->snl", q, W_UK)\nq_pe     = RoPE(q_c @ W_QR).view(Sq, N, R)\nnew_kv_c = h_t @ W_DKV\nnew_k_pe = RoPE(h_t @ W_KR)\nkv_c     = torch.cat([new_kv_c, cache_kv_c], dim=0)\nk_pe     = torch.cat([new_k_pe, cache_k_pe], dim=0)\n\n// MQA with QK headdim = Lkv + R\n//           V headdim = Lkv\n//      spda_o shape [Sq, N, Lkv]\n// NOTE: this is less compute-friendly since Lkv > P\n//       but is more data-movement friendly since its MQA vs MHA\nspda_o = scaled_dot_product_attention(\n    torch.cat([ql_nope, q_pe], dim=-1),\n    torch.cat([kv_c, k_pe], dim=-1),\n    kv_c\n)\n\no = einsum("snl,lnv->snv", spda_o.reshape(-1, N, Lkv), W_UV)\nreturn o.view(-1, N * V) @ self.num_heads @ W_O\n\n\n## Chunked Prefill\n\nFor chunked prefill we want to use the compute friendly algorithm. We are \nassuming sufficiently large Sq / Skv ratio, in the future may want to switch to \nthe data-movement friendly approach if the chunk (i.e. `Sq`) is small.\n\nHowever, the compute-friendly approach can potentially run out of memory if Skv\nis large due to: `k_nope = (kv_c @ W_UK).view(Skv, N, P)`\n\nTo mitigate this, we chunk the computation of attention with respect to the \ncurrent context (i.e. `cache_kv_c` and `cache_k_pe`) so that we can used a \nfixed workspace size.\n\nThe chunked prefill approach is as follows:\n\nMCC        Max chunk of context to process per iter, computed dynamically, \n           used to bound the memory usage\n\nq_c        = h_t @ W_DQ\nq_nope     = (q_c @ W_UQ).view(Sq, N, P)\nq_pe       = RoPE(q_c @ W_QR).view(Sq, N, R)\nnew_kv_c   = h_t @ W_DKV\nnew_k_pe   = RoPE(h_t @ W_KR)\nnew_k_nope = (new_kv_c @ W_UK.view(Lkv, N * P)).view(Sq, N, P)\nnew_v      = (new_kv_c @ W_UV.view(Lkv, N * V)).view(Sq, N, V)\n\n// MHA between queries and new KV\n//     with QK headdim = P + R\n//           V headdim = V\n//    curr_o   shape [Sq, N, V]\n//    curr_lse shape [N, Sq], this is just order FA returns\ncurr_o, curr_lse = scaled_dot_product_attention(\n    torch.cat([q_nope, q_pe], dim=-1),\n    torch.cat([new_k_nope, new_k_pe.unsqueeze(1).expand(-1, N, -1)], dim=-1),\n    new_v,\n    casual=True,\n    return_softmax_lse=True\n) \n\n// Compute attention with the already existing context\nfor chunk_idx in range(cdiv(C, MCC)):\n    chunk_start  = chunk_idx * MCC\n    chunk_end    = min(chunk_start + MCC, C)\n    Sc           = chunk_end - chunk_start\n    cache_kv_c_chunk   = cache_kv_c[chunk_start:chunk_end]\n    cache_k_pe_chunk   = cache_k_pe[chunk_start:chunk_end]\n    cache_k_nope_chunk = (cache_kv_c_chunk @ W_UK).view(-1, N, P)\n    cache_v_chunk      = (cache_kv_c_chunk @ W_UV).view(-1, N, V)\n\n    chunk_o, chunk_lse = scaled_dot_product_attention(\n        torch.cat([q_nope, q_pe], dim=-1),\n        torch.cat([cache_k_nope_chunk,\n                   cache_k_pe_chunk.unsqueeze(1).expand(-1, N, -1)],\n                   dim=-1),\n        cache_v_chunk,\n        casual=False,\n        return_softmax_lse=True\n    )\n\n    curr_o, curr_lse = merge_attn_states(\n        suffix_output=curr_o,\n        suffix_lse=curr_lse,\n        prefix_output=chunk_o,\n        prefix_lse=chunk_lse,\n    )\n\nreturn curr_o @ W_O\n'
if HAS_TRITON:
else:
    triton_attention = None
try:
    is_vllm_fa = True
except ImportError:
    is_vllm_fa = False
    try:
    except ImportError:
        flash_attn_varlen_func = None
if TYPE_CHECKING:
is_hip = current_platform.is_rocm()

class MLACommonBackend(AttentionBackend):

    @staticmethod
    def get_name() -> str:
        return 'TRITON_MLA'

    @staticmethod
    def get_metadata_cls() -> Type['AttentionMetadata']:
        return MLACommonMetadata

    @staticmethod
    def get_builder_cls() -> Type['MLACommonMetadataBuilder']:
        return MLACommonMetadataBuilder

    @staticmethod
    def get_state_cls() -> Type['MLACommonState']:
        return MLACommonState

    @staticmethod
    def get_kv_cache_shape(num_blocks: int, block_size: int, num_kv_heads: int, head_size: int) -> Tuple[int, ...]:
        return (num_blocks, block_size, head_size)

    @staticmethod
    def swap_blocks(src_kv_cache: torch.Tensor, dst_kv_cache: torch.Tensor, src_to_dst: torch.Tensor) -> None:
        ops.swap_blocks(src_kv_cache, dst_kv_cache, src_to_dst)

    @staticmethod
    def copy_blocks(kv_caches: List[torch.Tensor], src_to_dists: torch.Tensor) -> None:
        ops.copy_blocks_mla(kv_caches, src_to_dists)

    @staticmethod
    def get_supported_head_sizes() -> List[int]:
        return [576]
T = TypeVar('T', bound='MLACommonMetadata')

class MLACommonState(AttentionState, Generic[T]):

    def __init__(self, runner):
        self.runner = runner
        self._is_graph_capturing = False
        scheduler_config = runner.scheduler_config
        self.model_config = runner.model_config
        cache_config = runner.cache_config
        self.chunked_prefill_enabled = scheduler_config.chunked_prefill_enabled
        self.enable_prefix_caching = cache_config.enable_prefix_caching
        if self.chunked_prefill_enabled or self.enable_prefix_caching:
            self.context_chunk_workspace_size = min(max(8 * self.model_config.max_model_len, 4 * scheduler_config.max_num_seqs * cache_config.block_size), 128 * 1024)
            assert self.context_chunk_workspace_size >= scheduler_config.max_num_seqs * cache_config.block_size

    @contextmanager
    def graph_capture(self, max_batch_size: int):
        self._is_graph_capturing = True
        self._graph_slot_mapping = torch.full((max_batch_size,), PAD_SLOT_ID, dtype=torch.long, device=self.runner.device)
        self._graph_seq_lens = torch.ones(max_batch_size, dtype=torch.int32, device=self.runner.device)
        self._graph_block_tables = torch.from_numpy(self.runner.graph_block_tables).to(device=self.runner.device)
        self._positions = torch.zeros((max_batch_size,), dtype=torch.long, device=self.runner.device)
        yield
        self._is_graph_capturing = False
        del self._graph_slot_mapping
        del self._graph_seq_lens
        del self._graph_block_tables
        del self._positions

    def graph_clone(self, batch_size: int):
        assert self._is_graph_capturing
        return self.__class__(self.runner)

    def graph_capture_get_metadata_for_batch(self, batch_size: int, is_encoder_decoder_model: bool=False) -> T:
        assert self._is_graph_capturing
        attn_metadata = self.runner.attn_backend.make_metadata(multi_modal_placeholder_index_maps=None, enable_kv_scales_calculation=False, use_cuda_graph=True, num_prefills=0, num_prefill_tokens=0, num_decode_tokens=batch_size, slot_mapping=self._graph_slot_mapping[:batch_size], seq_lens=None, seq_lens_tensor=self._graph_seq_lens[:batch_size], max_query_len=1, max_decode_query_len=1, max_prefill_seq_len=0, max_decode_seq_len=self.runner.max_seq_len_to_capture, query_start_loc=None, seq_start_loc=None, context_lens_tensor=None, block_tables=self._graph_block_tables[:batch_size], head_dim=self.runner.model_config.get_head_size())
        if is_encoder_decoder_model:
            raise NotImplementedError('MLACommonState does not support encoder/decoder yet')
        return attn_metadata

    def get_graph_input_buffers(self, attn_metadata, is_encoder_decoder_model: bool=False):
        input_buffers = {'slot_mapping': attn_metadata.slot_mapping, 'seq_lens_tensor': attn_metadata.decode_metadata.seq_lens_tensor, 'block_tables': attn_metadata.decode_metadata.block_tables}
        if is_encoder_decoder_model:
            raise NotImplementedError('MLACommonState does not support encoder/decoder yet')
        return input_buffers

    def prepare_graph_input_buffers(self, input_buffers, attn_metadata, is_encoder_decoder_model: bool=False):
        input_buffers['seq_lens_tensor'].copy_(attn_metadata.decode_metadata.seq_lens_tensor, non_blocking=True)
        input_buffers['block_tables'].copy_(attn_metadata.decode_metadata.block_tables, non_blocking=True)
        if is_encoder_decoder_model:
            raise NotImplementedError('TritonMLAState does not support encoder/decoder yet')

    def begin_forward(self, model_input):
        if self.chunked_prefill_enabled or self.enable_prefix_caching:
            if not hasattr(self, 'context_chunk_workspace'):
                assert model_input.input_tokens is not None
                self.context_chunk_workspace = torch.empty((self.context_chunk_workspace_size, self.model_config.get_head_size()), dtype=self.model_config.dtype, device=model_input.input_tokens.device)
            model_input.attn_metadata.context_chunk_workspace = self.context_chunk_workspace

@dataclass
class MLACommonMetadata(AttentionMetadata):
    """Metadata for MLACommon. 
    
    NOTE: Please read the comment at the top of the file before trying to 
    understand this class

    NOTE: Any python object stored here is not updated when it is
    cuda-graph replayed. If you have values that need to be changed
    dynamically, it should be stored in tensor. The tensor has to be
    updated from `CUDAGraphRunner.forward` API.
    """
    use_cuda_graph: bool
    seq_lens: Optional[List[int]]
    seq_lens_tensor: Optional[torch.Tensor]
    max_prefill_seq_len: int
    max_decode_seq_len: int
    context_lens_tensor: Optional[torch.Tensor]
    block_tables: Optional[torch.Tensor]
    max_query_len: Optional[int] = None
    max_decode_query_len: Optional[int] = None
    query_start_loc: Optional[torch.Tensor] = None
    seq_start_loc: Optional[torch.Tensor] = None
    _cached_prefill_metadata: Optional[Any] = None
    _cached_decode_metadata: Optional[Any] = None
    num_prefill_tokens: int
    head_dim: Optional[int] = None
    is_profile_run: bool = False
    context_chunk_cu_seq_lens: Optional[torch.Tensor] = None
    context_chunk_starts: Optional[torch.Tensor] = None
    context_chunk_seq_tot: Optional[List[int]] = None
    context_chunk_max_seq_lens: Optional[List[int]] = None
    context_chunk_workspace: Optional[torch.Tensor] = None

    def __post_init__(self):
        supported_head_sizes = MLACommonBackend.get_supported_head_sizes()
        if self.head_dim is not None and self.head_dim not in supported_head_sizes:
            raise ValueError(f'Only {supported_head_sizes} are supported for head_dim,', f' received {self.head_dim}.')

    @property
    def prefill_metadata(self):
        if self.num_prefills == 0:
            return None
        if self._cached_prefill_metadata is not None:
            return self._cached_prefill_metadata
        assert self.seq_lens is not None
        assert self.seq_lens_tensor is not None
        query_start_loc = None if self.query_start_loc is None else self.query_start_loc[:self.num_prefills + 1]
        slot_mapping = None if self.slot_mapping is None else self.slot_mapping[:self.num_prefill_tokens]
        seq_lens = None if self.seq_lens is None else self.seq_lens[:self.num_prefills]
        seq_lens_tensor = None if self.seq_lens_tensor is None else self.seq_lens_tensor[:self.num_prefills]
        seq_start_loc = None if self.seq_start_loc is None else self.seq_start_loc[:self.num_prefills + 1]
        context_lens_tensor = None if self.context_lens_tensor is None else self.context_lens_tensor[:self.num_prefills]
        block_tables = None if self.block_tables is None else self.block_tables[:self.num_prefills]
        self._cached_prefill_metadata = self.__class__(use_cuda_graph=False, num_prefills=self.num_prefills, num_prefill_tokens=self.num_prefill_tokens, num_decode_tokens=0, slot_mapping=slot_mapping, multi_modal_placeholder_index_maps=None, enable_kv_scales_calculation=False, seq_lens=seq_lens, seq_lens_tensor=seq_lens_tensor, max_query_len=self.max_query_len, max_prefill_seq_len=self.max_prefill_seq_len, max_decode_query_len=0, max_decode_seq_len=0, query_start_loc=query_start_loc, seq_start_loc=seq_start_loc, context_lens_tensor=context_lens_tensor, block_tables=block_tables, head_dim=self.head_dim, is_profile_run=self.is_profile_run, context_chunk_cu_seq_lens=self.context_chunk_cu_seq_lens, context_chunk_starts=self.context_chunk_starts, context_chunk_seq_tot=self.context_chunk_seq_tot, context_chunk_max_seq_lens=self.context_chunk_max_seq_lens)
        return self._cached_prefill_metadata

    @property
    def decode_metadata(self):
        if self.num_decode_tokens == 0:
            return None
        if self._cached_decode_metadata is not None:
            return self._cached_decode_metadata
        assert self.seq_lens_tensor is not None
        slot_mapping = None if self.slot_mapping is None else self.slot_mapping[self.num_prefill_tokens:]
        seq_lens_tensor = None if self.seq_lens_tensor is None else self.seq_lens_tensor[self.num_prefills:]
        block_tables = None if self.block_tables is None else self.block_tables[self.num_prefills:]
        self._cached_decode_metadata = self.__class__(use_cuda_graph=self.use_cuda_graph, num_prefills=0, num_prefill_tokens=0, num_decode_tokens=self.num_decode_tokens, slot_mapping=slot_mapping, multi_modal_placeholder_index_maps=None, enable_kv_scales_calculation=False, seq_lens=None, seq_lens_tensor=seq_lens_tensor, max_decode_query_len=self.max_decode_query_len, max_query_len=self.max_query_len, max_prefill_seq_len=0, max_decode_seq_len=self.max_decode_seq_len, query_start_loc=self.query_start_loc[self.num_prefills:] - self.query_start_loc[self.num_prefills] if self.query_start_loc is not None else None, seq_start_loc=self.seq_start_loc[self.num_prefills:] if self.seq_start_loc is not None else None, context_lens_tensor=None, block_tables=block_tables, head_dim=self.head_dim, is_profile_run=self.is_profile_run)
        return self._cached_decode_metadata

    def advance_step(self, model_input: 'ModelInputForGPUWithSamplingMetadata', sampled_token_ids: Optional[torch.Tensor], block_size: int, num_seqs: int, num_queries: int, turn_prefills_into_decodes: bool=False):
        """
        Update metadata in-place to advance one decode step.
        """
        if num_seqs != num_queries:
            assert num_seqs > num_queries
        if turn_prefills_into_decodes:
            assert self.num_decode_tokens + self.num_prefills == num_seqs
            self.num_decode_tokens += self.num_prefills
            self.num_prefills = 0
            self.num_prefill_tokens = 0
            self.max_prefill_seq_len = 0
            self.max_query_len = 1
            self.slot_mapping = self.slot_mapping[:num_seqs]
        else:
            assert self.seq_lens is not None
            assert self.max_decode_seq_len == max(self.seq_lens)
        assert self.num_prefills == 0
        assert self.num_prefill_tokens == 0
        assert self.num_decode_tokens == num_seqs
        assert self.slot_mapping.shape == (num_seqs,)
        assert self.seq_lens is not None
        assert len(self.seq_lens) == num_seqs
        assert self.seq_lens_tensor is not None
        assert self.seq_lens_tensor.shape == (num_seqs,)
        assert self.max_query_len == 1
        assert self.max_prefill_seq_len == 0
        assert self.query_start_loc is not None
        assert self.query_start_loc.shape == (num_queries + 1,)
        assert self.seq_start_loc is not None
        assert self.seq_start_loc.shape == (num_seqs + 1,)
        assert self.context_lens_tensor is not None
        assert self.context_lens_tensor.shape == (num_queries,)
        assert self.block_tables is not None
        assert self.block_tables.shape[0] == num_seqs
        for i in range(num_queries):
            self.seq_lens[i] += 1
        self.max_decode_seq_len = max(self.seq_lens)
        self._ops_advance_step(num_seqs=num_seqs, num_queries=num_queries, block_size=block_size, input_tokens=model_input.input_tokens, sampled_token_ids=sampled_token_ids, input_positions=model_input.input_positions)

    def _ops_advance_step(self, num_seqs: int, num_queries: int, block_size: int, input_tokens: torch.Tensor, sampled_token_ids: torch.Tensor, input_positions: torch.Tensor) -> None:
        ops.advance_step_flashattn(num_seqs=num_seqs, num_queries=num_queries, block_size=block_size, input_tokens=input_tokens, sampled_token_ids=sampled_token_ids, input_positions=input_positions, seq_lens=self.seq_lens_tensor, slot_mapping=self.slot_mapping, block_tables=self.block_tables)

class MLACommonMetadataBuilder(AttentionMetadataBuilder[T], Generic[T]):
    """
    NOTE: Please read the comment at the top of the file before trying to 
    understand this class
    """
    BLOCK_TABLE_EXTENDER: list[list[int]] = []

    def __init__(self, input_builder: 'ModelInputForGPUBuilder'):
        self.input_builder = input_builder
        self.runner = input_builder.runner
        self.sliding_window = input_builder.sliding_window
        self.block_size = input_builder.block_size
        self.chunked_prefill_enabled = self.runner.scheduler_config.chunked_prefill_enabled
        self.enable_prefix_caching = self.runner.cache_config.enable_prefix_caching
        if self.chunked_prefill_enabled or self.enable_prefix_caching:
            attn_state = self.input_builder.runner.attn_state
            self.context_chunk_workspace_size = attn_state.context_chunk_workspace_size
            self.page_size = self.runner.block_size

    def prepare(self):
        self.slot_mapping: List[int] = []
        self.prefill_seq_lens: List[int] = []
        self.context_lens: List[int] = []
        self.block_tables: List[List[int]] = []
        self.curr_seq_lens: List[int] = []
        self.multimodal_placeholder_maps: Dict[str, MultiModalPlaceholderMap] = defaultdict(MultiModalPlaceholderMap)
        self.num_prefills = 0
        self.num_prefill_tokens = 0
        self.num_decode_tokens = 0
        self.has_prefix_cache_hit = False

    def _add_seq_group(self, inter_data: 'ModelInputForGPUBuilder.InterDataForSeqGroup', chunked_prefill_enabled: bool, prefix_cache_hit: bool):
        """Add a sequence group to the metadata. Specifically update/append
        1. context length.
        2. block table.
        3. slot mapping.
        """
        is_prompt = inter_data.is_prompt
        block_tables = inter_data.block_tables
        for seq_id, token_len, seq_len, curr_seq_len, query_len, context_len, curr_sliding_window_block in zip(inter_data.seq_ids, [len(t) for t in inter_data.input_tokens], inter_data.orig_seq_lens, inter_data.seq_lens, inter_data.query_lens, inter_data.context_lens, inter_data.curr_sliding_window_blocks):
            self.context_lens.append(context_len)
            if is_prompt:
                self.num_prefills += 1
                self.num_prefill_tokens += token_len
                self.prefill_seq_lens.append(seq_len)
            else:
                self.num_decode_tokens += query_len
                self.curr_seq_lens.append(curr_seq_len)
            block_table = []
            if prefix_cache_hit:
                block_table = block_tables[seq_id]
            elif (chunked_prefill_enabled or not is_prompt) and block_tables is not None:
                if curr_sliding_window_block == 0:
                    block_table = block_tables[seq_id]
                else:
                    block_table = block_tables[seq_id][-curr_sliding_window_block:]
            self.block_tables.append(block_table)
            is_profile_run = is_block_tables_empty(block_tables)
            start_idx = compute_slot_mapping_start_idx(is_prompt, query_len, context_len, self.sliding_window)
            compute_slot_mapping(is_profile_run, self.slot_mapping, seq_id, seq_len, context_len, start_idx, self.block_size, inter_data.block_tables)

    def _get_graph_runner_block_tables(self, num_seqs: int, block_tables: List[List[int]]) -> torch.Tensor:
        max_batch_size, max_blocks = self.runner.graph_block_tables.shape
        assert max_batch_size >= num_seqs
        graph_block_tables = self.runner.graph_block_tables[:num_seqs]
        for i, block_table in enumerate(block_tables):
            if block_table:
                num_blocks = len(block_table)
                if num_blocks <= max_blocks:
                    graph_block_tables[i, :num_blocks] = block_table
                else:
                    graph_block_tables[i, :max_blocks] = block_table[:max_blocks]
        return torch.from_numpy(graph_block_tables).to(device=self.runner.device, non_blocking=True)

    def build(self, seq_lens: List[int], query_lens: List[int], cuda_graph_pad_size: int, batch_size: int):
        """Build attention metadata with on-device tensors.

        Args:
            seq_lens: The maybe padded sequence lengths of the input sequences.
            query_lens: The query lengths of the input sequences.
            cuda_graph_pad_size: The padding size for cuda graph.
                                 -1 if cuda graph is not used.
            batch_size: The maybe padded batch size.
        """
        prefix_cache_hit = any([inter_data.prefix_cache_hit for inter_data in self.input_builder.inter_data_list])
        for inter_data in self.input_builder.inter_data_list:
            self._add_seq_group(inter_data, self.input_builder.chunked_prefill_enabled, prefix_cache_hit)
        device = self.runner.device
        use_captured_graph = cuda_graph_pad_size != -1
        max_query_len = max(query_lens)
        decode_query_lens = query_lens[self.num_prefills:]
        if len(decode_query_lens) > 0:
            max_decode_query_len = max(decode_query_lens)
        else:
            max_decode_query_len = 1
        max_prefill_seq_len = max(self.prefill_seq_lens, default=0)
        max_decode_seq_len = max(self.curr_seq_lens, default=0)
        num_decode_tokens = self.num_decode_tokens
        query_start_loc = list(accumulate(query_lens, initial=0))
        seq_start_loc = list(accumulate(seq_lens, initial=0))
        num_seqs = len(seq_lens)
        if use_captured_graph:
            self.slot_mapping.extend([PAD_SLOT_ID] * cuda_graph_pad_size)
            self.block_tables.extend(self.__class__.BLOCK_TABLE_EXTENDER * cuda_graph_pad_size)
            num_decode_tokens = batch_size - self.num_prefill_tokens
            block_tables = self._get_graph_runner_block_tables(num_seqs, self.block_tables)
        else:
            block_tables = make_tensor_with_pad(self.block_tables, pad=0, dtype=torch.int, device=device)
        assert max_query_len > 0, 'query_lens: {}'.format(query_lens)
        assert device is not None
        context_lens_tensor = async_tensor_h2d(self.context_lens, torch.int, device, self.runner.pin_memory)
        seq_lens_tensor = async_tensor_h2d(seq_lens, torch.int, device, self.runner.pin_memory)
        slot_mapping_tensor = async_tensor_h2d(self.slot_mapping, torch.long, device, self.runner.pin_memory)
        query_start_loc_tensor = async_tensor_h2d(query_start_loc, torch.int32, device, self.runner.pin_memory)
        seq_start_loc_tensor = async_tensor_h2d(seq_start_loc, torch.int32, device, self.runner.pin_memory)
        context_chunk_cu_seq_lens = None
        context_chunk_starts = None
        context_chunk_seq_tot = None
        context_chunk_max_seq_lens = None
        if (self.chunked_prefill_enabled or self.enable_prefix_caching) and self.num_prefills > 0 and (context_lens_tensor is not None) and (context_lens_tensor[:self.num_prefills].max() > 0):
            num_prefills_with_context = (context_lens_tensor[:self.num_prefills] > 0).sum().item()
            max_context_chunk = self.context_chunk_workspace_size // num_prefills_with_context
            max_context_chunk = round_down(max_context_chunk, self.page_size)
            assert max_context_chunk > 0
            num_chunks = cdiv(context_lens_tensor.max(), max_context_chunk)
            context_chunk_starts = torch.arange(num_chunks, device=device, dtype=torch.int32).unsqueeze(1).expand(-1, self.num_prefills) * max_context_chunk
            chunk_ends = torch.min(context_lens_tensor[:self.num_prefills].unsqueeze(0), context_chunk_starts + max_context_chunk)
            chunk_seq_lens = (chunk_ends - context_chunk_starts).clamp(min=0)
            _context_chunk_cu_seq_lens = chunk_seq_lens.cumsum(dim=1).to(torch.int32)
            zero = torch.zeros(num_chunks, dtype=torch.int32, device=device).unsqueeze(-1)
            context_chunk_cu_seq_lens = torch.cat([zero, _context_chunk_cu_seq_lens], dim=1)
            context_chunk_max_seq_lens = chunk_seq_lens.max(dim=1).values.tolist()
            context_chunk_seq_tot = chunk_seq_lens.sum(dim=1).tolist()
            assert max(context_chunk_seq_tot) <= self.context_chunk_workspace_size
        return self.runner.attn_backend.make_metadata(use_cuda_graph=use_captured_graph, num_prefills=self.num_prefills, slot_mapping=slot_mapping_tensor, num_prefill_tokens=self.num_prefill_tokens, num_decode_tokens=num_decode_tokens, multi_modal_placeholder_index_maps=None, enable_kv_scales_calculation=False, seq_lens=seq_lens, seq_lens_tensor=seq_lens_tensor, max_query_len=max_query_len, max_decode_query_len=max_decode_query_len, max_prefill_seq_len=max_prefill_seq_len, max_decode_seq_len=max_decode_seq_len, query_start_loc=query_start_loc_tensor, seq_start_loc=seq_start_loc_tensor, context_lens_tensor=context_lens_tensor, block_tables=block_tables, head_dim=self.runner.model_config.get_head_size(), is_profile_run=self.runner.in_profile_run, context_chunk_cu_seq_lens=context_chunk_cu_seq_lens, context_chunk_starts=context_chunk_starts, context_chunk_seq_tot=context_chunk_seq_tot, context_chunk_max_seq_lens=context_chunk_max_seq_lens)

class MLACommonImpl(MLAAttentionImpl[T], Generic[T]):
    """
    NOTE: Please read the comment at the top of the file before trying to 
    understand this class
    """

    def __init__(self, num_heads: int, head_size: int, scale: float, num_kv_heads: int, alibi_slopes: Optional[List[float]], sliding_window: Optional[int], kv_cache_dtype: str, logits_soft_cap: Optional[float], attn_type: str, kv_sharing_target_layer_name: Optional[str], q_lora_rank: Optional[int], kv_lora_rank: int, qk_nope_head_dim: int, qk_rope_head_dim: int, qk_head_dim: int, v_head_dim: int, kv_b_proj: ColumnParallelLinear) -> None:
        if kv_sharing_target_layer_name is not None:
            raise NotImplementedError('KV sharing not supported in V0.')
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim
        self.kv_b_proj = kv_b_proj
        self.triton_fa_func = triton_attention
        self.flash_attn_varlen_func = flash_attn_varlen_func
        self.vllm_flash_attn_version = get_flash_attn_version()
        if self.vllm_flash_attn_version is not None:
            self.flash_attn_varlen_func = functools.partial(flash_attn_varlen_func, fa_version=self.vllm_flash_attn_version)
        self._pad_v = self.vllm_flash_attn_version is None or not (self.vllm_flash_attn_version == 3 and current_platform.get_device_capability()[0] == 9)

    def _flash_attn_varlen_diff_headdims(self, q, k, v, softmax_scale, return_softmax_lse, **kwargs):
        maybe_padded_v = v
        if self._pad_v:
            maybe_padded_v = torch.nn.functional.pad(v, [0, q.shape[-1] - v.shape[-1]], value=0)
        if is_hip and envs.VLLM_USE_TRITON_FLASH_ATTN and (not return_softmax_lse):
            attn_out = self.triton_fa_func(q, k, maybe_padded_v, None, kwargs['cu_seqlens_q'], kwargs['cu_seqlens_k'], kwargs['max_seqlen_q'], kwargs['max_seqlen_k'], kwargs['causal'], softmax_scale, None)
        elif is_vllm_fa:
            attn_out = self.flash_attn_varlen_func(q=q, k=k, v=maybe_padded_v, return_softmax_lse=return_softmax_lse, softmax_scale=softmax_scale, **kwargs)
        else:
            attn_out = self.flash_attn_varlen_func(q=q, k=k, v=maybe_padded_v, return_attn_probs=return_softmax_lse, softmax_scale=softmax_scale, **kwargs)
        rest = None
        if isinstance(attn_out, tuple):
            attn_out, *rest = attn_out
        if return_softmax_lse:
            assert rest is not None
            return (attn_out, rest[0])
        return attn_out

    def _v_up_proj(self, x):
        x = x.view(-1, self.num_heads, self.kv_lora_rank).transpose(0, 1)
        x = torch.bmm(x, self.W_UV)
        return x.transpose(0, 1).reshape(-1, self.num_heads * self.v_head_dim)

    def process_weights_after_loading(self, act_dtype: torch.dtype):

        def get_layer_weight(layer):
            WEIGHT_NAMES = ('weight', 'qweight', 'weight_packed')
            for attr in WEIGHT_NAMES:
                if hasattr(layer, attr):
                    return getattr(layer, attr)
            raise AttributeError(f"Layer '{layer}' has no recognized weight attribute: {WEIGHT_NAMES}.")

        def get_and_maybe_dequant_weights(layer: LinearBase):
            if not isinstance(layer.quant_method, UnquantizedLinearMethod):
                eye = torch.eye(layer.input_size_per_partition, dtype=act_dtype, device=get_layer_weight(layer).device)
                dequant_weights = layer.quant_method.apply(layer, eye, bias=None)
                del eye
                return dequant_weights.T
            return layer.weight
        kv_b_proj_weight = get_and_maybe_dequant_weights(self.kv_b_proj).T
        assert kv_b_proj_weight.shape == (self.kv_lora_rank, self.num_heads * (self.qk_nope_head_dim + self.v_head_dim)), f'kv_b_proj_weight.shape={kv_b_proj_weight.shape!r}, self.kv_lora_rank={self.kv_lora_rank!r}, self.num_heads={self.num_heads!r}, self.qk_nope_head_dim={self.qk_nope_head_dim!r}, self.v_head_dim={self.v_head_dim!r}'
        kv_b_proj_weight = kv_b_proj_weight.view(self.kv_lora_rank, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        W_UK, W_UV = kv_b_proj_weight.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        self.W_UV = W_UV.transpose(0, 1)
        self.W_UK_T = W_UK.permute(1, 2, 0)

    def _compute_prefill_context(self, q: torch.Tensor, kv_c_and_k_pe_cache: torch.Tensor, attn_metadata: MLACommonMetadata):
        prefill_metadata = attn_metadata.prefill_metadata
        assert prefill_metadata is not None
        assert prefill_metadata.context_chunk_seq_tot is not None
        assert prefill_metadata.context_chunk_cu_seq_lens is not None
        assert prefill_metadata.context_chunk_starts is not None
        assert prefill_metadata.context_chunk_max_seq_lens is not None
        assert prefill_metadata.context_lens_tensor is not None
        output = None
        iters = len(prefill_metadata.context_chunk_seq_tot)
        assert attn_metadata.context_chunk_workspace is not None
        workspace = attn_metadata.context_chunk_workspace
        for i in range(iters):
            toks = prefill_metadata.context_chunk_seq_tot[i]
            ops.gather_cache(src_cache=kv_c_and_k_pe_cache, dst=workspace, block_table=prefill_metadata.block_tables, cu_seq_lens=prefill_metadata.context_chunk_cu_seq_lens[i], batch_size=prefill_metadata.num_prefills, seq_starts=prefill_metadata.context_chunk_starts[i])
            kv_c_normed = workspace[:toks][..., :self.kv_lora_rank]
            k_pe = workspace[:toks][..., self.kv_lora_rank:].unsqueeze(1)
            kv_nope = self.kv_b_proj(kv_c_normed)[0].view(-1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = kv_nope.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            k = torch.cat((k_nope, k_pe.expand((*k_nope.shape[:-1], -1))), dim=-1)
            attn_output, attn_softmax_lse = self._flash_attn_varlen_diff_headdims(q=q, k=k, v=v, cu_seqlens_q=prefill_metadata.query_start_loc, cu_seqlens_k=prefill_metadata.context_chunk_cu_seq_lens[i], max_seqlen_q=prefill_metadata.max_query_len, max_seqlen_k=prefill_metadata.context_chunk_max_seq_lens[i], softmax_scale=self.scale, causal=False, return_softmax_lse=True)
            if output is None:
                output = attn_output
                output_lse = attn_softmax_lse
            else:
                output_tmp = torch.empty_like(output)
                output_lse_tmp = torch.empty_like(output_lse)
                merge_attn_states(output=output_tmp, output_lse=output_lse_tmp, prefix_output=output, prefix_lse=output_lse, suffix_output=attn_output, suffix_lse=attn_softmax_lse)
                output = output_tmp
                output_lse = output_lse_tmp
        return (output, output_lse)

    def _forward_prefill(self, q: torch.Tensor, kv_c_normed: torch.Tensor, k_pe: torch.Tensor, kv_c_and_k_pe_cache: torch.Tensor, attn_metadata: MLACommonMetadata) -> torch.Tensor:
        prefill_metadata = attn_metadata.prefill_metadata
        assert prefill_metadata is not None
        has_context = prefill_metadata.context_lens_tensor is not None and prefill_metadata.context_lens_tensor.max() > 0
        kv_nope = self.kv_b_proj(kv_c_normed)[0].view(-1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = kv_nope.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k = torch.cat((k_nope, k_pe.expand((*k_nope.shape[:-1], -1))), dim=-1)
        output = self._flash_attn_varlen_diff_headdims(q=q, k=k, v=v, cu_seqlens_q=prefill_metadata.query_start_loc, cu_seqlens_k=prefill_metadata.query_start_loc, max_seqlen_q=prefill_metadata.max_prefill_seq_len, max_seqlen_k=prefill_metadata.max_prefill_seq_len, softmax_scale=self.scale, causal=True, return_softmax_lse=has_context)
        if has_context:
            suffix_output, suffix_lse = output
            context_output, context_lse = self._compute_prefill_context(q, kv_c_and_k_pe_cache, attn_metadata)
            output = torch.empty_like(suffix_output)
            merge_attn_states(output=output, prefix_output=context_output, prefix_lse=context_lse, suffix_output=suffix_output, suffix_lse=suffix_lse)
        if self._pad_v:
            output = output[..., :v.shape[-1]]
        return output.flatten(start_dim=-2)

    @abstractmethod
    def _forward_decode(self, ql_nope: torch.Tensor, q_pe: torch.Tensor, kv_c_and_k_pe_cache: torch.Tensor, attn_metadata: T) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, layer: AttentionLayer, q: torch.Tensor, k_c_normed: torch.Tensor, k_pe: torch.Tensor, kv_cache: torch.Tensor, attn_metadata: T, output: Optional[torch.Tensor]=None, output_scale: Optional[torch.Tensor]=None) -> torch.Tensor:
        if output is not None:
            raise NotImplementedError('output is not yet supported for MLAImplBase')
        if output_scale is not None:
            raise NotImplementedError('fused output quantization is not yet supported for MLAImplBase')
        if attn_metadata.is_profile_run and attn_metadata.context_chunk_workspace is not None:
            _ = torch.empty((attn_metadata.context_chunk_workspace.shape[0], self.num_heads, self.qk_nope_head_dim + self.v_head_dim), device=k_c_normed.device, dtype=k_c_normed.dtype)
        has_decode = attn_metadata.decode_metadata is not None
        has_prefill = attn_metadata.prefill_metadata is not None
        num_prefill_tokens: int = attn_metadata.num_prefill_tokens
        q = q.view(-1, self.num_heads, self.qk_head_dim)
        decode_q = q[num_prefill_tokens:]
        prefill_q = q[:num_prefill_tokens]
        prefill_k_pe = k_pe[:num_prefill_tokens]
        prefill_k_c_normed = k_c_normed[:num_prefill_tokens]
        if kv_cache.numel() > 0:
            ops.concat_and_cache_mla(k_c_normed, k_pe.squeeze(1), kv_cache, attn_metadata.slot_mapping.flatten(), kv_cache_dtype=self.kv_cache_dtype, scale=layer._k_scale)
        output = torch.empty(attn_metadata.num_prefill_tokens + attn_metadata.num_decode_tokens, self.v_head_dim * self.num_heads, device=q.device, dtype=q.dtype)
        if has_prefill:
            output[:num_prefill_tokens] = self._forward_prefill(prefill_q, prefill_k_c_normed, prefill_k_pe, kv_cache, attn_metadata)
        if has_decode:
            decode_q_nope, decode_q_pe = decode_q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
            decode_q_nope = decode_q_nope.transpose(0, 1)
            decode_ql_nope = torch.bmm(decode_q_nope, self.W_UK_T)
            decode_ql_nope = decode_ql_nope.transpose(0, 1)
            output[num_prefill_tokens:] = self._forward_decode(decode_ql_nope, decode_q_pe, kv_cache, attn_metadata)
        return output