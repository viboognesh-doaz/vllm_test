from abc import abstractmethod
from dataclasses import dataclass, field
from flash_attn import flash_attn_varlen_func
from flashinfer import BatchPrefillWithRaggedKVCacheWrapper
from flashinfer.prefill import cudnn_batch_prefill_with_kv_cache
from typing import TYPE_CHECKING, Generic, Optional, TypeVar, Union
from vllm import _custom_ops as ops
from vllm.attention.backends.abstract import AttentionBackend, AttentionLayer, AttentionMetadata, MLAAttentionImpl
from vllm.attention.backends.utils import get_mla_dims
from vllm.attention.ops.merge_attn_states import merge_attn_states
from vllm.attention.utils.fa_utils import get_flash_attn_version
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import ColumnParallelLinear, LinearBase, UnquantizedLinearMethod
from vllm.platforms import current_platform
from vllm.utils import cdiv, round_down
from vllm.v1.attention.backends.utils import AttentionMetadataBuilder, CommonAttentionMetadata, get_per_layer_parameters, infer_global_hyperparameters, reorder_batch_to_split_decodes_and_prefills, split_decodes_and_prefills
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.worker.gpu_input_batch import InputBatch
from vllm.vllm_flash_attn import flash_attn_varlen_func
import functools
import torch
import vllm.envs as envs
'\n# MLA Common Components\n\nThis file implements common components for MLA implementations.\n\nFirst we define:\n\nSq      as Q sequence length\nSkv     as KV sequence length\n\nMLA has two possible ways of computing, a data-movement friendly approach and a\ncompute friendly approach, we generally want to use the compute friendly\napproach for "prefill" (i.e. the ratio Sq / Skv is "small", is near 1)\nand the data-movement friendly approach for "decode" (i.e. the ratio\nSq / Skv is "large").\n\nNOTE what we deem small and large is currently determined by if its labelled\nprefill or decode by the scheduler, but this is something we should probably\ntune.\n\nMain reference: DeepseekV2 paper, and FlashInfer Implementation\n(https://arxiv.org/abs/2405.04434 and https://github.com/flashinfer-ai/flashinfer/pull/551).\n\nDeepseek\'s MLA attention works the following way:\n* Use a single latent vector to represent the per-token entry of the KV cache. \n* For decode (i.e. the memory friendly approach) the attention "simulates" a\nmulti-head attention, while the compute is similar to multi-query attention.\n\nBelow is example of both paths assuming batchsize = 1\n\n## More Extent Definitions:\n\nC           Context length, `Skv - Sq`\nH           hidden size\nN           number of attention heads\nLq          latent dimension for Q              1536 in DSV3\nLkv         latent dimension for K/V            512 in DSV3\nP           nope dimension, no rope.            128 in DSV3\nR           rope dimension, goes through rope.  64 in DSV3\nV           V head dim.                         128 in DSV3\n\n## Vector/Matrix Definitions\n\nh_t         hidden states (input to attention)  shape [Sq, H]\nq_c         latent/compressed Q                 shape [Sq, Lq]\nq_nope      uncompressed Q (no-rope)            shape [Sq, N, P]\nq_pe        uncompressed Q (rope)               shape [Sq, N, R]\nkv_c        latent/compressed KV                shape [Skv, Lkv]\nk_pe        decoupled k position embeddings     shape [Skv, R]\nnew_kv_c    new kv_c from current iter          shape [Sq, Lkv]\nnew_k_pe    new k_pe from current iter          shape [Sq, R]\ncache_kv_c  cached k_c from previous iters      shape [C, Lkv]\ncache_k_pe  cached k_pe from previous iters     shape [C, R]\nW_DQ        project h_t to q_c                  shape [H, Lq]\nW_UQ        project q_c to q_nope               shape [Lq, N * P]\nW_QR        project q_c to q_pe                 shape [Lq, N * R]\nW_DKV       project h_t to kv_c                 shape [H, Lkv]\nW_UK        project kv_c to k_nope              shape [Lkv, N, P]\nW_KR        project h_t to k_pe                 shape [H, R]\nW_UV        project kv_c to v                   shape [Lkv, N, V]\nW_O         project v to h_t                    shape [N * V, H]\n\n\n## Compute Friendly Approach (i.e. "_forward_prefill"):\n\nq_c      = h_t @ W_DQ\nq_nope   = (q_c @ W_UQ).view(Sq, N, P)\nq_pe     = RoPE(q_c @ W_QR).view(Sq, N, R)\nnew_kv_c = h_t @ W_DKV\nnew_k_pe = RoPE(h_t @ W_KR)\nkv_c     = torch.cat([new_kv_c, cache_kv_c], dim=0)\nk_pe     = torch.cat([new_k_pe, cache_k_pe], dim=0)\nk_nope   = (kv_c @ W_UK.view(Lkv, N * P)).view(Skv, N, P)\nv        = (kv_c @ W_UV.view(Lkv, N * V)).view(Skv, N, V)\n\n// MHA with QK headdim = P + R\n//           V headdim = V\n//      spda_o shape [Sq, N, V]\nspda_o = scaled_dot_product_attention(\n    torch.cat([q_nope, q_pe], dim=-1),\n    torch.cat([k_nope, k_pe.unsqueeze(1).expand(-1, N, -1)], dim=-1),\n    v\n) \nreturn spda_o @ W_O\n\nNOTE: in the actual code,\n    `kv_b_proj` is [W_UK; W_UV] concatenated per head\n    `q_b_proj` is [W_UQ; W_QR] concatenated per head\n    `out_proj` is W_O\n\n\n## Data-Movement Friendly Approach (i.e. "_forward_decode"):\n\nRuntime\nq_c      = h_t @ W_DQ\nq_nope   = (q_c @ W_UQ).view(-1, N, P)\nql_nope  = einsum("snh,lnh->snl", q, W_UK)\nq_pe     = RoPE(q_c @ W_QR).view(Sq, N, R)\nnew_kv_c = h_t @ W_DKV\nnew_k_pe = RoPE(h_t @ W_KR)\nkv_c     = torch.cat([new_kv_c, cache_kv_c], dim=0)\nk_pe     = torch.cat([new_k_pe, cache_k_pe], dim=0)\n\n// MQA with QK headdim = Lkv + R\n//           V headdim = Lkv\n//      spda_o shape [Sq, N, Lkv]\n// NOTE: this is less compute-friendly since Lkv > P\n//       but is more data-movement friendly since its MQA vs MHA\nspda_o = scaled_dot_product_attention(\n    torch.cat([ql_nope, q_pe], dim=-1),\n    torch.cat([kv_c, k_pe], dim=-1),\n    kv_c\n)\n\no = einsum("snl,lnv->snv", spda_o.reshape(-1, N, Lkv), W_UV)\nreturn o.view(-1, N * V) @ self.num_heads @ W_O\n\n\n## Chunked Prefill\n\nFor chunked prefill we want to use the compute friendly algorithm. We are \nassuming sufficiently large Sq / Skv ratio, in the future may want to switch to \nthe data-movement friendly approach if the chunk (i.e. `Sq`) is small.\n\nHowever, the compute-friendly approach can potentially run out of memory if Skv\nis large due to: `k_nope = (kv_c @ W_UK).view(Skv, N, P)`\n\nTo mitigate this, we chunk the computation of attention with respect to the \ncurrent context (i.e. `cache_kv_c` and `cache_k_pe`) so that we can used a \nfixed workspace size.\n\nThe chunked prefill approach is as follows:\n\nMCC        Max chunk of context to process per iter, computed dynamically, \n           used to bound the memory usage\n\nq_c        = h_t @ W_DQ\nq_nope     = (q_c @ W_UQ).view(Sq, N, P)\nq_pe       = RoPE(q_c @ W_QR).view(Sq, N, R)\nnew_kv_c   = h_t @ W_DKV\nnew_k_pe   = RoPE(h_t @ W_KR)\nnew_k_nope = (new_kv_c @ W_UK.view(Lkv, N * P)).view(Sq, N, P)\nnew_v      = (new_kv_c @ W_UV.view(Lkv, N * V)).view(Sq, N, V)\n\n// MHA between queries and new KV\n//     with QK headdim = P + R\n//           V headdim = V\n//    curr_o   shape [Sq, N, V]\n//    curr_lse shape [N, Sq], this is just order FA returns\ncurr_o, curr_lse = scaled_dot_product_attention(\n    torch.cat([q_nope, q_pe], dim=-1),\n    torch.cat([new_k_nope, new_k_pe.unsqueeze(1).expand(-1, N, -1)], dim=-1),\n    new_v,\n    casual=True,\n    return_softmax_lse=True\n) \n\n// Compute attention with the already existing context\nfor chunk_idx in range(cdiv(C, MCC)):\n    chunk_start  = chunk_idx * MCC\n    chunk_end    = min(chunk_start + MCC, C)\n    Sc           = chunk_end - chunk_start\n    cache_kv_c_chunk   = cache_kv_c[chunk_start:chunk_end]\n    cache_k_pe_chunk   = cache_k_pe[chunk_start:chunk_end]\n    cache_k_nope_chunk = (cache_kv_c_chunk @ W_UK).view(-1, N, P)\n    cache_v_chunk      = (cache_kv_c_chunk @ W_UV).view(-1, N, V)\n\n    chunk_o, chunk_lse = scaled_dot_product_attention(\n        torch.cat([q_nope, q_pe], dim=-1),\n        torch.cat([cache_k_nope_chunk,\n                   cache_k_pe_chunk.unsqueeze(1).expand(-1, N, -1)],\n                   dim=-1),\n        cache_v_chunk,\n        casual=False,\n        return_softmax_lse=True\n    )\n\n    curr_o, curr_lse = merge_attn_states(\n        suffix_output=curr_o,\n        suffix_lse=curr_lse,\n        prefix_output=chunk_o,\n        prefix_lse=chunk_lse,\n    )\n\nreturn curr_o @ W_O\n'
try:
    is_vllm_fa = True
except ImportError:
    if current_platform.is_rocm():
    is_vllm_fa = False
try:
    flashinfer_available = True
except ImportError:
    flashinfer_available = False
if TYPE_CHECKING:
logger = init_logger(__name__)
CUDNN_WORKSPACE_SIZE = 12800

class MLACommonBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        return 'TRITON_MLA_VLLM_V1'

    @staticmethod
    def get_metadata_cls() -> type['AttentionMetadata']:
        return MLACommonMetadata

    @staticmethod
    def get_builder_cls() -> type['MLACommonMetadataBuilder']:
        return MLACommonMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(num_blocks: int, block_size: int, num_kv_heads: int, head_size: int) -> tuple[int, ...]:
        return (num_blocks, block_size, head_size)

    @classmethod
    def get_supported_dtypes(cls) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [576]

    @classmethod
    def validate_head_size(cls, head_size: int) -> None:
        supported_head_sizes = cls.get_supported_head_sizes()
        if head_size not in supported_head_sizes:
            attn_type = cls.__name__.removesuffix('Backend')
            raise ValueError(f'Head size {head_size} is not supported by {attn_type}. Supported head sizes are: {supported_head_sizes}. Set VLLM_ATTENTION_BACKEND=FLEX_ATTENTION to use FlexAttention backend which supports all head sizes.')

@dataclass
class MLACommonPrefillMetadata:
    """ Prefill Specific Metadata """

    @dataclass
    class ChunkedContextMetadata:
        cu_seq_lens: torch.Tensor
        starts: torch.Tensor
        seq_tot: list[int]
        max_seq_lens: list[int]
        seq_lens: torch.Tensor
        workspace: torch.Tensor
    block_table: torch.Tensor
    query_start_loc: torch.Tensor
    max_query_len: int
    chunked_context: Optional[ChunkedContextMetadata] = None

@dataclass
class FlashInferPrefillMetadata(MLACommonPrefillMetadata):
    prefill_main: Optional['BatchPrefillWithRaggedKVCacheWrapper'] = None
    prefill_chunks: list['BatchPrefillWithRaggedKVCacheWrapper'] = field(default_factory=list)

@dataclass
class CudnnPrefillMetadata(MLACommonPrefillMetadata):

    class ChunkedContextMetadata(MLACommonPrefillMetadata.ChunkedContextMetadata):
        seq_lens: torch.Tensor
    query_seq_lens: Optional[torch.Tensor] = None
    cudnn_workspace: Optional[torch.Tensor] = None

@dataclass
class MLACommonDecodeMetadata:
    block_table: torch.Tensor
    seq_lens: torch.Tensor
D = TypeVar('D', bound=MLACommonDecodeMetadata)

@dataclass
class MLACommonMetadata(Generic[D]):
    """Metadata for MLACommon.

    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """
    num_reqs: int
    max_query_len: int
    num_actual_tokens: int
    query_start_loc: torch.Tensor
    slot_mapping: torch.Tensor
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    head_dim: Optional[int] = None
    decode: Optional[D] = None
    prefill: Optional[Union[MLACommonPrefillMetadata, FlashInferPrefillMetadata, CudnnPrefillMetadata]] = None

    def __post_init__(self):
        if self.head_dim is not None:
            MLACommonBackend.validate_head_size(self.head_dim)
M = TypeVar('M', bound=MLACommonMetadata)

def use_flashinfer_prefill() -> bool:
    if flashinfer_available and (not envs.VLLM_USE_CUDNN_PREFILL):
        return current_platform.has_device_capability(100)
    return False

def use_cudnn_prefill() -> bool:
    if flashinfer_available and envs.VLLM_USE_CUDNN_PREFILL:
        return current_platform.has_device_capability(100)
    return False
FLASHINFER_WORKSPACE_BUFFER_SIZE = 394 * 1024 * 1024

class MLACommonMetadataBuilder(AttentionMetadataBuilder[M]):
    """
    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    def __init__(self, kv_cache_spec: AttentionSpec, vllm_config: VllmConfig, device: torch.device, metadata_cls: Optional[type[M]]=None):
        self.metadata_cls = metadata_cls if metadata_cls is not None else MLACommonMetadata
        self.kv_cache_spec = kv_cache_spec
        self.device = device
        scheduler_config = vllm_config.scheduler_config
        self.model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        parallel_config = vllm_config.parallel_config
        self.chunked_prefill_enabled = scheduler_config.chunked_prefill_enabled
        self.num_heads = self.model_config.get_num_attention_heads(parallel_config)
        self.mla_dims = get_mla_dims(self.model_config)
        self.aot_schedule = current_platform.is_cuda()
        if self.aot_schedule:
            self.page_size = self.kv_cache_spec.block_size
        if self.chunked_prefill_enabled:
            self.chunked_prefill_workspace_size = min(max(8 * self.model_config.max_model_len, 4 * scheduler_config.max_num_seqs * cache_config.block_size), 128 * 1024)
            assert self.chunked_prefill_workspace_size >= scheduler_config.max_num_seqs * cache_config.block_size
            self.chunked_prefill_workspace = torch.empty((self.chunked_prefill_workspace_size, self.model_config.get_head_size()), dtype=self.model_config.dtype, device=device)
        self._use_cudnn_prefill = use_cudnn_prefill()
        self._use_fi_prefill = use_flashinfer_prefill()
        self.prefill_metadata_cls = FlashInferPrefillMetadata if self._use_fi_prefill else CudnnPrefillMetadata if self._use_cudnn_prefill else MLACommonPrefillMetadata
        if self._use_fi_prefill:
            self._workspace_buffer = torch.empty(FLASHINFER_WORKSPACE_BUFFER_SIZE, dtype=torch.uint8, device=device)
            self._fi_prefill_main: Optional[BatchPrefillWithRaggedKVCacheWrapper] = None
            self._fi_prefill_chunks: list[BatchPrefillWithRaggedKVCacheWrapper] = []
            self._global_hyperparameters = infer_global_hyperparameters(get_per_layer_parameters(vllm_config, MLACommonImpl))
        if self._use_cudnn_prefill:
            self.cudnn_workspace = torch.empty(CUDNN_WORKSPACE_SIZE * scheduler_config.max_num_seqs, dtype=torch.int8, device=device)

    def _build_fi_prefill_wrappers(self, prefill: FlashInferPrefillMetadata):
        qo_indptr = prefill.query_start_loc
        has_context = False
        if prefill.chunked_context is not None:
            chunked_context = prefill.chunked_context
            has_context = True
        if self._fi_prefill_main is None:
            self._fi_prefill_main = BatchPrefillWithRaggedKVCacheWrapper(self._workspace_buffer, 'NHD', backend='cutlass')
        if has_context:
            num_chunks = chunked_context.cu_seq_lens.shape[0]
            if len(self._fi_prefill_chunks) < num_chunks:
                for _ in range(len(self._fi_prefill_chunks), num_chunks):
                    self._fi_prefill_chunks.append(BatchPrefillWithRaggedKVCacheWrapper(self._workspace_buffer, 'NHD', backend='cutlass'))
            assert num_chunks <= len(self._fi_prefill_chunks)
        num_qo_heads = self.num_heads
        num_kv_heads = num_qo_heads
        assert self.kv_cache_spec.num_kv_heads == 1
        head_dim_qk = self.mla_dims.qk_nope_head_dim + self.mla_dims.qk_rope_head_dim
        head_dim_vo = self.mla_dims.v_head_dim
        kv_indptr = qo_indptr.clone()
        self._fi_prefill_main.plan(qo_indptr=qo_indptr, kv_indptr=kv_indptr, num_qo_heads=num_qo_heads, num_kv_heads=num_kv_heads, head_dim_qk=head_dim_qk, head_dim_vo=head_dim_vo, causal=True, sm_scale=self._global_hyperparameters.sm_scale, window_left=self._global_hyperparameters.window_left, logits_soft_cap=self._global_hyperparameters.logits_soft_cap, q_data_type=self.model_config.dtype, kv_data_type=self.kv_cache_spec.dtype)
        if has_context:
            for i in range(num_chunks):
                kv_indptr_chunk = chunked_context.cu_seq_lens[i]
                self._fi_prefill_chunks[i].plan(qo_indptr=qo_indptr, kv_indptr=kv_indptr_chunk, num_qo_heads=num_qo_heads, num_kv_heads=num_kv_heads, head_dim_qk=head_dim_qk, head_dim_vo=head_dim_vo, causal=False, sm_scale=self._global_hyperparameters.sm_scale, window_left=self._global_hyperparameters.window_left, logits_soft_cap=self._global_hyperparameters.logits_soft_cap, q_data_type=self.model_config.dtype, kv_data_type=self.kv_cache_spec.dtype)
        prefill.prefill_main = self._fi_prefill_main
        prefill.prefill_chunks = self._fi_prefill_chunks

    def reorder_batch(self, input_batch: 'InputBatch', scheduler_output: 'SchedulerOutput') -> bool:
        return reorder_batch_to_split_decodes_and_prefills(input_batch, scheduler_output, decode_threshold=1)

    def _build_decode(self, block_table_tensor: torch.Tensor, seq_lens: torch.Tensor):
        return MLACommonDecodeMetadata(block_table=block_table_tensor, seq_lens=seq_lens)

    def build_for_cudagraph_capture(self, common_attn_metadata: CommonAttentionMetadata) -> M:
        """
        This method builds the metadata for full cudagraph capture.
        Currently, only decode is supported for full cudagraphs with MLA.
        """
        m = common_attn_metadata
        assert m.num_reqs == m.num_actual_tokens, 'MLA only supports decode-only full CUDAGraph capture. Make sure all cudagraph capture sizes <= max_num_seq.'
        m.max_query_len = 1
        return self.build(0, m)

    def build(self, common_prefix_len: int, common_attn_metadata: CommonAttentionMetadata, fast_build: bool=False) -> M:
        num_reqs = common_attn_metadata.num_reqs
        num_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len
        device = self.device
        block_table_tensor = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping
        query_start_loc = common_attn_metadata.query_start_loc
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        seq_lens = common_attn_metadata.seq_lens
        query_seq_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        num_computed_tokens_cpu = common_attn_metadata.seq_lens_cpu - query_seq_lens_cpu
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = split_decodes_and_prefills(common_attn_metadata)
        assert num_decodes + num_prefills == num_reqs
        assert num_decode_tokens + num_prefill_tokens == num_tokens
        prefill_metadata = None
        if num_prefills > 0:
            reqs_start = num_decodes
            context_lens_cpu = num_computed_tokens_cpu[reqs_start:num_reqs]
            max_context_len_cpu = context_lens_cpu.max().item()
            num_prefills_with_context_cpu = (context_lens_cpu > 0).sum().item()
            prefill_query_start_loc = query_start_loc[reqs_start:] - query_start_loc[reqs_start]
            chunked_context_metadata = None
            if self.chunked_prefill_enabled and num_prefills > 0 and (max_context_len_cpu > 0):
                max_context_chunk = self.chunked_prefill_workspace_size // num_prefills_with_context_cpu
                if self.aot_schedule:
                    max_context_chunk = round_down(max_context_chunk, self.page_size)
                assert max_context_chunk > 0
                num_chunks = cdiv(max_context_len_cpu, max_context_chunk)
                chunk_starts = torch.arange(num_chunks, dtype=torch.int32).unsqueeze(1).expand(-1, num_prefills) * max_context_chunk
                chunk_ends = torch.min(context_lens_cpu.unsqueeze(0), chunk_starts + max_context_chunk)
                chunk_seq_lens = (chunk_ends - chunk_starts).clamp(min=0)
                cu_seq_lens_cpu = torch.zeros(num_chunks, num_prefills + 1, dtype=torch.int32, pin_memory=True)
                torch.cumsum(chunk_seq_lens, dim=1, out=cu_seq_lens_cpu[:, 1:], dtype=torch.int32)
                chunked_context_metadata_cls = CudnnPrefillMetadata.ChunkedContextMetadata if self._use_cudnn_prefill else MLACommonPrefillMetadata.ChunkedContextMetadata
                chunked_context_metadata = chunked_context_metadata_cls(cu_seq_lens=cu_seq_lens_cpu.to(device, non_blocking=True), starts=chunk_starts.to(device, non_blocking=True), seq_tot=chunk_seq_lens.sum(dim=1).tolist(), max_seq_lens=chunk_seq_lens.max(dim=1).values.tolist(), seq_lens=chunk_seq_lens, workspace=self.chunked_prefill_workspace)
                if self._use_cudnn_prefill:
                    chunked_context_metadata.seq_lens = chunk_seq_lens
                assert max(chunked_context_metadata.max_seq_lens) <= self.chunked_prefill_workspace_size
            prefill_metadata = self.prefill_metadata_cls(block_table=block_table_tensor[reqs_start:, ...], query_start_loc=prefill_query_start_loc, max_query_len=max_query_len, chunked_context=chunked_context_metadata)
            if self._use_cudnn_prefill:
                assert isinstance(prefill_metadata, CudnnPrefillMetadata)
                prefill_metadata.query_seq_lens = prefill_query_start_loc[1:] - prefill_query_start_loc[:-1]
                prefill_metadata.cudnn_workspace = self.cudnn_workspace
        decode_metadata = None
        if num_decodes > 0:
            decode_metadata = self._build_decode(block_table_tensor=block_table_tensor[:num_decodes, ...], seq_lens=seq_lens[:num_decodes])
        attn_metadata = self.metadata_cls(num_reqs=common_attn_metadata.num_reqs, max_query_len=common_attn_metadata.max_query_len, num_actual_tokens=num_tokens, query_start_loc=query_start_loc, slot_mapping=slot_mapping, head_dim=self.model_config.get_head_size(), num_decodes=num_decodes, num_decode_tokens=num_decode_tokens, num_prefills=num_prefills, prefill=prefill_metadata, decode=decode_metadata)
        if self._use_fi_prefill and num_prefills > 0:
            assert isinstance(attn_metadata.prefill, FlashInferPrefillMetadata)
            self._build_fi_prefill_wrappers(attn_metadata.prefill)
        return attn_metadata

    def can_run_in_cudagraph(self, common_attn_metadata: CommonAttentionMetadata) -> bool:
        return common_attn_metadata.max_query_len == 1

class MLACommonImpl(MLAAttentionImpl[M], Generic[M]):
    """
    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    def __init__(self, num_heads: int, head_size: int, scale: float, num_kv_heads: int, alibi_slopes: Optional[list[float]], sliding_window: Optional[int], kv_cache_dtype: str, logits_soft_cap: Optional[float], attn_type: str, kv_sharing_target_layer_name: Optional[str], q_lora_rank: Optional[int], kv_lora_rank: int, qk_nope_head_dim: int, qk_rope_head_dim: int, qk_head_dim: int, v_head_dim: int, kv_b_proj: ColumnParallelLinear) -> None:
        if kv_sharing_target_layer_name is not None:
            raise NotImplementedError('KV sharing is not supported for MLA')
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
        if use_flashinfer_prefill():
            logger.debug_once('Using FlashInfer prefill for MLA')
            self._run_prefill_context_chunk = self._run_prefill_context_chunk_fi
            self._run_prefill_new_tokens = self._run_prefill_new_tokens_fi
            self._pad_v = False
        elif use_cudnn_prefill():
            logger.debug_once('Using CUDNN prefill for MLA')
            self._run_prefill_context_chunk = self._run_prefill_context_chunk_cudnn
            self._run_prefill_new_tokens = self._run_prefill_new_tokens_cudnn
            self._pad_v = False
        else:
            logger.debug_once('Using FlashAttention prefill for MLA')
            self._run_prefill_context_chunk = self._run_prefill_context_chunk_fa
            self._run_prefill_new_tokens = self._run_prefill_new_tokens_fa
            self.flash_attn_varlen_func = flash_attn_varlen_func
            self.vllm_flash_attn_version = get_flash_attn_version()
            if self.vllm_flash_attn_version is not None:
                self.flash_attn_varlen_func = functools.partial(flash_attn_varlen_func, fa_version=self.vllm_flash_attn_version)
            self._pad_v = self.vllm_flash_attn_version is None or not (self.vllm_flash_attn_version == 3 and current_platform.get_device_capability()[0] == 9)

    def _flash_attn_varlen_diff_headdims(self, q, k, v, return_softmax_lse=False, softmax_scale=None, **kwargs):
        maybe_padded_v = v
        if self._pad_v:
            maybe_padded_v = torch.nn.functional.pad(v, [0, q.shape[-1] - v.shape[-1]], value=0)
        if is_vllm_fa:
            kwargs['return_softmax_lse'] = return_softmax_lse
        else:
            kwargs['return_attn_probs'] = return_softmax_lse
        attn_out = self.flash_attn_varlen_func(q=q, k=k, v=maybe_padded_v, softmax_scale=softmax_scale, **kwargs)
        lse = None
        if isinstance(attn_out, tuple):
            attn_out, lse = (attn_out[0], attn_out[1])
        if return_softmax_lse:
            return (attn_out, lse)
        return attn_out

    def _run_prefill_new_tokens_fa(self, prefill: MLACommonPrefillMetadata, q, k, v, return_softmax_lse):
        return self._flash_attn_varlen_diff_headdims(q=q, k=k, v=v, cu_seqlens_q=prefill.query_start_loc, cu_seqlens_k=prefill.query_start_loc, max_seqlen_q=prefill.max_query_len, max_seqlen_k=prefill.max_query_len, softmax_scale=self.scale, causal=True, return_softmax_lse=return_softmax_lse)

    def _run_prefill_new_tokens_fi(self, prefill: MLACommonPrefillMetadata, q, k, v, return_softmax_lse):
        assert isinstance(prefill, FlashInferPrefillMetadata)
        assert prefill.prefill_main is not None
        return prefill.prefill_main.run(q=q, k=k, v=v, return_lse=return_softmax_lse)

    def _run_prefill_new_tokens_cudnn(self, prefill: MLACommonPrefillMetadata, q, k, v, return_softmax_lse):
        assert isinstance(prefill, CudnnPrefillMetadata)
        assert prefill.query_seq_lens is not None
        output, lse = cudnn_batch_prefill_with_kv_cache(q=q, k_cache=k, v_cache=v, scale=self.scale, workspace_buffer=prefill.cudnn_workspace, max_token_per_sequence=prefill.max_query_len, max_sequence_kv=prefill.max_query_len, actual_seq_lens_q=prefill.query_seq_lens.view(-1, 1, 1, 1), actual_seq_lens_kv=prefill.query_seq_lens.view(-1, 1, 1, 1), causal=True, return_lse=True, is_cuda_graph_compatible=True)
        if return_softmax_lse:
            return (output, lse)
        return output

    def _run_prefill_context_chunk_fa(self, prefill: MLACommonPrefillMetadata, chunk_idx: int, q, k, v):
        assert prefill.chunked_context is not None
        return self._flash_attn_varlen_diff_headdims(q=q, k=k, v=v, cu_seqlens_q=prefill.query_start_loc, cu_seqlens_k=prefill.chunked_context.cu_seq_lens[chunk_idx], max_seqlen_q=prefill.max_query_len, max_seqlen_k=prefill.chunked_context.max_seq_lens[chunk_idx], softmax_scale=self.scale, causal=False, return_softmax_lse=True)

    def _run_prefill_context_chunk_fi(self, prefill: MLACommonPrefillMetadata, chunk_idx: int, q, k, v):
        assert isinstance(prefill, FlashInferPrefillMetadata)
        return prefill.prefill_chunks[chunk_idx].run(q=q, k=k, v=v, return_lse=True)

    def _run_prefill_context_chunk_cudnn(self, prefill: MLACommonPrefillMetadata, chunk_idx: int, q, k, v):
        assert isinstance(prefill, CudnnPrefillMetadata)
        assert prefill.chunked_context is not None
        assert prefill.chunked_context.seq_lens[chunk_idx] is not None
        assert prefill.query_seq_lens is not None
        return cudnn_batch_prefill_with_kv_cache(q=q, k_cache=k, v_cache=v, scale=self.scale, workspace_buffer=prefill.cudnn_workspace, max_token_per_sequence=prefill.max_query_len, max_sequence_kv=prefill.chunked_context.max_seq_lens[chunk_idx], actual_seq_lens_q=prefill.query_seq_lens.view(-1, 1, 1, 1), actual_seq_lens_kv=prefill.chunked_context.seq_lens[chunk_idx].view(-1, 1, 1, 1), causal=False, return_lse=True, is_cuda_graph_compatible=True)

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
        assert attn_metadata.prefill is not None
        prefill_metadata = attn_metadata.prefill
        assert prefill_metadata.chunked_context is not None
        output = None
        iters = len(prefill_metadata.chunked_context.seq_tot)
        workspace = prefill_metadata.chunked_context.workspace
        for i in range(iters):
            toks = prefill_metadata.chunked_context.seq_tot[i]
            ops.gather_cache(src_cache=kv_c_and_k_pe_cache, dst=workspace, block_table=prefill_metadata.block_table, cu_seq_lens=prefill_metadata.chunked_context.cu_seq_lens[i], batch_size=attn_metadata.num_prefills, seq_starts=prefill_metadata.chunked_context.starts[i])
            kv_c_normed = workspace[:toks][..., :self.kv_lora_rank]
            k_pe = workspace[:toks][..., self.kv_lora_rank:].unsqueeze(1)
            kv_nope = self.kv_b_proj(kv_c_normed)[0].view(-1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = kv_nope.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            k = torch.cat((k_nope, k_pe.expand((*k_nope.shape[:-1], -1))), dim=-1)
            attn_output, attn_softmax_lse = self._run_prefill_context_chunk(prefill=prefill_metadata, chunk_idx=i, q=q, k=k, v=v)
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
        assert attn_metadata.prefill is not None
        has_context = attn_metadata.prefill.chunked_context is not None
        kv_nope = self.kv_b_proj(kv_c_normed)[0].view(-1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = kv_nope.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k = torch.cat((k_nope, k_pe.expand((*k_nope.shape[:-1], -1))), dim=-1)
        output = self._run_prefill_new_tokens(prefill=attn_metadata.prefill, q=q, k=k, v=v, return_softmax_lse=has_context)
        if has_context:
            suffix_output, suffix_lse = output
            context_output, context_lse = self._compute_prefill_context(q, kv_c_and_k_pe_cache, attn_metadata)
            output = torch.empty_like(suffix_output)
            merge_attn_states(output=output, prefix_output=context_output, prefix_lse=context_lse, suffix_output=suffix_output, suffix_lse=suffix_lse)
        if self._pad_v:
            output = output[..., :v.shape[-1]]
        return output.flatten(start_dim=-2)

    @abstractmethod
    def _forward_decode(self, ql_nope: torch.Tensor, q_pe: torch.Tensor, kv_c_and_k_pe_cache: torch.Tensor, attn_metadata: M) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, layer: AttentionLayer, q: torch.Tensor, k_c_normed: torch.Tensor, k_pe: torch.Tensor, kv_cache: torch.Tensor, attn_metadata: M, output: Optional[torch.Tensor]=None, output_scale: Optional[torch.Tensor]=None) -> torch.Tensor:
        assert output is not None, 'Output tensor must be provided.'
        if output_scale is not None:
            raise NotImplementedError('fused output quantization is not yet supported for MLACommonImpl')
        if attn_metadata is None:
            return output.fill_(0)
        num_actual_toks = attn_metadata.num_actual_tokens
        output_padded = output
        output = output[:num_actual_toks, ...]
        q = q[:num_actual_toks, ...]
        k_c_normed = k_c_normed[:num_actual_toks, ...]
        k_pe = k_pe[:num_actual_toks, ...]
        assert attn_metadata.num_decodes is not None and attn_metadata.num_prefills is not None and (attn_metadata.num_decode_tokens is not None)
        has_decode = attn_metadata.num_decodes > 0
        has_prefill = attn_metadata.num_prefills > 0
        num_decode_tokens = attn_metadata.num_decode_tokens
        decode_q = q[:num_decode_tokens]
        prefill_q = q[num_decode_tokens:]
        prefill_k_pe = k_pe[num_decode_tokens:]
        prefill_k_c_normed = k_c_normed[num_decode_tokens:]
        if kv_cache.numel() > 0:
            ops.concat_and_cache_mla(k_c_normed, k_pe.squeeze(1), kv_cache, attn_metadata.slot_mapping.flatten(), kv_cache_dtype=self.kv_cache_dtype, scale=layer._k_scale)
        if has_prefill:
            output[num_decode_tokens:] = self._forward_prefill(prefill_q, prefill_k_c_normed, prefill_k_pe, kv_cache, attn_metadata)
        if has_decode:
            assert attn_metadata.decode is not None
            decode_q_nope, decode_q_pe = decode_q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
            decode_q_nope = decode_q_nope.transpose(0, 1)
            decode_ql_nope = torch.bmm(decode_q_nope, self.W_UK_T)
            decode_ql_nope = decode_ql_nope.transpose(0, 1)
            output[:num_decode_tokens] = self._forward_decode(decode_ql_nope, decode_q_pe, kv_cache, attn_metadata)
        return output_padded