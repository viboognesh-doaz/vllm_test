from dataclasses import dataclass
from flash_attn import flash_attn_varlen_func
from functools import cache
from typing import TYPE_CHECKING, List, Optional, Tuple, Type
from vllm import _custom_ops as ops
from vllm.attention.backends.abstract import AttentionBackend, AttentionImpl, AttentionLayer, AttentionMetadata, AttentionType
from vllm.attention.backends.utils import CommonAttentionState, CommonMetadataBuilder
from vllm.attention.ops.paged_attn import PagedAttention, PagedAttentionMetadata
from vllm.attention.ops.rocm_aiter_paged_attn import AITERPagedAttention
from vllm.attention.ops.triton_flash_attention import triton_attention
from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.platforms import current_platform
from vllm.platforms.rocm import use_rocm_custom_paged_attention
from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata
import itertools
import torch
import vllm.envs as envs
'Attention layer ROCm GPUs.'
if TYPE_CHECKING:
logger = init_logger(__name__)
_PARTITION_SIZE_ROCM = 256

@cache
def is_rocm_aiter_paged_attn_enabled() -> bool:
    return envs.VLLM_ROCM_USE_AITER_PAGED_ATTN and envs.VLLM_ROCM_USE_AITER

@cache
def _get_paged_attn_module() -> PagedAttention:
    """
    Initializes the appropriate PagedAttention module from `attention/ops`,
    which is used as helper function
    by `ROCmFlashAttentionImpl` and `ROCmFlashAttentionBackend`.

    The choice of attention module depends on whether
    AITER paged attention is enabled:
    - If enabled, `ROCmFlashAttentionImpl` uses `AITERPagedAttention`.
    - Otherwise, it defaults to using the original `PagedAttention`.
    """
    if is_rocm_aiter_paged_attn_enabled():
        return AITERPagedAttention()
    return PagedAttention()

class ROCmFlashAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        return 'ROCM_FLASH'

    @staticmethod
    def get_impl_cls() -> Type['ROCmFlashAttentionImpl']:
        return ROCmFlashAttentionImpl

    @staticmethod
    def get_metadata_cls() -> Type['AttentionMetadata']:
        return ROCmFlashAttentionMetadata

    @staticmethod
    def get_builder_cls() -> Type['ROCmFlashAttentionMetadataBuilder']:
        return ROCmFlashAttentionMetadataBuilder

    @staticmethod
    def get_state_cls() -> Type['CommonAttentionState']:
        return CommonAttentionState

    @staticmethod
    def get_kv_cache_shape(num_blocks: int, block_size: int, num_kv_heads: int, head_size: int) -> Tuple[int, ...]:
        paged_attn = _get_paged_attn_module()
        return paged_attn.get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def swap_blocks(src_kv_cache: torch.Tensor, dst_kv_cache: torch.Tensor, src_to_dst: torch.Tensor) -> None:
        paged_attn = _get_paged_attn_module()
        paged_attn.swap_blocks(src_kv_cache, dst_kv_cache, src_to_dst)

    @staticmethod
    def copy_blocks(kv_caches: List[torch.Tensor], src_to_dists: torch.Tensor) -> None:
        paged_attn = _get_paged_attn_module()
        paged_attn.copy_blocks(kv_caches, src_to_dists)

@dataclass
class ROCmFlashAttentionMetadata(AttentionMetadata, PagedAttentionMetadata):
    """Metadata for FlashAttentionBackend.

    NOTE: Any python object stored here is not updated when it is
    cuda-graph replayed. If you have values that need to be changed
    dynamically, it should be stored in tensor. The tensor has to be
    updated from `CUDAGraphRunner.forward` API.
    """
    seq_lens: Optional[List[int]]
    seq_lens_tensor: Optional[torch.Tensor]
    max_prefill_seq_len: int
    max_decode_seq_len: int
    use_cuda_graph: bool
    max_query_len: Optional[int] = None
    query_start_loc: Optional[torch.Tensor] = None
    seq_start_loc: Optional[torch.Tensor] = None
    context_lens_tensor: Optional[torch.Tensor] = None
    max_decode_query_len: Optional[int] = None
    _cached_prefill_metadata: Optional['ROCmFlashAttentionMetadata'] = None
    _cached_decode_metadata: Optional['ROCmFlashAttentionMetadata'] = None
    encoder_seq_lens: Optional[List[int]] = None
    encoder_seq_lens_tensor: Optional[torch.Tensor] = None
    max_encoder_seq_len: Optional[int] = None
    num_encoder_tokens: Optional[int] = None
    cross_slot_mapping: Optional[torch.Tensor] = None
    cross_block_tables: Optional[torch.Tensor] = None

    @property
    def prefill_metadata(self) -> Optional['ROCmFlashAttentionMetadata']:
        if self.num_prefills == 0:
            return None
        if self._cached_prefill_metadata is not None:
            return self._cached_prefill_metadata
        assert self.seq_lens is not None
        assert self.seq_lens_tensor is not None
        assert self.block_tables is not None
        self._cached_prefill_metadata = ROCmFlashAttentionMetadata(num_prefills=self.num_prefills, num_prefill_tokens=self.num_prefill_tokens, num_decode_tokens=0, slot_mapping=self.slot_mapping[:self.num_prefill_tokens], multi_modal_placeholder_index_maps=self.multi_modal_placeholder_index_maps, enable_kv_scales_calculation=self.enable_kv_scales_calculation, seq_lens=self.seq_lens[:self.num_prefills], seq_lens_tensor=self.seq_lens_tensor[:self.num_prefills], max_query_len=self.max_query_len, max_prefill_seq_len=self.max_prefill_seq_len, max_decode_seq_len=0, query_start_loc=None if self.query_start_loc is None else self.query_start_loc[:self.num_prefills + 1], seq_start_loc=None if self.seq_start_loc is None else self.seq_start_loc[:self.num_prefills + 1], context_lens_tensor=None if self.context_lens_tensor is None else self.context_lens_tensor[:self.num_prefills], block_tables=self.block_tables[:self.num_prefills], use_cuda_graph=False, encoder_seq_lens=self.encoder_seq_lens, encoder_seq_lens_tensor=self.encoder_seq_lens_tensor, max_encoder_seq_len=self.max_encoder_seq_len, cross_slot_mapping=self.cross_slot_mapping, cross_block_tables=self.cross_block_tables)
        return self._cached_prefill_metadata

    @property
    def decode_metadata(self) -> Optional['ROCmFlashAttentionMetadata']:
        if self.num_decode_tokens == 0:
            return None
        if self._cached_decode_metadata is not None:
            return self._cached_decode_metadata
        assert self.block_tables is not None
        assert self.seq_lens_tensor is not None
        self._cached_decode_metadata = ROCmFlashAttentionMetadata(num_prefills=0, num_prefill_tokens=0, num_decode_tokens=self.num_decode_tokens, slot_mapping=self.slot_mapping[self.num_prefill_tokens:], multi_modal_placeholder_index_maps=None, enable_kv_scales_calculation=True, seq_lens=None, seq_lens_tensor=self.seq_lens_tensor[self.num_prefills:], max_query_len=None, max_prefill_seq_len=0, max_decode_seq_len=self.max_decode_seq_len, query_start_loc=None, seq_start_loc=None, context_lens_tensor=None, block_tables=self.block_tables[self.num_prefills:], use_cuda_graph=self.use_cuda_graph, encoder_seq_lens=self.encoder_seq_lens, encoder_seq_lens_tensor=self.encoder_seq_lens_tensor, max_encoder_seq_len=self.max_encoder_seq_len, cross_slot_mapping=self.cross_slot_mapping, cross_block_tables=self.cross_block_tables)
        if self._cached_decode_metadata.query_start_loc is not None:
            qs = self._cached_decode_metadata.query_start_loc
            self._cached_decode_metadata.query_start_loc = qs - qs[0]
        return self._cached_decode_metadata

    def advance_step(self, model_input: 'ModelInputForGPUWithSamplingMetadata', sampled_token_ids: Optional[torch.Tensor], block_size: int, num_seqs: int, num_queries: int, turn_prefills_into_decodes: bool=False):
        """
        Update metadata in-place to advance one decode step.
        """
        assert not turn_prefills_into_decodes, 'Chunked prefill is not supported with rocm_flash_attn yet.turn_prefills_into_decodes is a Multi-Step + Chunked-Prefill specific parameter.'
        if num_seqs != num_queries:
            assert num_seqs > num_queries
            assert self.use_cuda_graph
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
        assert self.max_decode_seq_len == max(self.seq_lens)
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
        ops.advance_step_flashattn(num_seqs=num_seqs, num_queries=num_queries, block_size=block_size, input_tokens=model_input.input_tokens, sampled_token_ids=sampled_token_ids, input_positions=model_input.input_positions, seq_lens=self.seq_lens_tensor, slot_mapping=self.slot_mapping, block_tables=self.block_tables)

class ROCmFlashAttentionMetadataBuilder(CommonMetadataBuilder[ROCmFlashAttentionMetadata]):
    _metadata_cls = ROCmFlashAttentionMetadata

def _make_alibi_bias(alibi_slopes: torch.Tensor, dtype: torch.dtype, seq_lens: Optional[List[int]], make_attn_mask: bool=True) -> List[torch.Tensor]:
    attn_biases = []
    if seq_lens:
        for seq_len in seq_lens:
            bias = torch.arange(seq_len, dtype=dtype)
            bias = bias[None, :] - bias[:, None]
            num_heads = alibi_slopes.shape[0]
            bias = bias[None, :].repeat((num_heads, 1, 1)).to(alibi_slopes.device)
            bias.mul_(alibi_slopes[:, None, None])
            if make_attn_mask:
                inf_mask = torch.empty((1, seq_len, seq_len), dtype=bias.dtype).fill_(-torch.inf).triu_(diagonal=1).to(alibi_slopes.device)
                attn_biases.append((bias + inf_mask).to(dtype))
            else:
                attn_biases.append(bias.to(dtype))
    return attn_biases

def _get_seq_len_block_table_args(attn_metadata: ROCmFlashAttentionMetadata, attn_type: str) -> tuple:
    """
    The particular choice of sequence-length
    attributes which should be extracted from attn_metadata is dependent
    on the type of attention operation.

    Decoder attn -> select entirely decoder self-attention-related fields
    Encoder/decoder cross-attn -> select encoder sequence lengths
    Encoder attn -> select encoder sequence lengths fields
    Encoder-only attn -> select prefill sequence lengths with 
        bidirectional attention
    
    Arguments:

    * attn_metadata: Attention metadata structure associated with attention op
    * attn_type: encoder attention, decoder self-attention,
                encoder/decoder cross-attention, encoder-only

    Returns:

    * Appropriate sequence-lengths tensors for query and key
    * Appropriate max sequence-length scalar
    * Causal masking flag
    """
    if attn_type == AttentionType.ENCODER:
        assert attn_metadata.encoder_seq_lens is not None
        assert attn_metadata.encoder_seq_lens_tensor is not None
        query_seq_start_loc = torch.tensor(list(itertools.accumulate([0] + attn_metadata.encoder_seq_lens)), device=attn_metadata.encoder_seq_lens_tensor.device, dtype=attn_metadata.encoder_seq_lens_tensor.dtype)
        causal_mask = False
        return (query_seq_start_loc, attn_metadata.max_encoder_seq_len, query_seq_start_loc, attn_metadata.max_encoder_seq_len, attn_metadata.encoder_seq_lens, causal_mask)
    elif attn_type == AttentionType.ENCODER_ONLY:
        assert attn_metadata.seq_lens is not None
        assert attn_metadata.seq_lens_tensor is not None
        query_seq_start_loc = torch.tensor(list(itertools.accumulate([0] + attn_metadata.seq_lens)), device=attn_metadata.seq_lens_tensor.device, dtype=attn_metadata.seq_lens_tensor.dtype)
        max_seq_len = attn_metadata.max_prefill_seq_len
        causal_mask = False
        return (query_seq_start_loc, max_seq_len, query_seq_start_loc, max_seq_len, attn_metadata.seq_lens, causal_mask)
    elif attn_type == AttentionType.DECODER:
        assert attn_metadata.seq_lens is not None
        assert attn_metadata.seq_lens_tensor is not None
        query_seq_start_loc = torch.tensor(list(itertools.accumulate([0] + attn_metadata.seq_lens)), device=attn_metadata.seq_lens_tensor.device, dtype=attn_metadata.seq_lens_tensor.dtype)
        max_seq_len = attn_metadata.max_prefill_seq_len
        causal_mask = True
        return (query_seq_start_loc, max_seq_len, query_seq_start_loc, max_seq_len, attn_metadata.seq_lens, causal_mask)
    elif attn_type == AttentionType.ENCODER_DECODER:
        assert attn_metadata.seq_lens is not None
        assert attn_metadata.encoder_seq_lens_tensor is not None
        query_start_loc = torch.tensor(list(itertools.accumulate([0] + attn_metadata.seq_lens)), device=attn_metadata.encoder_seq_lens_tensor.device, dtype=attn_metadata.encoder_seq_lens_tensor.dtype)
        assert attn_metadata.encoder_seq_lens is not None
        assert attn_metadata.seq_lens_tensor is not None
        key_seq_start_loc = torch.tensor(list(itertools.accumulate([0] + attn_metadata.encoder_seq_lens)), device=attn_metadata.seq_lens_tensor.device, dtype=attn_metadata.seq_lens_tensor.dtype)
        causal_mask = False
        return (query_start_loc, attn_metadata.max_prefill_seq_len, key_seq_start_loc, attn_metadata.max_encoder_seq_len, attn_metadata.seq_lens, causal_mask)
    else:
        raise AttributeError(f'Invalid attention type {str(attn_type)}')

class ROCmFlashAttentionImpl(AttentionImpl):
    """
    If the input tensors contain prompt tokens, the layout is as follows:
    |<--------------- num_prompt_tokens -------------->|
    |<--prompt_0-->|<--prompt_1-->|...|<--prompt_N-1-->|

    Otherwise, the layout is as follows:
    |<------------------ num_generation_tokens (M) ----------------->|
    |<--generation_0-->|..........|<--generation_M-1-->|<--padding-->|

    Generation tokens can contain padding when cuda-graph is used.
    Currently, prompt tokens don't contain any padding.

    The prompts might have different lengths, while the generation tokens
    always have length 1.

    If chunked prefill is enabled, prefill tokens and decode tokens can be
    batched together in a flattened 1D query.

    |<----- num_prefill_tokens ---->|<------- num_decode_tokens ----------->|	
    |<-prompt_0->|...|<-prompt_N-1->|<-generation_0->|...|<-generation_M-1->|

    Currently, cuda graph is disabled for chunked prefill, meaning there's no
    padding between prefill and decode tokens.
    """

    def __init__(self, num_heads: int, head_size: int, scale: float, num_kv_heads: int, alibi_slopes: Optional[List[float]], sliding_window: Optional[int], kv_cache_dtype: str, logits_soft_cap: Optional[float]=None, attn_type: str=AttentionType.DECODER, kv_sharing_target_layer_name: Optional[str]=None, use_irope: bool=False) -> None:
        if kv_sharing_target_layer_name is not None:
            raise NotImplementedError('KV sharing is not supported in V0 ROCM_FLASH backend.')
        if use_irope:
            logger.warning_once('Using irope in ROCm Flash Attention is not supported yet, it will fail back to global attention for long context.')
        if use_irope:
            logger.warning('Using irope in V0 is not supported yet, it will fall back to global attention for long context.')
        if logits_soft_cap is None:
            self.logits_soft_cap = 0.0
        else:
            self.logits_soft_cap = logits_soft_cap
        self.attn_type = attn_type
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        self.sliding_window = (sliding_window, sliding_window) if sliding_window is not None else (-1, -1)
        self.kv_cache_dtype = kv_cache_dtype
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.paged_attn_module = _get_paged_attn_module()
        supported_head_sizes = self.paged_attn_module.get_supported_head_sizes()
        if head_size not in supported_head_sizes:
            raise ValueError(f'Head size {head_size} is not supported by PagedAttention. Supported head sizes are: {supported_head_sizes}.')
        self.use_naive_attn = False
        self.use_triton_flash_attn = envs.VLLM_USE_TRITON_FLASH_ATTN
        if self.use_triton_flash_attn:
            if logits_soft_cap is not None:
                raise ValueError('ROCm Triton FlashAttention does not support attention logits soft capping. please try using the ROCm CK FA backend instead by setting the env var `VLLM_USE_TRITON_FLASH_ATTN=0`')
            self.triton_attn_func = triton_attention
            logger.debug('Using Triton FA in ROCmBackend')
            if self.sliding_window != (-1, -1):
                logger.warning('ROCm Triton FA does not currently support sliding window attention. If using half precision, please try using the ROCm CK FA backend instead by setting the env var `VLLM_USE_TRITON_FLASH_ATTN=0`')
        else:
            if not current_platform.has_device_capability(90):
                self.use_naive_attn = True
            else:
                try:
                    self.fa_attn_func = flash_attn_varlen_func
                    logger.debug('Using CK FA in ROCmBackend')
                except ModuleNotFoundError:
                    self.use_naive_attn = True
            if self.use_naive_attn:
                if logits_soft_cap is not None:
                    raise ValueError('ROCm Naive FlashAttention does not support attention logits soft capping.')
                self.sdpa_attn_func = _sdpa_attention
                logger.debug('Using naive (SDPA) attention in ROCmBackend')
        self.aiter_kv_scales_initialized = False
        self.force_fp8_attention = get_current_vllm_config() is not None and get_current_vllm_config().model_config.override_attention_dtype == 'fp8'

    def repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """torch.repeat_interleave(x, dim=1, repeats=n_rep)"""
        tokens, n_kv_heads, head_dim = x.shape
        return x[:, :, None, :].expand(tokens, n_kv_heads, n_rep, head_dim).reshape(tokens, n_kv_heads * n_rep, head_dim)

    def fused_output_quant_supported(self, dtype: torch.dtype, static: bool, group_shape: GroupShape):
        if self.use_triton_flash_attn:
            return dtype == current_platform.fp8_dtype() and static and (group_shape == GroupShape.PER_TENSOR)
        return False

    def forward(self, layer: AttentionLayer, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, kv_cache: torch.Tensor, attn_metadata: ROCmFlashAttentionMetadata, output: Optional[torch.Tensor]=None, output_scale: Optional[torch.Tensor]=None) -> torch.Tensor:
        """Forward pass with FlashAttention and PagedAttention.

        For decoder-only models: query, key and value must be non-None.

        For encoder/decoder models:
        * ROCmFlashAttentionImpl.forward() may be invoked for both self- and 
            cross-attention layers.
        * For self-attention: query, key and value must be non-None.
        * For cross-attention:
            * Query must be non-None
            * During prefill, key and value must be non-None; key and value
              get cached for use during decode.
            * During decode, key and value may be None, since:
              (1) key and value tensors were cached during prefill, and
              (2) cross-attention key and value tensors do not grow during
                  decode
        
        A note on how the attn_type (attention type enum) argument impacts
        attention forward() behavior:
    
            * DECODER: normal decoder-only behavior;
                use decoder self-attention block table
            * ENCODER: no KV caching; pass encoder sequence
                attributes (encoder_seq_lens/encoder_seq_lens_tensor/
                max_encoder_seq_len) to kernel, in lieu of decoder
                sequence attributes (seq_lens/seq_lens_tensor/max_seq_len)
            * ENCODER_DECODER: cross-attention behavior;
                use cross-attention block table for caching KVs derived
                from encoder hidden states; since KV sequence lengths
                will match encoder sequence lengths, pass encoder sequence
                attributes to kernel (encoder_seq_lens/encoder_seq_lens_tensor/
                max_encoder_seq_len)
            * ENCODER_ONLY: bidirectional attention with no KV caching;
                use prefill sequence attributes

        Args:
            query: shape = [num_tokens, num_heads * head_size]
            key: shape = [num_tokens, num_kv_heads * head_size]
            value: shape = [num_tokens, num_kv_heads * head_size]
            kv_cache = [2, num_blocks, block_size * num_kv_heads * head_size]
                NOTE: kv_cache will be an empty tensor with shape [0]
                for profiling run.
            attn_metadata: Metadata for attention.
            attn_type: Select attention type, between encoder attention,
                       decoder self-attention, or encoder/decoder cross-
                       attention. Defaults to decoder self-attention,
                       which is the vLLM default generally
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        assert output is not None, 'Output tensor must be provided.'
        if output_scale is not None and (not self.use_triton_flash_attn):
            raise NotImplementedError('fused output quantization only supported for Triton implementation in ROCMFlashAttentionImpl for now')
        query = query.view(-1, self.num_heads, self.head_size)
        if key is not None:
            assert value is not None
            key = key.view(-1, self.num_kv_heads, self.head_size)
            value = value.view(-1, self.num_kv_heads, self.head_size)
        else:
            assert value is None
        paged_attn = self.paged_attn_module
        if is_rocm_aiter_paged_attn_enabled() and kv_cache.dtype.itemsize == 1 and (not self.aiter_kv_scales_initialized) and (kv_cache.shape != torch.Size([0])):
            num_blocks = kv_cache.shape[1]
            block_size = kv_cache.shape[2] // (self.num_kv_heads * self.head_size)
            k_scale = torch.empty((self.num_kv_heads, num_blocks * block_size), dtype=torch.float32, device=kv_cache.device)
            v_scale = torch.empty((self.num_kv_heads, num_blocks * block_size), dtype=torch.float32, device=kv_cache.device)
            self.aiter_kv_scales_initialized = True
            k_scale.fill_(layer._k_scale.item())
            v_scale.fill_(layer._v_scale.item())
            layer._k_scale = k_scale
            layer._v_scale = v_scale
        if self.attn_type not in [AttentionType.ENCODER, AttentionType.ENCODER_ONLY] and kv_cache.numel() > 0:
            key_cache, value_cache = paged_attn.split_kv_cache(kv_cache, self.num_kv_heads, self.head_size)
            if key is not None and value is not None:
                paged_attn.write_to_paged_cache(key, value, key_cache, value_cache, attn_metadata.slot_mapping if self.attn_type != AttentionType.ENCODER_DECODER else attn_metadata.cross_slot_mapping, self.kv_cache_dtype, layer._k_scale, layer._v_scale)
        if self.attn_type != AttentionType.ENCODER:
            num_prefill_tokens = attn_metadata.num_prefill_tokens
        elif self.attn_type == AttentionType.ENCODER_ONLY:
            num_prefill_tokens = query.shape[0]
        else:
            assert attn_metadata.num_encoder_tokens is not None
            num_prefill_tokens = attn_metadata.num_encoder_tokens
        decode_query = query[num_prefill_tokens:]
        query = query[:num_prefill_tokens]
        if key is not None and value is not None and (self.attn_type not in [AttentionType.ENCODER_DECODER, AttentionType.ENCODER_ONLY]):
            key = key[:num_prefill_tokens]
            value = value[:num_prefill_tokens]
        if (prefill_meta := attn_metadata.prefill_metadata):
            if self.attn_type == AttentionType.DECODER and (kv_cache.numel() == 0 or prefill_meta.block_tables is None or prefill_meta.block_tables.numel() == 0):
                query_seq_start_loc, query_max_seq_len, key_seq_start_loc, key_max_seq_len, seq_lens, causal_mask = (prefill_meta.seq_start_loc, prefill_meta.max_prefill_seq_len, prefill_meta.seq_start_loc, prefill_meta.max_prefill_seq_len, attn_metadata.seq_lens, True)
            else:
                query_seq_start_loc, query_max_seq_len, key_seq_start_loc, key_max_seq_len, seq_lens, causal_mask = _get_seq_len_block_table_args(prefill_meta, self.attn_type)
            if kv_cache.numel() == 0 or prefill_meta.block_tables.numel() == 0:
                attn_masks = None
                if self.use_triton_flash_attn:
                    if self.alibi_slopes is not None:
                        attn_masks = _make_alibi_bias(self.alibi_slopes, query.dtype, seq_lens, make_attn_mask=causal_mask)
                    use_fp8_scales = layer._q_scale and layer._k_scale and layer._v_scale and layer._prob_scale and (self.kv_cache_dtype == 'fp8' or self.force_fp8_attention)
                    full_scales = (layer._q_scale.item(), layer._k_scale.item(), layer._v_scale.item(), layer._prob_scale.item()) if use_fp8_scales else None
                    self.triton_attn_func(query, key, value, output[:num_prefill_tokens], query_seq_start_loc, key_seq_start_loc, query_max_seq_len, key_max_seq_len, causal_mask, self.scale, attn_masks[0][None] if attn_masks is not None else None, full_scales, output_scale)
                elif self.use_naive_attn:
                    if self.num_kv_heads != self.num_heads:
                        key = self.repeat_kv(key, self.num_queries_per_kv)
                        value = self.repeat_kv(value, self.num_queries_per_kv)
                    if self.alibi_slopes is not None:
                        attn_masks = _make_alibi_bias(self.alibi_slopes, query.dtype, attn_metadata.seq_lens, make_attn_mask=causal_mask)
                    query = query.movedim(0, query.dim() - 2)
                    key = key.movedim(0, key.dim() - 2)
                    value = value.movedim(0, value.dim() - 2)
                    self.sdpa_attn_func(query, key, value, output[:num_prefill_tokens], query_seq_start_loc, num_prefill_tokens, self.num_heads, self.head_size, self.scale, attn_masks)
                else:
                    output[:num_prefill_tokens] = self.fa_attn_func(q=query, k=key, v=value, cu_seqlens_q=query_seq_start_loc, cu_seqlens_k=key_seq_start_loc, max_seqlen_q=prefill_meta.max_prefill_seq_len, max_seqlen_k=key_max_seq_len, softmax_scale=self.scale, causal=causal_mask, window_size=self.sliding_window, alibi_slopes=self.alibi_slopes, softcap=self.logits_soft_cap)
            elif self.attn_type != AttentionType.ENCODER_ONLY:
                output[:num_prefill_tokens] = paged_attn.forward_prefix(query, key, value, self.kv_cache_dtype, key_cache, value_cache, prefill_meta.block_tables, prefill_meta.query_start_loc, prefill_meta.seq_lens_tensor, prefill_meta.max_query_len, self.alibi_slopes, self.sliding_window[0], layer._k_scale, layer._v_scale)
        if (decode_meta := attn_metadata.decode_metadata) and self.attn_type != AttentionType.ENCODER_ONLY:
            num_seqs, num_heads, head_size = decode_query.shape
            block_size = value_cache.shape[3]
            gqa_ratio = num_heads // self.num_kv_heads
            use_custom = use_rocm_custom_paged_attention(decode_query.dtype, head_size, block_size, gqa_ratio, decode_meta.max_decode_seq_len, self.sliding_window, self.kv_cache_dtype, self.alibi_slopes)
            if use_custom:
                max_seq_len = decode_meta.max_decode_seq_len if self.attn_type != AttentionType.ENCODER_DECODER else decode_meta.max_encoder_seq_len
                assert max_seq_len is not None
                max_num_partitions = (max_seq_len + _PARTITION_SIZE_ROCM - 1) // _PARTITION_SIZE_ROCM
                assert _PARTITION_SIZE_ROCM % block_size == 0
                tmp_output = torch.empty(size=(num_seqs, num_heads, max_num_partitions, head_size), dtype=query.dtype, device=output.device)
                exp_sums = torch.empty(size=(num_seqs, num_heads, max_num_partitions), dtype=torch.float32, device=output.device)
                max_logits = torch.empty_like(exp_sums)
                query_start_loc = None
                ops.paged_attention_rocm(output[num_prefill_tokens:], exp_sums, max_logits, tmp_output, decode_query, key_cache, value_cache, self.num_kv_heads, self.scale, decode_meta.block_tables if self.attn_type != AttentionType.ENCODER_DECODER else decode_meta.cross_block_tables, decode_meta.seq_lens_tensor if self.attn_type != AttentionType.ENCODER_DECODER else decode_meta.encoder_seq_lens_tensor, query_start_loc, block_size, max_seq_len, self.alibi_slopes, self.kv_cache_dtype, layer._k_scale, layer._v_scale, output_scale)
            else:
                if output_scale is None:
                    out_pa = output[num_prefill_tokens:]
                else:
                    out_pa = torch.empty_like(output[num_prefill_tokens:], dtype=query.dtype)
                out_pa[:] = paged_attn.forward_decode(decode_query, key_cache, value_cache, decode_meta.block_tables if self.attn_type != AttentionType.ENCODER_DECODER else decode_meta.cross_block_tables, decode_meta.seq_lens_tensor if self.attn_type != AttentionType.ENCODER_DECODER else decode_meta.encoder_seq_lens_tensor, decode_meta.max_decode_seq_len if self.attn_type != AttentionType.ENCODER_DECODER else decode_meta.max_encoder_seq_len, self.kv_cache_dtype, self.num_kv_heads, self.scale, self.alibi_slopes, layer._k_scale, layer._v_scale)
                if output_scale is not None:
                    out_uq = out_pa.view(-1, self.num_heads * self.head_size)
                    out_q = output.view(-1, self.num_heads * self.head_size)
                    ops.scaled_fp8_quant(out_uq, output_scale, output=out_q[num_prefill_tokens:])
        return output.view(-1, self.num_heads * self.head_size)

def _sdpa_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, output: torch.Tensor, seq_lens: torch.Tensor, num_tokens: int, num_heads: int, head_size: int, scale: float, attn_masks: Optional[List[torch.Tensor]]=None) -> torch.Tensor:
    start = 0
    assert output.shape == (num_tokens, num_heads, head_size)
    assert output.dtype == query.dtype
    assert output.device == query.device
    for i, seq_len in enumerate(seq_lens):
        end = start + seq_len
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            sub_out = torch.nn.functional.scaled_dot_product_attention(query[:, start:end, :], key[:, start:end, :], value[:, start:end, :], dropout_p=0.0, is_causal=attn_masks is None, attn_mask=attn_masks[i] if attn_masks else None, scale=scale).movedim(query.dim() - 2, 0)
            output[start:end, :, :] = sub_out
            start = end
    return output