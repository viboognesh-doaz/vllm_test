from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Type
from vllm.attention.backends.abstract import AttentionBackend, AttentionImpl, AttentionLayer, AttentionMetadata, AttentionType
from vllm.attention.backends.utils import CommonAttentionState, CommonMetadataBuilder, get_num_prefill_decode_query_kv_tokens, get_seq_len_block_table_args, is_all_cross_attn_metadata_set, is_all_encoder_attn_metadata_set
from vllm.attention.ops.paged_attn import PagedAttention, PagedAttentionMetadata
from vllm.logger import init_logger
from xformers import ops as xops
from xformers.ops.fmha.attn_bias import AttentionBias, BlockDiagonalCausalMask, BlockDiagonalMask, LowerTriangularMaskWithTensorBias
import torch
'Attention layer with xFormers and PagedAttention.'
logger = init_logger(__name__)

class XFormersBackend(AttentionBackend):

    @staticmethod
    def get_name() -> str:
        return 'XFORMERS'

    @staticmethod
    def get_impl_cls() -> Type['XFormersImpl']:
        return XFormersImpl

    @staticmethod
    def get_metadata_cls() -> Type['AttentionMetadata']:
        return XFormersMetadata

    @staticmethod
    def get_builder_cls() -> Type['XFormersMetadataBuilder']:
        return XFormersMetadataBuilder

    @staticmethod
    def get_state_cls() -> Type['CommonAttentionState']:
        return CommonAttentionState

    @staticmethod
    def get_kv_cache_shape(num_blocks: int, block_size: int, num_kv_heads: int, head_size: int) -> Tuple[int, ...]:
        return PagedAttention.get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def swap_blocks(src_kv_cache: torch.Tensor, dst_kv_cache: torch.Tensor, src_to_dst: Dict[int, int]) -> None:
        PagedAttention.swap_blocks(src_kv_cache, dst_kv_cache, src_to_dst)

    @staticmethod
    def copy_blocks(kv_caches: List[torch.Tensor], src_to_dists: torch.Tensor) -> None:
        PagedAttention.copy_blocks(kv_caches, src_to_dists)

@dataclass
class XFormersMetadata(AttentionMetadata, PagedAttentionMetadata):
    """Metadata for XFormersbackend.

    NOTE: Any python object stored here is not updated when it is
    cuda-graph replayed. If you have values that need to be changed
    dynamically, it should be stored in tensor. The tensor has to be
    updated from `CUDAGraphRunner.forward` API.
    """
    seq_lens_tensor: Optional[torch.Tensor]
    max_prefill_seq_len: int
    max_decode_seq_len: int
    use_cuda_graph: bool
    seq_lens: Optional[List[int]] = None
    seq_start_loc: Optional[torch.Tensor] = None
    context_lens_tensor: Optional[torch.Tensor] = None
    max_query_len: Optional[int] = None
    max_decode_query_len: Optional[int] = None
    query_start_loc: Optional[torch.Tensor] = None
    _cached_prefill_metadata: Optional['XFormersMetadata'] = None
    _cached_decode_metadata: Optional['XFormersMetadata'] = None
    encoder_seq_lens: Optional[List[int]] = None
    encoder_seq_lens_tensor: Optional[torch.Tensor] = None
    encoder_seq_start_loc: Optional[torch.Tensor] = None
    max_encoder_seq_len: Optional[int] = None
    num_encoder_tokens: Optional[int] = None
    cross_slot_mapping: Optional[torch.Tensor] = None
    cross_block_tables: Optional[torch.Tensor] = None

    def __post_init__(self):
        self.attn_bias: Optional[List[AttentionBias]] = None
        self.encoder_attn_bias: Optional[List[AttentionBias]] = None
        self.cross_attn_bias: Optional[List[AttentionBias]] = None

    @property
    def is_all_encoder_attn_metadata_set(self):
        """
        All attention metadata required for encoder attention is set.
        """
        return is_all_encoder_attn_metadata_set(self)

    @property
    def is_all_cross_attn_metadata_set(self):
        """
        All attention metadata required for enc/dec cross-attention is set.

        Superset of encoder attention required metadata.
        """
        return is_all_cross_attn_metadata_set(self)

    @property
    def prefill_metadata(self) -> Optional['XFormersMetadata']:
        if self.num_prefills == 0:
            return None
        if self._cached_prefill_metadata is not None:
            return self._cached_prefill_metadata
        assert self.seq_lens is not None or self.encoder_seq_lens is not None
        assert self.seq_lens_tensor is not None or self.encoder_seq_lens_tensor is not None
        query_start_loc = None if self.query_start_loc is None else self.query_start_loc[:self.num_prefills + 1]
        seq_start_loc = None if self.seq_start_loc is None else self.seq_start_loc[:self.num_prefills + 1]
        slot_mapping = None if self.slot_mapping is None else self.slot_mapping[:self.num_prefill_tokens]
        seq_lens = None if self.seq_lens is None else self.seq_lens[:self.num_prefills]
        seq_lens_tensor = None if self.seq_lens_tensor is None else self.seq_lens_tensor[:self.num_prefills]
        context_lens_tensor = None if self.context_lens_tensor is None else self.context_lens_tensor[:self.num_prefills]
        block_tables = None if self.block_tables is None else self.block_tables[:self.num_prefills]
        self._cached_prefill_metadata = XFormersMetadata(num_prefills=self.num_prefills, num_prefill_tokens=self.num_prefill_tokens, num_decode_tokens=0, slot_mapping=slot_mapping, multi_modal_placeholder_index_maps=self.multi_modal_placeholder_index_maps, enable_kv_scales_calculation=self.enable_kv_scales_calculation, seq_lens=seq_lens, seq_lens_tensor=seq_lens_tensor, max_query_len=self.max_query_len, max_prefill_seq_len=self.max_prefill_seq_len, max_decode_seq_len=0, query_start_loc=query_start_loc, seq_start_loc=seq_start_loc, context_lens_tensor=context_lens_tensor, block_tables=block_tables, use_cuda_graph=False, encoder_seq_lens=self.encoder_seq_lens, encoder_seq_lens_tensor=self.encoder_seq_lens_tensor, max_encoder_seq_len=self.max_encoder_seq_len, cross_slot_mapping=self.cross_slot_mapping, cross_block_tables=self.cross_block_tables)
        return self._cached_prefill_metadata

    @property
    def decode_metadata(self) -> Optional['XFormersMetadata']:
        if self.num_decode_tokens == 0:
            return None
        if self._cached_decode_metadata is not None:
            return self._cached_decode_metadata
        assert self.seq_lens_tensor is not None or self.encoder_seq_lens_tensor is not None
        slot_mapping = None if self.slot_mapping is None else self.slot_mapping[self.num_prefill_tokens:]
        seq_lens_tensor = None if self.seq_lens_tensor is None else self.seq_lens_tensor[self.num_prefills:]
        block_tables = None if self.block_tables is None else self.block_tables[self.num_prefills:]
        self._cached_decode_metadata = XFormersMetadata(num_prefills=0, num_prefill_tokens=0, num_decode_tokens=self.num_decode_tokens, slot_mapping=slot_mapping, multi_modal_placeholder_index_maps=None, enable_kv_scales_calculation=True, seq_lens_tensor=seq_lens_tensor, max_prefill_seq_len=0, max_decode_seq_len=self.max_decode_seq_len, block_tables=block_tables, use_cuda_graph=self.use_cuda_graph, encoder_seq_lens=self.encoder_seq_lens, encoder_seq_lens_tensor=self.encoder_seq_lens_tensor, max_encoder_seq_len=self.max_encoder_seq_len, cross_slot_mapping=self.cross_slot_mapping, cross_block_tables=self.cross_block_tables)
        if self._cached_decode_metadata.query_start_loc is not None:
            qs = self._cached_decode_metadata.query_start_loc
            self._cached_decode_metadata.query_start_loc = qs - qs[0]
        return self._cached_decode_metadata

def _get_attn_bias(attn_metadata: XFormersMetadata, attn_type: str) -> Optional[AttentionBias]:
    """
    Extract appropriate attention bias from attention metadata
    according to attention type.

    Arguments:

    * attn_metadata: Attention metadata structure associated with attention
    * attn_type: encoder attention, decoder self-attention,
                 encoder/decoder cross-attention

    Returns:
    * Appropriate attention bias value given the attention type
    """
    if attn_type == AttentionType.DECODER or attn_type == AttentionType.ENCODER_ONLY:
        return attn_metadata.attn_bias
    elif attn_type == AttentionType.ENCODER:
        return attn_metadata.encoder_attn_bias
    elif attn_type == AttentionType.ENCODER_DECODER:
        return attn_metadata.cross_attn_bias
    else:
        raise AttributeError(f'Invalid attention type {str(attn_type)}')

def _set_attn_bias(attn_metadata: XFormersMetadata, attn_bias: List[Optional[AttentionBias]], attn_type: str) -> None:
    """
    Update appropriate attention bias field of attention metadata,
    according to attention type.

    Arguments:

    * attn_metadata: Attention metadata structure associated with attention
    * attn_bias: The desired attention bias value
    * attn_type: encoder attention, decoder self-attention,
                 encoder/decoder cross-attention
    """
    if attn_type == AttentionType.DECODER or attn_type == AttentionType.ENCODER_ONLY:
        attn_metadata.attn_bias = attn_bias
    elif attn_type == AttentionType.ENCODER:
        attn_metadata.encoder_attn_bias = attn_bias
    elif attn_type == AttentionType.ENCODER_DECODER:
        attn_metadata.cross_attn_bias = attn_bias
    else:
        raise AttributeError(f'Invalid attention type {str(attn_type)}')

class XFormersMetadataBuilder(CommonMetadataBuilder[XFormersMetadata]):
    _metadata_cls = XFormersMetadata

class XFormersImpl(AttentionImpl[XFormersMetadata]):
    """
    If the input tensors contain prompt tokens, the layout is as follows:
    |<--------------- num_prefill_tokens ----------------->|	
    |<--prefill_0-->|<--prefill_1-->|...|<--prefill_N-1--->|

    Otherwise, the layout is as follows:	
    |<----------------- num_decode_tokens ------------------>|	
    |<--decode_0-->|..........|<--decode_M-1-->|<--padding-->|

    Generation tokens can contain padding when cuda-graph is used.
    Currently, prompt tokens don't contain any padding.

    The prompts might have different lengths, while the generation tokens
    always have length 1.

    If chunked prefill is enabled, prefill tokens and decode tokens can be
    batched together in a flattened 1D query.

    |<----- num_prefill_tokens ---->|<------- num_decode_tokens --------->|
    |<-prefill_0->|...|<-prefill_N-1->|<--decode_0-->|...|<--decode_M-1-->|

    Currently, cuda graph is disabled for chunked prefill, meaning there's no
    padding between prefill and decode tokens.
    """

    def __init__(self, num_heads: int, head_size: int, scale: float, num_kv_heads: int, alibi_slopes: Optional[List[float]], sliding_window: Optional[int], kv_cache_dtype: str, logits_soft_cap: Optional[float]=None, attn_type: str=AttentionType.DECODER, kv_sharing_target_layer_name: Optional[str]=None, use_irope: bool=False) -> None:
        if kv_sharing_target_layer_name is not None:
            raise NotImplementedError('KV sharing is not supported in V0 XFORMERS backend.')
        if logits_soft_cap is not None:
            logger.warning_once('XFormers does not support logits soft cap. Outputs may be slightly off.')
        if use_irope:
            logger.warning_once('Using irope in XFormers is not supported yet, it will fall back to global attention for long context.')
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        self.sliding_window = sliding_window
        self.kv_cache_dtype = kv_cache_dtype
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        supported_head_sizes = PagedAttention.get_supported_head_sizes()
        if head_size not in supported_head_sizes:
            raise ValueError(f'Head size {head_size} is not supported by PagedAttention. Supported head sizes are: {supported_head_sizes}.')
        self.attn_type = attn_type

    def forward(self, layer: AttentionLayer, query: torch.Tensor, key: Optional[torch.Tensor], value: Optional[torch.Tensor], kv_cache: torch.Tensor, attn_metadata: 'XFormersMetadata', output: Optional[torch.Tensor]=None, output_scale: Optional[torch.Tensor]=None) -> torch.Tensor:
        """Forward pass with xFormers and PagedAttention.

        For decoder-only models: query, key and value must be non-None.

        For encoder/decoder models:
        * XFormersImpl.forward() may be invoked for both self- and cross-
          attention layers.
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
                sequence attributes (seq_lens/seq_lens_tensor/max_seq_len).
                Used for encoder branch of encoder-decoder models.
            * ENCODER_ONLY: no kv_caching, uses the normal attention 
                attributes (seq_lens/seq_lens_tensor/max_seq_len).
            * ENCODER_DECODER: cross-attention behavior;
                use cross-attention block table for caching KVs derived
                from encoder hidden states; since KV sequence lengths
                will match encoder sequence lengths, pass encoder sequence
                attributes to kernel (encoder_seq_lens/encoder_seq_lens_tensor/
                max_encoder_seq_len)
    
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
        if output_scale is not None:
            raise NotImplementedError('fused output quantization is not yet supported for XFormersImpl')
        attn_type = self.attn_type
        if attn_type == AttentionType.ENCODER and (not attn_metadata.is_all_encoder_attn_metadata_set):
            raise AttributeError('Encoder attention requires setting encoder metadata attributes.')
        elif attn_type == AttentionType.ENCODER_DECODER and (not attn_metadata.is_all_cross_attn_metadata_set):
            raise AttributeError('Encoder/decoder cross-attention requires setting cross-attention metadata attributes.')
        query = query.view(-1, self.num_heads, self.head_size)
        if key is not None:
            assert value is not None
            key = key.view(-1, self.num_kv_heads, self.head_size)
            value = value.view(-1, self.num_kv_heads, self.head_size)
        else:
            assert value is None
        if attn_type != AttentionType.ENCODER and kv_cache.numel() > 0:
            key_cache, value_cache = PagedAttention.split_kv_cache(kv_cache, self.num_kv_heads, self.head_size)
            if key is not None and value is not None:
                if attn_type == AttentionType.ENCODER_DECODER:
                    updated_slot_mapping = attn_metadata.cross_slot_mapping
                else:
                    updated_slot_mapping = attn_metadata.slot_mapping
                PagedAttention.write_to_paged_cache(key, value, key_cache, value_cache, updated_slot_mapping, self.kv_cache_dtype, layer._k_scale, layer._v_scale)
        num_prefill_query_tokens, num_prefill_kv_tokens, num_decode_query_tokens = get_num_prefill_decode_query_kv_tokens(attn_metadata, attn_type)
        output = torch.empty_like(query)
        decode_query = query[num_prefill_query_tokens:]
        query = query[:num_prefill_query_tokens]
        if key is not None and value is not None:
            key = key[:num_prefill_kv_tokens]
            value = value[:num_prefill_kv_tokens]
        assert query.shape[0] == num_prefill_query_tokens
        assert decode_query.shape[0] == num_decode_query_tokens
        if (prefill_meta := attn_metadata.prefill_metadata):
            if kv_cache.numel() == 0 or prefill_meta.block_tables.numel() == 0:
                out = self._run_memory_efficient_xformers_forward(query, key, value, prefill_meta, attn_type=attn_type)
                assert out.shape == output[:num_prefill_query_tokens].shape
                output[:num_prefill_query_tokens] = out
            else:
                assert attn_type != AttentionType.ENCODER_ONLY, 'Encoder-only models should not have prefix attention.'
                assert prefill_meta.query_start_loc is not None
                assert prefill_meta.max_query_len is not None
                out = PagedAttention.forward_prefix(query, key, value, self.kv_cache_dtype, key_cache, value_cache, prefill_meta.block_tables, prefill_meta.query_start_loc, prefill_meta.seq_lens_tensor, prefill_meta.max_query_len, self.alibi_slopes, self.sliding_window, layer._k_scale, layer._v_scale)
                assert output[:num_prefill_query_tokens].shape == out.shape
                output[:num_prefill_query_tokens] = out
        if (decode_meta := attn_metadata.decode_metadata):
            assert attn_type != AttentionType.ENCODER_ONLY, 'Encoder-only models should not have decode metadata.'
            seq_lens_arg, max_seq_len_arg, block_tables_arg = get_seq_len_block_table_args(decode_meta, False, attn_type)
            output[num_prefill_query_tokens:] = PagedAttention.forward_decode(decode_query, key_cache, value_cache, block_tables_arg, seq_lens_arg, max_seq_len_arg, self.kv_cache_dtype, self.num_kv_heads, self.scale, self.alibi_slopes, layer._k_scale, layer._v_scale)
        return output.view(-1, self.num_heads * self.head_size)

    def _run_memory_efficient_xformers_forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attn_metadata: XFormersMetadata, attn_type: str=AttentionType.DECODER) -> torch.Tensor:
        """Attention for 1D query of multiple prompts. Multiple prompt
        tokens are flattened in to `query` input.

        See https://facebookresearch.github.io/xformers/components/ops.html
        for API spec.

        Args:
            output: shape = [num_prefill_tokens, num_heads, head_size]
            query: shape = [num_prefill_tokens, num_heads, head_size]
            key: shape = [num_prefill_tokens, num_kv_heads, head_size]
            value: shape = [num_prefill_tokens, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
            attn_type: Select attention type, between encoder attention,
                       decoder self-attention, or encoder/decoder cross-
                       attention. Defaults to decoder self-attention,
                       which is the vLLM default generally
        """
        original_query = query
        if self.num_kv_heads != self.num_heads:
            query = query.view(query.shape[0], self.num_kv_heads, self.num_queries_per_kv, query.shape[-1])
            key = key[:, :, None, :].expand(key.shape[0], self.num_kv_heads, self.num_queries_per_kv, key.shape[-1])
            value = value[:, :, None, :].expand(value.shape[0], self.num_kv_heads, self.num_queries_per_kv, value.shape[-1])
        attn_bias = _get_attn_bias(attn_metadata, attn_type)
        if attn_bias is None:
            if self.alibi_slopes is None:
                if attn_type == AttentionType.ENCODER_DECODER:
                    assert attn_metadata.seq_lens is not None
                    assert attn_metadata.encoder_seq_lens is not None
                    attn_bias = BlockDiagonalMask.from_seqlens(attn_metadata.seq_lens, attn_metadata.encoder_seq_lens, device=query.device)
                elif attn_type == AttentionType.ENCODER:
                    assert attn_metadata.encoder_seq_lens is not None
                    attn_bias = BlockDiagonalMask.from_seqlens(attn_metadata.encoder_seq_lens, device=query.device)
                elif attn_type == AttentionType.ENCODER_ONLY:
                    assert attn_metadata.seq_lens is not None
                    attn_bias = BlockDiagonalMask.from_seqlens(attn_metadata.seq_lens, device=query.device)
                elif attn_type == AttentionType.DECODER:
                    assert attn_metadata.seq_lens is not None
                    attn_bias = BlockDiagonalCausalMask.from_seqlens(attn_metadata.seq_lens, device=query.device)
                else:
                    raise ValueError('Unknown AttentionType: %s', attn_type)
                if self.sliding_window is not None:
                    attn_bias = attn_bias.make_local_attention(self.sliding_window)
                attn_bias = [attn_bias]
            else:
                assert attn_type == AttentionType.DECODER
                assert attn_metadata.seq_lens is not None
                attn_bias = _make_alibi_bias(self.alibi_slopes, self.num_kv_heads, query.dtype, attn_metadata.seq_lens)
            _set_attn_bias(attn_metadata, attn_bias, attn_type)
        if self.alibi_slopes is None:
            query = query.unsqueeze(0)
            key = key.unsqueeze(0)
            value = value.unsqueeze(0)
            out = xops.memory_efficient_attention_forward(query, key, value, attn_bias=attn_bias[0], p=0.0, scale=self.scale)
            return out.view_as(original_query)
        assert attn_metadata.seq_lens is not None
        output = torch.empty_like(original_query)
        start = 0
        for i, seq_len in enumerate(attn_metadata.seq_lens):
            end = start + seq_len
            out = xops.memory_efficient_attention_forward(query[None, start:end], key[None, start:end], value[None, start:end], attn_bias=attn_bias[i], p=0.0, scale=self.scale)
            output[start:end].copy_(out.view_as(original_query[start:end]))
            start += seq_len
        return output

def _make_alibi_bias(alibi_slopes: torch.Tensor, num_kv_heads: int, dtype: torch.dtype, seq_lens: List[int]) -> List[AttentionBias]:
    attn_biases: List[AttentionBias] = []
    for seq_len in seq_lens:
        bias = torch.arange(seq_len, dtype=dtype)
        bias = bias[None, :] - bias[:, None]
        padded_len = (seq_len + 7) // 8 * 8
        num_heads = alibi_slopes.shape[0]
        bias = torch.empty(1, num_heads, seq_len, padded_len, device=alibi_slopes.device, dtype=dtype)[:, :, :, :seq_len].copy_(bias)
        bias.mul_(alibi_slopes[:, None, None])
        attn_biases.append(LowerTriangularMaskWithTensorBias(bias))
    return attn_biases