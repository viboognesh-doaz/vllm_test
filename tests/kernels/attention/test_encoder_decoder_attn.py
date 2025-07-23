from tests.kernels.utils import *
from typing import NamedTuple, Optional
from vllm.attention import Attention, AttentionMetadata, AttentionType
from vllm.attention.backends.utils import STR_NOT_IMPL_ENC_DEC_ROCM_HIP
from vllm.attention.selector import _Backend, _cached_get_attn_backend, global_force_attn_backend_context_manager
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.platforms import current_platform
import pytest
import torch
'\nTests:\n\n* E2E test of Encoder attention + Decoder self-attention +\n      Encoder/decoder cross-attention (collectively\n      "encoder/decoder attention")\n\n'

@pytest.fixture(scope='function', autouse=True)
def use_v0_only(monkeypatch):
    """
    Encoder-decoder is only supported on V0, so set 
    VLLM_USE_V1=0 for all tests in the module.
    """
    monkeypatch.setenv('VLLM_USE_V1', '0')
LIST_ENC_DEC_SUPPORTED_BACKENDS = [_Backend.XFORMERS, _Backend.FLASH_ATTN]
HEAD_SIZES = [64, 256]
NUM_HEADS = [1, 16]
BATCH_SIZES = [1, 16]
BLOCK_SIZES = [16]
CUDA_DEVICE = 'cuda:0'
MAX_DEC_SEQ_LENS = [128]
MAX_ENC_SEQ_LENS = [128]
HEAD_SIZES_FOR_UNSUPP = [HEAD_SIZES[0]]

class TestPoint(NamedTuple):
    """
    Encapsulates the attributes which define a single invocation
    of the test_e2e_enc_dec_attn() test

    Attributes:
        num_heads: The number of heads in the model.
        head_size: Head dimension
        backend_name: Name of the backend framework used.
        batch_size: Number of samples per batch.
        block_size: Size of each block of data processed.
        max_dec_seq_len: Maximum sequence length for the decoder.
        max_enc_seq_len: Maximum sequence length for the encoder.
        num_blocks: Number of blocks in the model.
    """
    num_heads: int
    head_size: int
    backend_name: str
    batch_size: int
    block_size: int
    max_dec_seq_len: int
    max_enc_seq_len: int
    num_blocks: int
    attn_type: AttentionType

class TestResources(NamedTuple):
    """
    Encapsulates key components for performing an
    encoder/decoder attention test

    Note that
    (1) attn automatically selects an attention backend
        based on platform info & a set of canned
        heuristics
    (2) attn_backend is thus *not the same backend
        instance* used by attn, but rather it is
        intended to be a
        *different instance* of the *same backend class*;
        it is assumed that the user of TestResources
        will leverage attn_backend for the purpose of
        constructing backend-compatible attention
        metadata instances

    Attributes:

    * scale: 1/sqrt(d) scale factor for attn
    * attn_backend: implementations of abstraction
                    attention interface using
                    a particular kernel library
                    i.e. XFormers
    * attn: Attention layer instance
    * kv_cache: shared key/value cache for all attention
    """
    scale: float
    attn: Attention
    kv_cache: torch.Tensor

def _make_test_resources(test_pt: TestPoint) -> TestResources:
    """
    Build key components for performing encoder/decoder attention test.

    Note that
    (1) The Attention instance constructed here, automatically selects
        an attention backend class based on platform info & a set of canned
        heuristics, so
    (2) The attention backend instance constructed here is thus *not
        the same backend instance* used by attn, but rather it is
        intended to be a *different instance* of the *same backend class*;
        therefore,
    (3) This function requires that test_pt.backend_name matches the backend
        class that Attention will automatically select when it is constructed.


    Arguments:

    * test_pt: TestPoint data structure; this function relies on the
               following fields: num_heads, head_size, num_blocks,
               block_size, backend_name

    Returns:

    * TestResources data structure.
    """
    scale = float(1.0 / test_pt.head_size ** 0.5)
    attn = Attention(test_pt.num_heads, test_pt.head_size, scale=scale, prefix=f'{test_pt.attn_type}', attn_type=test_pt.attn_type)
    if test_pt.num_blocks is None or test_pt.num_heads is None:
        return TestResources(scale, attn, torch.tensor([], dtype=torch.float32, device=CUDA_DEVICE))
    if test_pt.attn_type in (AttentionType.DECODER, AttentionType.ENCODER_DECODER):
        kv_cache = make_kv_cache(test_pt.num_blocks, test_pt.num_heads, test_pt.head_size, test_pt.block_size, device=CUDA_DEVICE, backend=test_pt.backend_name)
    else:
        kv_cache = torch.tensor([])
    attn.kv_cache = [kv_cache]
    return TestResources(scale, attn, kv_cache)

def _encoder_attn_setup(test_pt: TestPoint, test_rsrcs: TestResources) -> PhaseTestParameters:
    """
    Set up test vectors & data structures for encoder attention test.

    A triplet of synthetic query/key/value tensors are constructed.
    Given this is an encoder attention test, the key & value
    sequences will have the same length as the corresponding queries.

    The query/key/value tensors are passed to an ideal reference
    self-attention implementation to generate an ideal output tensor.

    Encoder inference does not populate the KV cache, therefore
    no KV cache memory mapping is constructed

    Arguments:

    * test_pt: TestPoint data structure; this function relies on the
               following fields: batch_size, num_heads, head_size,
               block_size, max_q_seq_len
    * test_rsrcs: TestResources data structure; this function relies on the
                  scale field


    Returns:

    * PhaseTestParameters data structure comprising (1) packed query/key/value
      tensors, (2) the ideal output of attention computed using a naive
      implementation, and (3) KVCache field set to None
    """
    num_heads, head_size, _, batch_size, _, _, max_q_seq_len, _, _ = test_pt
    scale = test_rsrcs.scale
    max_kv_seq_len = max_q_seq_len
    qkv_in, _, _ = make_qkv(batch_size, max_q_seq_len, max_kv_seq_len, num_heads, head_size, attn_type=AttentionType.ENCODER, device=CUDA_DEVICE)
    ideal_output = ref_masked_attention(qkv_in.query, qkv_in.key, qkv_in.value, scale=scale, q_seq_lens=qkv_in.q_seq_lens, kv_seq_lens=qkv_in.kv_seq_lens)
    packed_ideal_output, _ = pack_tensor(ideal_output, qkv_in.q_seq_lens, device=CUDA_DEVICE)
    packed_qkv = pack_qkv(qkv_in, device=CUDA_DEVICE)
    return PhaseTestParameters(PackedQKVO(packed_qkv, packed_ideal_output), None)

def _decoder_attn_setup(test_pt: TestPoint, test_rsrcs: TestResources, block_base_addr: int=0) -> tuple[QKVInputs, PhaseTestParameters, PhaseTestParameters, int]:
    """
    Set up test vectors & data structures for self-attention test.

    A triplet of synthetic query/key/value tensors are constructed ("baseline"
    query/key/value). Given this is a self-attention test, the key & value
    sequences will have the same length as the corresponding queries.

    "Prefill" query/key/value tensors are derived by masking out the last value
    in each baseline query/key/value. These tensors are used to test prefill &
    populate KV cache for a subsequent decode test.

    "Decode" query/key/value tensors are derived by extracting *only* the last
    value from each baseline query/key/value (i.e. complement of the prefill
    tensors.) These tensors are used to test decode, conditional on the kv cache
    being populated during the prefill test.

    The baseline query/key/value tensors are passed to an ideal reference
    self-attention implementation to generate a "Baseline" ideal output tensor.
    This tensor is split into the "Prefill" ideal output tensor (all but the
    last element of each output sequence) and the "Decode" ideal output tensor
    (*only* the last element of each output sequence); the "Prefill" and
    "Decode" ideal output tensors can be used to validate the prefill and decode
    test results, respectively.

    This function also constructs the self-attention KV cache memory mapping
    (slot mapping and block table), ensuring that the block table starts at
    block_base_addr

    Arguments:

    * test_pt: TestPoint data structure; this function relies on the
               following fields: batch_size, num_heads, head_size,
               block_size, max_q_seq_len
    * test_rsrcs: TestResources data structure; this function relies on the
                  scale field
    * block_base_addr: decoder self-attention block-table base address

    Returns:
    * qkv: Unpacked (batch_size x padded_seq_len x num_heads x
           head_size) query/key/value tensors
    * Prefill-phase decoder self-attention PhaseTestParameters data structure,
      including (1) packed (number_of_tokens x num_heads x head_size)
      query/key/value tensors along with (2) ideal attention output
      computed using a naive implementation, and (3) memory-mapping data
      structures appropriate for prefill phase.
    * Decode-phase decoder self-attention PhaseTestParameters data structure,
      including (1) packed (number_of_tokens x num_heads x head_size)
      query/key/value tensors along with (2) ideal attention output
      computed using a naive implementation, and (3) memory-mapping data
      structures appropriate for decode phase.
    * max_block_idx: max physical address in decoder self-attention block-table
                     (intended to be used as the base address for the encoder/
                      decoder cross-attention block-table, which is not
                      constructed in this function)
    """
    num_heads, head_size, _, batch_size, block_size, max_q_seq_len, _, _, _ = test_pt
    scale = test_rsrcs.scale
    max_kv_seq_len = max_q_seq_len
    qkv, prefill_qkv, decode_qkv = make_qkv(batch_size, max_q_seq_len, max_kv_seq_len, num_heads, head_size, attn_type=AttentionType.DECODER, device=CUDA_DEVICE)
    causal_mask = make_causal_mask(max_q_seq_len, max_kv_seq_len).to(CUDA_DEVICE)
    ideal_output = ref_masked_attention(qkv.query, qkv.key, qkv.value, scale=scale, custom_mask=causal_mask, q_seq_lens=qkv.q_seq_lens, kv_seq_lens=qkv.kv_seq_lens)
    prefill_ideal_output = torch.zeros_like(ideal_output)
    decode_ideal_output = torch.zeros_like(ideal_output[:, 0:1])
    for bdx, prefill_q_seq_len in enumerate(prefill_qkv.q_seq_lens):
        prefill_ideal_output[bdx, :prefill_q_seq_len] = ideal_output[bdx, :prefill_q_seq_len]
        decode_ideal_output[bdx, :] = ideal_output[bdx, prefill_q_seq_len:prefill_q_seq_len + 1]
    prefill_packed_ideal_output, _ = pack_tensor(prefill_ideal_output, prefill_qkv.q_seq_lens, device=CUDA_DEVICE)
    decode_packed_ideal_output, _ = pack_tensor(decode_ideal_output, [1 for _ in range(batch_size)], device=CUDA_DEVICE)
    prefill_block_tables = make_empty_block_tables_tensor(device=CUDA_DEVICE)
    decode_block_tables, slot_mapping_list, max_block_idx = make_block_tables_slot_mapping(block_size, qkv.q_seq_lens, device=CUDA_DEVICE, block_base_addr=block_base_addr)
    prefill_slot_mapping, decode_slot_mapping = split_slot_mapping(slot_mapping_list, qkv.q_seq_lens, device=CUDA_DEVICE)
    prefill_pckd_qkv = pack_qkv(prefill_qkv, device=CUDA_DEVICE)
    decode_pckd_qkv = pack_qkv(decode_qkv, device=CUDA_DEVICE)
    return (qkv, PhaseTestParameters(PackedQKVO(prefill_pckd_qkv, prefill_packed_ideal_output), KVMemoryMap(prefill_block_tables, prefill_slot_mapping)), PhaseTestParameters(PackedQKVO(decode_pckd_qkv, decode_packed_ideal_output), KVMemoryMap(decode_block_tables, decode_slot_mapping)), max_block_idx)

def _enc_dec_cross_attn_setup_reuses_query(decoder_qkv: QKVInputs, encoder_test_params: PhaseTestParameters, prefill_decoder_phase_test_params: PhaseTestParameters, test_pt: TestPoint, test_rsrcs: TestResources, block_base_addr: int=0) -> tuple[PhaseTestParameters, PhaseTestParameters]:
    """
    Set up test vectors & data structures for cross-attention test.

    A triplet of synthetic cross-attention key/value tensors are constructed
    ("baseline" key/value). Given this is a cross-attention test, we assume
    query tensors were already synthesized for a prior self-attention test and
    will be reused for cross-attention. The key & value sequences generated here
    may have a different length than the corresponding queries (as is often
    the case for cross-attention between decoder and encoder sequences.)

    Cross attention key & value tensors do not grow during autoregressive
    inference; thus this function obtains a single key/value pair suitable for
    both prefill and decode.

    The "baseline" query tensor is received as an argument. The "baseline"
    query/key/value tensors are passed to an ideal reference cross-attention
    implementation to generate a "baseline" ideal output tensor. This tensor is
    split into the "Prefill" ideal output tensor (all but the last element of
    each output sequence) and the "Decode" ideal output tensor (*only* the last
    element of each output sequence); the "Prefill" and "Decode" ideal output
    tensors can be used to validate the prefill and decode test results,
    respectively.

    This function also constructs the cross-attention KV cache memory mapping
    (slot mapping and block table), ensuring that the block table starts at
    block_base_addr.

    Arguments:

    * decoder_qkv: pre-existing unpacked (batch_size x padded_seq_len x
                   num_heads x head_size) decoder self-attention inputs;
                   this function relies on the query and q_seq_lens
                   fields
    * encoder_test_params: PhaseTestParameters data structure which was
                           used for encoder inference; KV cache field
                           is not used by this function
    * prefill_decoder_phase_test_params: PhaseTestParameters data structure
                                         used for prefill-phase decoder
                                         self-attention; all fields
                                         including KV cache required
    * test_pt: TestPoint data structure; this function relies on the
               following fields: batch_size, num_heads, head_size,
               block_size, max_q_seq_len
    * test_rsrcs: TestResources data structure; this function relies on the
                  scale field
    * block_base_addr: decoder self-attention block-table base address

    Returns:

    * Prefill-phase encoder/decoder cross-attention PhaseTestParameters data
      structure, including (1) packed
      (number_of_tokens x num_heads x head_size) query/key/value tensors
      along with (2) ideal attention output computed using a
      naive implementation, and (3) memory-mapping data structures appropriate
      for prefill phase.
    * Decode-phase encoder/decoder cross-attention PhaseTestParameters data
      structure, including (1) packed
      (number_of_tokens x num_heads x head_size) query/key/value tensors
      along with (2) ideal attention output computed using a
      naive implementation, and (3) memory-mapping data structures appropriate
      for decode phase.
    """
    assert encoder_test_params.packed_qkvo.packed_qkv is not None
    assert prefill_decoder_phase_test_params.packed_qkvo.packed_qkv is not None
    num_heads, head_size, _, batch_size, block_size, max_decoder_seq_len, max_encoder_seq_len, _, _ = test_pt
    scale = test_rsrcs.scale
    decoder_query = decoder_qkv.query
    decoder_seq_lens = decoder_qkv.q_seq_lens
    encoder_seq_lens = encoder_test_params.packed_qkvo.packed_qkv.q_seq_lens
    prefill_q_seq_lens = prefill_decoder_phase_test_params.packed_qkvo.packed_qkv.q_seq_lens
    assert prefill_q_seq_lens is not None
    cross_kv, _, _ = make_qkv(batch_size, max_decoder_seq_len, max_encoder_seq_len, num_heads, head_size, force_kv_seq_lens=encoder_seq_lens, attn_type=AttentionType.ENCODER_DECODER, device=CUDA_DEVICE)
    ideal_output = ref_masked_attention(decoder_query, cross_kv.key, cross_kv.value, scale=scale, q_seq_lens=decoder_seq_lens, kv_seq_lens=cross_kv.kv_seq_lens)
    prefill_ideal_output = torch.zeros_like(ideal_output)
    decode_ideal_output = torch.zeros_like(ideal_output[:, 0:1])
    for bdx, prefill_q_seq_len in enumerate(prefill_q_seq_lens):
        prefill_ideal_output[bdx, :prefill_q_seq_len] = ideal_output[bdx, :prefill_q_seq_len]
        decode_ideal_output[bdx, :] = ideal_output[bdx, prefill_q_seq_len:prefill_q_seq_len + 1]
    prefill_packed_ideal_output, _ = pack_tensor(prefill_ideal_output, prefill_q_seq_lens, device=CUDA_DEVICE)
    decode_packed_ideal_output, _ = pack_tensor(decode_ideal_output, [1 for _ in range(batch_size)], device=CUDA_DEVICE)
    prefill_block_tables = make_empty_block_tables_tensor(device=CUDA_DEVICE)
    decode_slot_mapping = make_empty_slot_mapping_tensor(device=CUDA_DEVICE)
    decode_block_tables, prefill_slot_mapping_list, _ = make_block_tables_slot_mapping(block_size, cross_kv.kv_seq_lens, block_base_addr=block_base_addr, device=CUDA_DEVICE)
    prefill_slot_mapping = maybe_make_long_tensor(prefill_slot_mapping_list, device=CUDA_DEVICE)
    packed_cross_kv = pack_qkv(cross_kv, device=CUDA_DEVICE)
    return (PhaseTestParameters(PackedQKVO(packed_cross_kv, prefill_packed_ideal_output), KVMemoryMap(prefill_block_tables, prefill_slot_mapping)), PhaseTestParameters(PackedQKVO(None, decode_packed_ideal_output), KVMemoryMap(decode_block_tables, decode_slot_mapping)))

def _run_encoder_attention_test(attn: Attention, encoder_test_params: PhaseTestParameters, attn_metadata: AttentionMetadata, test_pt: TestPoint, vllm_config: VllmConfig) -> torch.Tensor:
    """
    Run encoder attention.

    attn.forward() is passed attn_type=AttentionType.ENCODER in order
    to configure the kernel invocation for encoder attention

    Requires attn_metadata.num_decode_tokens == 0
    (There is no encoder execution in the decode-phase)

    Arguments:

    * attn: Attention wrapper instance
    * encoder_test_params: encoder PhaseTestParameters data structure;
                           this function relies on the packed
                           (number_of_tokens x num_heads x head_size)
                           query/key/value fields
    * attn_metadata: attention metadata for encoder/decoder-self attention
    * test_pt: The TestPoint object containing test details like number of
               model heads, head size, name of the backend being used etc.

    Returns:
    * Attention.forward() applied to packed {query,key,value} and
      & attn_metadata
    """
    assert attn_metadata.num_decode_tokens == 0
    packed_qkv = encoder_test_params.packed_qkvo.packed_qkv
    assert packed_qkv is not None
    with set_forward_context(attn_metadata, vllm_config):
        reshaped_query = packed_qkv.query.view(-1, test_pt.num_heads * test_pt.head_size)
        return attn.forward(reshaped_query, packed_qkv.key, packed_qkv.value)

def _run_decoder_self_attention_test(test_rsrcs: TestResources, decoder_test_params: PhaseTestParameters, attn_metadata: AttentionMetadata, test_pt: TestPoint, vllm_config: VllmConfig) -> torch.Tensor:
    """
    Run decoder self-attention test.

    attn.forward() is passed attn_type=AttentionType.DECODER
    in order to configure the kernel invocation for decoder self-attention.

    Arguments:

    * test_rsrcs: TestResources instance; this function relies on the kv_cache
                  and attn (Attention wrapper instance) fields
    * decoder_test_params: decoder PhaseTestParameters data structure;
                           this function relies on the packed
                           (number_of_tokens x num_heads x head_size)
                           query/key/value fields
    * attn_metadata: attention metadata for decoder-self attention
                     (contains KV cache memory-mapping)
    * test_pt: The TestPoint object containing test details like number of
               model heads, head size, name of the backend being used etc.

    Returns:
    * Attention.forward() applied to packed_{query,key,value}, kv_cache
      & attn_metadata
    """
    attn = test_rsrcs.attn
    packed_qkv = decoder_test_params.packed_qkvo.packed_qkv
    assert packed_qkv is not None
    with set_forward_context(attn_metadata, vllm_config):
        reshaped_query = packed_qkv.query.view(-1, test_pt.num_heads * test_pt.head_size)
        return attn.forward(reshaped_query, packed_qkv.key, packed_qkv.value)

def _run_encoder_decoder_cross_attention_test(test_rsrcs: TestResources, decoder_test_params: PhaseTestParameters, cross_test_params: Optional[PhaseTestParameters], attn_metadata: AttentionMetadata, test_pt: TestPoint, vllm_config: VllmConfig) -> torch.Tensor:
    """
    Run encoder/decoder cross-attention test.

    Via PhaseTestParameters data structures, consumes the same query utilized
    for decoder self-attention, plus a key/value specific to cross-attention.

    if cross_test_params is None or cross_test_params.packed_qkvo.packed_qkv
    is None, this reflects that in decode-phase cross attention there
    is no growth in the key and value tensors.

    attn.forward() is passed attn_type=AttentionType.ENCODER_DECODER
    in order to configure the kernel invocation for encoder/decoder cross-
    attention.

    Arguments:

    * test_rsrcs: TestResources instance; this function relies on the kv_cache
                  and attn (Attention wrapper instance) fields
    * decoder_test_params: decoder PhaseTestParameters data structure;
                           this function relies on the packed
                           (number_of_tokens x num_heads x head_size)
                           query field
    * cross_test_params: encoder/decoder PhaseTestParameters data structure;
                         this function relies on the packed
                         (number_of_tokens x num_heads x head_size)
                         key/value fields
    * attn_metadata: attention metadata for encoder/decoder-self attention
    * test_pt: The TestPoint object containing test details like number of
               model heads, head size, name of the backend being used etc.

    Returns:
    * Attention.forward() applied to packed_{query,key,value}, kv_cache
      & attn_metadata
    """
    assert decoder_test_params.packed_qkvo.packed_qkv is not None
    attn = test_rsrcs.attn
    if cross_test_params is None:
        key = None
        value = None
    else:
        cross_pckd_qkv = cross_test_params.packed_qkvo.packed_qkv
        key = None if cross_pckd_qkv is None else cross_pckd_qkv.key
        value = None if cross_pckd_qkv is None else cross_pckd_qkv.value
    with set_forward_context(attn_metadata, vllm_config):
        reshaped_query = decoder_test_params.packed_qkvo.packed_qkv.query.view(-1, test_pt.num_heads * test_pt.head_size)
        return attn.forward(reshaped_query, key, value)

@pytest.fixture(autouse=True)
def set_reset_environment(attn_backend):
    default_dtype = torch.get_default_dtype()
    if attn_backend.name == 'FLASH_ATTN':
        torch.set_default_dtype(torch.bfloat16)
    _cached_get_attn_backend.cache_clear()
    yield
    torch.set_default_dtype(default_dtype)

@pytest.mark.skipif(current_platform.is_rocm(), reason=STR_NOT_IMPL_ENC_DEC_ROCM_HIP)
@pytest.mark.parametrize('num_heads', NUM_HEADS)
@pytest.mark.parametrize('head_size', HEAD_SIZES)
@pytest.mark.parametrize('attn_backend', LIST_ENC_DEC_SUPPORTED_BACKENDS)
@pytest.mark.parametrize('batch_size', BATCH_SIZES)
@pytest.mark.parametrize('block_size', BLOCK_SIZES)
@pytest.mark.parametrize('max_dec_seq_len', MAX_DEC_SEQ_LENS)
@pytest.mark.parametrize('max_enc_seq_len', MAX_ENC_SEQ_LENS)
def test_encoder_only(num_heads: int, head_size: int, attn_backend: _Backend, batch_size: int, block_size: int, max_dec_seq_len: int, max_enc_seq_len: int):
    """
    End-to-end encoder-only attention test:

    * Construct fake test vectors for (1) encoder attention
    * Construct (1) attention metadata structure with prefill-phase
      encoder attention, and (2) an analogous attention metadata
      structure but for decode-phase
    * Test & validate encoder attention against ideal output

    No KV cache is required for encoder-only attention.

    Note on ROCm/HIP: currently encoder/decoder models are not supported on
    AMD GPUs, therefore this test simply is skipped if
    current_platform.is_rocm().

    This test globally forces an override of the usual backend
    auto-selection process, forcing the specific backend-under-test
    to be utilized.

    Arguments:

    * num_heads
    * head_size,
    * attn_backend: The attention backend to employ for testing
    * batch_size
    * block_size: KV cache block size
    * max_dec_seq_len: max length of decoder input sequences
    * max_enc_seq_len: max length of encoder input sequences
    """
    with global_force_attn_backend_context_manager(attn_backend):
        test_pt = TestPoint(num_heads, head_size, attn_backend.name, batch_size, block_size, max_dec_seq_len, max_enc_seq_len, 4096, AttentionType.ENCODER)
        vllm_config = VllmConfig()
        with set_current_vllm_config(vllm_config):
            test_rsrcs = _make_test_resources(test_pt)
        enc_test_params = _encoder_attn_setup(test_pt, test_rsrcs)
        prephase_attn_metadata: AttentionMetadata = make_test_metadata(attn_backend, True, None, decoder_test_params=None, encoder_test_params=enc_test_params, cross_test_params=None, device=CUDA_DEVICE)
        enc_pckd_act_out: torch.Tensor = _run_encoder_attention_test(test_rsrcs.attn, enc_test_params, prephase_attn_metadata, test_pt=test_pt, vllm_config=vllm_config)
        assert_actual_matches_ideal(enc_test_params, enc_pckd_act_out, attn_backend.name)

@pytest.mark.skipif(current_platform.is_rocm(), reason=STR_NOT_IMPL_ENC_DEC_ROCM_HIP)
@pytest.mark.parametrize('num_heads', NUM_HEADS)
@pytest.mark.parametrize('head_size', HEAD_SIZES)
@pytest.mark.parametrize('attn_backend', LIST_ENC_DEC_SUPPORTED_BACKENDS)
@pytest.mark.parametrize('batch_size', BATCH_SIZES)
@pytest.mark.parametrize('block_size', BLOCK_SIZES)
@pytest.mark.parametrize('max_dec_seq_len', MAX_DEC_SEQ_LENS)
@pytest.mark.parametrize('max_enc_seq_len', MAX_ENC_SEQ_LENS)
def test_e2e_enc_dec_attn(num_heads: int, head_size: int, attn_backend: _Backend, batch_size: int, block_size: int, max_dec_seq_len: int, max_enc_seq_len: int) -> None:
    """
    End-to-end encoder/decoder test:

    * Construct fake test vectors for (1) encoder attention,
      (2) decoder self-attention, and (3) encoder/decoder cross-attention
    * Construct (1) attention metadata structure with self- and cross-attention
      attributes for prefill-phase, and (2) an analogous attention metadata
      structure but for decode-phase
    * Test attention steps in the following order

        * Encoder attention
        * Prefill self-attention
        * Prefill cross-attention
        * Decode self-attention
        * Decode cross-attention
        * Besides being reflective of realistic use-cases, this order would
          exacerbate any accidental overlap in the self-/cross-attention
          block tables, which one hopes to avoid


    * Validate output correctness against ideal reference attention
      implementation

    Block tables are constructed such that cross-attention KV cache is in a
    higher, non-intersecting address-space than self-attention KV cache.

    Self- and cross-attention share the same query tensor but not the K/V
    tensors. Self-attention K/Vs must have the same seq len as Q while
    cross-attention K/Vs are allowed to differ in seq len, as is often the case
    for cross-attention.

    This test globally forces an override of the usual backend
    auto-selection process, forcing the specific backend-under-test
    to be utilized.

    Note on ROCm/HIP: currently encoder/decoder models are not supported on
    AMD GPUs, therefore this test simply is skipped if
    current_platform.is_rocm().

    Note on metadata: there is a single attention metadata structure shared by
    all prefill-phase attention operations (encoder, decoder, enc/dec cross),
    and a single one shared by all decode-phase attention operations
    (decoder & enc/dec cross.) This is intended to reflect the behavior
    of EncoderDecoderModelRunner, which constructs a single attention metadata
    structure for each prefill or decode run. A realistic scenario would rely
    on the attention backend to utilize the appropriate attention metadata
    fields according to the value of attn_metadata.attention_type. Thus,
    this test is organized so as to confirm that the backend-under-test can
    handle a shared prefill attention metadata structure & a shared decode    attention metadata structure.

    Arguments:

    * num_heads
    * head_size,
    * attn_backend: The attention backend to employ for testing
    * batch_size
    * block_size: KV cache block size
    * max_dec_seq_len: max length of decoder input sequences
    * max_enc_seq_len: max length of encoder input sequences
    """
    with global_force_attn_backend_context_manager(attn_backend):
        enc_test_pt = TestPoint(num_heads, head_size, attn_backend.name, batch_size, block_size, max_dec_seq_len, max_enc_seq_len, 4096, AttentionType.ENCODER)
        enc_dec_test_pt = TestPoint(num_heads, head_size, attn_backend.name, batch_size, block_size, max_dec_seq_len, max_enc_seq_len, 4096, AttentionType.ENCODER_DECODER)
        dec_test_pt = TestPoint(num_heads, head_size, attn_backend.name, batch_size, block_size, max_dec_seq_len, max_enc_seq_len, 4096, AttentionType.DECODER)
        vllm_config = VllmConfig()
        with set_current_vllm_config(vllm_config):
            enc_test_rsrcs = _make_test_resources(enc_test_pt)
            enc_dec_test_rsrcs = _make_test_resources(enc_dec_test_pt)
            dec_test_rsrcs = _make_test_resources(dec_test_pt)
        enc_test_params = _encoder_attn_setup(enc_test_pt, enc_test_rsrcs)
        dec_qkv, prephase_dec_test_params, decphase_dec_test_params, cross_block_base_addr = _decoder_attn_setup(dec_test_pt, dec_test_rsrcs)
        prephase_cross_test_params, decphase_cross_test_params = _enc_dec_cross_attn_setup_reuses_query(dec_qkv, enc_test_params, prephase_dec_test_params, enc_dec_test_pt, enc_dec_test_rsrcs, block_base_addr=cross_block_base_addr)
        assert prephase_dec_test_params.packed_qkvo.packed_qkv is not None
        prephase_attn_metadata: AttentionMetadata = make_test_metadata(attn_backend, True, prephase_dec_test_params.packed_qkvo.packed_qkv.q_seq_lens, decoder_test_params=prephase_dec_test_params, encoder_test_params=enc_test_params, cross_test_params=prephase_cross_test_params, device=CUDA_DEVICE)
        enc_pckd_act_out = _run_encoder_attention_test(enc_test_rsrcs.attn, enc_test_params, prephase_attn_metadata, test_pt=enc_test_pt, vllm_config=vllm_config)
        assert_actual_matches_ideal(enc_test_params, enc_pckd_act_out, attn_backend.name)
        prephase_dec_pckd_act_out = _run_decoder_self_attention_test(dec_test_rsrcs, prephase_dec_test_params, prephase_attn_metadata, test_pt=dec_test_pt, vllm_config=vllm_config)
        assert_actual_matches_ideal(prephase_dec_test_params, prephase_dec_pckd_act_out, attn_backend.name)
        prephase_cross_pckd_act_out = _run_encoder_decoder_cross_attention_test(enc_dec_test_rsrcs, prephase_dec_test_params, prephase_cross_test_params, prephase_attn_metadata, test_pt=enc_dec_test_pt, vllm_config=vllm_config)
        assert_actual_matches_ideal(prephase_cross_test_params, prephase_cross_pckd_act_out, attn_backend.name)
        decphase_attn_metadata: AttentionMetadata = make_test_metadata(attn_backend, False, dec_qkv.q_seq_lens, decoder_test_params=decphase_dec_test_params, encoder_test_params=enc_test_params, cross_test_params=decphase_cross_test_params, device=CUDA_DEVICE)
        decphase_dec_pckd_act_out = _run_decoder_self_attention_test(dec_test_rsrcs, decphase_dec_test_params, decphase_attn_metadata, test_pt=dec_test_pt, vllm_config=vllm_config)
        assert_actual_matches_ideal(decphase_dec_test_params, decphase_dec_pckd_act_out, attn_backend.name)
        decphase_cross_pckd_act_out = _run_encoder_decoder_cross_attention_test(enc_dec_test_rsrcs, decphase_dec_test_params, None, decphase_attn_metadata, test_pt=enc_dec_test_pt, vllm_config=vllm_config)
        assert_actual_matches_ideal(decphase_cross_test_params, decphase_cross_pckd_act_out, attn_backend.name)