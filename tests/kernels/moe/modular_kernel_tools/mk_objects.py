from vllm.model_executor.layers.fused_moe.batched_deep_gemm_moe import BatchedDeepGemmExperts
from vllm.model_executor.layers.fused_moe.batched_triton_or_deep_gemm_moe import BatchedTritonOrDeepGemmExperts
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.cutlass_moe import CutlassExpertsFp8
from vllm.model_executor.layers.fused_moe.deep_gemm_moe import DeepGemmExperts
from vllm.model_executor.layers.fused_moe.deepep_ht_prepare_finalize import DeepEPHTPrepareAndFinalize
from vllm.model_executor.layers.fused_moe.deepep_ll_prepare_finalize import DeepEPLLPrepareAndFinalize
from vllm.model_executor.layers.fused_moe.fused_batched_moe import BatchedTritonExperts, NaiveBatchedExperts
from vllm.model_executor.layers.fused_moe.layer import TritonExperts
from vllm.model_executor.layers.fused_moe.pplx_prepare_finalize import PplxPrepareAndFinalize
from vllm.model_executor.layers.fused_moe.prepare_finalize import MoEPrepareAndFinalizeNoEP
from vllm.model_executor.layers.fused_moe.triton_deep_gemm_moe import TritonOrDeepGemmExperts
from vllm.utils import has_deep_ep, has_pplx
import torch
if has_deep_ep():
if has_pplx():
MK_MULTI_GPU_PREPARE_FINALIZE_TYPES = []
if has_pplx():
    MK_MULTI_GPU_PREPARE_FINALIZE_TYPES += [PplxPrepareAndFinalize]
if has_deep_ep():
    MK_MULTI_GPU_PREPARE_FINALIZE_TYPES += [DeepEPHTPrepareAndFinalize, DeepEPLLPrepareAndFinalize]
MK_SINGLE_GPU_PREPARE_FINALIZE_TYPES = [MoEPrepareAndFinalizeNoEP]
MK_ALL_PREPARE_FINALIZE_TYPES = MK_MULTI_GPU_PREPARE_FINALIZE_TYPES + MK_SINGLE_GPU_PREPARE_FINALIZE_TYPES
MK_FUSED_EXPERT_TYPES = [BatchedDeepGemmExperts, BatchedTritonExperts, NaiveBatchedExperts, BatchedTritonOrDeepGemmExperts, CutlassExpertsFp8, DeepGemmExperts, TritonOrDeepGemmExperts, TritonExperts]
MK_QUANT_CONFIGS = [None, FusedMoEQuantConfig(quant_dtype=torch.float8_e4m3fn, per_out_ch_quant=True, per_act_token_quant=False, block_shape=None), FusedMoEQuantConfig(quant_dtype=torch.float8_e4m3fn, per_out_ch_quant=True, per_act_token_quant=True, block_shape=None), FusedMoEQuantConfig(quant_dtype=torch.float8_e4m3fn, per_out_ch_quant=False, per_act_token_quant=False, block_shape=None), FusedMoEQuantConfig(quant_dtype=torch.float8_e4m3fn, per_out_ch_quant=False, per_act_token_quant=True, block_shape=None), FusedMoEQuantConfig(quant_dtype=torch.float8_e4m3fn, per_out_ch_quant=False, per_act_token_quant=False, block_shape=[128, 128])]