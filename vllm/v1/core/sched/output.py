from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.lora.request import LoRARequest
from vllm.multimodal.inputs import MultiModalKwargs, PlaceholderRange
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.v1.request import Request
import numpy as np
import numpy.typing as npt
if TYPE_CHECKING:

@dataclass
class NewRequestData:
    req_id: str
    prompt_token_ids: list[int]
    mm_inputs: list[MultiModalKwargs]
    mm_hashes: list[str]
    mm_positions: list[PlaceholderRange]
    sampling_params: Optional[SamplingParams]
    pooling_params: Optional[PoolingParams]
    block_ids: tuple[list[int], ...]
    num_computed_tokens: int
    lora_request: Optional[LoRARequest]

    @classmethod
    def from_request(cls, request: Request, block_ids: tuple[list[int], ...]) -> NewRequestData:
        return cls(req_id=request.request_id, prompt_token_ids=request.prompt_token_ids, mm_inputs=request.mm_inputs, mm_hashes=request.mm_hashes, mm_positions=request.mm_positions, sampling_params=request.sampling_params, pooling_params=request.pooling_params, block_ids=block_ids, num_computed_tokens=request.num_computed_tokens, lora_request=request.lora_request)

    def __repr__(self):
        return f'NewRequestData(req_id={self.req_id},prompt_token_ids={self.prompt_token_ids},mm_inputs={self.mm_inputs},mm_hashes={self.mm_hashes},mm_positions={self.mm_positions},sampling_params={self.sampling_params},block_ids={self.block_ids},num_computed_tokens={self.num_computed_tokens},lora_request={self.lora_request})'

    def anon_repr(self):
        return f'NewRequestData(req_id={self.req_id},prompt_token_ids_len={len(self.prompt_token_ids)},mm_inputs={self.mm_inputs},mm_hashes={self.mm_hashes},mm_positions={self.mm_positions},sampling_params={self.sampling_params},block_ids={self.block_ids},num_computed_tokens={self.num_computed_tokens},lora_request={self.lora_request})'

@dataclass
class CachedRequestData:
    req_ids: list[str]
    resumed_from_preemption: list[bool]
    new_token_ids: list[list[int]]
    new_block_ids: list[tuple[list[int], ...]]
    num_computed_tokens: list[int]

    @property
    def num_reqs(self) -> int:
        return len(self.req_ids)

    @classmethod
    def make_empty(cls) -> CachedRequestData:
        return cls(req_ids=[], resumed_from_preemption=[], new_token_ids=[], new_block_ids=[], num_computed_tokens=[])

@dataclass
class SchedulerOutput:
    scheduled_new_reqs: list[NewRequestData]
    scheduled_cached_reqs: CachedRequestData
    num_scheduled_tokens: dict[str, int]
    total_num_scheduled_tokens: int
    scheduled_spec_decode_tokens: dict[str, list[int]]
    scheduled_encoder_inputs: dict[str, list[int]]
    num_common_prefix_blocks: list[int]
    finished_req_ids: set[str]
    free_encoder_input_ids: list[tuple[str, int]]
    structured_output_request_ids: dict[str, int]
    grammar_bitmask: Optional[npt.NDArray[np.int32]]
    kv_connector_metadata: Optional[KVConnectorMetadata] = None