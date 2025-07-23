from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional
from vllm.attention.backends.abstract import AttentionMetadata
from vllm.config import KVTransferConfig, VllmConfig
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole
from vllm.forward_context import ForwardContext
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import Request
import copy
import torch
if TYPE_CHECKING:
logger = init_logger(__name__)

@dataclass
class MultiKVConnectorMetadata(KVConnectorMetadata):
    metadata: tuple[KVConnectorMetadata, ...]
    extra_async_saves: Optional[dict[str, int]] = None

class MultiConnector(KVConnectorBase_V1):
    """
    A wrapper for using multiple KVConnectors at the same time.

    The current logic is:
    - Load KV from the first connector that advertises available tokens from
      get_num_new_matched_tokens(), based on the order in the config.
    - Save to all connectors.
    """

    def __init__(self, vllm_config: 'VllmConfig', role: KVConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)
        self._connectors: list[KVConnectorBase_V1] = []
        ktcs = vllm_config.kv_transfer_config.kv_connector_extra_config.get('connectors')
        assert ktcs is not None
        for ktc in ktcs:
            temp_config = copy.copy(vllm_config)
            engine_id = ktc.get('engine_id', vllm_config.kv_transfer_config.engine_id)
            temp_config.kv_transfer_config = KVTransferConfig(**ktc, engine_id=engine_id)
            self._connectors.append(KVConnectorFactory.create_connector_v1(temp_config, role))
        self._requests_to_connector: dict[str, int] = {}
        self._extra_async_saves: dict[str, int] = {}

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        for c in self._connectors:
            c.register_kv_caches(kv_caches)

    def bind_connector_metadata(self, connector_metadata: KVConnectorMetadata) -> None:
        assert isinstance(connector_metadata, MultiKVConnectorMetadata)
        if connector_metadata.extra_async_saves:
            self._extra_async_saves.update(connector_metadata.extra_async_saves)
        for c, cm in zip(self._connectors, connector_metadata.metadata):
            c.bind_connector_metadata(cm)

    def clear_connector_metadata(self) -> None:
        for c in self._connectors:
            c.clear_connector_metadata()

    def start_load_kv(self, forward_context: 'ForwardContext', **kwargs) -> None:
        for c in self._connectors:
            c.start_load_kv(forward_context, **kwargs)

    def wait_for_layer_load(self, layer_name: str) -> None:
        for c in self._connectors:
            c.wait_for_layer_load(layer_name)

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor, attn_metadata: 'AttentionMetadata', **kwargs) -> None:
        for c in self._connectors:
            c.save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)

    def wait_for_save(self):
        for c in self._connectors:
            c.wait_for_save()

    def get_finished(self, finished_req_ids: set[str]) -> tuple[Optional[set[str]], Optional[set[str]]]:
        finished_sending: set[str] = set()
        finished_recving: set[str] = set()
        for c in self._connectors:
            sending, recving = c.get_finished(finished_req_ids)
            if not recving and (not sending):
                continue
            finished_recving.update(recving or ())
            for req_id in sending or ():
                extra_pending = self._extra_async_saves.get(req_id)
                if extra_pending is None:
                    finished_sending.add(req_id)
                    continue
                assert extra_pending > 0
                if extra_pending == 1:
                    del self._extra_async_saves[req_id]
                else:
                    self._extra_async_saves[req_id] = extra_pending - 1
        return (finished_sending or None, finished_recving or None)

    def get_num_new_matched_tokens(self, request: 'Request', num_computed_tokens: int) -> tuple[int, bool]:
        to_return = (0, False)
        for i, c in enumerate(self._connectors):
            toks, load_async = c.get_num_new_matched_tokens(request, num_computed_tokens)
            if to_return[0] == 0 and toks > 0:
                self._requests_to_connector[request.request_id] = i
                to_return = (toks, load_async)
        return to_return

    def update_state_after_alloc(self, request: 'Request', blocks: 'KVCacheBlocks', num_external_tokens: int):
        chosen_connector = self._requests_to_connector.get(request.request_id, -1)
        empty_blocks = blocks.new_empty()
        for i, c in enumerate(self._connectors):
            if i == chosen_connector:
                c.update_state_after_alloc(request, blocks, num_external_tokens)
            else:
                c.update_state_after_alloc(request, empty_blocks, 0)

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> MultiKVConnectorMetadata:
        metadata = MultiKVConnectorMetadata(metadata=tuple((c.build_connector_meta(scheduler_output) for c in self._connectors)))
        if self._extra_async_saves:
            metadata.extra_async_saves = self._extra_async_saves
            self._extra_async_saves = {}
        return metadata

    def request_finished(self, request: 'Request', blocks: list[int]) -> tuple[bool, Optional[dict[str, Any]]]:
        async_saves = 0
        kv_txfer_params = None
        for c in self._connectors:
            async_save, txfer_params = c.request_finished(request, blocks)
            if async_save:
                async_saves += 1
            if txfer_params is not None:
                if kv_txfer_params is not None:
                    raise RuntimeError('Only one connector can produce KV transfer params')
                kv_txfer_params = txfer_params
        if async_saves > 1:
            self._extra_async_saves[request.request_id] = async_saves - 1
        self._requests_to_connector.pop(request.request_id, None)
        return (async_saves > 0, kv_txfer_params)