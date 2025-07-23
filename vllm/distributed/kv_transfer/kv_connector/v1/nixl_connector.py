from collections import defaultdict
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from nixl._api import nixl_agent as NixlWrapper
from transformers import Llama4TextConfig
from typing import TYPE_CHECKING, Any, Optional
from vllm import envs
from vllm.attention.backends.abstract import AttentionMetadata
from vllm.attention.selector import backend_name_to_enum, get_attn_backend
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole
from vllm.distributed.parallel_state import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size, get_tp_group
from vllm.distributed.utils import divide
from vllm.forward_context import ForwardContext
from vllm.logger import init_logger
from vllm.platforms import _Backend
from vllm.utils import make_zmq_path, make_zmq_socket, round_down
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import Request
from vllm.v1.request import RequestStatus
import contextlib
import math
import msgspec
import queue
import threading
import time
import torch
import uuid
import zmq
if TYPE_CHECKING:
Transfer = tuple[int, float]
EngineId = str
ReqId = str
GET_META_MSG = b'get_meta_msg'
logger = init_logger(__name__)
try:
    logger.info('NIXL is available')
except ImportError:
    logger.warning('NIXL is not available')
    NixlWrapper = None

class NixlAgentMetadata(msgspec.Struct, omit_defaults=True, dict=True):
    engine_id: str
    agent_metadata: bytes
    kv_caches_base_addr: list[int]
    num_blocks: int
    block_len: int
    attn_backend_name: str

@dataclass
class ReqMeta:
    local_block_ids: list[int]
    remote_block_ids: list[int]
    remote_host: str
    remote_port: int
    remote_engine_id: str
    tp_size: int

class NixlConnectorMetadata(KVConnectorMetadata):

    def __init__(self):
        self.reqs_to_recv: dict[ReqId, ReqMeta] = {}
        self.reqs_to_send: dict[ReqId, float] = {}

    def add_new_req(self, request_id: ReqId, local_block_ids: list[int], kv_transfer_params: dict[str, Any]):
        self.reqs_to_recv[request_id] = ReqMeta(local_block_ids=local_block_ids, remote_block_ids=kv_transfer_params['remote_block_ids'], remote_engine_id=kv_transfer_params['remote_engine_id'], remote_host=kv_transfer_params['remote_host'], remote_port=kv_transfer_params['remote_port'], tp_size=kv_transfer_params.get('tp_size', 1))

class NixlConnector(KVConnectorBase_V1):

    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole):
        assert vllm_config.kv_transfer_config is not None
        assert vllm_config.kv_transfer_config.engine_id is not None
        self.engine_id: EngineId = vllm_config.kv_transfer_config.engine_id
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler: Optional[NixlConnectorScheduler] = NixlConnectorScheduler(vllm_config, self.engine_id)
            self.connector_worker: Optional[NixlConnectorWorker] = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = NixlConnectorWorker(vllm_config, self.engine_id)

    def get_num_new_matched_tokens(self, request: 'Request', num_computed_tokens: int) -> tuple[int, bool]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_after_alloc(self, request: 'Request', blocks: 'KVCacheBlocks', num_external_tokens: int):
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(request, blocks, num_external_tokens)

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    def request_finished(self, request: 'Request', block_ids: list[int]) -> tuple[bool, Optional[dict[str, Any]]]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, block_ids)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(kv_caches)

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        """Get the finished recving and sending requests."""
        assert self.connector_worker is not None
        return self.connector_worker.get_finished()

    def start_load_kv(self, forward_context: 'ForwardContext', **kwargs) -> None:
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, NixlConnectorMetadata)
        self.connector_worker.start_load_kv(self._connector_metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """NixlConnector does not do layerwise saving."""
        pass

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor, attn_metadata: 'AttentionMetadata', **kwargs) -> None:
        """NixlConnector does not save explicitly."""
        pass

    def wait_for_save(self):
        """NixlConnector does not save explicitly."""
        pass

class NixlConnectorScheduler:
    """Implementation of Scheduler side methods"""

    def __init__(self, vllm_config: VllmConfig, engine_id: str):
        self.vllm_config = vllm_config
        self.block_size = vllm_config.cache_config.block_size
        self.engine_id: EngineId = engine_id
        self.side_channel_host = envs.VLLM_NIXL_SIDE_CHANNEL_HOST
        self.side_channel_port = envs.VLLM_NIXL_SIDE_CHANNEL_PORT + vllm_config.parallel_config.data_parallel_rank * vllm_config.parallel_config.tensor_parallel_size
        logger.info('Initializing NIXL Scheduler %s', engine_id)
        self._reqs_need_recv: dict[ReqId, tuple[Request, list[int]]] = {}
        self._reqs_need_send: dict[ReqId, float] = {}

    def get_num_new_matched_tokens(self, request: 'Request', num_computed_tokens: int) -> tuple[int, bool]:
        """
        For remote prefill, pull all prompt blocks from remote
        asynchronously relative to engine execution.
        
        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request
        Returns:
            * the number of tokens that can be loaded from the 
              external KV cache beyond what is already computed.
            * true if the external KV cache tokens will be loaded
              asynchronously (between scheduler steps).
        """
        params = request.kv_transfer_params
        logger.debug('NIXLConnector get_num_new_matched_tokens: num_computed_tokens=%s, kv_transfer_params=%s', num_computed_tokens, params)
        if params is not None and params.get('do_remote_prefill'):
            assert num_computed_tokens % self.block_size == 0
            rounded_num_prompt_tokens = round_down(len(request.prompt_token_ids), self.block_size)
            count = max(rounded_num_prompt_tokens - num_computed_tokens, 0)
            if count > 0:
                return (count, True)
        return (0, False)

    def update_state_after_alloc(self, request: 'Request', blocks: 'KVCacheBlocks', num_external_tokens: int):
        params = request.kv_transfer_params
        logger.debug('NIXLConnector update_state_after_alloc: num_external_tokens=%s, kv_transfer_params=%s', num_external_tokens, params)
        if params is not None and params.get('do_remote_prefill'):
            if params.get('remote_block_ids'):
                if all((p in params for p in ('remote_engine_id', 'remote_host', 'remote_port'))):
                    local_block_ids = blocks.get_unhashed_block_ids() if num_external_tokens > 0 else []
                    self._reqs_need_recv[request.request_id] = (request, local_block_ids)
                else:
                    logger.warning('Got invalid KVTransferParams: %s. This request will not utilize KVTransfer', params)
            else:
                assert num_external_tokens == 0
            params['do_remote_prefill'] = False

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        meta = NixlConnectorMetadata()
        for req_id, (req, block_ids) in self._reqs_need_recv.items():
            assert req.kv_transfer_params is not None
            meta.add_new_req(request_id=req_id, local_block_ids=block_ids, kv_transfer_params=req.kv_transfer_params)
        self._reqs_need_recv.clear()
        meta.reqs_to_send = self._reqs_need_send
        self._reqs_need_send = {}
        return meta

    def request_finished(self, request: 'Request', block_ids: list[int]) -> tuple[bool, Optional[dict[str, Any]]]:
        """
        Once a request is finished, determine whether request blocks
        should be freed now or will be sent asynchronously and freed later.
        """
        params = request.kv_transfer_params
        logger.debug('NIXLConnector request_finished, request_status=%s, kv_transfer_params=%s', request.status, params)
        if not params:
            return (False, None)
        if params.get('do_remote_prefill'):
            self._reqs_need_recv[request.request_id] = (request, [])
            params['do_remote_prefill'] = False
            return (False, None)
        if not params.get('do_remote_decode') or request.status != RequestStatus.FINISHED_LENGTH_CAPPED:
            return (False, None)
        all_full = request.num_computed_tokens % self.block_size == 0
        computed_block_ids = block_ids if all_full else block_ids[:-1]
        delay_free_blocks = len(computed_block_ids) > 0
        if delay_free_blocks:
            self._reqs_need_send[request.request_id] = time.perf_counter() + envs.VLLM_NIXL_ABORT_REQUEST_TIMEOUT
        return (delay_free_blocks, dict(do_remote_prefill=True, do_remote_decode=False, remote_block_ids=computed_block_ids, remote_engine_id=self.engine_id, remote_host=self.side_channel_host, remote_port=self.side_channel_port, tp_size=self.vllm_config.parallel_config.tensor_parallel_size))

class NixlConnectorWorker:
    """Implementation of Worker side methods"""

    def __init__(self, vllm_config: VllmConfig, engine_id: str):
        if NixlWrapper is None:
            logger.error('NIXL is not available')
            raise RuntimeError('NIXL is not available')
        logger.info('Initializing NIXL wrapper')
        logger.info('Initializing NIXL worker %s', engine_id)
        self.vllm_config = vllm_config
        self.block_size = vllm_config.cache_config.block_size
        self.nixl_wrapper = NixlWrapper(str(uuid.uuid4()), None)
        self._remote_agents: dict[EngineId, dict[int, str]] = defaultdict(dict)
        self.side_channel_port: int = envs.VLLM_NIXL_SIDE_CHANNEL_PORT + vllm_config.parallel_config.data_parallel_rank * vllm_config.parallel_config.tensor_parallel_size
        self.engine_id: EngineId = engine_id
        self.tp_rank = get_tensor_model_parallel_rank()
        self.world_size = get_tensor_model_parallel_world_size()
        self.tp_group = get_tp_group()
        self.kv_caches: dict[str, torch.Tensor] = {}
        self.kv_caches_base_addr: dict[EngineId, list[int]] = {}
        self.num_regions = 0
        self.num_layers = 0
        self.src_xfer_side_handle: int = 0
        self.dst_xfer_side_handles: dict[EngineId, int] = {}
        self.dst_num_blocks: dict[EngineId, int] = {}
        self._registered_descs: list[Any] = []
        self._recving_transfers = defaultdict[ReqId, list[Transfer]](list)
        self._reqs_to_send: dict[ReqId, float] = {}
        self._nixl_handshake_listener_t: Optional[threading.Thread] = None
        self._handshake_initiation_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='vllm-nixl-handshake-initiator')
        self._ready_requests = queue.Queue[tuple[ReqId, ReqMeta]]()
        self._handshake_futures: dict[EngineId, Future[dict[int, str]]] = {}
        self._handshake_lock = threading.RLock()
        self.vllm_config = vllm_config
        self.block_size = vllm_config.cache_config.block_size
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.block_window_per_layer: list[Optional[int]] = []
        self.use_mla = self.model_config.use_mla
        backend = get_attn_backend(self.model_config.get_head_size(), self.model_config.dtype, self.cache_config.cache_dtype, self.block_size, self.model_config.is_attention_free, use_mla=self.use_mla)
        self.backend_name = backend.get_name()
        attn_backend = backend_name_to_enum(self.backend_name)
        self._use_flashinfer = attn_backend == _Backend.FLASHINFER_VLLM_V1
        logger.debug('Detected attention backend %s', self.backend_name)
        self._tp_size: dict[EngineId, int] = {self.engine_id: self.world_size}
        self.consumer_notification_counts_by_req = defaultdict[ReqId, int](int)

    def __del__(self):
        """Cleanup background threads on destruction."""
        self._handshake_initiation_executor.shutdown(wait=False)
        if self._nixl_handshake_listener_t:
            self._nixl_handshake_listener_t.join(timeout=0)

    @staticmethod
    def _nixl_handshake_listener(metadata: NixlAgentMetadata, ready_event: threading.Event, base_port: int, tp_rank: int):
        """Background thread for getting new NIXL handshakes."""
        encoder = msgspec.msgpack.Encoder()
        encoded_data = encoder.encode(metadata)
        size_in_bytes = len(encoded_data)
        logger.debug('Size of encoded NixlAgentMetadata: %s bytes', str(size_in_bytes))
        host = envs.VLLM_NIXL_SIDE_CHANNEL_HOST
        path = make_zmq_path('tcp', host, base_port + tp_rank)
        logger.debug('Starting listening on path: %s', path)
        with zmq_ctx(zmq.ROUTER, path) as sock:
            ready_event.set()
            while True:
                identity, _, msg = sock.recv_multipart()
                if msg != GET_META_MSG:
                    logger.warning('Connection listener got unexpected message %s', msg)
                sock.send_multipart((identity, b'', encoded_data))

    def _nixl_handshake(self, host: str, port: int, remote_tp_size: int, expected_engine_id: str) -> dict[int, str]:
        """Do a NIXL handshake with a remote instance."""
        start_time = time.perf_counter()
        tp_ratio = self._tp_size[self.engine_id] // remote_tp_size
        p_remote_rank = self.tp_rank // tp_ratio
        path = make_zmq_path('tcp', host, port + p_remote_rank)
        logger.debug('Querying metadata on path: %s at remote rank %s', path, p_remote_rank)
        with zmq_ctx(zmq.REQ, path) as sock:
            sock.send(GET_META_MSG)
            metadata_bytes = sock.recv()
            decoder = msgspec.msgpack.Decoder(NixlAgentMetadata)
            metadata = decoder.decode(metadata_bytes)
            got_metadata_time = time.perf_counter()
            logger.debug('NIXL handshake: get metadata took: %s', got_metadata_time - start_time)
            if metadata.engine_id != expected_engine_id:
                raise RuntimeError(f'Remote NIXL agent engine ID mismatch. Expected {expected_engine_id},received {metadata.engine_id}.')
            remote_agent_name = self.add_remote_agent(metadata, p_remote_rank, remote_tp_size)
            setup_agent_time = time.perf_counter()
            logger.debug('NIXL handshake: add agent took: %s', setup_agent_time - got_metadata_time)
        return {p_remote_rank: remote_agent_name}

    def _background_nixl_handshake(self, req_id: str, remote_engine_id: EngineId, meta: ReqMeta):
        fut = self._handshake_futures.get(remote_engine_id)
        if fut is None:
            fut = self._handshake_initiation_executor.submit(self._nixl_handshake, meta.remote_host, meta.remote_port, meta.tp_size, remote_engine_id)
            self._handshake_futures[remote_engine_id] = fut

            def done_callback(f: Future[dict[int, str]], eid=remote_engine_id):
                with self._handshake_lock:
                    del self._handshake_futures[eid]
                    try:
                        self._remote_agents[eid] = f.result()
                    except Exception:
                        logger.exception('Handshake with %s failed', eid)
            fut.add_done_callback(done_callback)

        def request_ready(_f: Future[Any], entry=(req_id, meta)):
            self._ready_requests.put(entry)
        fut.add_done_callback(request_ready)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Register the KV Cache data in nixl."""
        _, first_kv_cache = next(iter(kv_caches.items()))
        kv_elem_size = first_kv_cache.element_size()
        use_mla = len(first_kv_cache.shape) == 3
        assert use_mla == self.use_mla
        if use_mla:
            self.num_blocks = first_kv_cache.shape[0]
            block_rank = 2
            block_shape = first_kv_cache.shape[-block_rank:]
            block_size, kv_latent_dim = block_shape
            self.slot_size_bytes = kv_elem_size * kv_latent_dim
        else:
            if self._use_flashinfer:
                self.num_blocks = first_kv_cache.shape[0]
                block_rank = 4
            else:
                self.num_blocks = first_kv_cache.shape[1]
                block_rank = 3
            block_shape = first_kv_cache.shape[-block_rank:]
            block_size, n_kv_heads, head_dim = block_shape[-3:]
            self.slot_size_bytes = kv_elem_size * n_kv_heads * head_dim
        assert block_size == self.block_size
        self.block_len = kv_elem_size * math.prod(block_shape)
        logger.info('Registering KV_Caches: use_mla: %s, num_blocks: %s, block_shape: %s, per_layer_kv_cache_shape: %s', use_mla, self.num_blocks, block_shape, first_kv_cache.shape)
        self.dst_num_blocks[self.engine_id] = self.num_blocks
        self.kv_caches = kv_caches
        kv_caches_base_addr = []
        caches_data = []
        for cache_or_caches in kv_caches.values():
            cache_list = [cache_or_caches] if use_mla or self._use_flashinfer else cache_or_caches
            for cache in cache_list:
                base_addr = cache.data_ptr()
                region_len = self.num_blocks * self.block_len
                caches_data.append((base_addr, region_len, cache.device.index, ''))
                kv_caches_base_addr.append(base_addr)
        self.kv_caches_base_addr[self.engine_id] = kv_caches_base_addr
        self.num_regions = len(caches_data)
        self.num_layers = len(self.kv_caches.keys())
        if self.vllm_config.model_config.hf_config.model_type == 'llama4':
            assert isinstance(self.vllm_config.model_config.hf_text_config, Llama4TextConfig)
            llama4_config = self.vllm_config.model_config.hf_text_config
            no_rope_layers = llama4_config.no_rope_layers
            chunk_size = llama4_config.attention_chunk_size
            chunk_block_size = math.ceil(chunk_size / self.block_size)
            for layer_idx in range(self.num_layers):
                is_local_attention = no_rope_layers[layer_idx] != 0
                block_window = chunk_block_size if is_local_attention else None
                self.block_window_per_layer.append(block_window)
            logger.debug('Llama 4 block window per layer mapping: %s', self.block_window_per_layer)
            assert len(self.block_window_per_layer) == self.num_layers
        descs = self.nixl_wrapper.get_reg_descs(caches_data, 'VRAM')
        logger.debug('Registering descs: %s', caches_data)
        self.nixl_wrapper.register_memory(descs)
        logger.debug('Done registering descs')
        self._registered_descs.append(descs)
        blocks_data = []
        for base_addr in self.kv_caches_base_addr[self.engine_id]:
            for block_id in range(self.num_blocks):
                block_offset = block_id * self.block_len
                addr = base_addr + block_offset
                blocks_data.append((addr, self.block_len, self.tp_rank))
        logger.debug('Created %s blocks for src engine %s and rank %s', len(blocks_data), self.engine_id, self.tp_rank)
        descs = self.nixl_wrapper.get_xfer_descs(blocks_data, 'VRAM')
        self.src_xfer_side_handle = self.nixl_wrapper.prep_xfer_dlist('NIXL_INIT_AGENT', descs)
        metadata = NixlAgentMetadata(engine_id=self.engine_id, agent_metadata=self.nixl_wrapper.get_agent_metadata(), kv_caches_base_addr=self.kv_caches_base_addr[self.engine_id], num_blocks=self.num_blocks, block_len=self.block_len, attn_backend_name=self.backend_name)
        ready_event = threading.Event()
        self._nixl_handshake_listener_t = threading.Thread(target=self._nixl_handshake_listener, args=(metadata, ready_event, self.side_channel_port, self.tp_rank), daemon=True, name='nixl_handshake_listener')
        self._nixl_handshake_listener_t.start()
        ready_event.wait()

    def add_remote_agent(self, nixl_agent_meta: NixlAgentMetadata, remote_tp_rank: int=0, remote_tp_size: int=1) -> str:
        """
        Add the remote NIXL agent and prepare the descriptors for reading cache
        blocks from remote.

        In particular, handle both homogeneous and heterogeneous TP. The former
        requires local rank_i to read from remote rank_i. 
        The latter, assuming D.world_size > P.world_size, requires that two or 
        more local TP worker share the xfer from a single TP worker.

        Here's an example:

        rank_offset     p_remote_tp_rank
        (kv split no)    
        --------------------------------
            0                 0      Worker0  ---- 1st half of KV ----> Worker0  [ KV Cache ]
                                                                        /
            1                 0      Worker1  ---- 2nd half of KV -----/

            0                 1      Worker2  ---- 1st half of KV ----> Worker1  [ KV Cache ]
                                                                        /
            1                 1      Worker3  ---- 2nd half of KV -----/


                                Decoder TP workers                     Prefix TP workers
                                  (world_size=4)                         (world_size=2)
                                                 tp_ratio = 4 // 2 = 2                  
                                
        Considering the KV Caches, if P-Worker_i has cache size [2, num_blocksP, kv_heads, block_size, head_dim]  
        then D-Worker_j has [2, num_blocksD, kv_heads//tp_ratio, block_size, head_dim]. Mind the "HND" layout format.
        Assuming num_blocksD >= num_blocksP, D-Worker0 reads from P-Worker0 by preparing the kv_heads//tp_ratio 
        first heads from all the slots of all the blocks. D-Worker1 will do the same, but reading the second split
        along the kv_heads dimension, and so forth until "tp_ratio" D TP workers have pulled from P-Worker0.   
        
        Note that the above will also hold true for the homogeneous TP case, where tp_ratio evaluates to 1.

        Regarding MLA case, the cache is replicated across TP workers so the rank_offset will just always be 0
        so that the whole cache is shared by "tp_ratio" D TP workers.
        """
        engine_id = nixl_agent_meta.engine_id
        if remote_tp_rank in self._remote_agents.get(engine_id, {}):
            return self._remote_agents[engine_id][remote_tp_rank]
        if engine_id not in self._tp_size:
            self._tp_size[engine_id] = remote_tp_size
        else:
            assert self._tp_size[engine_id] == remote_tp_size
        assert nixl_agent_meta.attn_backend_name == self.backend_name
        remote_agent_name = self.nixl_wrapper.add_remote_agent(nixl_agent_meta.agent_metadata)
        tp_ratio = divide(self._tp_size[self.engine_id], self._tp_size[engine_id])
        assert tp_ratio > 0, 'Decode TP cannot be smaller than prefill TP'
        total_num_kv_heads = self.model_config.get_total_num_kv_heads()
        is_kv_replicated = self._tp_size[engine_id] // total_num_kv_heads >= 1
        if self.use_mla or is_kv_replicated:
            remote_block_size = nixl_agent_meta.block_len // self.slot_size_bytes
            assert self.block_len == nixl_agent_meta.block_len
        else:
            remote_block_size = nixl_agent_meta.block_len // (self.slot_size_bytes * tp_ratio)
            if self._use_flashinfer:
                remote_block_size //= 2
            assert nixl_agent_meta.block_len == self.block_len * tp_ratio, 'Remote P worker KV layer cache must be of shape [2, N, local_kv_heads*tp_ratio, block_size, head_dim] and same dtype.'
        assert self.block_size == remote_block_size, f'Remote P worker with different block size is not supported self.block_size={self.block_size!r} remote_block_size={remote_block_size!r}'
        if engine_id in self.dst_num_blocks:
            assert self.dst_num_blocks[engine_id] == nixl_agent_meta.num_blocks
        else:
            self.dst_num_blocks[engine_id] = nixl_agent_meta.num_blocks
        blocks_data = []
        self.kv_caches_base_addr[engine_id] = nixl_agent_meta.kv_caches_base_addr
        rank_offset = self.tp_rank % tp_ratio * self.block_len if not (self.use_mla or is_kv_replicated) else 0
        for base_addr in nixl_agent_meta.kv_caches_base_addr:
            for block_id in range(nixl_agent_meta.num_blocks):
                block_offset = block_id * nixl_agent_meta.block_len
                addr = base_addr + block_offset + rank_offset
                blocks_data.append((addr, self.block_len, remote_tp_rank))
        logger.debug('Created %s blocks for dst engine %s with remote rank %s and local rank %s', len(blocks_data), engine_id, remote_tp_rank, self.tp_rank)
        descs = self.nixl_wrapper.get_xfer_descs(blocks_data, 'VRAM')
        self.dst_xfer_side_handles[engine_id] = self.nixl_wrapper.prep_xfer_dlist(remote_agent_name, descs)
        return remote_agent_name

    def get_finished(self) -> tuple[set[str], set[str]]:
        """
        Get requests that are done sending or recving on this specific worker.
        The scheduler process (via the MultiprocExecutor) will use this output
        to track which workers are done.
        """
        done_sending = self._get_new_notifs()
        done_recving = self._pop_done_transfers(self._recving_transfers)
        if len(done_sending) > 0 or len(done_recving) > 0:
            logger.debug('Rank %s, get_finished: %s requests done sending and %s requests done recving', self.tp_rank, len(done_sending), len(done_recving))
        now = time.perf_counter()
        while self._reqs_to_send:
            req_id, expires = next(iter(self._reqs_to_send.items()))
            if now < expires:
                break
            del self._reqs_to_send[req_id]
            done_sending.add(req_id)
        return (done_sending, done_recving)

    def _get_new_notifs(self) -> set[str]:
        """
        Get req_ids which got a remote xfer message. When multiple consumers
        are reading from the same producer (heterogeneous TP scenario), wait
        for all consumers to be done pulling.
        """
        notified_req_ids: set[str] = set()
        for notifs in self.nixl_wrapper.get_new_notifs().values():
            for notif in notifs:
                req_id, tp_ratio = notif.decode('utf-8').rsplit(':', 1)
                self.consumer_notification_counts_by_req[req_id] += 1
                if self.consumer_notification_counts_by_req[req_id] == int(tp_ratio):
                    notified_req_ids.add(req_id)
                    del self.consumer_notification_counts_by_req[req_id]
                    del self._reqs_to_send[req_id]
        return notified_req_ids

    def _pop_done_transfers(self, transfers: dict[str, list[tuple[int, float]]]) -> set[str]:
        """
        Pop completed xfers by checking for DONE state.
        Args:
            transfers: dict of req_id -> list[running_xfer]
        Returns:
            set of req_ids that have all done xfers
        """
        done_req_ids: set[str] = set()
        for req_id, handles in list(transfers.items()):
            in_progress = False
            for handle, _xfer_stime in handles:
                xfer_state = self.nixl_wrapper.check_xfer_state(handle)
                if xfer_state == 'DONE':
                    self.nixl_wrapper.release_xfer_handle(handle)
                elif xfer_state == 'PROC':
                    in_progress = True
                    continue
                else:
                    raise RuntimeError('Transfer failed with state %s', xfer_state)
            if not in_progress:
                done_req_ids.add(req_id)
                del transfers[req_id]
        return done_req_ids

    def start_load_kv(self, metadata: NixlConnectorMetadata):
        """
        Start loading by triggering non-blocking nixl_xfer.
        We check for these trnxs to complete in each step().
        """
        for req_id, meta in metadata.reqs_to_recv.items():
            remote_engine_id = meta.remote_engine_id
            logger.debug('start_load_kv for request %s from remote engine %s. Num local_block_ids: %s. Num remote_block_ids: %s. ', req_id, remote_engine_id, len(meta.local_block_ids), len(meta.remote_block_ids))
            if remote_engine_id not in self._remote_agents:
                with self._handshake_lock:
                    if remote_engine_id not in self._remote_agents:
                        self._background_nixl_handshake(req_id, remote_engine_id, meta)
                        continue
            self._read_blocks_for_req(req_id, meta)
        while not self._ready_requests.empty():
            self._read_blocks_for_req(*self._ready_requests.get_nowait())
        self._reqs_to_send.update(metadata.reqs_to_send)

    def _read_blocks_for_req(self, req_id: str, meta: ReqMeta):
        logger.debug('Remote agent %s available, calling _read_blocks for req %s', meta.remote_engine_id, req_id)
        self._read_blocks(request_id=req_id, dst_engine_id=meta.remote_engine_id, local_block_ids=meta.local_block_ids, remote_block_ids=meta.remote_block_ids)

    def _read_blocks(self, local_block_ids: list[int], remote_block_ids: list[int], dst_engine_id: str, request_id: str):
        tp_ratio = self._tp_size[self.engine_id] // self._tp_size[dst_engine_id]
        notif_id = f'{request_id}:{tp_ratio}'.encode()
        num_local_blocks = len(local_block_ids)
        if num_local_blocks == 0:
            remote_rank = self.tp_rank // tp_ratio
            agent_name = self._remote_agents[dst_engine_id][remote_rank]
            self.nixl_wrapper.send_notif(agent_name, notif_msg=notif_id)
            return
        num_remote_blocks = len(remote_block_ids)
        assert num_local_blocks <= num_remote_blocks
        if num_local_blocks < num_remote_blocks:
            remote_block_ids = remote_block_ids[-num_local_blocks:]
        local_xfer_side_handle = self.src_xfer_side_handle
        remote_xfer_side_handle = self.dst_xfer_side_handles[dst_engine_id]
        local_block_descs_ids: list[int] = []
        remote_block_descs_ids: list[int] = []
        if not self.block_window_per_layer:
            remote_block_descs_ids = self._get_block_descs_ids(dst_engine_id, remote_block_ids)
            local_block_descs_ids = self._get_block_descs_ids(self.engine_id, local_block_ids)
        else:
            for layer_idx, block_window in enumerate(self.block_window_per_layer):
                if block_window is None:
                    layer_local_block_ids = local_block_ids
                    layer_remote_block_ids = remote_block_ids
                else:
                    layer_local_block_ids = local_block_ids[-block_window:]
                    layer_remote_block_ids = remote_block_ids[-block_window:]
                layer_local_desc_ids = self._get_block_descs_ids(self.engine_id, layer_local_block_ids, layer_idx)
                layer_remote_desc_ids = self._get_block_descs_ids(dst_engine_id, layer_remote_block_ids, layer_idx)
                local_block_descs_ids.extend(layer_local_desc_ids)
                remote_block_descs_ids.extend(layer_remote_desc_ids)
        assert len(local_block_descs_ids) == len(remote_block_descs_ids)
        handle = self.nixl_wrapper.make_prepped_xfer('READ', local_xfer_side_handle, local_block_descs_ids, remote_xfer_side_handle, remote_block_descs_ids, notif_msg=notif_id)
        self.nixl_wrapper.transfer(handle)
        self._recving_transfers[request_id].append((handle, time.perf_counter()))

    def _get_block_descs_ids(self, engine_id: str, block_ids: list[int], layer_idx: Optional[int]=None) -> list[int]:
        """
        Get the descs ids for a set of block ids.
        If layer_idx is provided, we use the region_ids for the given layer.
        Otherwise, we use all regions.
        """
        if layer_idx is None:
            region_ids = range(self.num_regions)
        else:
            assert layer_idx < self.num_layers
            if self.num_layers < self.num_regions:
                assert 2 * self.num_layers == self.num_regions
                region_ids = range(2 * layer_idx, 2 * layer_idx + 2)
            else:
                assert self.num_layers == self.num_regions
                region_ids = range(layer_idx, layer_idx + 1)
        num_blocks = self.dst_num_blocks[engine_id]
        descs_ids: list[int] = []
        for reg_id in region_ids:
            for block_id in block_ids:
                descs_ids.append(reg_id * num_blocks + block_id)
        return descs_ids

@contextlib.contextmanager
def zmq_ctx(socket_type: Any, addr: str) -> Iterator[zmq.Socket]:
    """Context manager for a ZMQ socket"""
    if socket_type not in (zmq.ROUTER, zmq.REQ):
        raise ValueError(f'Unexpected socket type: {socket_type}')
    ctx: Optional[zmq.Context] = None
    try:
        ctx = zmq.Context()
        yield make_zmq_socket(ctx=ctx, path=addr, socket_type=socket_type, bind=socket_type == zmq.ROUTER)
    finally:
        if ctx is not None:
            ctx.destroy(linger=0)