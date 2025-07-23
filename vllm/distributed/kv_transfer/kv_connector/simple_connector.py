from typing import TYPE_CHECKING, Optional, Union
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase
from vllm.distributed.kv_transfer.kv_connector.utils import model_aware_kv_ops_helper as kv_helper
from vllm.distributed.kv_transfer.kv_lookup_buffer.simple_buffer import SimpleBuffer
from vllm.distributed.kv_transfer.kv_pipe.mooncake_pipe import MooncakePipe
from vllm.distributed.kv_transfer.kv_pipe.pynccl_pipe import PyNcclPipe
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors
from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata
import os
import torch
'\nSimple KV Cache Connector for Distributed Machine Learning Inference\n\nThe SimpleConnector transfers KV caches between prefill vLLM worker (KV cache\nproducer) and decode vLLM worker (KV cache consumer) using PyNcclPipe or\nMooncakePipe.\n\nBut the logic can be extended to support other pipe and lookup buffer.\n'
if TYPE_CHECKING:
logger = init_logger(__name__)

class SimpleConnector(KVConnectorBase):

    def __init__(self, rank: int, local_rank: int, config: VllmConfig):
        self.config = config.kv_transfer_config
        self.kv_helper = kv_helper(config)
        if self.config.kv_connector == 'PyNcclConnector':
            logger.info('Initializing PyNcclConfig under kv_transfer_config %s', self.config)
        elif self.config.kv_connector == 'MooncakeConnector':
            use_mooncake_distributed_pipe = os.getenv('MOONCAKE_CONFIG_PATH') is not None
            if not use_mooncake_distributed_pipe:
                raise ValueError("To use MooncakeConnector, you need to pass the ENV: 'MOONCAKE_CONFIG_PATH=/path/to/mooncake_config.json'.")
            else:
                logger.info('Initializing MooncakeConfig under kv_transfer_config %s', self.config)
        self.lookup_buffer_size = self.config.kv_buffer_size
        self.producer_buffer: Optional[SimpleBuffer] = None
        self.consumer_buffer: Optional[SimpleBuffer] = None
        self.producer_data_pipe: Union[PyNcclPipe, MooncakePipe]
        self.consumer_data_pipe: Union[PyNcclPipe, MooncakePipe]
        self.producer_signal_pipe: Union[PyNcclPipe, MooncakePipe]
        self.consumer_signal_pipe: Union[PyNcclPipe, MooncakePipe]
        port_offset_base = 2 * rank
        if self.config.is_kv_producer:
            if self.config.kv_connector == 'PyNcclConnector':
                self.producer_data_pipe = PyNcclPipe(local_rank=local_rank, config=self.config, port_offset=port_offset_base)
                self.producer_signal_pipe = PyNcclPipe(local_rank=local_rank, config=self.config, port_offset=port_offset_base + 1, device='cpu')
            elif self.config.kv_connector == 'MooncakeConnector':
                self.producer_data_pipe = MooncakePipe(local_rank=local_rank, config=self.config)
                self.producer_signal_pipe = self.producer_data_pipe
            self.producer_buffer = SimpleBuffer(self.producer_signal_pipe, self.producer_data_pipe, self.config.kv_buffer_size)
        else:
            if self.config.kv_connector == 'PyNcclConnector':
                self.consumer_data_pipe = PyNcclPipe(local_rank=local_rank, config=self.config, port_offset=port_offset_base)
                self.consumer_signal_pipe = PyNcclPipe(local_rank=local_rank, config=self.config, port_offset=port_offset_base + 1, device='cpu')
            elif self.config.kv_connector == 'MooncakeConnector':
                self.consumer_data_pipe = MooncakePipe(local_rank=local_rank, config=self.config)
                self.consumer_signal_pipe = self.consumer_data_pipe
            self.consumer_buffer = SimpleBuffer(self.consumer_signal_pipe, self.consumer_data_pipe, self.config.kv_buffer_size)

    def select(self, input_tokens: Optional[torch.Tensor], roi: Optional[torch.Tensor]) -> list[Optional[torch.Tensor]]:
        assert self.consumer_buffer is not None, 'Please initialize the consumer buffer before calling select.'
        return self.consumer_buffer.drop_select(input_tokens, roi)

    def insert(self, input_tokens: torch.Tensor, roi: torch.Tensor, key: torch.Tensor, value: torch.Tensor, hidden: torch.Tensor) -> None:
        assert self.producer_buffer is not None, 'Please initialize the producer buffer before calling insert.'
        self.producer_buffer.insert(input_tokens, roi, key, value, hidden)

    def send_kv_caches_and_hidden_states(self, model_executable: torch.nn.Module, model_input: 'ModelInputForGPUWithSamplingMetadata', kv_caches: list[torch.Tensor], hidden_or_intermediate_states: Union[torch.Tensor, IntermediateTensors]) -> None:
        input_tokens_tensor = model_input.input_tokens
        seq_lens = model_input.attn_metadata.seq_lens
        slot_mapping_flat = model_input.attn_metadata.slot_mapping.flatten()
        num_prefill_tokens = model_input.attn_metadata.num_prefill_tokens
        start_layer = model_executable.model.start_layer
        end_layer = model_executable.model.end_layer
        num_heads, head_size = self.kv_helper.get_model_args(model_executable)
        for idx, slen in enumerate(seq_lens):
            start_pos = sum(seq_lens[:idx])
            end_pos = start_pos + slen
            if start_pos >= num_prefill_tokens:
                logger.warning("You have some decode requests while using SimpleConnector. Their KVCache won't be sent.")
                break
            current_tokens = input_tokens_tensor[start_pos:end_pos]
            keys, values = ([], [])
            for layer_id in range(start_layer, end_layer):
                kv_cache = kv_caches[layer_id - start_layer]
                key_cache, value_cache = self.kv_helper.get_kv_from_cache(kv_cache, num_heads, head_size)
                current_slot_mapping = slot_mapping_flat[start_pos:end_pos]
                keys.append(key_cache[current_slot_mapping].unsqueeze(0))
                values.append(value_cache[current_slot_mapping].unsqueeze(0))
            keys = torch.cat(keys, dim=0)
            values = torch.cat(values, dim=0)
            self.insert(current_tokens, torch.ones_like(current_tokens, dtype=bool), keys, values, hidden_or_intermediate_states[start_pos:end_pos])
        logger.debug('[rank%d]: KV send DONE.', torch.distributed.get_rank())

    def recv_kv_caches_and_hidden_states(self, model_executable: torch.nn.Module, model_input: 'ModelInputForGPUWithSamplingMetadata', kv_caches: list[torch.Tensor]) -> tuple[Union[torch.Tensor, IntermediateTensors], bool, 'ModelInputForGPUWithSamplingMetadata']:
        bypass_model_exec = True
        input_tokens_tensor = model_input.input_tokens
        seq_lens = model_input.attn_metadata.seq_lens
        num_prefill_tokens = model_input.attn_metadata.num_prefill_tokens
        slot_mapping = model_input.attn_metadata.slot_mapping.flatten()
        start_layer = model_executable.model.start_layer
        end_layer = model_executable.model.end_layer
        hidden_or_intermediate_states_for_one_req = []
        input_tokens_list = []
        num_computed_tokens_list = []
        start_pos_list = []
        for idx, slen in enumerate(seq_lens):
            start_pos = sum(seq_lens[:idx])
            end_pos = start_pos + slen
            if start_pos >= num_prefill_tokens:
                logger.warning('You should set --enable_chunked_prefill=False and --max_num_batched_tokens should be equal to --max_seq_len_to_capture')
                bypass_model_exec = False
                assert start_pos == num_prefill_tokens
                break
            current_tokens = input_tokens_tensor[start_pos:end_pos]
            num_tokens = slen
            input_tokens_list.append(current_tokens)
            start_pos_list.append(start_pos)
            ret = self.select(current_tokens, torch.ones_like(current_tokens, dtype=bool))
            if ret[0] is None:
                bypass_model_exec = False
                num_computed_tokens_list.append(0)
                continue
            roi: torch.Tensor = ret[1]
            keys: torch.Tensor = ret[2]
            values: torch.Tensor = ret[3]
            hidden: torch.Tensor = ret[4]
            num_computed_tokens = roi.shape[0]
            num_computed_tokens_list.append(num_computed_tokens)
            if not all([num_computed_tokens == num_tokens, hidden is not None]):
                bypass_model_exec = False
            end_pos = start_pos + num_computed_tokens
            for cur_layer in range(start_layer, end_layer):
                layer_id = cur_layer - start_layer
                kv_cache = kv_caches[layer_id]
                layer = model_executable.model.layers[cur_layer]
                remote_k, remote_v = (keys[layer_id], values[layer_id])
                self.kv_helper.put_kv_to_cache(model_executable, remote_k, remote_v, layer, kv_cache, slot_mapping, start_pos, end_pos)
            hidden_or_intermediate_states_for_one_req.append(hidden)
        if not bypass_model_exec:
            logger.warning('[rank%d]: Failed to receive all KVs and hidden states, redo model forwarding.', torch.distributed.get_rank())
            hidden_or_intermediate_states = None
        else:
            logger.debug('[rank%d]: Successfully received all KVs and hidden states, skip model forwarding.', torch.distributed.get_rank())
            hidden_or_intermediate_states = torch.cat(hidden_or_intermediate_states_for_one_req, dim=0)
        return (hidden_or_intermediate_states, bypass_model_exec, model_input)

    def close(self):
        self.producer_data_pipe.close()
        self.consumer_data_pipe.close()
        if self.config.kv_connector == 'PyNcclConnector':
            self.producer_signal_pipe.close()
            self.consumer_signal_pipe.close()
        elif self.config.kv_connector == 'MooncakeConnector':
            pass