from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from vllm.distributed import broadcast_tensor_dict, get_pp_group
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import ExecuteModelRequest
from vllm.worker.model_runner_base import BroadcastableModelInput
from vllm.worker.multi_step_model_runner import MultiStepModelRunner, StatefulModelInput
from vllm.worker.worker import Worker, WorkerInput
import dataclasses
import torch

@dataclass
class MultiStepState:
    worker_input: WorkerInput
    model_input: StatefulModelInput

class MultiStepWorker(Worker):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        base_model_runner = self.model_runner
        self.model_runner = MultiStepModelRunner(base_model_runner, vllm_config=base_model_runner.vllm_config, kv_cache_dtype=self.cache_config.cache_dtype, is_driver_worker=base_model_runner.is_driver_worker)
        pipeline_parallel_size = self.parallel_config.pipeline_parallel_size
        self.multi_step_states: List[Optional[MultiStepState]] = [None] * pipeline_parallel_size
        self.temp_output = None

    def _get_driver_input_and_broadcast(self, execute_model_req: ExecuteModelRequest) -> Tuple[BroadcastableModelInput, WorkerInput, Dict[str, torch.Tensor]]:
        """
        Get the driver input and broadcast it to other workers.
        """
        assert self.is_driver_worker
        virtual_engine = execute_model_req.virtual_engine
        is_first_multi_step = execute_model_req.is_first_multi_step
        if is_first_multi_step:
            worker_input: WorkerInput = self.prepare_worker_input(execute_model_req=execute_model_req)
            model_input: StatefulModelInput = self.model_runner.prepare_model_input(execute_model_req.seq_group_metadata_list, execute_model_req.virtual_engine, execute_model_req.finished_requests_ids)
            if execute_model_req.async_callback:
                model_input.frozen_model_input = dataclasses.replace(model_input.frozen_model_input, async_callback=execute_model_req.async_callback)
        else:
            multi_step_state = self.multi_step_states[virtual_engine]
            worker_input = multi_step_state.worker_input
            model_input = multi_step_state.model_input
            frozen_model_input = model_input.frozen_model_input
            assert frozen_model_input is not None
            assert frozen_model_input.attn_metadata is not None
            frozen_model_input.attn_metadata._cached_prefill_metadata = None
            frozen_model_input.attn_metadata._cached_decode_metadata = None
        model_input.is_first_multi_step = is_first_multi_step
        model_input.is_last_step = execute_model_req.is_last_step
        if not is_first_multi_step:
            self._prepare_last_sampled_token_ids_for_tp_workers(execute_model_req=execute_model_req, model_input=model_input)
        if self.do_metadata_broadcast:
            broadcast_data = worker_input.as_broadcastable_tensor_dict()
            broadcast_data.update(model_input.as_broadcastable_tensor_dict())
            broadcast_tensor_dict(broadcast_data, src=0)
        return (model_input, worker_input, {})

    def _prepare_last_sampled_token_ids_for_tp_workers(self, execute_model_req: ExecuteModelRequest, model_input: StatefulModelInput) -> None:
        """ 
        Prepare the last sampled token ids for TP workers. If it's the last 
        PP rank, then the last sampled token ids are already in the model_input.
        If it is NOT the last PP rank, then we need to get the last sampled
        token that is cached in the execute_model_req.
        """
        if get_pp_group().is_last_rank:
            assert model_input.cached_outputs[-1].sampler_output.sampled_token_ids is None
            assert model_input.cached_outputs[-1].sampled_token_ids is not None
            model_input.last_sampled_token_ids = model_input.cached_outputs[-1].sampled_token_ids
            for output in model_input.cached_outputs[:-1]:
                if output.pythonized:
                    output.sampled_token_ids = None
        else:
            assert execute_model_req.last_sampled_token_ids is not None
            model_input.last_sampled_token_ids = execute_model_req.last_sampled_token_ids.cuda()
            model_input.add_sampler_output(SamplerOutput(outputs=[], sampled_token_ids=None), model_input.last_sampled_token_ids)
            for output in model_input.cached_outputs[:-1]:
                output.sampled_token_ids = None
            assert model_input.cached_outputs[-1].sampled_token_ids is not None

    def prepare_input(self, execute_model_req: Optional[ExecuteModelRequest]=None) -> Optional[Tuple[StatefulModelInput, WorkerInput, Dict[str, torch.Tensor]]]:
        """
        Depending on the current state of the request and multi step worker,
        this method may skip the normal _prepare_model_input and
        _prepare_worker_input methods and instead used cached values.
        """
        if self.is_driver_worker:
            if execute_model_req is None:
                if self.do_metadata_broadcast:
                    broadcast_tensor_dict({}, src=0)
                return None
            virtual_engine = execute_model_req.virtual_engine
            model_input, worker_input, kwargs = self._get_driver_input_and_broadcast(execute_model_req)
            assert isinstance(model_input, StatefulModelInput)
            if execute_model_req.is_first_multi_step:
                self.multi_step_states[virtual_engine] = MultiStepState(worker_input=worker_input, model_input=model_input)
        else:
            broadcast_data = self._get_worker_input_from_broadcast()
            if broadcast_data is None:
                return None
            model_input, worker_input, kwargs = broadcast_data
            assert isinstance(model_input, StatefulModelInput)
            virtual_engine = worker_input.virtual_engine
            if model_input.is_first_multi_step:
                pass
            else:
                assert isinstance(model_input, StatefulModelInput)
                model_input.add_sampler_output(SamplerOutput(outputs=[], sampled_token_ids=None), model_input.last_sampled_token_ids)
        assert model_input is not None
        assert worker_input is not None
        return (model_input, worker_input, kwargs)