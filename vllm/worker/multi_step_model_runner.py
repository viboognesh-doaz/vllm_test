from ..model_executor.model_loader.tensorizer import TensorizerConfig
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from vllm.attention.backends.abstract import AttentionBackend
from vllm.distributed import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.sampler import PromptLogprobs, SampleLogprobs, SamplerOutput, SamplingMetadata, get_logprobs, get_pythonized_sample_results
from vllm.platforms import current_platform
from vllm.sequence import CompletionSequenceGroupOutput, IntermediateTensors, Logprob, SequenceGroupMetadata, SequenceOutput
from vllm.utils import PyObjectCache, async_tensor_h2d, current_stream
from vllm.worker.model_runner import GPUModelRunnerBase, ModelInputForGPUWithSamplingMetadata
from vllm.worker.model_runner_base import BroadcastableModelInput, _init_attn_metadata_from_tensor_dict, _init_frozen_model_input_from_tensor_dict, _init_sampling_metadata_from_tensor_dict
import dataclasses
import functools
import torch
if TYPE_CHECKING:
logger = init_logger(__name__)
MULTI_STEP_ATTENTION_BACKENDS = ['FLASH_ATTN', 'ROCM_FLASH', 'FLASHINFER', 'NO_ATTENTION']
MULTI_STEP_CHUNKED_PREFILL_ATTENTION_BACKENDS = ['FLASH_ATTN', 'FLASHINFER']

def _get_supported_attention_backends(chunked_prefill_enabled: bool) -> List[str]:
    if chunked_prefill_enabled:
        return MULTI_STEP_CHUNKED_PREFILL_ATTENTION_BACKENDS
    else:
        return MULTI_STEP_ATTENTION_BACKENDS

def seq_output_builder():
    return SequenceOutput(0, 0, {0: Logprob(logprob=float('inf'), rank=None, decoded_token=None)})

def completion_seq_group_output_builder():
    return CompletionSequenceGroupOutput([], None)

class PythonizationCache:

    def __init__(self):
        self.cached_seq_output = PyObjectCache(seq_output_builder)
        self.cached_completion_seq_group_output = PyObjectCache(completion_seq_group_output_builder)

    def reset(self):
        self.cached_seq_output.reset()
        self.cached_completion_seq_group_output.reset()

@dataclass
class ModelOutput:
    """The output of a single model forward pass.

    The sampler_output_ready_event is set when the tensors in
    sampler_output are ready (the model+sampler forward pass has
    completed). We use the event to synchronize the GPU->CPU transfer,
    which we want to only run when the data has been written to the
    GPU tensors. Until the event is ready, the tensors in sampler_output
    will have garbage data.

    There are two scenarios:
    1. The output tensors are ready and we can pythonize them immediately.
    2. The output tensors are not ready and we need to wait for the event to be
    ready.
    """
    sampler_output: SamplerOutput
    sampler_output_ready_event: torch.cuda.Event
    sampled_token_ids: Optional[torch.Tensor] = None
    pythonized: bool = False
    logprobs: Optional['torch.Tensor'] = None
    pythonization_cache: Optional[PythonizationCache] = None

    def pythonize(self, input_metadata: 'StatefulModelInput', copy_stream: torch.cuda.Stream, pinned_sampled_token_buffer: torch.Tensor) -> None:
        """Pythonize the output. Blocking."""
        if not self.pythonized:
            self._pythonize_sampler_output(input_metadata, copy_stream, pinned_sampled_token_buffer, True)
            self.pythonized = True

    def maybe_pythonize(self, input_metadata: 'StatefulModelInput', copy_stream: torch.cuda.Stream, pinned_sampled_token_buffer: torch.Tensor) -> None:
        """Pythonize the output if ready, else return None. Non-blocking."""
        if not self.pythonized:
            self.pythonized = self._pythonize_sampler_output(input_metadata, copy_stream, pinned_sampled_token_buffer, False)

    def _pythonize_sampler_output(self, input_metadata: 'StatefulModelInput', copy_stream: torch.cuda.Stream, pinned_sampled_token_buffer: torch.Tensor, blocking: bool) -> bool:
        """
        If blocking is set, will block until the forward pass for the output is
        ready and pythonize the output. Upon completing Pythonization, erases
        self.logprobs (note that a non-blocking call that is performed when
        the sampler output is not yet ready, will not erase self.logprobs.)
        """
        assert self.sampled_token_ids is not None
        if not blocking and (not self.sampler_output_ready_event.query()):
            return False
        if blocking:
            self.sampler_output_ready_event.synchronize()
        with torch.cuda.stream(copy_stream):
            _pythonize_sampler_output(input_metadata, self.sampler_output, pinned_sampled_token_buffer, self.sampled_token_ids, self.logprobs, self.pythonization_cache)
        self.logprobs = None
        return True

@dataclass(frozen=False)
class StatefulModelInput(BroadcastableModelInput):
    frozen_model_input: Optional[ModelInputForGPUWithSamplingMetadata] = None
    cached_outputs: List[ModelOutput] = field(default_factory=list)
    last_sampled_token_ids: Optional[torch.Tensor] = None
    current_step: int = 0
    is_multi_step: bool = True
    is_last_step: bool = False
    is_first_multi_step: bool = False
    base_output_proc_callback: Optional[Callable] = None
    step_cuda_events: List[current_platform.Event] = field(default_factory=lambda: [current_platform.Event(blocking=True)] * 2)
    num_seqs: int = -1
    num_queries: int = -1
    num_single_step_prefills: int = 0

    def as_broadcastable_tensor_dict(self) -> Dict[str, Any]:
        assert self.frozen_model_input is not None
        tensor_dict = self.frozen_model_input.as_broadcastable_tensor_dict()
        new_tensor_dict = {'last_sampled_token_ids': self.last_sampled_token_ids, 'current_step': self.current_step, 'is_multi_step': self.is_multi_step, 'is_last_step': self.is_last_step, 'is_first_multi_step': self.is_first_multi_step, 'num_seqs': self.num_seqs, 'num_queries': self.num_queries, 'num_single_step_prefills': self.num_single_step_prefills}
        tensor_dict.update(new_tensor_dict)
        return tensor_dict

    @classmethod
    def from_broadcasted_tensor_dict(cls, tensor_dict: Dict[str, Any], attn_backend: Optional['AttentionBackend']=None) -> 'StatefulModelInput':
        tensor_dict = _init_sampling_metadata_from_tensor_dict(tensor_dict)
        if attn_backend is not None:
            tensor_dict = _init_attn_metadata_from_tensor_dict(attn_backend, tensor_dict)
        tensor_dict = _init_frozen_model_input_from_tensor_dict(ModelInputForGPUWithSamplingMetadata, tensor_dict)
        return cls(**tensor_dict)

    def record_step_event(self, current_stream: torch.cuda.Stream):
        self.step_cuda_events[self.current_step & 1] = torch.cuda.Event(blocking=True)
        self.step_cuda_events[self.current_step & 1].record(current_stream)

    def wait_previous_step(self):
        self.step_cuda_events[self.current_step + 1 & 1].wait()

    def add_sampler_output(self, sampler_output: SamplerOutput, sampled_token_ids: Optional[torch.Tensor]=None):
        self.cached_outputs.append(ModelOutput(sampler_output=sampler_output, sampler_output_ready_event=None, sampled_token_ids=sampled_token_ids, pythonized=False))

    def maybe_advance_sampling_metadata(self, device: str, pin_memory: bool):
        """
        sampling_metadata.selected_token_indices is constructed for the
        first-step in Multi-Step. However, when chunked-prefill is enabled with
        multi-step, the scheduled prompts are fully processed in the
        first-step and are processed as decodes in the rest of the steps.
        This function updates the sampling_metadata.selected_token_indices
        to account for this conversion.

        Example:
        Let 2 prompts and 2 decodes be scheduled together. Let the
        num-tokens to process for the 2 prompts be 5 and 8 respectively.

        In that case, sampling_metadata.sampled_token_indices will be,
        [4, 12, 13, 14] as it is constructed for the first-step in
        multi-step.
        However, the prompts turns to decodes after the first-step
        and the num-tokens for the previously-prompt sequences will
        be 1 and 1 as they are decodes now. The self.sampled_token_indices
        must be updated to [0,1,2,3].
        """
        assert self.current_step == 1 and self.num_single_step_prefills > 0
        if not get_pp_group().is_last_rank:
            return
        assert self.frozen_model_input is not None
        assert self.frozen_model_input.sampling_metadata is not None
        self.frozen_model_input.sampling_metadata.selected_token_indices = async_tensor_h2d(list(range(self.num_queries)), dtype=torch.long, target_device=device, pin_memory=pin_memory)

    def maybe_advance_frozen_model_input(self, device: str, pin_memory: bool):
        """
        Advancing the datastructures of StatefulModelInput::frozen_model_input
        is only required when prefills are scheduled with decodes to run in
        multi-step. This advancement/correction is required to account for
        the conversion of Prefills to Decodes after the first multi-step.
        """
        if self.current_step != 1 or self.num_single_step_prefills == 0:
            return
        assert self.frozen_model_input is not None
        fmi = self.frozen_model_input
        assert fmi.input_tokens is not None
        assert fmi.input_tokens.shape[0] >= self.num_seqs
        fmi_new_input_tokens: torch.Tensor = fmi.input_tokens[:self.num_seqs]
        assert fmi.input_positions is not None
        assert fmi.input_positions.shape[0] >= self.num_seqs
        fmi_new_input_positions: torch.Tensor = fmi.input_positions[:self.num_seqs]
        assert fmi.lora_mapping is None
        assert fmi.lora_requests is not None
        assert len(fmi.lora_requests) == 0
        assert fmi.attn_metadata is not None
        assert fmi.prompt_adapter_mapping is None
        assert fmi.prompt_adapter_requests is not None
        assert len(fmi.prompt_adapter_requests) == 0
        assert fmi.multi_modal_kwargs is not None
        assert len(fmi.multi_modal_kwargs) == 0
        self.frozen_model_input = dataclasses.replace(self.frozen_model_input, input_tokens=fmi_new_input_tokens, input_positions=fmi_new_input_positions)
        self.maybe_advance_sampling_metadata(device, pin_memory)

class MultiStepModelRunner(GPUModelRunnerBase[StatefulModelInput]):

    def __init__(self, base_model_runner: GPUModelRunnerBase, *args, **kwargs):
        super().__init__(*args, **kwargs)
        supported_attention_backends: List[str] = _get_supported_attention_backends(self.scheduler_config.chunked_prefill_enabled)
        if self.attn_backend.get_name() not in supported_attention_backends:
            ms_config_str: str = 'Multi-Step + Chunked-Prefill' if self.scheduler_config.chunked_prefill_enabled else 'Multi-Step'
            raise ValueError(f'{ms_config_str} not supported for attention backend: {self.attn_backend.get_name()}. Set VLLM_ATTENTION_BACKEND to a value from {supported_attention_backends}.')
        self._base_model_runner: GPUModelRunnerBase = base_model_runner
        self.is_multi_step = self.scheduler_config.is_multi_step
        self.pinned_sampled_token_ids: Optional[torch.Tensor] = None
        self.pythonization_cache = PythonizationCache() if self.parallel_config.pipeline_parallel_size == 1 else None

    @functools.cached_property
    def _copy_stream(self):
        return torch.cuda.Stream()

    def make_model_input_from_broadcasted_tensor_dict(self, tensor_dict: Dict[str, Any]) -> StatefulModelInput:
        model_input = StatefulModelInput.from_broadcasted_tensor_dict(tensor_dict, attn_backend=self.attn_backend)
        return model_input

    def prepare_model_input(self, seq_group_metadata_list: List[SequenceGroupMetadata], virtual_engine: int=0, finished_requests_ids: Optional[List[str]]=None) -> StatefulModelInput:
        frozen_model_input: ModelInputForGPUWithSamplingMetadata = self._base_model_runner.prepare_model_input(seq_group_metadata_list, virtual_engine, finished_requests_ids)
        assert frozen_model_input.query_lens is not None
        assert frozen_model_input.seq_lens is not None
        assert frozen_model_input.attn_metadata is not None
        num_queries = len(frozen_model_input.query_lens)
        num_seqs = len(frozen_model_input.seq_lens)
        num_single_step_prefills = frozen_model_input.attn_metadata.num_prefills
        model_input = StatefulModelInput(frozen_model_input=frozen_model_input, num_seqs=num_seqs, num_queries=num_queries, num_single_step_prefills=num_single_step_prefills)
        return model_input

    def _async_process_outputs(self, model_input: StatefulModelInput, output_proc_callback: Callable):
        output_proc_callback()
        cont = True
        for step_num, model_output in enumerate(model_input.cached_outputs):
            if not model_output.pythonized:
                model_output.maybe_pythonize(model_input, self._copy_stream, self.pinned_sampled_token_ids)
                if model_output.pythonized:
                    ctx = output_proc_callback.keywords['ctx']
                    ctx.append_output(outputs=[model_output.sampler_output], seq_group_metadata_list=ctx.seq_group_metadata_list, scheduler_outputs=ctx.scheduler_outputs, is_async=False, is_last_step=False, is_first_step_output=step_num == 0)
                    output_proc_callback()
                else:
                    cont = False
            if not cont:
                break

    def _final_process_outputs(self, model_input: StatefulModelInput, output_proc_callback: Optional[Callable]) -> List[SamplerOutput]:
        assert model_input.frozen_model_input is not None
        has_async_callback = output_proc_callback is not None
        outputs = []
        for step_num, output in enumerate(model_input.cached_outputs):
            is_last_step = step_num == len(model_input.cached_outputs) - 1
            if has_async_callback:
                assert output_proc_callback is not None
                output_proc_callback()
                if not output.pythonized:
                    output.pythonize(model_input, self._copy_stream, self.pinned_sampled_token_ids)
                    if not is_last_step:
                        ctx = output_proc_callback.keywords['ctx']
                        ctx.append_output(outputs=[output.sampler_output], seq_group_metadata_list=ctx.seq_group_metadata_list, scheduler_outputs=ctx.scheduler_outputs, is_async=False, is_last_step=False, is_first_step_output=step_num == 0)
                    else:
                        outputs.append(output.sampler_output)
            else:
                output.pythonize(model_input, self._copy_stream, self.pinned_sampled_token_ids)
                outputs.append(output.sampler_output)
        return outputs

    @torch.inference_mode()
    def execute_model(self, model_input: StatefulModelInput, kv_caches: List[torch.Tensor], intermediate_tensors: Optional[IntermediateTensors]=None, num_steps: int=1) -> Optional[Union[List[SamplerOutput], IntermediateTensors]]:
        """ 
        Execute the model for a single step and update multi-step
        metadata
        """
        assert num_steps == 1, 'MultiStepModelRunner only supports num_steps=1'
        frozen_model_input = model_input.frozen_model_input
        assert frozen_model_input is not None
        if not model_input.is_multi_step:
            return self._base_model_runner.execute_model(frozen_model_input, None, intermediate_tensors, num_steps)
        if self.is_driver_worker and get_pp_group().is_last_rank:
            if self.pinned_sampled_token_ids is None:
                self.pinned_sampled_token_ids = torch.zeros((self.scheduler_config.max_num_seqs, 1), dtype=torch.long, device='cpu', pin_memory=True)
            self._base_model_runner.sampler.include_gpu_probs_tensor = True
            if frozen_model_input.sampling_metadata:
                frozen_model_input.sampling_metadata.skip_sampler_cpu_output = True
        stream = current_stream()
        if not model_input.is_first_multi_step:
            model_input.wait_previous_step()
            model_input = self._advance_step(model_input, model_input.cached_outputs[-1].sampler_output)
            frozen_model_input = model_input.frozen_model_input
            assert frozen_model_input is not None
        if model_input.base_output_proc_callback is None:
            assert frozen_model_input is not None
            model_input.base_output_proc_callback = frozen_model_input.async_callback
        if frozen_model_input.async_callback is not None:
            assert model_input.base_output_proc_callback is not None
            async_callback = functools.partial(self._async_process_outputs, model_input=model_input, output_proc_callback=model_input.base_output_proc_callback)
            model_input.frozen_model_input = dataclasses.replace(model_input.frozen_model_input, async_callback=async_callback)
            frozen_model_input = model_input.frozen_model_input
            assert frozen_model_input is not None
        output = self._base_model_runner.execute_model(frozen_model_input, None, intermediate_tensors, num_steps=1)
        model_input.record_step_event(stream)
        if get_pp_group().is_last_rank and self.is_driver_worker:
            assert isinstance(output, list)
            assert len(output) == 1, 'MultiStepModelRunner requires single-step base_models'
            output_ready_event = torch.cuda.Event()
            output_ready_event.record(stream)
            if self.parallel_config.pipeline_parallel_size > 1:
                output[0].sampled_token_ids_cpu = output[0].sampled_token_ids.cpu()
            model_input.cached_outputs.append(ModelOutput(output[0], output_ready_event, output[0].sampled_token_ids, False, output[0].logprobs, self.pythonization_cache))
            output[0].sampled_token_ids = None
            output[0].sampled_token_probs = None
            output[0].logprobs = None
            if frozen_model_input.async_callback is None:
                for model_output in model_input.cached_outputs:
                    model_output.maybe_pythonize(model_input, self._copy_stream, self.pinned_sampled_token_ids)
        model_input.current_step += 1
        if not get_pp_group().is_last_rank:
            assert isinstance(output, IntermediateTensors)
            return output
        if not self.is_driver_worker:
            return []
        if model_input.is_last_step:
            outputs = self._final_process_outputs(model_input, model_input.base_output_proc_callback)
            if self.pythonization_cache:
                self.pythonization_cache.reset()
            return outputs
        return output

    def _update_sampling_metadata(self, sampling_metadata: SamplingMetadata, num_seqs: Optional[int], num_queries: int):
        assert sampling_metadata.num_prompts == 0
        assert len(sampling_metadata.seq_groups) == num_queries
        assert sampling_metadata.selected_token_indices.shape == (num_queries,)
        for i in range(num_queries):
            seq_group = sampling_metadata.seq_groups[i]
            assert seq_group.is_prompt is False
            assert seq_group.prompt_logprob_indices == []
            assert seq_group.sample_indices == [i]
            assert seq_group.seq_len is None
            assert seq_group.query_len is None

    def _advance_step(self, model_input: StatefulModelInput, out: SamplerOutput) -> StatefulModelInput:
        model_input.maybe_advance_frozen_model_input(self.device, self.pin_memory)
        frozen_model_input = model_input.frozen_model_input
        assert frozen_model_input is not None
        assert frozen_model_input.input_tokens is not None
        assert frozen_model_input.input_tokens.shape[0] == model_input.num_seqs
        assert frozen_model_input.attn_metadata is not None
        sampled_token_ids = model_input.cached_outputs[-1].sampled_token_ids
        num_seqs = model_input.num_seqs
        num_queries = model_input.num_queries
        frozen_model_input = model_input.frozen_model_input
        assert frozen_model_input is not None
        attn_metadata = frozen_model_input.attn_metadata
        assert attn_metadata is not None
        turn_prefills_into_decodes: bool = model_input.current_step == 1 and model_input.num_single_step_prefills != 0
        attn_metadata.advance_step(frozen_model_input, sampled_token_ids, self.block_size, num_seqs, num_queries, turn_prefills_into_decodes=turn_prefills_into_decodes)
        return model_input

    def load_model(self) -> None:
        self._base_model_runner.load_model()
        self.model_memory_usage = self._base_model_runner.model_memory_usage

    def save_sharded_state(self, path: str, pattern: Optional[str]=None, max_size: Optional[int]=None) -> None:
        return self._base_model_runner.save_sharded_state(path, pattern, max_size)

    def save_tensorized_model(self, tensorizer_config: TensorizerConfig) -> None:
        return self._base_model_runner.save_tensorized_model(tensorizer_config)

    def profile_run(self) -> None:
        return self._base_model_runner.profile_run()

    def remove_all_loras(self):
        return self._base_model_runner.remove_all_loras()

    def capture_model(self, kv_caches: List[List]) -> None:
        return self._base_model_runner.capture_model(kv_caches)

    @property
    def vocab_size(self) -> int:
        return self._base_model_runner.vocab_size
DeferredLogprobsReturnType = Tuple[Optional[List[Optional[PromptLogprobs]]], Optional[List[SampleLogprobs]]]

def deferred_pythonize_logprobs(output: SamplerOutput, sampling_metadata: SamplingMetadata, logprobs_tensor: Optional[torch.Tensor]) -> DeferredLogprobsReturnType:
    """Perform deferred logprob Pythonization.

    1. Pythonize GPU-side sampler result tensors into CPU-side sampler result.
    2. Pythonize GPU-side logprobs tensor into CPU-side logprobs lists,
       utilizing  the Pythonized sampler result computed in step 1.
    
    These deferred computations are not required for single-step scheduling
    or the `profile_run()` phase of multi-step scheduling.

    Args:
        output: sampler output (under deferred Pythonization)
        sampling_metadata
        
    Returns:
        prompt_logprobs (CPU), sample_logprobs (CPU)
    """
    sampler_result = get_pythonized_sample_results(output.deferred_sample_results_args)
    output.deferred_sample_results_args = None
    prompt_logprobs, sample_logprobs = get_logprobs(logprobs_tensor, sampling_metadata, sampler_result)
    assert len(prompt_logprobs) == len(sampling_metadata.seq_groups)
    assert len(sample_logprobs) == len(sampling_metadata.seq_groups)
    return (prompt_logprobs, sample_logprobs)

def _pythonize_sampler_output(model_input: StatefulModelInput, output: SamplerOutput, pinned_sampled_token_buffer: torch.Tensor, sampled_token_ids: torch.Tensor, logprobs_tensor: Optional[torch.Tensor], cache: Optional[PythonizationCache]) -> None:
    """ This function is only called when the output tensors are ready.
    See [`ModelOutput`][vllm.worker.multi_step_model_runner.ModelOutput].

    Modifies `output.outputs` and `pinned_sampled_token_buffer` in-place,
    adding a Pythonized output data structure
    ([`CompletionSequenceGroupOutput`][vllm.sequence.CompletionSequenceGroupOutput])
    for each [`SequenceGroup`][vllm.sequence.SequenceGroup].

    Args:
      model_input
      output: sampler output
      pinned_sampled_token_token_buffer: CPU-side pinned memory
                                         (receives copy of
                                         GPU-side token buffer.)
      sampled_token_ids: GPU-side token buffer
      logprobs_tensor: GPU-side tensor containing 
                       logprobs computed during sampling
    """
    assert model_input.frozen_model_input is not None
    frozen_model_input = model_input.frozen_model_input
    assert frozen_model_input.sampling_metadata is not None
    sampling_metadata = frozen_model_input.sampling_metadata
    assert not output.outputs
    pinned_buffer = pinned_sampled_token_buffer[:model_input.num_queries]
    seq_groups = sampling_metadata.seq_groups
    prompt_logprobs_are_requested_for_prefill = any([sg.sampling_params.prompt_logprobs is not None and sg.is_prompt for sg in seq_groups])
    any_logprobs_are_requested = prompt_logprobs_are_requested_for_prefill or any([sg.sampling_params.logprobs is not None for sg in seq_groups])
    if prompt_logprobs_are_requested_for_prefill:
        sample_idx_tensor = torch.tensor([sdx for sg in seq_groups for sdx in sg.sample_indices])
        pinned_buffer = pinned_buffer.copy_(sampled_token_ids[sample_idx_tensor, :], non_blocking=False)
    else:
        pinned_buffer = pinned_buffer.copy_(sampled_token_ids, non_blocking=False)
    samples_list = pinned_buffer.tolist()
    skip_sampler_cpu_output = frozen_model_input.sampling_metadata.skip_sampler_cpu_output
    do_pythonize_logprobs = skip_sampler_cpu_output and any_logprobs_are_requested
    prompt_logprobs, sample_logprobs = deferred_pythonize_logprobs(output, sampling_metadata, logprobs_tensor) if do_pythonize_logprobs else (None, None)
    for sgdx, (seq_group, sample_result) in enumerate(zip(seq_groups, samples_list)):
        if seq_group.sampling_params.logits_processors:
            assert len(seq_group.sampling_params.logits_processors) == 0, 'Logits Processors are not supported in multi-step decoding'
        if do_pythonize_logprobs:
            assert prompt_logprobs is not None
            assert sample_logprobs is not None
            group_prompt_logprobs, group_sample_logprobs = (prompt_logprobs[sgdx], sample_logprobs[sgdx])
        elif any_logprobs_are_requested:
            group_prompt_logprobs, group_sample_logprobs = (output.outputs[sgdx].prompt_logprobs, [sample.logprobs for sample in output.outputs[sgdx].samples])
        seq_ids = seq_group.seq_ids
        next_token_ids = sample_result
        parent_ids = [0]
        seq_outputs: List[SequenceOutput]
        if cache is not None:
            completion_seq_group_output: CompletionSequenceGroupOutput = cache.cached_completion_seq_group_output.get_object()
            completion_seq_group_output.samples.clear()
            seq_outputs = completion_seq_group_output.samples
        else:
            seq_outputs = []
        for tdx, (parent_id, next_token_id) in enumerate(zip(parent_ids, next_token_ids)):
            if cache is not None:
                seq_output: SequenceOutput = cache.cached_seq_output.get_object()
                seq_output.parent_seq_id = seq_ids[parent_id]
                seq_output.output_token = next_token_id
                if any_logprobs_are_requested:
                    seq_output.logprobs = group_sample_logprobs[tdx]
                else:
                    logprobs = next(iter(seq_output.logprobs.values()))
                    seq_output.logprobs.clear()
                    logprobs.logprob = float('inf')
                    logprobs.rank = None
                    logprobs.decoded_token = None
                    seq_output.logprobs[next_token_id] = logprobs
                seq_outputs.append(seq_output)
            else:
                seq_outputs.append(SequenceOutput(seq_ids[parent_id], next_token_id, group_sample_logprobs[tdx] if any_logprobs_are_requested else {next_token_id: Logprob(logprob=float('inf'), rank=None, decoded_token=None)}))
        if cache is not None:
            completion_seq_group_output.prompt_logprobs = group_prompt_logprobs if any_logprobs_are_requested else None
            output.outputs.append(completion_seq_group_output)
        else:
            output.outputs.append(CompletionSequenceGroupOutput(seq_outputs, group_prompt_logprobs if any_logprobs_are_requested else None))
    assert len(output.outputs) > 0