from copy import copy
from typing import Optional
from vllm.outputs import CompletionOutput
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.v1.metrics.stats import IterationStats

class ParentRequest:
    """Info, state & processing for parallel sampling request.

    Store parent request ID and sampling params.
    Facilitate generating child request sampling params.
    """
    request_id: str
    sampling_params: SamplingParams
    child_requests: set[str]
    output_aggregator: list[CompletionOutput]
    max_num_generation_tokens: int
    cached_child_sampling_params: Optional[SamplingParams]

    def __init__(self, request_id: str, sampling_params: SamplingParams) -> None:
        self.request_id = request_id
        self.sampling_params = sampling_params
        self.child_requests = set()
        self.output_aggregator = [None] * sampling_params.n if sampling_params.output_kind == RequestOutputKind.FINAL_ONLY else []
        self.max_num_generation_tokens = 0
        self.cached_child_sampling_params = None

    def _get_child_sampling_params(self, index: int) -> SamplingParams:
        """Efficiently obtain child `sampling_params`

        If `sampling_params.seed` is not `None` then 
        each child request requires a unique clone of
        parent `sampling_params` with a unique seed.

        Args:
          index: index within `n` child requests

        Returns:
          Child `sampling_params` instance.
        """
        seed = self.sampling_params.seed
        if self.cached_child_sampling_params:
            return self.cached_child_sampling_params
        child_sampling_params = copy(self.sampling_params)
        child_sampling_params.n = 1
        if seed is None:
            self.cached_child_sampling_params = child_sampling_params
        else:
            child_sampling_params.seed = seed + index
        return child_sampling_params

    def get_child_info(self, index: int) -> tuple[str, SamplingParams]:
        """Get child request ID and sampling params.
        
        Args:
          index: index within `n` child requests.
        
        Returns:
          (request ID, sampling_params) tuple
        """
        child_req_id = f'{index}_{self.request_id}'
        self.child_requests.add(child_req_id)
        return (child_req_id, self._get_child_sampling_params(index))

    @property
    def n(self) -> int:
        return self.sampling_params.n

    def get_outputs(self, child_request_id: str, completion_output: CompletionOutput) -> tuple[str, list[CompletionOutput], bool]:
        if completion_output.finished():
            self.child_requests.remove(child_request_id)
        if self.sampling_params.output_kind != RequestOutputKind.FINAL_ONLY:
            outputs = [completion_output]
        else:
            self.output_aggregator[completion_output.index] = completion_output
            outputs = [] if self.child_requests else self.output_aggregator
        finished = not self.child_requests
        return (self.request_id, outputs, finished)

    def observe_num_generation_tokens(self, num_generation_tokens: int):
        self.max_num_generation_tokens = max(num_generation_tokens, self.max_num_generation_tokens)
        return self.max_num_generation_tokens

    @staticmethod
    def observe_finished_request(parent_req: Optional['ParentRequest'], iteration_stats: IterationStats, num_generation_tokens: int):
        n_param = parent_req.n if parent_req is not None else 1
        if parent_req is not None:
            num_generation_tokens = parent_req.observe_num_generation_tokens(num_generation_tokens)
        if parent_req is None or not parent_req.child_requests:
            iteration_stats.max_num_generation_tokens_iter.append(num_generation_tokens)
            iteration_stats.n_params_iter.append(n_param)