from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Optional
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.reasoning import ReasoningParser
from vllm.reasoning import ReasoningParserManager
from vllm.transformers_utils.tokenizer_group import init_tokenizer_from_configs
from vllm.utils import LazyLoader
from vllm.v1.request import Request
from vllm.v1.structured_output.backend_guidance import GuidanceBackend
from vllm.v1.structured_output.backend_outlines import OutlinesBackend
from vllm.v1.structured_output.backend_types import StructuredOutputBackend, StructuredOutputGrammar
from vllm.v1.structured_output.backend_xgrammar import XgrammarBackend
import multiprocessing
import numpy as np
import numpy.typing as npt
import torch
if TYPE_CHECKING:
else:
    torch = LazyLoader('torch', globals(), 'torch')
logger = init_logger(__name__)

class StructuredOutputManager:
    """Engine-level manager for structured output requests."""

    def __init__(self, vllm_config: VllmConfig):
        self.backend: Optional[StructuredOutputBackend] = None
        self.reasoner: Optional[ReasoningParser] = None
        self.vllm_config = vllm_config
        self._grammar_bitmask: Optional[torch.Tensor] = None
        self._full_mask = torch.tensor(-1, dtype=torch.int32)
        if not self.vllm_config.model_config.skip_tokenizer_init:
            max_workers = max(1, (multiprocessing.cpu_count() + 1) // 2)
            self.executor = ThreadPoolExecutor(max_workers=max_workers)
            self.tokenizer = init_tokenizer_from_configs(model_config=self.vllm_config.model_config, scheduler_config=self.vllm_config.scheduler_config, lora_config=self.vllm_config.lora_config).get_lora_tokenizer(None)
            reasoning_backend = self.vllm_config.decoding_config.reasoning_backend
            if reasoning_backend:
                reasoner_cls = ReasoningParserManager.get_reasoning_parser(reasoning_backend)
                self.reasoner = reasoner_cls(tokenizer=self.tokenizer)

    def grammar_init(self, request: Request) -> None:
        if request.structured_output_request is None:
            return
        if TYPE_CHECKING:
            assert request.sampling_params is not None and request.sampling_params.guided_decoding is not None
        if self.backend is None:
            assert request.sampling_params is not None
            backend = request.sampling_params.guided_decoding.backend
            vocab_size = self.vllm_config.model_config.get_vocab_size()
            if backend == 'xgrammar':
                self.backend = XgrammarBackend(self.vllm_config, tokenizer=self.tokenizer, vocab_size=vocab_size)
            elif backend == 'guidance':
                self.backend = GuidanceBackend(self.vllm_config, tokenizer=self.tokenizer, vocab_size=vocab_size)
            elif backend == 'outlines':
                self.backend = OutlinesBackend(self.vllm_config, tokenizer=self.tokenizer, vocab_size=vocab_size)
            else:
                raise ValueError(f'Unsupported structured output backend: {backend}')
        grammar = self.executor.submit(self._async_create_grammar, request)
        request.structured_output_request.grammar = grammar

    def _async_create_grammar(self, request: Request) -> StructuredOutputGrammar:
        key = request.structured_output_request.structured_output_key
        request_type, grammar_spec = key
        assert self.backend is not None
        return self.backend.compile_grammar(request_type, grammar_spec)

    def grammar_bitmask(self, requests: dict[str, Request], structured_output_request_ids: dict[str, int], scheduled_spec_decode_tokens: dict[str, list[int]]) -> Optional[npt.NDArray[np.int32]]:
        if not structured_output_request_ids:
            return None
        max_num_spec_tokens = 0
        if self.vllm_config.speculative_config is not None:
            max_num_spec_tokens = self.vllm_config.speculative_config.num_speculative_tokens
        if self._grammar_bitmask is None:
            assert self.backend is not None
            max_batch_size = self.vllm_config.scheduler_config.max_num_seqs
            self._grammar_bitmask = self.backend.allocate_token_bitmask(max_batch_size * (1 + max_num_spec_tokens))
        bitmask_tensor = self._grammar_bitmask
        cumulative_index = 0
        ordered_seq = sorted(structured_output_request_ids.items(), key=lambda x: x[1])
        bitmask_tensor[:len(ordered_seq) * (1 + max_num_spec_tokens)].fill_(self._full_mask)
        for req_id, _ in ordered_seq:
            request = requests[req_id]
            structured_output_request = request.structured_output_request
            if TYPE_CHECKING:
                assert structured_output_request is not None
                assert structured_output_request.grammar is not None
            apply_bitmask: bool = True
            if self.reasoner is not None:
                if structured_output_request.reasoning_ended is None:
                    structured_output_request.reasoning_ended = self.reasoner.is_reasoning_end(request.prompt_token_ids)
                apply_bitmask = structured_output_request.reasoning_ended
            state_advancements = 0
            req_tokens = scheduled_spec_decode_tokens.get(req_id, []) + [None]
            for i, token in enumerate(req_tokens):
                if apply_bitmask and (not structured_output_request.grammar.is_terminated()):
                    structured_output_request.grammar.fill_bitmask(bitmask_tensor, cumulative_index)
                    if token is not None:
                        assert structured_output_request.grammar.accept_tokens(req_id, [token])
                        state_advancements += 1
                cumulative_index += 1
            if state_advancements > 0:
                structured_output_request.grammar.rollback(state_advancements)
        if cumulative_index < bitmask_tensor.shape[0]:
            bitmask_tensor = bitmask_tensor[:cumulative_index]
        return bitmask_tensor.numpy()

    def should_advance(self, request: Request) -> bool:
        if not request.use_structured_output:
            return False
        if TYPE_CHECKING:
            assert request.structured_output_request is not None
            assert request.structured_output_request.grammar is not None
        if self.reasoner is not None:
            structured_req = request.structured_output_request
            if structured_req.reasoning_ended:
                return True
            if self.reasoner.is_reasoning_end(request.all_token_ids):
                structured_req.reasoning_ended = True
            return False
        else:
            return True

    def clear_backend(self) -> None:
        if self.backend is not None:
            self.backend.destroy()