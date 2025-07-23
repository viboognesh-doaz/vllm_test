from __future__ import annotations
from transformers import PreTrainedTokenizer
from typing import TYPE_CHECKING
from vllm.config import ModelConfig
from vllm.logger import init_logger
from vllm.logits_process import LogitsProcessor
from vllm.model_executor.guided_decoding.guidance_decoding import get_local_guidance_guided_decoding_logits_processor
from vllm.model_executor.guided_decoding.guidance_decoding import get_local_guidance_guided_decoding_logits_processor
from vllm.model_executor.guided_decoding.lm_format_enforcer_decoding import get_local_lm_format_enforcer_guided_decoding_logits_processor
from vllm.model_executor.guided_decoding.lm_format_enforcer_decoding import get_local_lm_format_enforcer_guided_decoding_logits_processor
from vllm.model_executor.guided_decoding.outlines_decoding import get_local_outlines_guided_decoding_logits_processor
from vllm.model_executor.guided_decoding.outlines_decoding import get_outlines_guided_decoding_logits_processor
from vllm.model_executor.guided_decoding.utils import convert_lark_to_gbnf, grammar_is_likely_lark, has_lmf_unsupported_json_features, has_xgrammar_unsupported_json_features
from vllm.model_executor.guided_decoding.xgrammar_decoding import get_local_xgrammar_guided_decoding_logits_processor
from vllm.model_executor.guided_decoding.xgrammar_decoding import get_local_xgrammar_guided_decoding_logits_processor
from vllm.model_executor.guided_decoding.xgrammar_decoding import xgr_installed
from vllm.reasoning import ReasoningParserManager
from vllm.sampling_params import GuidedDecodingParams
if TYPE_CHECKING:
logger = init_logger(__name__)

def maybe_backend_fallback(guided_params: GuidedDecodingParams) -> GuidedDecodingParams:

    def fallback_or_error(guided_params: GuidedDecodingParams, message: str, fallback: str) -> None:
        """Change the backend to the specified fallback with a warning log,
        or raise a ValueError if the `disable_fallback` option is specified."""
        if guided_params.disable_fallback:
            raise ValueError(message)
        logger.warning('%s Falling back to use %s instead.', message, fallback)
        guided_params.backend = fallback
    if guided_params.backend == 'auto':
        guided_params.backend = 'xgrammar'
    if guided_params.backend == 'lm-format-enforcer':
        if guided_params.grammar is not None:
            fallback_or_error(guided_params, 'lm-format-enforcer does not support grammar guided decoding.', 'xgrammar')
        elif guided_params.json is not None and has_lmf_unsupported_json_features(guided_params.json):
            fallback_or_error(guided_params, 'lm-format-enforcer does not support advanced JSON schema features like patterns or numeric ranges.', 'outlines')
    if guided_params.backend == 'xgrammar':
        if guided_params.json is not None and has_xgrammar_unsupported_json_features(guided_params.json):
            fallback_or_error(guided_params, 'xgrammar does not support advanced JSON schema features like string length, item limits, or property bounds.', 'outlines')
        elif guided_params.grammar is not None and grammar_is_likely_lark(guided_params.grammar):
            try:
                convert_lark_to_gbnf(guided_params.grammar)
            except Exception:
                fallback_or_error(guided_params, 'xgrammar does not support Lark grammars and the grammar failed to convert to GBNF.', 'guidance')
        elif not xgr_installed:
            fallback_or_error(guided_params, 'xgrammar module cannot be imported successfully.', 'guidance')
    if guided_params.backend == 'outlines':
        if guided_params.json_object is not None:
            fallback_or_error(guided_params, 'outlines does not support json_object.', 'guidance')
        elif guided_params.grammar is not None:
            if grammar_is_likely_lark(guided_params.grammar):
                fallback_or_error(guided_params, 'outlines no longer supports grammars.', 'guidance')
            else:
                fallback_or_error(guided_params, 'outlines no longer supports grammars.', 'xgrammar')
    return guided_params

async def get_guided_decoding_logits_processor(guided_params: GuidedDecodingParams, tokenizer: PreTrainedTokenizer, model_config: ModelConfig, reasoning_backend: str | None=None) -> LogitsProcessor | None:
    reasoner = None
    if reasoning_backend:
        reasoner_class = ReasoningParserManager.get_reasoning_parser(reasoning_backend)
        reasoner = reasoner_class(tokenizer)
    guided_params = maybe_backend_fallback(guided_params)
    if guided_params.backend == 'outlines':
        return await get_outlines_guided_decoding_logits_processor(guided_params, tokenizer, reasoner)
    if guided_params.backend == 'lm-format-enforcer':
        return get_local_lm_format_enforcer_guided_decoding_logits_processor(guided_params, tokenizer)
    if guided_params.backend == 'xgrammar':
        return get_local_xgrammar_guided_decoding_logits_processor(guided_params, tokenizer, model_config, reasoner)
    if guided_params.backend == 'guidance':
        return get_local_guidance_guided_decoding_logits_processor(guided_params, tokenizer)
    raise ValueError(f"Unknown guided decoding backend '{guided_params.backend}'. Must be one of 'outlines, 'lm-format-enforcer', 'xgrammar', 'guidance'")

def get_local_guided_decoding_logits_processor(guided_params: GuidedDecodingParams, tokenizer: PreTrainedTokenizer, model_config: ModelConfig, reasoning_backend: str | None=None) -> LogitsProcessor | None:
    guided_params = maybe_backend_fallback(guided_params)
    reasoner = None
    if reasoning_backend:
        reasoner_class = ReasoningParserManager.get_reasoning_parser(reasoning_backend)
        reasoner = reasoner_class(tokenizer)
    if guided_params.backend == 'outlines':
        return get_local_outlines_guided_decoding_logits_processor(guided_params, tokenizer, reasoner)
    if guided_params.backend == 'lm-format-enforcer':
        return get_local_lm_format_enforcer_guided_decoding_logits_processor(guided_params, tokenizer)
    if guided_params.backend == 'xgrammar':
        return get_local_xgrammar_guided_decoding_logits_processor(guided_params, tokenizer, model_config, reasoner)
    if guided_params.backend == 'guidance':
        return get_local_guidance_guided_decoding_logits_processor(guided_params, tokenizer)
    raise ValueError(f"Unknown guided decoding backend '{guided_params.backend}'. Must be one of 'outlines, 'lm-format-enforcer', 'xgrammar', 'guidance'")