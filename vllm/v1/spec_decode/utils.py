from vllm.sampling_params import SamplingParams
_SAMPLING_EPS = 1e-05

def is_spec_decode_unsupported(sampling_params: SamplingParams) -> bool:
    """True if request is incompatible with speculative decoding"""
    return sampling_params.frequency_penalty != 0.0 or sampling_params.presence_penalty != 0.0 or sampling_params.repetition_penalty != 1.0 or (sampling_params.min_p > _SAMPLING_EPS) or (sampling_params.logprobs is not None)