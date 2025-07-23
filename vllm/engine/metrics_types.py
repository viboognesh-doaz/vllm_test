from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List
from vllm.config import SupportsMetricsInfo, VllmConfig
import time
'\nThese types are defined in this file to avoid importing vllm.engine.metrics\nand therefore importing prometheus_client.\n\nThis is required due to usage of Prometheus multiprocess mode to enable \nmetrics after splitting out the uvicorn process from the engine process.\n\nPrometheus multiprocess mode requires setting PROMETHEUS_MULTIPROC_DIR\nbefore prometheus_client is imported. Typically, this is done by setting\nthe env variable before launch, but since we are a library, we need to\ndo this in Python code and lazily import prometheus_client.\n'

@dataclass
class Stats:
    """Created by LLMEngine for use by StatLogger."""
    now: float
    num_running_sys: int
    num_waiting_sys: int
    num_swapped_sys: int
    gpu_cache_usage_sys: float
    cpu_cache_usage_sys: float
    cpu_prefix_cache_hit_rate: float
    gpu_prefix_cache_hit_rate: float
    num_prompt_tokens_iter: int
    num_generation_tokens_iter: int
    num_tokens_iter: int
    time_to_first_tokens_iter: List[float]
    time_per_output_tokens_iter: List[float]
    num_preemption_iter: int
    time_e2e_requests: List[float]
    time_queue_requests: List[float]
    time_inference_requests: List[float]
    time_prefill_requests: List[float]
    time_decode_requests: List[float]
    num_prompt_tokens_requests: List[int]
    num_generation_tokens_requests: List[int]
    n_requests: List[int]
    max_num_generation_tokens_requests: List[int]
    max_tokens_requests: List[int]
    finished_reason_requests: List[str]
    waiting_lora_adapters: List[str]
    running_lora_adapters: List[str]
    max_lora: str

class StatLoggerBase(ABC):
    """Base class for StatLogger."""

    def __init__(self, local_interval: float, vllm_config: VllmConfig) -> None:
        self.num_prompt_tokens: List[int] = []
        self.num_generation_tokens: List[int] = []
        self.last_local_log = time.time()
        self.local_interval = local_interval

    @abstractmethod
    def log(self, stats: Stats) -> None:
        raise NotImplementedError

    @abstractmethod
    def info(self, type: str, obj: SupportsMetricsInfo) -> None:
        raise NotImplementedError