from dataclasses import dataclass
from prometheus_client import Metric as PromMetric
from prometheus_client import REGISTRY
from prometheus_client.samples import Sample
from typing import Optional

@dataclass
class Metric:
    """A base class for prometheus metrics.

    Each metric may be associated with key=value labels, and
    in some cases a single vLLM instance may have multiple
    metrics with the same name but different sets of labels.
    """
    name: str
    labels: dict[str, str]

@dataclass
class Counter(Metric):
    """A monotonically increasing integer counter."""
    value: int

@dataclass
class Vector(Metric):
    """An ordered array of integer counters.

    This type - which doesn't exist in Prometheus - models one very
    specific metric, vllm:spec_decode_num_accepted_tokens_per_pos.
    """
    values: list[int]

@dataclass
class Gauge(Metric):
    """A numerical value that can go up or down."""
    value: float

@dataclass
class Histogram(Metric):
    """Observations recorded in configurable buckets.

    Buckets are represented by a dictionary. The key is
    the upper limit of the bucket, and the value is the
    observed count in that bucket. A '+Inf' key always
    exists.

    The count property is the total count across all
    buckets, identical to the count of the '+Inf' bucket.

    The sum property is the total sum of all observed
    values.
    """
    count: int
    sum: float
    buckets: dict[str, int]

def get_metrics_snapshot() -> list[Metric]:
    """An API for accessing in-memory Prometheus metrics.

    Example:
        >>> for metric in llm.get_metrics():
        ...     if isinstance(metric, Counter):
        ...         print(f"{metric} = {metric.value}")
        ...     elif isinstance(metric, Gauge):
        ...         print(f"{metric} = {metric.value}")
        ...     elif isinstance(metric, Histogram):
        ...         print(f"{metric}")
        ...         print(f"    sum = {metric.sum}")
        ...         print(f"    count = {metric.count}")
        ...         for bucket_le, value in metrics.buckets.items():
        ...             print(f"    {bucket_le} = {value}")
    """
    collected: list[Metric] = []
    for metric in REGISTRY.collect():
        if not metric.name.startswith('vllm:'):
            continue
        if metric.type == 'gauge':
            samples = _get_samples(metric)
            for s in samples:
                collected.append(Gauge(name=metric.name, labels=s.labels, value=s.value))
        elif metric.type == 'counter':
            samples = _get_samples(metric, '_total')
            if metric.name == 'vllm:spec_decode_num_accepted_tokens_per_pos':
                for labels, values in _digest_num_accepted_by_pos_samples(samples):
                    collected.append(Vector(name=metric.name, labels=labels, values=values))
            else:
                for s in samples:
                    collected.append(Counter(name=metric.name, labels=s.labels, value=int(s.value)))
        elif metric.type == 'histogram':
            bucket_samples = _get_samples(metric, '_bucket')
            count_samples = _get_samples(metric, '_count')
            sum_samples = _get_samples(metric, '_sum')
            for labels, buckets, count_value, sum_value in _digest_histogram(bucket_samples, count_samples, sum_samples):
                collected.append(Histogram(name=metric.name, labels=labels, buckets=buckets, count=count_value, sum=sum_value))
        else:
            raise AssertionError(f'Unknown metric type {metric.type}')
    return collected

def _get_samples(metric: PromMetric, suffix: Optional[str]=None) -> list[Sample]:
    name = metric.name + suffix if suffix is not None else metric.name
    return [s for s in metric.samples if s.name == name]

def _strip_label(labels: dict[str, str], key_to_remove: str) -> dict[str, str]:
    labels_copy = labels.copy()
    labels_copy.pop(key_to_remove)
    return labels_copy

def _digest_histogram(bucket_samples: list[Sample], count_samples: list[Sample], sum_samples: list[Sample]) -> list[tuple[dict[str, str], dict[str, int], int, float]]:
    buckets_by_labels: dict[frozenset[tuple[str, str]], dict[str, int]] = {}
    for s in bucket_samples:
        bucket = s.labels['le']
        labels_key = frozenset(_strip_label(s.labels, 'le').items())
        if labels_key not in buckets_by_labels:
            buckets_by_labels[labels_key] = {}
        buckets_by_labels[labels_key][bucket] = int(s.value)
    counts_by_labels: dict[frozenset[tuple[str, str]], int] = {}
    for s in count_samples:
        labels_key = frozenset(s.labels.items())
        counts_by_labels[labels_key] = int(s.value)
    sums_by_labels: dict[frozenset[tuple[str, str]], float] = {}
    for s in sum_samples:
        labels_key = frozenset(s.labels.items())
        sums_by_labels[labels_key] = s.value
    assert set(buckets_by_labels.keys()) == set(counts_by_labels.keys()) == set(sums_by_labels.keys())
    output = []
    label_keys = list(buckets_by_labels.keys())
    for k in label_keys:
        labels = dict(k)
        output.append((labels, buckets_by_labels[k], counts_by_labels[k], sums_by_labels[k]))
    return output

def _digest_num_accepted_by_pos_samples(samples: list[Sample]) -> list[tuple[dict[str, str], list[int]]]:
    max_pos = 0
    values_by_labels: dict[frozenset[tuple[str, str]], dict[int, int]] = {}
    for s in samples:
        position = int(s.labels['position'])
        max_pos = max(max_pos, position)
        labels_key = frozenset(_strip_label(s.labels, 'position').items())
        if labels_key not in values_by_labels:
            values_by_labels[labels_key] = {}
        values_by_labels[labels_key][position] = int(s.value)
    output = []
    for labels_key, values_by_position in values_by_labels.items():
        labels = dict(labels_key)
        values = [0] * (max_pos + 1)
        for pos, val in values_by_position.items():
            values[pos] = val
        output.append((labels, values))
    return output