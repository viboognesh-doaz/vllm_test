from collections.abc import Mapping
from opentelemetry.context.context import Context
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.environment_variables import OTEL_EXPORTER_OTLP_TRACES_PROTOCOL
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import SpanKind, Tracer, set_tracer_provider
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from typing import Optional
from vllm.logger import init_logger
from vllm.utils import run_once
import os
import traceback
TRACE_HEADERS = ['traceparent', 'tracestate']
logger = init_logger(__name__)
_is_otel_imported = False
otel_import_error_traceback: Optional[str] = None
try:
    _is_otel_imported = True
except ImportError:
    otel_import_error_traceback = traceback.format_exc()

    class Context:
        pass

    class BaseSpanAttributes:
        pass

    class SpanKind:
        pass

    class Tracer:
        pass

def is_otel_available() -> bool:
    return _is_otel_imported

def init_tracer(instrumenting_module_name: str, otlp_traces_endpoint: str) -> Optional[Tracer]:
    if not is_otel_available():
        raise ValueError(f'OpenTelemetry is not available. Unable to initialize a tracer. Ensure OpenTelemetry packages are installed. Original error:\n{otel_import_error_traceback}')
    trace_provider = TracerProvider()
    span_exporter = get_span_exporter(otlp_traces_endpoint)
    trace_provider.add_span_processor(BatchSpanProcessor(span_exporter))
    set_tracer_provider(trace_provider)
    tracer = trace_provider.get_tracer(instrumenting_module_name)
    return tracer

def get_span_exporter(endpoint):
    protocol = os.environ.get(OTEL_EXPORTER_OTLP_TRACES_PROTOCOL, 'grpc')
    if protocol == 'grpc':
    elif protocol == 'http/protobuf':
    else:
        raise ValueError(f"Unsupported OTLP protocol '{protocol}' is configured")
    return OTLPSpanExporter(endpoint=endpoint)

def extract_trace_context(headers: Optional[Mapping[str, str]]) -> Optional[Context]:
    if is_otel_available():
        headers = headers or {}
        return TraceContextTextMapPropagator().extract(headers)
    else:
        return None

def extract_trace_headers(headers: Mapping[str, str]) -> Mapping[str, str]:
    return {h: headers[h] for h in TRACE_HEADERS if h in headers}

class SpanAttributes:
    GEN_AI_USAGE_COMPLETION_TOKENS = 'gen_ai.usage.completion_tokens'
    GEN_AI_USAGE_PROMPT_TOKENS = 'gen_ai.usage.prompt_tokens'
    GEN_AI_REQUEST_MAX_TOKENS = 'gen_ai.request.max_tokens'
    GEN_AI_REQUEST_TOP_P = 'gen_ai.request.top_p'
    GEN_AI_REQUEST_TEMPERATURE = 'gen_ai.request.temperature'
    GEN_AI_RESPONSE_MODEL = 'gen_ai.response.model'
    GEN_AI_REQUEST_ID = 'gen_ai.request.id'
    GEN_AI_REQUEST_N = 'gen_ai.request.n'
    GEN_AI_USAGE_NUM_SEQUENCES = 'gen_ai.usage.num_sequences'
    GEN_AI_LATENCY_TIME_IN_QUEUE = 'gen_ai.latency.time_in_queue'
    GEN_AI_LATENCY_TIME_TO_FIRST_TOKEN = 'gen_ai.latency.time_to_first_token'
    GEN_AI_LATENCY_E2E = 'gen_ai.latency.e2e'
    GEN_AI_LATENCY_TIME_IN_SCHEDULER = 'gen_ai.latency.time_in_scheduler'
    GEN_AI_LATENCY_TIME_IN_MODEL_FORWARD = 'gen_ai.latency.time_in_model_forward'
    GEN_AI_LATENCY_TIME_IN_MODEL_EXECUTE = 'gen_ai.latency.time_in_model_execute'

def contains_trace_headers(headers: Mapping[str, str]) -> bool:
    return any((h in headers for h in TRACE_HEADERS))

@run_once
def log_tracing_disabled_warning() -> None:
    logger.warning('Received a request with trace context but tracing is disabled')