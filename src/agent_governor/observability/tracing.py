"""OTel distributed tracing with graceful no-op degradation."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = None


F = TypeVar("F", bound=Callable[..., Any])


class TracingManager:
    """Centralized OTel tracing with lazy init and no-op fallback.

    Creates a TracerProvider with optional OTLP exporter.
    If OTel packages are not installed, all methods are no-ops.

    Example:
        mgr = TracingManager(otlp_endpoint="http://localhost:4317")
        tracer = mgr.get_tracer()
        with tracer.start_as_current_span("my_span"):
            ...
        mgr.shutdown()
    """

    def __init__(
        self,
        service_name: str = "agent-governor",
        otlp_endpoint: str = "",
    ) -> None:
        self._service_name = service_name
        self._otlp_endpoint = otlp_endpoint
        self._tracer: Any = None
        self._provider: Any = None
        self._initialized = False
        if OTEL_AVAILABLE:
            self._init()

    def _init(self) -> None:
        """Initialize the OTel tracer provider with BatchSpanProcessor."""
        if self._initialized:
            return
        resource = Resource.create({"service.name": self._service_name})
        provider = TracerProvider(resource=resource)
        if self._otlp_endpoint:
            exporter = OTLPSpanExporter(endpoint=self._otlp_endpoint)
            processor = BatchSpanProcessor(
                exporter,
                max_queue_size=2048,
                max_export_batch_size=512,
                schedule_delay_millis=5000,
            )
            provider.add_span_processor(processor)
        self._provider = provider
        self._tracer = provider.get_tracer("agent_governor")
        self._initialized = True

    def get_tracer(self) -> Any:
        """Get the OTel tracer. Returns None if OTel is unavailable."""
        if not OTEL_AVAILABLE:
            return None
        return self._tracer

    def shutdown(self) -> None:
        """Shutdown the tracer provider, flushing any pending spans."""
        if OTEL_AVAILABLE and self._provider:
            self._provider.shutdown()


def traced(
    span_name: str = "",
    attributes: dict[str, str] | None = None,
) -> Callable[[F], F]:
    """Decorator: wraps async function in an OTel span. No-op if OTel absent.

    Args:
        span_name: Span name. Defaults to function name.
        attributes: Static attributes to set on the span.

    Example:
        @traced(span_name="llm_call", attributes={"service": "chat"})
        async def my_function(...):
            ...
    """
    attrs = attributes or {}

    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not OTEL_AVAILABLE:
                return await func(*args, **kwargs)
            tracer = trace.get_tracer("agent_governor")
            name = span_name or func.__name__
            with tracer.start_as_current_span(name) as span:
                for k, v in attrs.items():
                    span.set_attribute(k, v)
                result = await func(*args, **kwargs)
                if result is not None and hasattr(result, "status"):
                    span.set_attribute("agent_governor.status", str(result.status))
                return result

        return wrapper  # type: ignore[return-value]

    return decorator


class TracedExecutor:
    """Wraps any Executor protocol impl, adding a child span per execute().

    Usage:
        executor = TracedExecutor(inner=llm_executor, tracing=mgr)
        result = await executor.execute(task)
    """

    def __init__(self, inner: Any, tracing: TracingManager | None = None) -> None:
        self._inner = inner
        self._tracing = tracing

    async def execute(self, task: Any) -> Any:
        """Execute with a child tracing span."""
        if not OTEL_AVAILABLE:
            return await self._inner.execute(task)

        tracer = self._tracing.get_tracer() if self._tracing else trace.get_tracer("agent_governor")
        if not tracer:
            return await self._inner.execute(task)
        task_name = getattr(task, "name", "unknown")
        with tracer.start_as_current_span(f"executor.{task_name}") as span:
            span.set_attribute("agent_governor.task.name", task_name)
            model = getattr(task, "model", "")
            if model:
                span.set_attribute("agent_governor.model", model)
            try:
                result = await self._inner.execute(task)
                if hasattr(result, "status"):
                    span.set_attribute("agent_governor.status", str(result.status))
                return result
            except Exception as e:
                span.set_attribute("agent_governor.status", "error")
                span.record_exception(e)
                raise
