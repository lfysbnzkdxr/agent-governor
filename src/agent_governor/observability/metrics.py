"""OTel metrics collection with no-op fallback."""

from __future__ import annotations

from typing import Any

try:
    from opentelemetry import metrics
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

    OTEL_METRICS_AVAILABLE = True
except ImportError:
    OTEL_METRICS_AVAILABLE = False


class MetricsCollector:
    """OTel metrics with graceful degradation.

    Records call count, errors, latency, and token usage as OTel metrics.
    If OTel packages are not installed, all methods are no-ops.

    Metric naming convention:
        - agent_governor.calls.total (Counter, labels: model, status)
        - agent_governor.errors.total (Counter, labels: model, error_type)
        - agent_governor.budget_exceeded.total (Counter)
        - agent_governor.latency_ms (Histogram, labels: model)
        - agent_governor.tokens_per_call (Histogram, labels: model)
    """

    def __init__(self, meter_name: str = "agent_governor", otlp_endpoint: str = "") -> None:
        self._meter_name = meter_name
        self._otlp_endpoint = otlp_endpoint
        self._meter: Any = None
        self._provider: Any = None
        self._instruments: dict[str, Any] = {}
        self._initialized = False
        if OTEL_METRICS_AVAILABLE:
            self._init()

    def _init(self) -> None:
        """Initialize the OTel meter provider."""
        if self._initialized:
            return
        if self._otlp_endpoint:
            reader = PeriodicExportingMetricReader(
                OTLPMetricExporter(endpoint=self._otlp_endpoint),
                export_interval_millis=60000,
            )
            provider = MeterProvider(metric_readers=[reader])
        else:
            provider = MeterProvider()
        self._provider = provider
        self._meter = provider.get_meter(self._meter_name)
        self._initialized = True

    def _get_counter(self, name: str, unit: str = "", description: str = "") -> Any:
        """Get or create a Counter instrument."""
        if name not in self._instruments:
            if self._meter:
                self._instruments[name] = self._meter.create_counter(
                    name, unit=unit, description=description,
                )
            else:
                return None
        return self._instruments[name]

    def _get_histogram(self, name: str, unit: str = "", description: str = "") -> Any:
        """Get or create a Histogram instrument."""
        if name not in self._instruments:
            if self._meter:
                self._instruments[name] = self._meter.create_histogram(
                    name, unit=unit, description=description,
                )
            else:
                return None
        return self._instruments[name]

    def record_call(self, model: str, status: str) -> None:
        """Record an LLM call."""
        if not OTEL_METRICS_AVAILABLE:
            return
        counter = self._get_counter(
            "agent_governor.calls.total",
            description="Total LLM calls",
        )
        if counter:
            counter.add(1, {"model": model, "status": status})

    def record_error(self, model: str, error_type: str) -> None:
        """Record an LLM error."""
        if not OTEL_METRICS_AVAILABLE:
            return
        counter = self._get_counter(
            "agent_governor.errors.total",
            description="Total LLM errors",
        )
        if counter:
            counter.add(1, {"model": model, "error_type": error_type})

    def record_budget_exceeded(self, model: str = "") -> None:
        """Record a budget exceeded event."""
        if not OTEL_METRICS_AVAILABLE:
            return
        counter = self._get_counter(
            "agent_governor.budget_exceeded.total",
            description="Budget exceeded events",
        )
        if counter:
            counter.add(1, {"model": model})

    def record_latency(self, model: str, latency_ms: float) -> None:
        """Record call latency in milliseconds."""
        if not OTEL_METRICS_AVAILABLE:
            return
        histogram = self._get_histogram(
            "agent_governor.latency_ms",
            unit="ms",
            description="Call latency in milliseconds",
        )
        if histogram:
            histogram.record(latency_ms, {"model": model})

    def record_tokens(self, model: str, tokens: int) -> None:
        """Record token usage per call."""
        if not OTEL_METRICS_AVAILABLE:
            return
        histogram = self._get_histogram(
            "agent_governor.tokens_per_call",
            description="Tokens per call",
        )
        if histogram:
            histogram.record(tokens, {"model": model})

    def shutdown(self) -> None:
        """Shutdown the meter provider."""
        if OTEL_METRICS_AVAILABLE and self._provider:
            self._provider.shutdown()
