"""Observability layer - tracing, audit logging, and cost tracking."""

from agent_governor.observability.audit import AuditLogger
from agent_governor.observability.cost import CostTracker

__all__ = [
    "AuditLogger",
    "CostTracker",
]

try:
    from agent_governor.observability.metrics import OTEL_METRICS_AVAILABLE, MetricsCollector
    from agent_governor.observability.tracing import (
        OTEL_AVAILABLE,
        TracedExecutor,
        TracingManager,
        traced,
    )

    __all__ += [
        "MetricsCollector",
        "OTEL_METRICS_AVAILABLE",
        "TracingManager",
        "TracedExecutor",
        "traced",
        "OTEL_AVAILABLE",
    ]
except ImportError:
    pass
