# Observability

Agent Governor provides OpenTelemetry-based observability for distributed
tracing and metrics collection.

## Tracing

Enable tracing via configuration:

```python
from agent_governor import Governor

governor = Governor(
    tracing_enabled=True,
    otlp_endpoint="http://localhost:4317",  # OTLP collector
)
```

When OTel packages are not installed, tracing is automatically disabled
without any import errors.

### Span Coverage

Tracing spans are created for:
- `governor.execute()` — Top-level LLM call span
- `dispatcher.dispatch()` — Dispatcher routing span
- `dag.node.<node_id>` — Per-node DAG execution span

### Manual Tracing

```python
from agent_governor.observability.tracing import traced

@traced(span_name="my_custom_span", attributes={"key": "value"})
async def my_function():
    ...

# Or use TracedExecutor to wrap any executor
from agent_governor.observability.tracing import TracedExecutor

traced_exec = TracedExecutor(inner=my_executor)
result = await traced_exec.execute(task)
```

## Metrics

Metrics are collected automatically when tracing is enabled:

```python
governor = Governor(tracing_enabled=True)
```

### Available Metrics

| Metric Name | Type | Labels |
|------------|------|--------|
| `agent_governor.calls.total` | Counter | model, status |
| `agent_governor.errors.total` | Counter | model, error_type |
| `agent_governor.budget_exceeded.total` | Counter | model |
| `agent_governor.latency_ms` | Histogram | model |
| `agent_governor.tokens_per_call` | Histogram | model |

## Audit Logging

Audit logs are written as JSON lines for easy ingestion:

```python
governor = Governor(
    audit_enabled=True,
    audit_log_file="audit.log",  # File path, empty = stdout
)
```

### Cost Report CLI

Generate cost reports from audit logs:

```bash
# Install with optional dependencies
pip install "agent-governor[all]"

# Run report
agent-governor-report audit.log

# JSON output grouped by caller
agent-governor-report audit.log --format json --group-by caller

# Limit records
agent-governor-report audit.log --limit 100
```
