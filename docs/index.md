# Agent Governor

Lightweight Python Agent governance and execution middleware.

## Features

- **Budget Control** — Set per-call and total budget limits
- **Model Routing** — Whitelist enforcement, fallback model routing
- **Audit Logging** — Structured JSON audit logs for all LLM calls
- **Cost Tracking** — Per-model cost calculation and tracking
- **Multi-Tenant** — Per-tenant budgets, rate limits, model isolation
- **Policy Engine** — Declarative YAML/JSON rules (budget, rate_limit, model_whitelist)
- **Quality Gates** — Validation pipeline (schema, budget, LLM-as-judge)
- **DAG Orchestration** — Multi-step task graphs with parallel execution
- **Resilience** — Circuit breaker, concurrency control, rate limiting
- **Observability** — OpenTelemetry tracing and metrics (optional)
- **Extensible Executors** — LLM, MCP, and HTTP execution backends *(deprecated, removal in v1.0)*

## Installation

```bash
pip install agent-governor
```

### Optional Dependencies

```bash
# DAG workflows (networkx)
pip install "agent-governor[dag]"

# Telemetry (OpenTelemetry)
pip install "agent-governor[telemetry]"

# Redis state backend
pip install "agent-governor[redis]"

# Everything
pip install "agent-governor[all]"
```

!!! warning "Deprecated"
    `mcp` and `http` extras are deprecated and will be removed in v1.0.
    Use `mcp` SDK and `httpx` directly instead.

## Quick Start

```python
import asyncio
from agent_governor import Governor

async def main():
    governor = Governor(
        budget_usd=5.0,
        models=["gpt-4o", "gpt-3.5-turbo"],
        fallback_model="gpt-3.5-turbo",
    )
    result = await governor.execute(
        messages=[{"role": "user", "content": "Hello!"}],
        model="gpt-4o",
    )
    print(result)

asyncio.run(main())
```
