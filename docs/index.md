# Agent Governor

Lightweight Python Agent governance and execution middleware.

## Features

- **Budget Control** — Set per-call and total budget limits
- **Model Routing** — Whitelist enforcement, fallback model routing
- **Audit Logging** — Structured JSON audit logs for all LLM calls
- **Cost Tracking** — Per-model cost calculation and tracking
- **Quality Gates** — Validation pipeline (schema, budget, LLM-as-judge)
- **DAG Orchestration** — Multi-step task graphs with parallel execution
- **Resilience** — Circuit breaker, concurrency control, rate limiting
- **Observability** — OpenTelemetry tracing and metrics (optional)
- **Extensible Executors** — LLM, MCP, and HTTP execution backends

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

# MCP tools
pip install "agent-governor[mcp]"

# HTTP executor
pip install "agent-governor[http]"

# Everything
pip install "agent-governor[all]"
```

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
