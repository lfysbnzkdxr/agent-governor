# Agent Governor

Lightweight Agent governance and execution middleware — cost control, audit logging, and model routing.

## Installation

```bash
pip install agent-governor
```

## Quick Start

```python
from agent_governor import Governor

governor = Governor(
    budget_usd=10.0,
    models=["gpt-4o", "qwen-plus"],
    fallback_model="qwen-turbo",
)

response = await governor.execute(
    messages=[{"role": "user", "content": "Hello!"}],
    model="gpt-4o",
)
```

## Features

- **Budget Control**: Set USD budget limits with pre-call enforcement
- **Model Whitelist**: Restrict which models can be called
- **Fallback Routing**: Automatic failover to backup models
- **Audit Logging**: Structured JSON logs for every LLM call
- **Cost Tracking**: Per-model token/cost accumulation
- **Concurrency Control**: Semaphore-based rate limiting

## License

MIT
