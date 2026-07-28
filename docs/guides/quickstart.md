# Getting Started

## Basic Usage

The `Governor` class is the main entry point:

```python
from agent_governor import Governor

governor = Governor(
    budget_usd=10.0,              # Total budget limit
    models=["gpt-4o", "qwen-plus"],  # Model whitelist
    fallback_model="qwen-turbo",  # Fallback on failure
)
```

### Async Execution

```python
result = await governor.execute(
    messages=[{"role": "user", "content": "Explain quantum computing"}],
    model="gpt-4o",
    name="my_query",  # For audit logging
)
print(result)
```

### Synchronous Execution

```python
result = governor.execute_sync(
    messages=[{"role": "user", "content": "Hello!"}],
    model="gpt-4o",
)
```

### Advanced Usage

For access to full result metadata (tokens, cost, latency):

```python
from agent_governor.models import Task

task = Task(
    name="my_task",
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
)
result = await governor.execute_task(task)
print(f"Tokens: {result.usage.total_tokens}")
print(f"Cost: ${result.cost_usd:.6f}")
print(f"Latency: {result.latency_ms:.1f}ms")
```

### Using the @governed Decorator

```python
from agent_governor import Governor, governed

governor = Governor(budget_usd=5.0)

@governed(governor)
async def ask_llm(prompt: str) -> str:
    """This function's LLM calls are automatically governed."""
    ...
```

## Configuration

```python
from agent_governor import GovernorConfig

config = GovernorConfig(
    budget_usd=10.0,
    models=["gpt-4o"],
    default_model="gpt-4o",
    max_concurrency=5,
    default_timeout=30.0,
    audit_enabled=True,
    tracing_enabled=True,
    otlp_endpoint="http://localhost:4317",
)
governor = Governor(config=config)
```
