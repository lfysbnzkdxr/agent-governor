"""Quickstart example for Agent Governor.

Demonstrates the simplest usage: replace litellm.completion() with governor.execute()
to get budget control + audit logging + model routing.

Prerequisites:
    pip install agent-governor
    export OPENAI_API_KEY="your-key"  # or any litellm-supported provider
"""

import asyncio

from agent_governor import Governor, governed


async def main() -> None:
    # === Level 1: Simplest usage ===
    governor = Governor(
        budget_usd=1.0,  # $1 budget limit
        models=["gpt-4o", "gpt-4o-mini"],  # Model whitelist
        fallback_model="gpt-4o-mini",  # Fallback on failure
        default_model="gpt-4o",
    )

    # Direct execute - returns content string
    print("=== Direct Execute ===")
    response = await governor.execute(
        messages=[{"role": "user", "content": "What is 2+2? Answer in one word."}],
        model="gpt-4o",
        name="math_question",
    )
    print(f"Response: {response}")

    # Check stats
    stats = governor.get_stats()
    print(f"\nCost so far: ${stats['total_cost_usd']:.6f}")
    print(f"Budget remaining: ${stats['budget_remaining']:.6f}")

    # === Using @governed decorator ===
    print("\n=== @governed Decorator ===")

    @governed(governor)
    async def ask_ai(question: str) -> str:
        return await governor.execute(
            messages=[{"role": "user", "content": question}],
            model="gpt-4o-mini",
            name="ask_ai",
        )

    answer = await ask_ai("What is the capital of France? One word answer.")
    print(f"Answer: {answer}")

    # === Advanced: Full TaskResult access ===
    print("\n=== Full TaskResult ===")
    from agent_governor.models import Task

    task = Task(
        name="detailed_call",
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say hello"}],
    )
    result = await governor.execute_task(task)
    print(f"Content: {result.content}")
    print(f"Model used: {result.model}")
    print(f"Tokens: {result.usage.total_tokens}")
    print(f"Cost: ${result.cost_usd:.6f}")
    print(f"Latency: {result.latency_ms:.1f}ms")

    # === Audit summary ===
    print("\n=== Audit Summary ===")
    summary = governor.audit_logger.get_summary()
    print(f"Total calls: {summary['total_calls']}")
    print(f"Total cost: ${summary['total_cost_usd']:.6f}")


if __name__ == "__main__":
    asyncio.run(main())
