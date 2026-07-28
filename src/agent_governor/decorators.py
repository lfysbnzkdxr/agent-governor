"""Decorators for Agent Governor - @governed decorator for simple usage."""

from __future__ import annotations

from typing import Any, Callable, TypeVar

from agent_governor.governor import Governor

F = TypeVar("F", bound=Callable[..., Any])


def governed(governor: Governor) -> Callable[[F], F]:
    """Decorator that marks a function as governed by Agent Governor.

    Attaches the governor instance to the function for introspection
    and provides a clean semantic marker in code.

    Example:
        governor = Governor(budget_usd=10.0, models=["gpt-4o"])

        @governed(governor)
        async def answer_question(question: str) -> str:
            return await governor.execute(
                messages=[{"role": "user", "content": question}],
                model="gpt-4o",
                name="answer_question",
            )

    Args:
        governor: The Governor instance to associate with the function.

    Returns:
        Decorator function.
    """

    def decorator(func: F) -> F:
        # Attach governor reference for introspection
        setattr(func, "_governor", governor)
        return func

    return decorator
