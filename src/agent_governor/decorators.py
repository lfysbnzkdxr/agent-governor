"""Decorators for Agent Governor - @governed decorator for simple usage."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from agent_governor.governor import Governor

F = TypeVar("F", bound=Callable[..., Any])


class _GovernedFunc:
    """Wrapper to attach governor reference to a function."""

    def __init__(self, func: F, governor: Governor) -> None:
        self._func = func
        self._governor = governor

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return await self._func(*args, **kwargs)


def governed(governor: Governor) -> Callable[[F], _GovernedFunc]:
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

    def decorator(func: F) -> _GovernedFunc:
        return _GovernedFunc(func, governor)

    return decorator
