"""Tests for @governed decorator and _GovernedFunc wrapper."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_governor.decorators import _GovernedFunc, governed


class TestGovernedFunc:
    """Tests for _GovernedFunc wrapper."""

    @pytest.mark.asyncio
    async def test_call_delegates_to_wrapped_func(self) -> None:
        """_GovernedFunc.__call__ should delegate to the wrapped function."""
        mock_func = AsyncMock(return_value="result")
        mock_governor = MagicMock()

        wrapped = _GovernedFunc(mock_func, mock_governor)
        result = await wrapped("arg1", key="val")

        assert result == "result"
        mock_func.assert_awaited_once_with("arg1", key="val")

    @pytest.mark.asyncio
    async def test_call_propagates_exceptions(self) -> None:
        """_GovernedFunc.__call__ should propagate exceptions from wrapped func."""
        mock_func = AsyncMock(side_effect=ValueError("boom"))
        mock_governor = MagicMock()

        wrapped = _GovernedFunc(mock_func, mock_governor)
        with pytest.raises(ValueError, match="boom"):
            await wrapped()

    def test_governor_attribute_accessible(self) -> None:
        """_GovernedFunc should expose the governor reference."""
        mock_func = AsyncMock()
        mock_governor = MagicMock()

        wrapped = _GovernedFunc(mock_func, mock_governor)
        assert wrapped._governor is mock_governor
        assert wrapped._func is mock_func


class TestGovernedDecorator:
    """Tests for the @governed decorator."""

    def test_returns_governed_func(self) -> None:
        """@governed should return a _GovernedFunc instance."""
        mock_governor = MagicMock()

        @governed(mock_governor)
        async def my_func() -> str:
            return "hello"

        assert isinstance(my_func, _GovernedFunc)
        assert my_func._governor is mock_governor

    @pytest.mark.asyncio
    async def test_decorated_func_executes(self) -> None:
        """Decorated function should still execute normally."""
        mock_governor = MagicMock()

        @governed(mock_governor)
        async def add(a: int, b: int) -> int:
            return a + b

        result = await add(3, 4)
        assert result == 7

    @pytest.mark.asyncio
    async def test_decorated_func_with_governor_execute(self) -> None:
        """Decorated function can call governor.execute internally."""
        mock_governor = MagicMock()
        mock_governor.execute = AsyncMock(return_value="governed response")

        @governed(mock_governor)
        async def ask(question: str) -> str:
            return await mock_governor.execute(
                messages=[{"role": "user", "content": question}],
                model="gpt-4o",
            )

        result = await ask("What is 2+2?")
        assert result == "governed response"
        mock_governor.execute.assert_awaited_once()
