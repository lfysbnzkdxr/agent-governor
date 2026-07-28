"""Tests for OTel Tracing (TracingManager, @traced, TracedExecutor)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_governor.observability.tracing import (
    OTEL_AVAILABLE,
    TracedExecutor,
    TracingManager,
    traced,
)

# Import the module so we can use patch.object
import agent_governor.observability.tracing as tracing_mod


class TestTracingManager:
    """Tests for TracingManager."""

    def test_init_noop_when_otel_not_available(self):
        """TracingManager should not crash when OTel is unavailable."""
        with patch.object(tracing_mod, "OTEL_AVAILABLE", False):
            mgr = TracingManager()
            assert mgr.get_tracer() is None
            mgr.shutdown()  # Should not raise

    def test_init_with_empty_endpoint(self):
        """TracingManager should handle empty endpoint gracefully."""
        mgr = TracingManager(otlp_endpoint="")
        if OTEL_AVAILABLE:
            assert mgr.get_tracer() is not None
        else:
            assert mgr.get_tracer() is None

    def test_shutdown_noop_when_not_initialized(self):
        """Shutdown without init should not raise."""
        TracingManager().shutdown()

    def test_get_tracer_returns_none_without_otel(self):
        """get_tracer should return None when OTel is absent."""
        with patch.object(tracing_mod, "OTEL_AVAILABLE", False):
            mgr = TracingManager()
            assert mgr.get_tracer() is None


@pytest.mark.asyncio
class TestTracedDecorator:
    """Tests for the @traced decorator."""

    async def test_decorator_noop_without_otel(self):
        """@traced should be a no-op when OTel is unavailable."""
        calls = []

        @traced()
        async def my_func(x: int) -> int:
            calls.append(x)
            return x * 2

        with patch.object(tracing_mod, "OTEL_AVAILABLE", False):
            result = await my_func(5)

        assert result == 10
        assert calls == [5]

    async def test_decorator_preserves_signature(self):
        """@traced should preserve function signature and behavior."""

        @traced(span_name="custom")
        async def add(a: int, b: int) -> int:
            return a + b

        result = await add(3, 4)
        assert result == 7

    async def test_decorator_with_otel(self):
        """@traced should create a span when OTel is available (mock)."""
        mock_span = MagicMock()
        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span

        calls = []

        @traced(span_name="test_span", attributes={"key": "val"})
        async def my_func(x: int) -> int:
            calls.append(x)
            return x * 2

        # Patch OTEL_AVAILABLE to True and provide a mock trace module
        mock_trace = MagicMock()
        mock_trace.get_tracer.return_value = mock_tracer
        with patch.object(tracing_mod, "OTEL_AVAILABLE", True):
            with patch.object(tracing_mod, "trace", mock_trace):
                result = await my_func(5)

        assert result == 10
        assert calls == [5]
        mock_span.set_attribute.assert_any_call("key", "val")

    async def test_decorator_records_status_on_result(self):
        """@traced should record status attribute when result has it."""
        mock_span = MagicMock()
        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span

        class FakeResult:
            status = "completed"

        @traced()
        async def my_func() -> FakeResult:
            return FakeResult()

        mock_trace = MagicMock()
        mock_trace.get_tracer.return_value = mock_tracer
        with patch.object(tracing_mod, "OTEL_AVAILABLE", True):
            with patch.object(tracing_mod, "trace", mock_trace):
                result = await my_func()

        assert result.status == "completed"
        mock_span.set_attribute.assert_any_call("agent_governor.status", "completed")


@pytest.mark.asyncio
class TestTracedExecutor:
    """Tests for TracedExecutor."""

    async def test_wraps_inner_without_otel(self):
        """TracedExecutor should delegate to inner when OTel is unavailable."""
        mock_inner = MagicMock()
        mock_inner.execute = AsyncMock(return_value="result")

        executor = TracedExecutor(inner=mock_inner)
        with patch.object(tracing_mod, "OTEL_AVAILABLE", False):
            result = await executor.execute(MagicMock())

        assert result == "result"
        mock_inner.execute.assert_awaited_once()

    async def test_wraps_inner_with_otel(self):
        """TracedExecutor should create span around inner execute."""
        mock_inner = MagicMock()
        mock_inner.execute = AsyncMock(return_value="result")
        mock_span = MagicMock()
        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span

        task = MagicMock()
        task.name = "test_task"
        task.model = "gpt-4o"

        executor = TracedExecutor(inner=mock_inner)
        mock_trace = MagicMock()
        mock_trace.get_tracer.return_value = mock_tracer
        with patch.object(tracing_mod, "OTEL_AVAILABLE", True):
            with patch.object(tracing_mod, "trace", mock_trace):
                result = await executor.execute(task)

        assert result == "result"
        mock_span.set_attribute.assert_any_call("agent_governor.task.name", "test_task")
        mock_span.set_attribute.assert_any_call("agent_governor.model", "gpt-4o")

    async def test_records_exception(self):
        """TracedExecutor should record exception on span."""
        mock_inner = MagicMock()
        mock_inner.execute = AsyncMock(side_effect=ValueError("test error"))
        mock_span = MagicMock()
        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span

        task = MagicMock()
        task.name = "failing_task"
        task.model = ""

        executor = TracedExecutor(inner=mock_inner)
        mock_trace = MagicMock()
        mock_trace.get_tracer.return_value = mock_tracer
        with patch.object(tracing_mod, "OTEL_AVAILABLE", True):
            with patch.object(tracing_mod, "trace", mock_trace):
                with pytest.raises(ValueError, match="test error"):
                    await executor.execute(task)

        mock_span.set_attribute.assert_any_call("agent_governor.status", "error")
        mock_span.record_exception.assert_called_once()


class TestOTELAvailability:
    """Tests for OTEL_AVAILABLE flag."""

    def test_otel_availability_flag_exists(self):
        """OTEL_AVAILABLE should be defined as bool."""
        assert isinstance(OTEL_AVAILABLE, bool)
