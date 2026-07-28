"""Tests for async circuit breaker."""

import asyncio
import time

import pytest

from agent_governor.data.circuit_breaker import CircuitBreaker, CircuitState
from agent_governor.exceptions import CircuitOpenError


class TestCircuitBreakerStates:
    async def test_initial_state_is_closed(self):
        cb = CircuitBreaker()
        assert cb.state == CircuitState.CLOSED

    async def test_stays_closed_below_threshold(self):
        cb = CircuitBreaker(failure_threshold=3)
        await cb.record_failure()
        await cb.record_failure()
        assert cb.state == CircuitState.CLOSED

    async def test_opens_at_threshold(self):
        cb = CircuitBreaker(failure_threshold=3)
        for _ in range(3):
            await cb.record_failure()
        assert cb.state == CircuitState.OPEN

    async def test_open_rejects_calls(self):
        cb = CircuitBreaker(failure_threshold=1, name="test_model")
        await cb.record_failure()
        assert cb.state == CircuitState.OPEN

        with pytest.raises(CircuitOpenError):
            await cb._before_call()

    async def test_transitions_to_half_open_after_timeout(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)
        await cb.record_failure()
        assert cb.state == CircuitState.OPEN

        await asyncio.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN

    async def test_half_open_success_closes(self):
        cb = CircuitBreaker(
            failure_threshold=1, recovery_timeout=0.1, success_threshold=2
        )
        await cb.record_failure()
        await asyncio.sleep(0.15)

        # In HALF_OPEN now
        assert cb.state == CircuitState.HALF_OPEN
        await cb._before_call()  # Allow call
        await cb.record_success()
        await cb._before_call()
        await cb.record_success()

        assert cb.state == CircuitState.CLOSED

    async def test_half_open_failure_reopens(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)
        await cb.record_failure()
        await asyncio.sleep(0.15)

        assert cb.state == CircuitState.HALF_OPEN
        await cb._before_call()
        await cb.record_failure()

        assert cb.state == CircuitState.OPEN

    async def test_half_open_limits_calls(self):
        cb = CircuitBreaker(
            failure_threshold=1, recovery_timeout=0.1, half_open_max_calls=2
        )
        await cb.record_failure()
        await asyncio.sleep(0.15)

        # Allow 2 calls
        await cb._before_call()
        await cb._before_call()
        # Third should be rejected
        with pytest.raises(CircuitOpenError):
            await cb._before_call()


class TestCircuitBreakerSlidingWindow:
    async def test_old_failures_expire(self):
        cb = CircuitBreaker(failure_threshold=3, window_size=0.1)
        await cb.record_failure()
        await cb.record_failure()
        await asyncio.sleep(0.15)  # Let them expire
        await cb.record_failure()

        # Only 1 failure in window, should stay closed
        assert cb.state == CircuitState.CLOSED

    async def test_recent_failures_count(self):
        cb = CircuitBreaker(failure_threshold=3, window_size=10.0)
        await cb.record_failure()
        await cb.record_failure()
        await cb.record_failure()
        assert cb.state == CircuitState.OPEN


class TestCircuitBreakerCallInterface:
    async def test_call_success(self):
        cb = CircuitBreaker()

        async def ok_fn():
            return "result"

        result = await cb.call(ok_fn)
        assert result == "result"
        assert cb.stats["total_successes"] == 1

    async def test_call_failure_records(self):
        cb = CircuitBreaker(failure_threshold=5)

        async def fail_fn():
            raise ValueError("oops")

        with pytest.raises(ValueError):
            await cb.call(fail_fn)
        assert cb.stats["total_failures"] == 1

    async def test_context_manager_success(self):
        cb = CircuitBreaker()
        async with cb:
            pass
        assert cb.stats["total_successes"] == 1

    async def test_context_manager_failure(self):
        cb = CircuitBreaker(failure_threshold=5)
        with pytest.raises(RuntimeError):
            async with cb:
                raise RuntimeError("fail")
        assert cb.stats["total_failures"] == 1

    async def test_manual_reset(self):
        cb = CircuitBreaker(failure_threshold=1)
        await cb.record_failure()
        assert cb.state == CircuitState.OPEN

        await cb.reset()
        assert cb.state == CircuitState.CLOSED


class TestCircuitBreakerStats:
    async def test_stats_structure(self):
        cb = CircuitBreaker(name="my_breaker", failure_threshold=5)
        stats = cb.stats
        assert stats["name"] == "my_breaker"
        assert stats["state"] == "closed"
        assert stats["total_calls"] == 0
        assert stats["failure_threshold"] == 5

    async def test_rejected_count(self):
        cb = CircuitBreaker(failure_threshold=1)
        await cb.record_failure()

        with pytest.raises(CircuitOpenError):
            await cb._before_call()
        assert cb.stats["total_rejected"] == 1
