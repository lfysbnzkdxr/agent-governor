"""Async circuit breaker with three-state state machine and sliding window.

Implements the Circuit Breaker pattern natively for asyncio:
- CLOSED: Normal operation, tracking failures in a sliding window.
- OPEN: All calls rejected immediately. Transitions to HALF_OPEN after timeout.
- HALF_OPEN: Limited trial calls allowed. Success closes, failure re-opens.

Uses asyncio.Lock (not threading.Lock) to avoid blocking the event loop.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from enum import StrEnum
from typing import TYPE_CHECKING, Any, TypeVar

import structlog

from agent_governor.exceptions import CircuitOpenError

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

logger = structlog.get_logger(__name__)

T = TypeVar("T")


class CircuitState(StrEnum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Async-native circuit breaker with sliding window failure tracking.

    Example:
        cb = CircuitBreaker(failure_threshold=5, recovery_timeout=30.0)

        async def call_api():
            async with cb:
                return await some_flaky_api()

        # Or use the decorator pattern:
        result = await cb.call(some_flaky_api, arg1, arg2)
    """

    def __init__(
        self,
        *,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 3,
        window_size: int = 60,
        success_threshold: int = 2,
        name: str = "",
    ) -> None:
        """Initialize the circuit breaker.

        Args:
            failure_threshold: Number of failures in window to trip the breaker.
            recovery_timeout: Seconds to wait in OPEN before transitioning to HALF_OPEN.
            half_open_max_calls: Max trial calls allowed in HALF_OPEN state.
            window_size: Sliding window size in seconds for failure counting.
            success_threshold: Consecutive successes in HALF_OPEN needed to close.
            name: Identifier for logging.
        """
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._half_open_max_calls = half_open_max_calls
        self._window_size = window_size
        self._success_threshold = success_threshold
        self._name = name or "default"

        self._state = CircuitState.CLOSED
        self._lock = asyncio.Lock()

        # Sliding window of failure timestamps
        self._failures: deque[float] = deque()

        # OPEN state
        self._opened_at: float = 0.0

        # HALF_OPEN state
        self._half_open_calls: int = 0
        self._half_open_successes: int = 0

        # Stats
        self._total_calls: int = 0
        self._total_failures: int = 0
        self._total_successes: int = 0
        self._total_rejected: int = 0

    @property
    def state(self) -> CircuitState:
        """Current circuit state (may transition OPEN->HALF_OPEN on read)."""
        if (
            self._state == CircuitState.OPEN
            and time.monotonic() - self._opened_at >= self._recovery_timeout
        ):
            return CircuitState.HALF_OPEN
        return self._state

    @property
    def name(self) -> str:
        return self._name

    @property
    def stats(self) -> dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            "name": self._name,
            "state": self.state.value,
            "total_calls": self._total_calls,
            "total_failures": self._total_failures,
            "total_successes": self._total_successes,
            "total_rejected": self._total_rejected,
            "failure_threshold": self._failure_threshold,
            "recovery_timeout": self._recovery_timeout,
        }

    async def __aenter__(self) -> CircuitBreaker:
        """Acquire permission to make a call."""
        await self._before_call()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Record call outcome."""
        if exc_type is None:
            await self.record_success()
        else:
            await self.record_failure()

    async def call(
        self,
        func: Callable[..., Coroutine[Any, Any, T]],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Execute a function through the circuit breaker.

        Args:
            func: Async function to call.
            *args, **kwargs: Arguments to pass to the function.

        Returns:
            The function's return value.

        Raises:
            CircuitOpenError: If the circuit is open.
        """
        await self._before_call()
        try:
            result = await func(*args, **kwargs)
            await self.record_success()
            return result
        except Exception:
            await self.record_failure()
            raise

    async def _before_call(self) -> None:
        """Check if a call is allowed. Raises CircuitOpenError if not."""
        async with self._lock:
            self._total_calls += 1
            current_state = self._get_effective_state()

            if current_state == CircuitState.CLOSED:
                return  # Always allow in CLOSED

            if current_state == CircuitState.OPEN:
                self._total_rejected += 1
                raise CircuitOpenError(
                    model=self._name,
                    recovery_timeout=self._recovery_timeout,
                )

            # HALF_OPEN: allow limited calls
            if self._half_open_calls >= self._half_open_max_calls:
                self._total_rejected += 1
                raise CircuitOpenError(
                    model=self._name,
                    recovery_timeout=self._recovery_timeout,
                )
            self._half_open_calls += 1

    async def record_success(self) -> None:
        """Record a successful call."""
        async with self._lock:
            self._total_successes += 1
            current_state = self._get_effective_state()

            if current_state == CircuitState.HALF_OPEN:
                self._half_open_successes += 1
                if self._half_open_successes >= self._success_threshold:
                    self._transition_to(CircuitState.CLOSED)

    async def record_failure(self) -> None:
        """Record a failed call."""
        async with self._lock:
            self._total_failures += 1
            now = time.monotonic()
            current_state = self._get_effective_state()

            if current_state == CircuitState.HALF_OPEN:
                # Any failure in HALF_OPEN re-opens
                self._transition_to(CircuitState.OPEN)
                return

            if current_state == CircuitState.CLOSED:
                # Add to sliding window
                self._failures.append(now)
                self._prune_window(now)

                if len(self._failures) >= self._failure_threshold:
                    self._transition_to(CircuitState.OPEN)

    async def reset(self) -> None:
        """Manually reset the circuit breaker to CLOSED."""
        async with self._lock:
            self._transition_to(CircuitState.CLOSED)

    def _get_effective_state(self) -> CircuitState:
        """Get state, handling OPEN -> HALF_OPEN time-based transition."""
        if (
            self._state == CircuitState.OPEN
            and time.monotonic() - self._opened_at >= self._recovery_timeout
        ):
            self._transition_to(CircuitState.HALF_OPEN)
        return self._state

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state with logging and reset of counters."""
        old_state = self._state
        self._state = new_state

        if new_state == CircuitState.OPEN:
            self._opened_at = time.monotonic()
            self._half_open_calls = 0
            self._half_open_successes = 0
        elif new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0
            self._half_open_successes = 0
        elif new_state == CircuitState.CLOSED:
            self._failures.clear()
            self._half_open_calls = 0
            self._half_open_successes = 0

        if old_state != new_state:
            logger.info(
                "circuit_state_change",
                breaker=self._name,
                from_state=old_state.value,
                to_state=new_state.value,
            )

    def _prune_window(self, now: float) -> None:
        """Remove failures outside the sliding window."""
        cutoff = now - self._window_size
        while self._failures and self._failures[0] < cutoff:
            self._failures.popleft()
