"""Concurrency control: keyed semaphores and token bucket rate limiter.

Provides:
- KeyedSemaphore: Per-route-key concurrency isolation (e.g., per-model limits).
- TokenBucket: Async token bucket rate limiter for throughput control.
- ConcurrencyManager: Combines both for unified concurrency governance.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class KeyedSemaphore:
    """Per-key semaphore isolation for concurrency control.

    Each key (e.g., model name) gets its own semaphore with a configurable
    limit. A global semaphore can also be applied across all keys.

    Example:
        ks = KeyedSemaphore(
            default_limit=10,
            key_limits={"gpt-4o": 5, "qwen-plus": 20},
            global_limit=50,
        )
        async with ks.acquire("gpt-4o"):
            await call_gpt4o()
    """

    def __init__(
        self,
        *,
        default_limit: int = 10,
        key_limits: dict[str, int] | None = None,
        global_limit: int = 0,
    ) -> None:
        """Initialize KeyedSemaphore.

        Args:
            default_limit: Default concurrency limit per key.
            key_limits: Per-key concurrency limits (overrides default).
            global_limit: Global limit across all keys. 0 = unlimited.
        """
        self._default_limit = default_limit
        self._key_limits = key_limits or {}
        self._semaphores: dict[str, asyncio.Semaphore] = {}
        self._global_semaphore = asyncio.Semaphore(global_limit) if global_limit > 0 else None

    def _get_semaphore(self, key: str) -> asyncio.Semaphore:
        """Get or create a semaphore for the given key."""
        if key not in self._semaphores:
            limit = self._key_limits.get(key, self._default_limit)
            self._semaphores[key] = asyncio.Semaphore(limit)
        return self._semaphores[key]

    def acquire(self, key: str) -> _KeyedAcquireContext:
        """Acquire concurrency slot for a key.

        Returns an async context manager.
        """
        return _KeyedAcquireContext(self, key)

    async def _acquire(self, key: str) -> None:
        """Internal: acquire both global and per-key semaphores."""
        if self._global_semaphore:
            await self._global_semaphore.acquire()
        await self._get_semaphore(key).acquire()

    async def _release(self, key: str) -> None:
        """Internal: release both per-key and global semaphores."""
        self._get_semaphore(key).release()
        if self._global_semaphore:
            self._global_semaphore.release()

    @property
    def active_keys(self) -> list[str]:
        """Get list of keys with created semaphores."""
        return list(self._semaphores.keys())


class _KeyedAcquireContext:
    """Async context manager for keyed semaphore acquisition."""

    def __init__(self, ks: KeyedSemaphore, key: str) -> None:
        self._ks = ks
        self._key = key

    async def __aenter__(self) -> _KeyedAcquireContext:
        await self._ks._acquire(self._key)
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self._ks._release(self._key)


class TokenBucket:
    """Async token bucket rate limiter.

    Controls throughput by allowing bursts up to bucket capacity
    while maintaining a steady refill rate.

    Example:
        bucket = TokenBucket(rate=10.0, capacity=20)
        async with bucket:
            await make_api_call()
    """

    def __init__(self, *, rate: float, capacity: float = 0.0) -> None:
        """Initialize TokenBucket.

        Args:
            rate: Tokens added per second (sustained throughput).
            capacity: Maximum bucket capacity (burst size).
                Defaults to rate (1 second of burst).
        """
        self._rate = rate
        self._capacity = capacity if capacity > 0 else rate
        self._tokens = self._capacity
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    @property
    def rate(self) -> float:
        return self._rate

    @property
    def capacity(self) -> float:
        return self._capacity

    @property
    def available_tokens(self) -> float:
        """Current available tokens (approximate, without lock)."""
        elapsed = time.monotonic() - self._last_refill
        return min(self._capacity, self._tokens + elapsed * self._rate)

    async def acquire(self, tokens: float = 1.0) -> None:
        """Acquire tokens, waiting if necessary.

        Args:
            tokens: Number of tokens to acquire.
        """
        while True:
            async with self._lock:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                # Calculate wait time
                deficit = tokens - self._tokens
                wait_time = deficit / self._rate

            await asyncio.sleep(wait_time)

    async def try_acquire(self, tokens: float = 1.0) -> bool:
        """Try to acquire tokens without waiting.

        Returns:
            True if tokens were acquired, False otherwise.
        """
        async with self._lock:
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    async def __aenter__(self) -> TokenBucket:
        await self.acquire()
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass  # Nothing to release for token bucket

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
        self._last_refill = now


class ConcurrencyManager:
    """Unified concurrency governance combining keyed semaphores and rate limiting.

    Example:
        manager = ConcurrencyManager(
            key_limits={"gpt-4o": 5, "qwen-plus": 20},
            rate_limits={"gpt-4o": 2.0},  # 2 req/s for gpt-4o
            global_limit=50,
        )
        async with manager.acquire("gpt-4o"):
            await call_api()
    """

    def __init__(
        self,
        *,
        default_limit: int = 10,
        key_limits: dict[str, int] | None = None,
        global_limit: int = 0,
        rate_limits: dict[str, float] | None = None,
        default_rate: float = 0.0,
    ) -> None:
        """Initialize ConcurrencyManager.

        Args:
            default_limit: Default per-key concurrency.
            key_limits: Per-key concurrency limits.
            global_limit: Global concurrency limit. 0 = unlimited.
            rate_limits: Per-key rate limits (requests/second).
            default_rate: Default rate limit. 0 = unlimited.
        """
        self._semaphore = KeyedSemaphore(
            default_limit=default_limit,
            key_limits=key_limits,
            global_limit=global_limit,
        )
        self._rate_limits = rate_limits or {}
        self._default_rate = default_rate
        self._buckets: dict[str, TokenBucket] = {}

    def _get_bucket(self, key: str) -> TokenBucket | None:
        """Get or create a token bucket for the key."""
        rate = self._rate_limits.get(key, self._default_rate)
        if rate <= 0:
            return None
        if key not in self._buckets:
            self._buckets[key] = TokenBucket(rate=rate)
        return self._buckets[key]

    def acquire(self, key: str) -> _ManagerAcquireContext:
        """Acquire both concurrency slot and rate limit token."""
        return _ManagerAcquireContext(self, key)

    async def _acquire(self, key: str) -> None:
        """Internal: acquire semaphore + rate limit."""
        # Rate limit first (wait for token)
        bucket = self._get_bucket(key)
        if bucket:
            await bucket.acquire()
        # Then concurrency limit
        await self._semaphore._acquire(key)

    async def _release(self, key: str) -> None:
        """Internal: release semaphore."""
        await self._semaphore._release(key)

    @property
    def stats(self) -> dict[str, Any]:
        """Get concurrency manager statistics."""
        return {
            "active_keys": self._semaphore.active_keys,
            "rate_limited_keys": list(self._buckets.keys()),
        }


class _ManagerAcquireContext:
    """Async context manager for ConcurrencyManager."""

    def __init__(self, manager: ConcurrencyManager, key: str) -> None:
        self._manager = manager
        self._key = key

    async def __aenter__(self) -> _ManagerAcquireContext:
        await self._manager._acquire(self._key)
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self._manager._release(self._key)
