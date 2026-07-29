"""Tests for concurrency control: KeyedSemaphore and TokenBucket."""

import asyncio
import time
from contextlib import suppress

import pytest

from agent_governor.data.concurrency import (
    ConcurrencyManager,
    KeyedSemaphore,
    TokenBucket,
)


class TestKeyedSemaphore:
    async def test_basic_acquire_release(self):
        ks = KeyedSemaphore(default_limit=2)
        async with ks.acquire("model_a"):
            pass  # Should not block

    async def test_per_key_isolation(self):
        ks = KeyedSemaphore(default_limit=1)
        # Acquire key "a" - should not block key "b"
        await ks._acquire("a")
        # "b" should still be acquirable
        async with ks.acquire("b"):
            pass
        await ks._release("a")

    async def test_limit_enforced(self):
        ks = KeyedSemaphore(default_limit=1)
        await ks._acquire("x")

        # Second acquire should block (test with timeout)
        acquired = False

        async def try_acquire():
            nonlocal acquired
            await ks._acquire("x")
            acquired = True

        task = asyncio.create_task(try_acquire())
        await asyncio.sleep(0.05)
        assert not acquired  # Still blocked

        await ks._release("x")
        await asyncio.sleep(0.05)
        assert acquired
        await ks._release("x")
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    async def test_custom_key_limits(self):
        ks = KeyedSemaphore(default_limit=10, key_limits={"gpt-4o": 2})
        # Should use limit 2 for gpt-4o
        await ks._acquire("gpt-4o")
        await ks._acquire("gpt-4o")
        # Third should block - verify via try with timeout
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(ks._acquire("gpt-4o"), timeout=0.05)
        # Cleanup
        await ks._release("gpt-4o")
        await ks._release("gpt-4o")

    async def test_global_limit(self):
        ks = KeyedSemaphore(default_limit=10, global_limit=2)
        await ks._acquire("a")
        await ks._acquire("b")
        # Third should block due to global limit
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(ks._acquire("c"), timeout=0.05)
        await ks._release("a")
        await ks._release("b")

    async def test_active_keys(self):
        ks = KeyedSemaphore()
        async with ks.acquire("model_x"):
            assert "model_x" in ks.active_keys


class TestTokenBucket:
    async def test_immediate_acquire_within_capacity(self):
        bucket = TokenBucket(rate=10.0, capacity=10.0)
        start = time.monotonic()
        await bucket.acquire()
        elapsed = time.monotonic() - start
        assert elapsed < 0.05  # Should be near-instant

    async def test_burst_capacity(self):
        bucket = TokenBucket(rate=10.0, capacity=5.0)
        # Should be able to acquire 5 immediately
        for _ in range(5):
            acquired = await bucket.try_acquire()
            assert acquired

    async def test_rate_limiting_delays(self):
        bucket = TokenBucket(rate=10.0, capacity=1.0)
        await bucket.acquire()  # Drain the bucket
        start = time.monotonic()
        await bucket.acquire()  # Should wait ~0.1s
        elapsed = time.monotonic() - start
        assert elapsed >= 0.05  # At least some delay

    async def test_try_acquire_fails_when_empty(self):
        bucket = TokenBucket(rate=1.0, capacity=1.0)
        assert await bucket.try_acquire() is True
        assert await bucket.try_acquire() is False

    async def test_refill_over_time(self):
        bucket = TokenBucket(rate=100.0, capacity=5.0)
        # Drain
        for _ in range(5):
            await bucket.try_acquire()
        # Wait for refill
        await asyncio.sleep(0.05)  # 100 tokens/s * 0.05s = 5 tokens
        assert await bucket.try_acquire() is True

    async def test_context_manager(self):
        bucket = TokenBucket(rate=100.0, capacity=10.0)
        async with bucket:
            pass  # Should not raise

    async def test_available_tokens_property(self):
        bucket = TokenBucket(rate=10.0, capacity=10.0)
        assert bucket.available_tokens <= 10.0
        assert bucket.available_tokens > 0


class TestConcurrencyManager:
    async def test_basic_acquire(self):
        mgr = ConcurrencyManager(default_limit=5)
        async with mgr.acquire("model_a"):
            pass

    async def test_with_rate_limit(self):
        mgr = ConcurrencyManager(
            default_limit=10,
            rate_limits={"slow_model": 100.0},  # High rate for fast test
        )
        async with mgr.acquire("slow_model"):
            pass

    async def test_stats(self):
        mgr = ConcurrencyManager(rate_limits={"m1": 10.0})
        async with mgr.acquire("m1"):
            stats = mgr.stats
            assert "m1" in stats["active_keys"]
