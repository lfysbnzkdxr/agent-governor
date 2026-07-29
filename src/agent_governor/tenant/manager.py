"""TenantManager - per-tenant cost tracking, rate limiting, and concurrency control.

Aggregates existing CostTracker and TokenBucket components into a unified
multi-tenant governance layer without modifying their source code.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

from agent_governor.data.concurrency import TokenBucket
from agent_governor.exceptions import RateLimitExceededError, TenantBudgetExceededError
from agent_governor.observability.cost import CostTracker
from agent_governor.tenant.config import MultiTenantConfig, TenantPolicy

logger = structlog.get_logger(__name__)


class TenantManager:
    """Manages per-tenant governance contexts.

    Each tenant gets an isolated CostTracker, TokenBucket (rate limit),
    and Semaphore (concurrency control).

    Example:
        config = MultiTenantConfig(
            tenants={"team-a": TenantPolicy(budget_usd=100, rate_limit_rps=5)},
            default_policy=TenantPolicy(budget_usd=10),
        )
        manager = TenantManager(config)

        # Check budget before call
        manager.check_budget("team-a")

        # Rate limit + concurrency control
        async with manager.acquire("team-a"):
            result = await call_llm()

        # Record usage after call
        manager.record_usage("team-a", "gpt-4o", prompt_tokens=100, completion_tokens=50)
    """

    def __init__(self, config: MultiTenantConfig) -> None:
        self._config = config
        self._trackers: dict[str, CostTracker] = {}
        self._buckets: dict[str, TokenBucket] = {}
        self._semaphores: dict[str, asyncio.Semaphore] = {}

    @property
    def config(self) -> MultiTenantConfig:
        """Get the multi-tenant configuration."""
        return self._config

    def _get_tracker(self, tenant_id: str) -> CostTracker:
        """Get or create a CostTracker for the tenant."""
        if tenant_id not in self._trackers:
            self._trackers[tenant_id] = CostTracker()
        return self._trackers[tenant_id]

    def _get_bucket(self, tenant_id: str, policy: TenantPolicy) -> TokenBucket | None:
        """Get or create a TokenBucket for the tenant's rate limit."""
        if policy.rate_limit_rps <= 0:
            return None
        if tenant_id not in self._buckets:
            self._buckets[tenant_id] = TokenBucket(rate=policy.rate_limit_rps)
        return self._buckets[tenant_id]

    def _get_semaphore(self, tenant_id: str, policy: TenantPolicy) -> asyncio.Semaphore:
        """Get or create a Semaphore for the tenant's concurrency limit."""
        if tenant_id not in self._semaphores:
            self._semaphores[tenant_id] = asyncio.Semaphore(policy.max_concurrency)
        return self._semaphores[tenant_id]

    def _resolve_policy(self, tenant_id: str) -> TenantPolicy:
        """Resolve the policy for a tenant, raising if none found."""
        policy = self._config.get_policy(tenant_id)
        if policy is None:
            # No policy configured and no default — allow with defaults
            return TenantPolicy()
        return policy

    def check_budget(self, tenant_id: str) -> None:
        """Check if tenant is within budget. Raises TenantBudgetExceededError if exceeded."""
        policy = self._resolve_policy(tenant_id)
        if policy.budget_usd <= 0:
            return  # No budget limit

        tracker = self._get_tracker(tenant_id)
        if not tracker.check_budget(policy.budget_usd):
            raise TenantBudgetExceededError(
                tenant_id,
                current_cost=tracker.total_cost,
                budget=policy.budget_usd,
            )

    def check_model_allowed(self, tenant_id: str, model: str) -> bool:
        """Check if a model is allowed for the tenant."""
        policy = self._resolve_policy(tenant_id)
        if not policy.allowed_models:
            return True  # Empty = inherit global (allow all)
        return model in policy.allowed_models

    @asynccontextmanager
    async def acquire(self, tenant_id: str) -> AsyncIterator[None]:
        """Acquire rate limit token and concurrency slot for a tenant.

        Usage:
            async with manager.acquire("team-a"):
                await call_llm()
        """
        policy = self._resolve_policy(tenant_id)

        # Rate limit (non-blocking check)
        bucket = self._get_bucket(tenant_id, policy)
        if bucket:
            acquired = await bucket.try_acquire()
            if not acquired:
                raise RateLimitExceededError(tenant_id, retry_after=1.0 / policy.rate_limit_rps)

        # Concurrency control (blocking)
        semaphore = self._get_semaphore(tenant_id, policy)
        await semaphore.acquire()
        try:
            yield
        finally:
            semaphore.release()

    def record_usage(
        self, tenant_id: str, model: str, prompt_tokens: int, completion_tokens: int
    ) -> float:
        """Record token usage for a tenant. Returns cost in USD."""
        tracker = self._get_tracker(tenant_id)
        cost = tracker.record_usage(model, prompt_tokens, completion_tokens)
        logger.debug(
            "tenant_usage_recorded",
            tenant_id=tenant_id,
            model=model,
            tokens=prompt_tokens + completion_tokens,
            cost_usd=f"${cost:.6f}",
        )
        return cost

    def get_tenant_summary(self, tenant_id: str) -> dict[str, Any]:
        """Get usage summary for a specific tenant."""
        tracker = self._get_tracker(tenant_id)
        policy = self._resolve_policy(tenant_id)
        summary = tracker.get_summary()
        summary["tenant_id"] = tenant_id
        summary["budget_usd"] = policy.budget_usd
        summary["budget_remaining"] = (
            max(0.0, policy.budget_usd - tracker.total_cost)
            if policy.budget_usd > 0
            else float("inf")
        )
        return summary

    def get_all_summaries(self) -> dict[str, dict[str, Any]]:
        """Get usage summaries for all tracked and configured tenants."""
        # Union of configured tenants (even if zero usage) and dynamically tracked ones
        tenant_ids = set(self._config.tenants.keys()) | set(self._trackers.keys())
        return {tid: self.get_tenant_summary(tid) for tid in tenant_ids}

    def reset(self, tenant_id: str = "") -> None:
        """Reset tracking data. If tenant_id is empty, reset all."""
        if tenant_id:
            if tenant_id in self._trackers:
                self._trackers[tenant_id].reset()
        else:
            for tracker in self._trackers.values():
                tracker.reset()
        logger.info("tenant_manager_reset", tenant_id=tenant_id or "all")
