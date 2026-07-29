"""Tenant configuration models for multi-tenant governance."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TenantPolicy(BaseModel):
    """Policy configuration for a single tenant.

    Example:
        policy = TenantPolicy(
            budget_usd=100.0,
            rate_limit_rps=5.0,
            max_concurrency=3,
            allowed_models=["gpt-4o", "qwen-plus"],
        )
    """

    budget_usd: float = Field(
        default=0.0, ge=0, description="Tenant budget limit in USD. 0 = unlimited"
    )
    rate_limit_rps: float = Field(
        default=0.0, ge=0, description="Max requests per second. 0 = unlimited"
    )
    max_concurrency: int = Field(
        default=5, ge=1, description="Max concurrent LLM calls for this tenant"
    )
    allowed_models: list[str] = Field(
        default_factory=list, description="Model whitelist. Empty = inherit global"
    )


class MultiTenantConfig(BaseModel):
    """Configuration for multi-tenant governance.

    Example:
        config = MultiTenantConfig(
            tenants={
                "team-sales": TenantPolicy(budget_usd=500, rate_limit_rps=10),
                "team-support": TenantPolicy(budget_usd=200, rate_limit_rps=5),
            },
            default_policy=TenantPolicy(budget_usd=50, rate_limit_rps=2),
        )
    """

    tenants: dict[str, TenantPolicy] = Field(
        default_factory=dict, description="Per-tenant policies"
    )
    default_policy: TenantPolicy | None = Field(
        default=None, description="Fallback policy for unknown tenants"
    )

    def get_policy(self, tenant_id: str) -> TenantPolicy | None:
        """Get policy for a tenant, falling back to default."""
        return self.tenants.get(tenant_id, self.default_policy)
