"""Governor configuration model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from agent_governor.tenant.config import TenantPolicy


class GovernorConfig(BaseModel):
    """Configuration for the Governor instance.

    Example:
        config = GovernorConfig(
            budget_usd=10.0,
            models=["gpt-4o", "qwen-plus"],
            fallback_model="qwen-turbo",
        )
    """

    # Budget control
    budget_usd: float = Field(
        default=0.0, ge=0, description="Total budget limit in USD. 0 = unlimited"
    )
    budget_warning_ratio: float = Field(
        default=0.8, ge=0, le=1.0, description="Ratio at which budget warning is triggered"
    )

    # Model routing
    models: list[str] = Field(
        default_factory=list, description="Allowed model whitelist. Empty = all allowed"
    )
    fallback_model: str = Field(default="", description="Fallback model when primary fails")
    default_model: str = Field(
        default="gpt-4o", description="Default model if not specified in call"
    )

    # Concurrency
    max_concurrency: int = Field(default=10, ge=1, description="Max concurrent LLM calls")

    # Timeout
    default_timeout: float = Field(default=60.0, gt=0, description="Default timeout in seconds")

    # Audit
    audit_enabled: bool = Field(default=True, description="Enable audit logging")
    audit_log_file: str = Field(default="", description="Audit log file path. Empty = stdout")

    # Retry
    max_retries: int = Field(default=2, ge=0, description="Max retries on failure")
    retry_delay: float = Field(default=1.0, gt=0, description="Delay between retries in seconds")

    # Tracing (Phase 3)
    tracing_enabled: bool = Field(default=False, description="Enable OTel distributed tracing")
    otlp_endpoint: str = Field(
        default="", description="OTLP collector endpoint URL (e.g. http://localhost:4317)"
    )

    # Multi-tenant (Phase 4)
    tenant_policies: dict[str, TenantPolicy] = Field(
        default_factory=dict, description="Per-tenant policy configs (TenantPolicy instances)"
    )
    default_tenant_policy: TenantPolicy | None = Field(
        default=None, description="Default policy for unconfigured tenants"
    )

    def is_model_allowed(self, model: str) -> bool:
        """Check if a model is in the whitelist."""
        if not self.models:
            return True
        return model in self.models
