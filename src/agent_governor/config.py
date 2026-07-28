"""Governor configuration model."""

from __future__ import annotations

from pydantic import BaseModel, Field


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
    budget_usd: float = Field(default=0.0, ge=0, description="Total budget limit in USD. 0 = unlimited")
    budget_warning_ratio: float = Field(
        default=0.8, ge=0, le=1.0, description="Ratio at which budget warning is triggered"
    )

    # Model routing
    models: list[str] = Field(default_factory=list, description="Allowed model whitelist. Empty = all allowed")
    fallback_model: str = Field(default="", description="Fallback model when primary fails")
    default_model: str = Field(default="gpt-4o", description="Default model if not specified in call")

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

    def is_model_allowed(self, model: str) -> bool:
        """Check if a model is in the whitelist."""
        if not self.models:
            return True
        return model in self.models
