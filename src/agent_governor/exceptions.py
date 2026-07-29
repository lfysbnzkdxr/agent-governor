"""Exception hierarchy for Agent Governor."""

from __future__ import annotations


class GovernorError(Exception):
    """Base exception for all Agent Governor errors."""

    def __init__(self, message: str = "", *, detail: str = "") -> None:
        self.detail = detail
        super().__init__(message)


class BudgetExceededError(GovernorError):
    """Raised when the cost budget is exceeded."""

    def __init__(
        self,
        message: str = "Budget exceeded",
        *,
        current_cost: float = 0.0,
        budget: float = 0.0,
    ) -> None:
        self.current_cost = current_cost
        self.budget = budget
        super().__init__(
            message,
            detail=f"current_cost=${current_cost:.4f}, budget=${budget:.4f}",
        )


class ModelNotAllowedError(GovernorError):
    """Raised when a model is not in the allowed whitelist."""

    def __init__(self, model: str, allowed_models: list[str] | None = None) -> None:
        self.model = model
        self.allowed_models = allowed_models or []
        super().__init__(
            f"Model '{model}' is not allowed",
            detail=f"allowed_models={self.allowed_models}",
        )


class CircuitOpenError(GovernorError):
    """Raised when the circuit breaker is open and calls are rejected."""

    def __init__(self, model: str = "", recovery_timeout: float = 0.0) -> None:
        self.model = model
        self.recovery_timeout = recovery_timeout
        super().__init__(
            f"Circuit breaker is open for model '{model}'",
            detail=f"recovery_timeout={recovery_timeout}s",
        )


class ExecutionTimeoutError(GovernorError):
    """Raised when a task execution exceeds its timeout."""

    def __init__(self, task_id: str = "", timeout: float = 0.0) -> None:
        self.task_id = task_id
        self.timeout = timeout
        super().__init__(
            f"Task '{task_id}' timed out after {timeout}s",
            detail=f"task_id={task_id}, timeout={timeout}s",
        )


class ValidationError(GovernorError):
    """Raised when a quality gate validation fails."""

    def __init__(self, message: str = "Validation failed", *, gate_name: str = "") -> None:
        self.gate_name = gate_name
        super().__init__(message, detail=f"gate={gate_name}")


class TenantBudgetExceededError(GovernorError):
    """Raised when a tenant's cost budget is exceeded."""

    def __init__(
        self,
        tenant_id: str = "",
        *,
        current_cost: float = 0.0,
        budget: float = 0.0,
    ) -> None:
        self.tenant_id = tenant_id
        self.current_cost = current_cost
        self.budget = budget
        super().__init__(
            f"Tenant '{tenant_id}' budget exceeded",
            detail=f"tenant_id={tenant_id}, current_cost=${current_cost:.4f}, budget=${budget:.4f}",
        )


class RateLimitExceededError(GovernorError):
    """Raised when a tenant's rate limit is exceeded."""

    def __init__(
        self,
        tenant_id: str = "",
        *,
        retry_after: float = 0.0,
    ) -> None:
        self.tenant_id = tenant_id
        self.retry_after = retry_after
        super().__init__(
            f"Tenant '{tenant_id}' rate limit exceeded",
            detail=f"tenant_id={tenant_id}, retry_after={retry_after:.2f}s",
        )
