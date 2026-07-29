"""Agent Governor - Lightweight Agent governance and execution middleware."""

from agent_governor.config import GovernorConfig
from agent_governor.decorators import governed
from agent_governor.exceptions import (
    BudgetExceededError,
    CircuitOpenError,
    ExecutionTimeoutError,
    GovernorError,
    ModelNotAllowedError,
    RateLimitExceededError,
    TenantBudgetExceededError,
    ValidationError,
)
from agent_governor.governor import Governor
from agent_governor.models import (
    CallRecord,
    DagExecutionResult,
    DagNodeResult,
    DagNodeStatus,
    Task,
    TaskResult,
    TokenUsage,
    ValidationResult,
)
from agent_governor.policy import PolicyDecision, PolicyEngine, PolicyRule
from agent_governor.tenant import BillingReport, MultiTenantConfig, TenantManager, TenantPolicy

__all__ = [
    # Core
    "Governor",
    "GovernorConfig",
    "governed",
    # Exceptions
    "GovernorError",
    "BudgetExceededError",
    "ModelNotAllowedError",
    "ExecutionTimeoutError",
    "CircuitOpenError",
    "ValidationError",
    "TenantBudgetExceededError",
    "RateLimitExceededError",
    # Models
    "Task",
    "TaskResult",
    "CallRecord",
    "TokenUsage",
    "ValidationResult",
    "DagExecutionResult",
    "DagNodeResult",
    "DagNodeStatus",
    # Multi-tenant
    "TenantManager",
    "TenantPolicy",
    "MultiTenantConfig",
    "BillingReport",
    # Policy
    "PolicyEngine",
    "PolicyRule",
    "PolicyDecision",
]

__version__ = "0.4.0"
