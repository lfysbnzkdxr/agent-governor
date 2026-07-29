"""Core Protocol definitions for Agent Governor.

All core interfaces are defined as typing.Protocol to enable
structural subtyping - third parties can adapt without inheritance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from agent_governor.models import CallRecord, Task, TaskResult, ValidationResult


@runtime_checkable
class Executor(Protocol):
    """Protocol for task executors (LLM, MCP, HTTP, etc.)."""

    async def execute(self, task: Task) -> TaskResult:
        """Execute a task and return the result."""
        ...


@runtime_checkable
class QualityGate(Protocol):
    """Protocol for quality gate validators (richer than Validator).

    Returns a structured ValidationResult instead of a simple bool,
    enabling scoring, reasons, and detailed diagnostics.
    """

    @property
    def name(self) -> str:
        """Gate identifier for logging and reporting."""
        ...

    async def check(self, task: Task, result: TaskResult) -> ValidationResult:
        """Validate a task result. Returns structured validation outcome."""
        ...


@runtime_checkable
class Validator(Protocol):
    """Protocol for simple validators (bool-based, Phase 1 compat)."""

    async def validate(self, task: Task, result: TaskResult) -> bool:
        """Validate a task result. Returns True if passed."""
        ...


@runtime_checkable
class StateBackend(Protocol):
    """Protocol for state persistence backends (SQLite, Redis, etc.)."""

    async def save(self, key: str, state: dict[str, Any]) -> None:
        """Persist state under the given key."""
        ...

    async def load(self, key: str) -> dict[str, Any] | None:
        """Load state by key. Returns None if not found."""
        ...

    async def delete(self, key: str) -> None:
        """Delete state by key."""
        ...


@runtime_checkable
class AuditLoggerProtocol(Protocol):
    """Protocol for audit loggers."""

    async def log(self, record: CallRecord) -> None:
        """Log a call record."""
        ...


@runtime_checkable
class CostTrackerProtocol(Protocol):
    """Protocol for cost tracking."""

    def record_usage(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Record token usage and return the cost in USD."""
        ...

    @property
    def total_cost(self) -> float:
        """Get total accumulated cost in USD."""
        ...

    def check_budget(self, budget_usd: float) -> bool:
        """Check if total cost is within budget. Returns True if within budget."""
        ...
