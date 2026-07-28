"""BudgetGate - validates that a task result stays within cost/token budgets."""

from __future__ import annotations

from agent_governor.models import Task, TaskResult, ValidationResult


class BudgetGate:
    """Validates that a single task result is within cost and token budgets.

    This gate checks per-call budgets, complementing the Governor-level
    total budget check which tracks cumulative spend.

    Example:
        gate = BudgetGate(max_cost_usd=0.10, max_tokens=5000)
        result = await gate.check(task, task_result)
    """

    def __init__(
        self,
        *,
        max_cost_usd: float = 0.0,
        max_tokens: int = 0,
        max_latency_ms: float = 0.0,
    ) -> None:
        """Initialize BudgetGate.

        Args:
            max_cost_usd: Maximum allowed cost per call. 0 = no limit.
            max_tokens: Maximum allowed total tokens. 0 = no limit.
            max_latency_ms: Maximum allowed latency. 0 = no limit.
        """
        self._max_cost_usd = max_cost_usd
        self._max_tokens = max_tokens
        self._max_latency_ms = max_latency_ms

    @property
    def name(self) -> str:
        return "budget_gate"

    async def check(self, task: Task, result: TaskResult) -> ValidationResult:
        """Check if the result is within budget constraints."""
        violations: list[str] = []
        score = 1.0

        # Cost check
        if self._max_cost_usd > 0 and result.cost_usd > self._max_cost_usd:
            violations.append(
                f"cost ${result.cost_usd:.4f} exceeds limit ${self._max_cost_usd:.4f}"
            )
            score *= self._max_cost_usd / result.cost_usd if result.cost_usd > 0 else 0

        # Token check
        total_tokens = result.usage.total_tokens
        if self._max_tokens > 0 and total_tokens > self._max_tokens:
            violations.append(
                f"tokens {total_tokens} exceeds limit {self._max_tokens}"
            )
            score *= self._max_tokens / total_tokens if total_tokens > 0 else 0

        # Latency check
        if self._max_latency_ms > 0 and result.latency_ms > self._max_latency_ms:
            violations.append(
                f"latency {result.latency_ms:.0f}ms exceeds limit {self._max_latency_ms:.0f}ms"
            )
            score *= self._max_latency_ms / result.latency_ms if result.latency_ms > 0 else 0

        if violations:
            return ValidationResult(
                passed=False,
                gate_name=self.name,
                score=max(0.0, score),
                reason="; ".join(violations),
                details={
                    "cost_usd": result.cost_usd,
                    "total_tokens": total_tokens,
                    "latency_ms": result.latency_ms,
                },
            )

        return ValidationResult(
            passed=True,
            gate_name=self.name,
            score=1.0,
            reason="Within budget",
        )
