"""Quality gate validation framework.

Provides ValidationPipeline for chaining multiple QualityGate instances
in sequence. Gates are evaluated in order; the pipeline short-circuits
on first failure (unless configured otherwise).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from agent_governor.models import Task, TaskResult, ValidationResult

if TYPE_CHECKING:
    from agent_governor.protocols import QualityGate

logger = structlog.get_logger(__name__)


class ValidationPipeline:
    """Chains multiple QualityGate instances into a validation pipeline.

    Gates are evaluated sequentially. By default, the pipeline
    short-circuits on the first failing gate.

    Example:
        pipeline = ValidationPipeline([
            SchemaGate(ReportModel),
            BudgetGate(max_cost_usd=0.5),
        ])
        result = await pipeline.run(task, task_result)
        if not result.passed:
            print(f"Failed at: {result.gate_name} - {result.reason}")
    """

    def __init__(
        self,
        gates: list[QualityGate] | None = None,
        *,
        short_circuit: bool = True,
    ) -> None:
        """Initialize the pipeline.

        Args:
            gates: List of quality gates to evaluate in order.
            short_circuit: If True, stop on first failing gate.
        """
        self._gates: list[QualityGate] = list(gates or [])
        self._short_circuit = short_circuit

    @property
    def gates(self) -> list[QualityGate]:
        """Get the list of gates in the pipeline."""
        return list(self._gates)

    def add_gate(self, gate: QualityGate) -> ValidationPipeline:
        """Add a gate to the end of the pipeline. Returns self for chaining."""
        self._gates.append(gate)
        return self

    async def run(self, task: Task, result: TaskResult) -> ValidationResult:
        """Run all gates against a task result.

        Args:
            task: The original task.
            result: The task result to validate.

        Returns:
            A single ValidationResult. If all gates pass, passed=True.
            If any gate fails, passed=False with the failing gate info.
            When short_circuit=False, all gates run and details accumulates.
        """
        if not self._gates:
            return ValidationResult(passed=True, gate_name="empty_pipeline")

        all_details: dict[str, Any] = {}
        min_score = 1.0

        for gate in self._gates:
            gate_result = await gate.check(task, result)
            all_details[gate.name] = {
                "passed": gate_result.passed,
                "score": gate_result.score,
                "reason": gate_result.reason,
            }
            min_score = min(min_score, gate_result.score)

            logger.debug(
                "gate_evaluated",
                gate=gate.name,
                passed=gate_result.passed,
                score=gate_result.score,
                task_id=task.id,
            )

            if not gate_result.passed and self._short_circuit:
                return ValidationResult(
                    passed=False,
                    gate_name=gate.name,
                    score=gate_result.score,
                    reason=gate_result.reason,
                    details=all_details,
                )

        # Determine overall pass/fail
        all_passed = all(d["passed"] for d in all_details.values())

        if all_passed:
            return ValidationResult(
                passed=True,
                gate_name="pipeline",
                score=min_score,
                details=all_details,
            )
        else:
            # Find first failing gate for the summary
            first_fail = next((name for name, d in all_details.items() if not d["passed"]), "")
            return ValidationResult(
                passed=False,
                gate_name=first_fail,
                score=min_score,
                reason=f"Pipeline failed at gate '{first_fail}'",
                details=all_details,
            )

    async def run_all(self, task: Task, result: TaskResult) -> list[ValidationResult]:
        """Run all gates and return individual results (no short-circuit).

        Useful for reporting/diagnostics when you want every gate's verdict.
        """
        results: list[ValidationResult] = []
        for gate in self._gates:
            gate_result = await gate.check(task, result)
            results.append(gate_result)
        return results
