"""Tests for quality gate validation framework."""

import pytest
from pydantic import BaseModel

from agent_governor.control.validator import ValidationPipeline
from agent_governor.control.validators.budget import BudgetGate
from agent_governor.control.validators.llm_judge import LLMJudgeGate
from agent_governor.control.validators.schema import SchemaGate
from agent_governor.models import Task, TaskResult, TaskStatus, TokenUsage, ValidationResult


# === Fixtures ===


def make_task(**kwargs) -> Task:
    return Task(name="test_task", messages=[{"role": "user", "content": "hello"}], **kwargs)


def make_result(content: str = "ok", cost: float = 0.01, tokens: int = 100, **kwargs) -> TaskResult:
    return TaskResult(
        task_id="t1",
        status=TaskStatus.COMPLETED,
        content=content,
        model="gpt-4o",
        usage=TokenUsage(prompt_tokens=tokens // 2, completion_tokens=tokens // 2),
        cost_usd=cost,
        latency_ms=500.0,
        **kwargs,
    )


class ReportSchema(BaseModel):
    title: str
    summary: str


class AlwaysPassGate:
    @property
    def name(self) -> str:
        return "always_pass"

    async def check(self, task: Task, result: TaskResult) -> ValidationResult:
        return ValidationResult(passed=True, gate_name=self.name, score=1.0)


class AlwaysFailGate:
    @property
    def name(self) -> str:
        return "always_fail"

    async def check(self, task: Task, result: TaskResult) -> ValidationResult:
        return ValidationResult(passed=False, gate_name=self.name, score=0.0, reason="forced fail")


# === ValidationPipeline Tests ===


class TestValidationPipeline:
    async def test_empty_pipeline_passes(self):
        pipeline = ValidationPipeline()
        result = await pipeline.run(make_task(), make_result())
        assert result.passed is True
        assert result.gate_name == "empty_pipeline"

    async def test_single_passing_gate(self):
        pipeline = ValidationPipeline([AlwaysPassGate()])
        result = await pipeline.run(make_task(), make_result())
        assert result.passed is True
        assert result.score == 1.0

    async def test_single_failing_gate_short_circuits(self):
        pipeline = ValidationPipeline([AlwaysFailGate(), AlwaysPassGate()])
        result = await pipeline.run(make_task(), make_result())
        assert result.passed is False
        assert result.gate_name == "always_fail"
        # Second gate should not appear in details due to short-circuit
        assert "always_pass" not in result.details

    async def test_no_short_circuit_runs_all(self):
        pipeline = ValidationPipeline(
            [AlwaysFailGate(), AlwaysPassGate()], short_circuit=False
        )
        result = await pipeline.run(make_task(), make_result())
        assert result.passed is False
        assert "always_pass" in result.details
        assert "always_fail" in result.details

    async def test_add_gate_chaining(self):
        pipeline = ValidationPipeline()
        pipeline.add_gate(AlwaysPassGate()).add_gate(AlwaysPassGate())
        assert len(pipeline.gates) == 2

    async def test_run_all_returns_individual_results(self):
        pipeline = ValidationPipeline([AlwaysPassGate(), AlwaysFailGate()])
        results = await pipeline.run_all(make_task(), make_result())
        assert len(results) == 2
        assert results[0].passed is True
        assert results[1].passed is False


# === SchemaGate Tests ===


class TestSchemaGate:
    async def test_valid_json_passes(self):
        gate = SchemaGate(ReportSchema)
        content = '{"title": "AI Report", "summary": "A summary"}'
        result = await gate.check(make_task(), make_result(content=content))
        assert result.passed is True
        assert result.score == 1.0

    async def test_invalid_json_fails(self):
        gate = SchemaGate(ReportSchema)
        result = await gate.check(make_task(), make_result(content="not json"))
        assert result.passed is False
        assert "not valid JSON" in result.reason

    async def test_schema_mismatch_fails(self):
        gate = SchemaGate(ReportSchema)
        content = '{"title": "Report"}'  # missing 'summary'
        result = await gate.check(make_task(), make_result(content=content))
        assert result.passed is False
        assert "error(s)" in result.reason

    async def test_gate_name_includes_schema(self):
        gate = SchemaGate(ReportSchema)
        assert gate.name == "schema_gate_ReportSchema"


# === BudgetGate Tests ===


class TestBudgetGate:
    async def test_within_budget_passes(self):
        gate = BudgetGate(max_cost_usd=1.0, max_tokens=1000)
        result = await gate.check(make_task(), make_result(cost=0.01, tokens=100))
        assert result.passed is True

    async def test_cost_exceeded_fails(self):
        gate = BudgetGate(max_cost_usd=0.005)
        result = await gate.check(make_task(), make_result(cost=0.01))
        assert result.passed is False
        assert "cost" in result.reason

    async def test_tokens_exceeded_fails(self):
        gate = BudgetGate(max_tokens=50)
        result = await gate.check(make_task(), make_result(tokens=100))
        assert result.passed is False
        assert "tokens" in result.reason

    async def test_latency_exceeded_fails(self):
        gate = BudgetGate(max_latency_ms=100.0)
        result = await gate.check(make_task(), make_result())  # 500ms
        assert result.passed is False
        assert "latency" in result.reason

    async def test_no_limits_always_passes(self):
        gate = BudgetGate()
        result = await gate.check(make_task(), make_result(cost=999, tokens=99999))
        assert result.passed is True


# === LLMJudgeGate Tests ===


class TestLLMJudgeGate:
    async def test_high_score_passes(self):
        async def mock_llm(messages):
            return '{"score": 0.9, "reason": "good quality"}'

        gate = LLMJudgeGate(llm_call=mock_llm, threshold=0.7)
        result = await gate.check(make_task(), make_result(content="great answer"))
        assert result.passed is True
        assert result.score == 0.9

    async def test_low_score_fails(self):
        async def mock_llm(messages):
            return '{"score": 0.3, "reason": "poor quality"}'

        gate = LLMJudgeGate(llm_call=mock_llm, threshold=0.7)
        result = await gate.check(make_task(), make_result(content="bad answer"))
        assert result.passed is False
        assert result.score == 0.3

    async def test_llm_failure_is_inconclusive_pass(self):
        async def mock_llm(messages):
            raise RuntimeError("API down")

        gate = LLMJudgeGate(llm_call=mock_llm)
        result = await gate.check(make_task(), make_result())
        assert result.passed is True  # Inconclusive = pass
        assert "inconclusive" in result.reason.lower()

    async def test_numeric_fallback_parse(self):
        async def mock_llm(messages):
            return "I rate this 8 out of 10"

        gate = LLMJudgeGate(llm_call=mock_llm, threshold=0.7)
        result = await gate.check(make_task(), make_result())
        assert result.score == 0.8  # 8/10
