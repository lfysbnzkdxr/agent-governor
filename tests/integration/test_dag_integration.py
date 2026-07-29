"""Integration tests: DAG + quality gates + checkpoint recovery."""

import pytest
from pydantic import BaseModel

from agent_governor.control.commander import Commander
from agent_governor.control.dag import DagExecutionEngine, TaskGraph
from agent_governor.control.state import StateManager
from agent_governor.control.validator import ValidationPipeline
from agent_governor.control.validators.budget import BudgetGate
from agent_governor.control.validators.schema import SchemaGate
from agent_governor.models import (
    DagNodeStatus,
    Task,
    TaskResult,
    TaskStatus,
    TokenUsage,
    ValidationResult,
)

# === Test Helpers ===


class AnalysisOutput(BaseModel):
    summary: str
    confidence: float


def make_success_executor(content: str = '{"summary": "ok", "confidence": 0.95}'):
    async def executor(task: Task) -> TaskResult:
        return TaskResult(
            task_id=task.id,
            status=TaskStatus.COMPLETED,
            content=content,
            model="mock",
            usage=TokenUsage(prompt_tokens=50, completion_tokens=50),
            cost_usd=0.005,
            latency_ms=100.0,
        )

    return executor


def make_fail_executor():
    async def executor(task: Task) -> TaskResult:
        raise RuntimeError("simulated failure")

    return executor


# === Integration Tests ===


class TestDagWithQualityGates:
    """Test 3-node DAG with quality gate validation."""

    async def test_three_node_dag_all_pass(self):
        """search -> analyze -> validate, all succeed."""
        graph = TaskGraph("research_pipeline")
        graph.add_node(
            "search",
            task=Task(name="search", messages=[{"role": "user", "content": "find papers"}]),
            executor=make_success_executor('{"results": ["paper1", "paper2"]}'),
        )
        graph.add_node(
            "analyze",
            task=Task(name="analyze"),
            executor=make_success_executor('{"summary": "AI is cool", "confidence": 0.92}'),
            depends_on=["search"],
            gate=SchemaGate(AnalysisOutput),
        )
        graph.add_node(
            "budget_check",
            task=Task(name="budget_check"),
            executor=make_success_executor(),
            depends_on=["analyze"],
            gate=BudgetGate(max_cost_usd=1.0, max_tokens=500),
        )

        engine = DagExecutionEngine(default_executor=make_success_executor())
        result = await engine.execute(graph)

        assert result.status == TaskStatus.COMPLETED
        assert result.node_results["search"].status == DagNodeStatus.COMPLETED
        assert result.node_results["analyze"].status == DagNodeStatus.COMPLETED
        assert result.node_results["analyze"].validation.passed is True
        assert result.node_results["budget_check"].status == DagNodeStatus.COMPLETED

    async def test_dag_fails_on_schema_gate(self):
        """analyze node produces invalid schema -> pipeline fails."""
        graph = TaskGraph("fail_pipeline")
        graph.add_node(
            "search",
            executor=make_success_executor(),
        )
        graph.add_node(
            "analyze",
            executor=make_success_executor("not valid json at all"),
            depends_on=["search"],
            gate=SchemaGate(AnalysisOutput),
        )
        graph.add_node(
            "report",
            executor=make_success_executor(),
            depends_on=["analyze"],
        )

        engine = DagExecutionEngine(default_executor=make_success_executor(), fail_fast=True)
        result = await engine.execute(graph)

        assert result.status == TaskStatus.FAILED
        assert result.node_results["analyze"].status == DagNodeStatus.FAILED
        assert result.node_results["analyze"].validation.passed is False
        assert result.node_results["report"].status == DagNodeStatus.SKIPPED

    async def test_dag_with_validation_pipeline(self):
        """Node with multi-gate validation pipeline."""
        pipeline = ValidationPipeline(
            [
                SchemaGate(AnalysisOutput),
                BudgetGate(max_cost_usd=1.0),
            ]
        )

        # Create a gate adapter that runs the pipeline
        class PipelineGate:
            @property
            def name(self) -> str:
                return "pipeline_gate"

            async def check(self, task, result) -> ValidationResult:
                return await pipeline.run(task, result)

        graph = TaskGraph("multi_gate")
        graph.add_node(
            "analyze",
            executor=make_success_executor('{"summary": "good", "confidence": 0.88}'),
            gate=PipelineGate(),
        )

        engine = DagExecutionEngine(default_executor=make_success_executor())
        result = await engine.execute(graph)

        assert result.status == TaskStatus.COMPLETED
        assert result.node_results["analyze"].validation.passed is True


class TestDagWithCheckpointRecovery:
    """Test DAG execution with checkpoint save and crash recovery."""

    async def test_checkpoint_save_and_recover(self, tmp_path):
        """Simulate crash after first node, then recover."""
        db_path = str(tmp_path / "checkpoints.db")

        # Phase 1: Execute partially (first node succeeds, second "crashes")
        state = StateManager(db_path)
        await state.initialize()

        graph = TaskGraph("recovery_test")
        graph.add_node("step1", executor=make_success_executor("step1_done"))
        graph.add_node("step2", executor=make_fail_executor(), depends_on=["step1"])

        engine = DagExecutionEngine(default_executor=make_success_executor())
        result1 = await engine.execute(graph)

        # Save checkpoints
        await state.save_graph_state(graph.id, result1.status.value)
        for nid, nr in result1.node_results.items():
            await state.save_node(graph.id, nid, nr)

        assert result1.node_results["step1"].status == DagNodeStatus.COMPLETED
        assert result1.node_results["step2"].status == DagNodeStatus.FAILED

        # Phase 2: "Recover" - verify we can read back state
        completed = await state.get_completed_nodes(graph.id)
        assert "step1" in completed
        assert "step2" not in completed

        # Phase 3: Re-execute with fixed executor
        graph2 = TaskGraph("recovery_test")
        graph2._id = graph.id  # Same ID for recovery
        graph2.add_node("step1", executor=make_success_executor("step1_done"))
        graph2.add_node(
            "step2", executor=make_success_executor("step2_fixed"), depends_on=["step1"]
        )

        result2 = await engine.execute(graph2)
        assert result2.status == TaskStatus.COMPLETED

        await state.close()


class TestCommanderIntegration:
    """Test Commander high-level orchestration."""

    async def test_execute_steps(self):
        commander = Commander(executor=make_success_executor())
        result = await commander.execute_steps(
            [
                {"id": "a", "task": Task(name="a")},
                {"id": "b", "task": Task(name="b"), "depends_on": ["a"]},
                {"id": "c", "task": Task(name="c"), "depends_on": ["a"]},
            ]
        )

        assert result.status == TaskStatus.COMPLETED
        assert len(result.node_results) == 3

    async def test_execute_with_state_manager(self, tmp_path):
        db_path = str(tmp_path / "cmd.db")
        state = StateManager(db_path)
        await state.initialize()

        commander = Commander(
            executor=make_success_executor(),
            state_manager=state,
        )

        graph = TaskGraph("cmd_test")
        graph.add_node("x", executor=make_success_executor())
        graph.add_node("y", executor=make_success_executor(), depends_on=["x"])

        result = await commander.execute_graph(graph)
        assert result.status == TaskStatus.COMPLETED

        # Verify state was persisted
        graph_state = await state.load_graph_state(graph.id)
        assert graph_state is not None
        assert graph_state["status"] == "completed"

        checkpoints = await state.load_node_checkpoints(graph.id)
        assert len(checkpoints) == 2

        await state.close()

    async def test_decompose_and_execute_single_node(self):
        """Without decompose_fn, runs as single node."""
        commander = Commander(executor=make_success_executor())
        result = await commander.decompose_and_execute("Do something complex")

        assert result.status == TaskStatus.COMPLETED
        assert "main" in result.node_results

    async def test_decompose_and_execute_with_fn(self):
        """With custom decompose function."""

        async def mock_decompose(description: str):
            return [
                {"id": "research", "description": "Search for info", "depends_on": []},
                {"id": "write", "description": "Write report", "depends_on": ["research"]},
            ]

        commander = Commander(executor=make_success_executor())
        result = await commander.decompose_and_execute(
            "Write a report",
            decompose_fn=mock_decompose,
        )

        assert result.status == TaskStatus.COMPLETED
        assert "research" in result.node_results
        assert "write" in result.node_results

    async def test_decompose_empty_subtasks_raises(self):
        """decompose_and_execute should raise ValueError for empty decomposition."""

        async def empty_decompose(description: str):
            return []

        commander = Commander(executor=make_success_executor())
        with pytest.raises(ValueError, match="no subtasks"):
            await commander.decompose_and_execute(
                "Do something",
                decompose_fn=empty_decompose,
            )

    async def test_recover_without_state_manager_raises(self):
        """recover_and_execute should raise RuntimeError without state manager."""
        commander = Commander(executor=make_success_executor())
        graph = TaskGraph("no_state")
        graph.add_node("a", executor=make_success_executor())

        with pytest.raises(RuntimeError, match="StateManager required"):
            await commander.recover_and_execute(graph)

    async def test_recover_with_no_checkpoints(self, tmp_path):
        """recover_and_execute should execute fresh when no checkpoints exist."""
        db_path = str(tmp_path / "recover.db")
        state = StateManager(db_path)
        await state.initialize()

        commander = Commander(
            executor=make_success_executor(),
            state_manager=state,
        )
        graph = TaskGraph("fresh_recover")
        graph.add_node("a", executor=make_success_executor())

        result = await commander.recover_and_execute(graph)
        assert result.status == TaskStatus.COMPLETED
        assert "a" in result.node_results

        await state.close()

    async def test_execute_graph_with_resume(self, tmp_path):
        """execute_graph with resume=True should check completed nodes."""
        db_path = str(tmp_path / "resume.db")
        state = StateManager(db_path)
        await state.initialize()

        commander = Commander(
            executor=make_success_executor(),
            state_manager=state,
        )
        graph = TaskGraph("resume_test")
        graph.add_node("a", executor=make_success_executor())
        graph.add_node("b", executor=make_success_executor(), depends_on=["a"])

        # First run
        result = await commander.execute_graph(graph)
        assert result.status == TaskStatus.COMPLETED

        # Resume run (should still work, no completed nodes to skip in engine)
        result2 = await commander.execute_graph(graph, resume=True)
        assert result2.status == TaskStatus.COMPLETED

        await state.close()

    async def test_execute_steps_with_options(self):
        """execute_steps should pass through timeout and max_retries."""
        commander = Commander(executor=make_success_executor())
        result = await commander.execute_steps(
            [
                {
                    "id": "a",
                    "task": Task(name="a"),
                    "timeout": 30.0,
                    "max_retries": 2,
                    "metadata": {"key": "val"},
                },
            ]
        )
        assert result.status == TaskStatus.COMPLETED
