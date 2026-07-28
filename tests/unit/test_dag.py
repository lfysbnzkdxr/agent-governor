"""Tests for DAG definition and execution engine."""

import asyncio

import pytest

from agent_governor.control.dag import (
    CycleDetectedError,
    DagExecutionEngine,
    TaskGraph,
)
from agent_governor.models import (
    DagNodeStatus,
    Task,
    TaskResult,
    TaskStatus,
    ValidationResult,
)


# === Helpers ===


def make_executor(content: str = "done", fail: bool = False, delay: float = 0.0):
    """Create a mock executor."""

    async def executor(task: Task) -> TaskResult:
        if delay:
            await asyncio.sleep(delay)
        if fail:
            raise RuntimeError("executor failed")
        return TaskResult(
            task_id=task.id,
            status=TaskStatus.COMPLETED,
            content=content,
            model="mock",
            cost_usd=0.001,
        )

    return executor


class FailGate:
    @property
    def name(self) -> str:
        return "fail_gate"

    async def check(self, task, result) -> ValidationResult:
        return ValidationResult(passed=False, gate_name=self.name, reason="gate failed")


# === TaskGraph Tests ===


class TestTaskGraph:
    def test_add_node_basic(self):
        g = TaskGraph("test")
        g.add_node("a", task=Task(name="a"))
        assert "a" in g.nodes
        assert g.name == "test"

    def test_add_duplicate_node_raises(self):
        g = TaskGraph()
        g.add_node("a")
        with pytest.raises(ValueError, match="already exists"):
            g.add_node("a")

    def test_add_node_missing_dependency_raises(self):
        g = TaskGraph()
        with pytest.raises(ValueError, match="not found"):
            g.add_node("b", depends_on=["a"])

    def test_topological_layers_linear(self):
        g = TaskGraph()
        g.add_node("a")
        g.add_node("b", depends_on=["a"])
        g.add_node("c", depends_on=["b"])
        layers = g.topological_layers()
        assert layers == [["a"], ["b"], ["c"]]

    def test_topological_layers_parallel(self):
        g = TaskGraph()
        g.add_node("a")
        g.add_node("b")
        g.add_node("c", depends_on=["a", "b"])
        layers = g.topological_layers()
        assert len(layers) == 2
        assert set(layers[0]) == {"a", "b"}
        assert layers[1] == ["c"]

    def test_cycle_detection(self):
        g = TaskGraph()
        g.add_node("a")
        g.add_node("b", depends_on=["a"])
        # Manually inject a cycle
        g._edges["a"] = {"b"}
        with pytest.raises(CycleDetectedError):
            g.validate()

    def test_get_dependencies(self):
        g = TaskGraph()
        g.add_node("a")
        g.add_node("b", depends_on=["a"])
        assert g.get_dependencies("b") == {"a"}
        assert g.get_dependencies("a") == set()

    def test_get_dependents(self):
        g = TaskGraph()
        g.add_node("a")
        g.add_node("b", depends_on=["a"])
        g.add_node("c", depends_on=["a"])
        assert set(g.get_dependents("a")) == {"b", "c"}

    def test_chaining_api(self):
        g = TaskGraph("chain")
        result = g.add_node("a").add_node("b", depends_on=["a"])
        assert result is g


# === DagExecutionEngine Tests ===


class TestDagExecutionEngine:
    async def test_single_node_execution(self):
        g = TaskGraph("single")
        g.add_node("n1", task=Task(name="n1"))

        engine = DagExecutionEngine(default_executor=make_executor("hello"))
        result = await engine.execute(g)

        assert result.status == TaskStatus.COMPLETED
        assert result.node_results["n1"].status == DagNodeStatus.COMPLETED
        assert result.node_results["n1"].result.content == "hello"

    async def test_linear_chain_execution(self):
        g = TaskGraph("chain")
        g.add_node("a", task=Task(name="a"))
        g.add_node("b", task=Task(name="b"), depends_on=["a"])
        g.add_node("c", task=Task(name="c"), depends_on=["b"])

        engine = DagExecutionEngine(default_executor=make_executor("step"))
        result = await engine.execute(g)

        assert result.status == TaskStatus.COMPLETED
        assert all(
            nr.status == DagNodeStatus.COMPLETED
            for nr in result.node_results.values()
        )

    async def test_parallel_execution(self):
        call_order = []

        async def tracking_executor(task: Task) -> TaskResult:
            call_order.append(task.name)
            await asyncio.sleep(0.01)
            return TaskResult(task_id=task.id, status=TaskStatus.COMPLETED, content="ok")

        g = TaskGraph("parallel")
        g.add_node("a", task=Task(name="a"))
        g.add_node("b", task=Task(name="b"))
        g.add_node("c", task=Task(name="c"), depends_on=["a", "b"])

        engine = DagExecutionEngine(default_executor=tracking_executor)
        result = await engine.execute(g)

        assert result.status == TaskStatus.COMPLETED
        # c should be last
        assert call_order[-1] == "c"

    async def test_node_failure_fail_fast(self):
        g = TaskGraph("fail")
        g.add_node("a", executor=make_executor(fail=True))
        g.add_node("b", depends_on=["a"])

        engine = DagExecutionEngine(
            default_executor=make_executor(), fail_fast=True
        )
        result = await engine.execute(g)

        assert result.status == TaskStatus.FAILED
        assert result.node_results["a"].status == DagNodeStatus.FAILED
        assert result.node_results["b"].status == DagNodeStatus.SKIPPED

    async def test_node_retry(self):
        attempts = []

        async def flaky_executor(task: Task) -> TaskResult:
            attempts.append(1)
            if len(attempts) < 3:
                raise RuntimeError("flaky")
            return TaskResult(task_id=task.id, status=TaskStatus.COMPLETED, content="ok")

        g = TaskGraph("retry")
        g.add_node("a", executor=flaky_executor, max_retries=3)

        engine = DagExecutionEngine(default_executor=make_executor())
        result = await engine.execute(g)

        assert result.status == TaskStatus.COMPLETED
        assert result.node_results["a"].attempts == 3

    async def test_node_timeout(self):
        g = TaskGraph("timeout")
        g.add_node("slow", executor=make_executor(delay=5.0), timeout=0.1)

        engine = DagExecutionEngine(default_executor=make_executor())
        result = await engine.execute(g)

        assert result.status == TaskStatus.FAILED
        assert "timed out" in result.node_results["slow"].error

    async def test_quality_gate_validation(self):
        g = TaskGraph("gated")
        g.add_node("a", executor=make_executor("content"), gate=FailGate())

        engine = DagExecutionEngine(default_executor=make_executor())
        result = await engine.execute(g)

        assert result.status == TaskStatus.FAILED
        node = result.node_results["a"]
        assert node.status == DagNodeStatus.FAILED
        assert node.validation is not None
        assert node.validation.passed is False

    async def test_no_executor_fails_gracefully(self):
        g = TaskGraph("no_exec")
        g.add_node("a")

        engine = DagExecutionEngine(default_executor=None)
        result = await engine.execute(g)

        assert result.status == TaskStatus.FAILED
        assert "No executor" in result.node_results["a"].error

    async def test_upstream_results_injected(self):
        received_metadata = {}

        async def capture_executor(task: Task) -> TaskResult:
            received_metadata.update(task.metadata)
            return TaskResult(task_id=task.id, status=TaskStatus.COMPLETED, content="from_a")

        g = TaskGraph("inject")
        g.add_node("a", executor=make_executor("upstream_content"))
        g.add_node("b", executor=capture_executor, depends_on=["a"])

        engine = DagExecutionEngine(default_executor=make_executor())
        await engine.execute(g)

        assert "_upstream" in received_metadata
        assert received_metadata["_upstream"]["a"] == "upstream_content"

    async def test_total_cost_accumulated(self):
        g = TaskGraph("cost")
        g.add_node("a", executor=make_executor())
        g.add_node("b", executor=make_executor())

        engine = DagExecutionEngine(default_executor=make_executor())
        result = await engine.execute(g)

        assert result.total_cost_usd == pytest.approx(0.002, abs=1e-6)
