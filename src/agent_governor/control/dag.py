"""DAG definition and async execution engine.

Provides TaskGraph for defining task dependency graphs (DSL) and
DagExecutionEngine for executing them with Kahn's layered parallelism,
per-node retry, timeout, and quality gate validation.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Callable, Coroutine
from typing import TYPE_CHECKING, Any

import structlog

from agent_governor.exceptions import GovernorError
from agent_governor.models import (
    DagExecutionResult,
    DagNodeResult,
    DagNodeStatus,
    Task,
    TaskResult,
    TaskStatus,
)

if TYPE_CHECKING:
    from agent_governor.observability.tracing import TracingManager
    from agent_governor.protocols import QualityGate

logger = structlog.get_logger(__name__)

# Type for node executor: takes Task, returns TaskResult
NodeExecutor = Callable[[Task], Coroutine[Any, Any, TaskResult]]


class CycleDetectedError(GovernorError):
    """Raised when a cycle is detected in the task graph."""

    def __init__(self, cycle: list[str] | None = None) -> None:
        self.cycle = cycle or []
        super().__init__(
            f"Cycle detected in task graph: {' -> '.join(self.cycle)}",
            detail=f"cycle={self.cycle}",
        )


class TaskGraph:
    """Directed Acyclic Graph for multi-step task orchestration.

    Provides a fluent DSL for defining tasks and their dependencies.
    Uses networkx internally for graph operations.

    Example:
        graph = TaskGraph("research_report")
        graph.add_node("search", task=search_task)
        graph.add_node("analyze", task=analyze_task, depends_on=["search"])
        graph.add_node("summarize", task=summary_task, depends_on=["analyze"])
        graph.validate()  # Raises CycleDetectedError if cyclic
    """

    def __init__(self, name: str = "", graph_id: str = "") -> None:
        """Initialize a TaskGraph.

        Args:
            name: Human-readable graph name.
            graph_id: Unique identifier. Auto-generated if not provided.
        """
        self._name = name
        self._id = graph_id or uuid.uuid4().hex[:12]
        self._nodes: dict[str, _NodeDef] = {}
        self._edges: dict[str, set[str]] = {}  # node -> set of dependencies

    @property
    def id(self) -> str:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @property
    def nodes(self) -> dict[str, _NodeDef]:
        return dict(self._nodes)

    def add_node(
        self,
        node_id: str,
        *,
        task: Task | None = None,
        executor: NodeExecutor | None = None,
        gate: QualityGate | None = None,
        depends_on: list[str] | None = None,
        max_retries: int = 0,
        timeout: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TaskGraph:
        """Add a node to the graph.

        Args:
            node_id: Unique node identifier.
            task: Task to execute at this node.
            executor: Custom executor function. If None, uses the engine's default.
            gate: Quality gate to validate the node result.
            depends_on: List of node IDs this node depends on.
            max_retries: Max retries for this node.
            timeout: Timeout in seconds for this node.
            metadata: Additional metadata.

        Returns:
            Self for chaining.

        Raises:
            ValueError: If node_id already exists or dependency not found.
        """
        if node_id in self._nodes:
            raise ValueError(f"Node '{node_id}' already exists in graph")

        deps = set(depends_on or [])
        for dep in deps:
            if dep not in self._nodes:
                raise ValueError(
                    f"Dependency '{dep}' not found. Add dependencies before dependents."
                )

        self._nodes[node_id] = _NodeDef(
            node_id=node_id,
            task=task,
            executor=executor,
            gate=gate,
            max_retries=max_retries,
            timeout=timeout,
            metadata=metadata or {},
        )
        self._edges[node_id] = deps

        return self

    def validate(self) -> None:
        """Validate the graph has no cycles.

        Raises:
            CycleDetectedError: If a cycle is detected.
        """
        cycle = self._detect_cycle()
        if cycle:
            raise CycleDetectedError(cycle)

    def topological_layers(self) -> list[list[str]]:
        """Get nodes grouped by topological layers (Kahn's algorithm).

        Returns a list of layers, where each layer contains node IDs
        that can be executed in parallel.

        Raises:
            CycleDetectedError: If the graph contains a cycle.
        """
        # Compute in-degrees
        in_degree: dict[str, int] = {node: 0 for node in self._nodes}
        dependents: dict[str, list[str]] = {node: [] for node in self._nodes}

        for node, deps in self._edges.items():
            in_degree[node] = len(deps)
            for dep in deps:
                dependents[dep].append(node)

        # Kahn's algorithm - layered
        layers: list[list[str]] = []
        queue = [n for n, d in in_degree.items() if d == 0]

        processed = 0
        while queue:
            layers.append(sorted(queue))
            next_queue: list[str] = []
            for node in queue:
                processed += 1
                for dependent in dependents[node]:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        next_queue.append(dependent)
            queue = next_queue

        if processed != len(self._nodes):
            cycle = self._detect_cycle()
            raise CycleDetectedError(cycle)

        return layers

    def get_dependencies(self, node_id: str) -> set[str]:
        """Get direct dependencies of a node."""
        return set(self._edges.get(node_id, set()))

    def get_dependents(self, node_id: str) -> list[str]:
        """Get nodes that depend on the given node."""
        return [n for n, deps in self._edges.items() if node_id in deps]

    def _detect_cycle(self) -> list[str] | None:
        """Detect cycle using DFS. Returns cycle path if found."""
        white, gray, black = 0, 1, 2
        color: dict[str, int] = {n: white for n in self._nodes}
        parent: dict[str, str | None] = {n: None for n in self._nodes}

        def dfs(node: str) -> list[str] | None:
            color[node] = gray
            for dep in self._edges.get(node, set()):
                if dep not in color:
                    continue
                if color[dep] == gray:
                    # Found cycle - reconstruct path
                    cycle = [dep, node]
                    current = node
                    while parent[current] is not None and parent[current] != dep:
                        current = parent[current]  # type: ignore[assignment]
                        cycle.append(current)
                    cycle.reverse()
                    return cycle
                if color[dep] == white:
                    parent[dep] = node
                    result = dfs(dep)
                    if result:
                        return result
            color[node] = black
            return None

        for node in self._nodes:
            if color[node] == white:
                result = dfs(node)
                if result:
                    return result
        return None


class _NodeDef:
    """Internal node definition."""

    __slots__ = (
        "node_id",
        "task",
        "executor",
        "gate",
        "max_retries",
        "timeout",
        "metadata",
    )

    def __init__(
        self,
        node_id: str,
        task: Task | None = None,
        executor: NodeExecutor | None = None,
        gate: QualityGate | None = None,
        max_retries: int = 0,
        timeout: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.node_id = node_id
        self.task = task
        self.executor = executor
        self.gate = gate
        self.max_retries = max_retries
        self.timeout = timeout
        self.metadata = metadata or {}


class DagExecutionEngine:
    """Async execution engine for TaskGraph.

    Executes nodes in topological layers with:
    - Layer-parallel execution via asyncio.TaskGroup
    - Per-node retry with configurable attempts
    - Per-node timeout
    - Quality gate validation after execution
    - Upstream result injection into downstream tasks

    Example:
        engine = DagExecutionEngine(default_executor=my_executor)
        result = await engine.execute(graph)
    """

    def __init__(
        self,
        default_executor: NodeExecutor | None = None,
        *,
        default_timeout: float = 120.0,
        default_max_retries: int = 1,
        fail_fast: bool = True,
        tracing: TracingManager | None = None,
    ) -> None:
        """Initialize the execution engine.

        Args:
            default_executor: Default executor for nodes without a custom one.
            default_timeout: Default timeout per node in seconds.
            default_max_retries: Default retry count per node.
            fail_fast: If True, cancel remaining nodes on first failure.
            tracing: Optional TracingManager for OTel span creation.
        """
        self._default_executor = default_executor
        self._default_timeout = default_timeout
        self._default_max_retries = default_max_retries
        self._fail_fast = fail_fast
        self._tracing = tracing

    async def execute(
        self,
        graph: TaskGraph,
        *,
        context: dict[str, Any] | None = None,
    ) -> DagExecutionResult:
        """Execute a task graph.

        Args:
            graph: The task graph to execute.
            context: Shared context dict accessible by all nodes.

        Returns:
            DagExecutionResult with per-node results.
        """
        graph.validate()
        layers = graph.topological_layers()
        ctx = context or {}

        exec_result = DagExecutionResult(graph_id=graph.id)
        node_results: dict[str, DagNodeResult] = {}
        start_time = time.perf_counter()
        has_failure = False

        logger.info(
            "dag_execution_started",
            graph_id=graph.id,
            graph_name=graph.name,
            layers=len(layers),
            total_nodes=len(graph.nodes),
        )

        for _layer_idx, layer in enumerate(layers):
            if has_failure and self._fail_fast:
                # Skip remaining layers
                for node_id in layer:
                    node_results[node_id] = DagNodeResult(
                        node_id=node_id,
                        status=DagNodeStatus.SKIPPED,
                        error="Skipped due to upstream failure",
                    )
                continue

            # Execute all nodes in this layer concurrently
            async with asyncio.TaskGroup() as tg:
                tasks_map: dict[str, asyncio.Task[DagNodeResult]] = {}
                for node_id in layer:
                    node_def = graph.nodes[node_id]
                    t = tg.create_task(self._execute_node(node_def, node_results, ctx))
                    tasks_map[node_id] = t

            # Collect results
            for node_id, t in tasks_map.items():
                node_result = t.result()
                node_results[node_id] = node_result
                if node_result.status == DagNodeStatus.FAILED:
                    has_failure = True

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Compute totals
        total_cost = sum(
            nr.result.cost_usd for nr in node_results.values() if nr.result is not None
        )

        exec_result.node_results = node_results
        exec_result.total_latency_ms = elapsed_ms
        exec_result.total_cost_usd = total_cost
        exec_result.status = TaskStatus.FAILED if has_failure else TaskStatus.COMPLETED

        if has_failure:
            failed_nodes = [
                nid for nid, nr in node_results.items() if nr.status == DagNodeStatus.FAILED
            ]
            exec_result.error = f"Failed nodes: {failed_nodes}"

        logger.info(
            "dag_execution_completed",
            graph_id=graph.id,
            status=exec_result.status.value,
            latency_ms=f"{elapsed_ms:.1f}",
            total_cost=f"${total_cost:.6f}",
        )

        return exec_result

    async def _execute_node(
        self,
        node_def: _NodeDef,
        upstream_results: dict[str, DagNodeResult],
        context: dict[str, Any],
    ) -> DagNodeResult:
        """Execute a single node with retry and validation."""
        node_id = node_def.node_id
        max_retries = node_def.max_retries or self._default_max_retries
        timeout = node_def.timeout or self._default_timeout

        node_result = DagNodeResult(
            node_id=node_id,
            status=DagNodeStatus.RUNNING,
            started_at=time.time(),
        )

        # Resolve executor
        executor = node_def.executor or self._default_executor
        if executor is None:
            node_result.status = DagNodeStatus.FAILED
            node_result.error = "No executor configured for node"
            node_result.completed_at = time.time()
            return node_result

        # Build task with upstream context
        task = node_def.task or Task(name=node_id)
        if node_def.metadata:
            task.metadata.update(node_def.metadata)

        # Inject upstream results into task metadata
        upstream_contents: dict[str, str] = {}
        for dep_id in upstream_results:
            dep_result = upstream_results[dep_id]
            if dep_result.result and dep_result.result.content:
                upstream_contents[dep_id] = dep_result.result.content
        if upstream_contents:
            task.metadata["_upstream"] = upstream_contents
        task.metadata["_context"] = context

        # Execute with retry and optional tracing
        last_error: Exception | None = None
        attempts = max_retries + 1

        for attempt in range(attempts):
            node_result.attempts = attempt + 1
            try:
                if self._tracing:
                    tracer = self._tracing.get_tracer()
                    if tracer:
                        with tracer.start_as_current_span(f"dag.node.{node_id}") as span:
                            span.set_attribute("agent_governor.node.id", node_id)
                            span.set_attribute("agent_governor.task.name", task.name or "")
                            result = await asyncio.wait_for(
                                executor(task),
                                timeout=timeout,
                            )
                            if hasattr(result, "status"):
                                span.set_attribute("agent_governor.status", str(result.status))
                    else:
                        result = await asyncio.wait_for(
                            executor(task),
                            timeout=timeout,
                        )
                else:
                    result = await asyncio.wait_for(
                        executor(task),
                        timeout=timeout,
                    )

                node_result.result = result

                # Run quality gate if configured
                if node_def.gate:
                    validation = await node_def.gate.check(task, result)
                    node_result.validation = validation
                    if not validation.passed:
                        node_result.status = DagNodeStatus.FAILED
                        node_result.error = (
                            f"Validation failed at gate '{validation.gate_name}': "
                            f"{validation.reason}"
                        )
                        node_result.completed_at = time.time()
                        logger.warning(
                            "node_validation_failed",
                            node_id=node_id,
                            gate=validation.gate_name,
                            reason=validation.reason,
                        )
                        return node_result

                # Success
                node_result.status = DagNodeStatus.COMPLETED
                node_result.completed_at = time.time()
                logger.debug("node_completed", node_id=node_id, attempt=attempt + 1)
                return node_result

            except TimeoutError:
                last_error = GovernorError(f"Node '{node_id}' timed out after {timeout}s")
            except Exception as e:
                last_error = e

            if attempt < attempts - 1:
                logger.debug(
                    "node_retry",
                    node_id=node_id,
                    attempt=attempt + 1,
                    error=str(last_error),
                )
                await asyncio.sleep(0.5 * (attempt + 1))  # Linear backoff

        # All attempts failed
        node_result.status = DagNodeStatus.FAILED
        node_result.error = str(last_error) if last_error else "Unknown error"
        node_result.completed_at = time.time()
        logger.warning(
            "node_failed",
            node_id=node_id,
            attempts=attempts,
            error=node_result.error,
        )
        return node_result
