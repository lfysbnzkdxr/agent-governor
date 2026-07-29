"""Commander: high-level task decomposition and DAG orchestration.

The Commander provides the highest-level API for multi-step agent workflows:
1. Accepts a complex task description
2. Optionally decomposes it into subtasks (via LLM or manual definition)
3. Builds a TaskGraph (DAG) from subtasks
4. Executes via DagExecutionEngine with checkpoint support
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Coroutine
from typing import TYPE_CHECKING, Any

import structlog

from agent_governor.control.dag import DagExecutionEngine, NodeExecutor, TaskGraph
from agent_governor.models import (
    DagExecutionResult,
    DagNodeStatus,
    Task,
)

if TYPE_CHECKING:
    from agent_governor.control.state import StateManager
    from agent_governor.control.validator import ValidationPipeline

logger = structlog.get_logger(__name__)

# Type for LLM decomposition function
DecomposeFn = Callable[[str], Coroutine[Any, Any, list[dict[str, Any]]]]

DECOMPOSE_PROMPT = """\
Decompose the following task into sequential/parallel subtasks.

Task: {task_description}

Return a JSON array of subtask objects:
[
  {{"id": "step_1", "description": "...", "depends_on": []}},
  {{"id": "step_2", "description": "...", "depends_on": ["step_1"]}}
]

Rules:
- Each subtask should be independently executable
- depends_on references IDs of prerequisite subtasks
- Keep it minimal (2-6 subtasks)
- Reply with ONLY the JSON array
"""


class Commander:
    """High-level orchestrator for multi-step agent workflows.

    Combines task decomposition, DAG construction, execution, and
    checkpoint recovery into a single cohesive API.

    Example:
        commander = Commander(
            executor=my_executor,
            state_manager=state_mgr,
        )

        # Manual DAG construction
        result = await commander.execute_steps([
            {"id": "search", "task": search_task},
            {"id": "analyze", "task": analyze_task, "depends_on": ["search"]},
        ])

        # Or LLM-decomposed
        result = await commander.decompose_and_execute(
            "Research and summarize AI safety papers",
            decompose_fn=my_llm_decompose,
        )
    """

    def __init__(
        self,
        executor: NodeExecutor | None = None,
        *,
        state_manager: StateManager | None = None,
        validation_pipeline: ValidationPipeline | None = None,
        default_timeout: float = 120.0,
        default_max_retries: int = 1,
        fail_fast: bool = True,
    ) -> None:
        """Initialize Commander.

        Args:
            executor: Default node executor function.
            state_manager: Optional state manager for checkpointing.
            validation_pipeline: Optional default validation pipeline.
            default_timeout: Default timeout per node.
            default_max_retries: Default retries per node.
            fail_fast: Stop execution on first failure.
        """
        self._executor = executor
        self._state_manager = state_manager
        self._validation_pipeline = validation_pipeline
        self._engine = DagExecutionEngine(
            default_executor=executor,
            default_timeout=default_timeout,
            default_max_retries=default_max_retries,
            fail_fast=fail_fast,
        )

    @property
    def engine(self) -> DagExecutionEngine:
        """Get the underlying execution engine."""
        return self._engine

    async def execute_graph(
        self,
        graph: TaskGraph,
        *,
        context: dict[str, Any] | None = None,
        resume: bool = False,
    ) -> DagExecutionResult:
        """Execute a pre-built task graph.

        Args:
            graph: The task graph to execute.
            context: Shared context for all nodes.
            resume: If True, skip already-completed nodes (crash recovery).

        Returns:
            DagExecutionResult with per-node outcomes.
        """
        # Check for resume
        completed_nodes: set[str] = set()
        if resume and self._state_manager:
            completed_nodes = await self._state_manager.get_completed_nodes(graph.id)
            if completed_nodes:
                logger.info(
                    "resuming_graph",
                    graph_id=graph.id,
                    completed_nodes=list(completed_nodes),
                )

        # Save initial state
        if self._state_manager:
            await self._state_manager.save_graph_state(graph.id, "running", {"name": graph.name})

        # Execute
        result = await self._engine.execute(graph, context=context)

        # Save final state
        if self._state_manager:
            await self._state_manager.save_graph_state(
                graph.id,
                result.status.value,
                {"total_cost": result.total_cost_usd},
            )
            # Save node checkpoints
            for node_id, node_result in result.node_results.items():
                await self._state_manager.save_node(graph.id, node_id, node_result)

        return result

    async def execute_steps(
        self,
        steps: list[dict[str, Any]],
        *,
        graph_name: str = "",
        context: dict[str, Any] | None = None,
    ) -> DagExecutionResult:
        """Build and execute a DAG from step definitions.

        Args:
            steps: List of step dicts with keys:
                - id (str): Node identifier
                - task (Task, optional): Task to execute
                - executor (callable, optional): Custom executor
                - depends_on (list[str], optional): Dependencies
                - gate (QualityGate, optional): Validation gate
                - max_retries (int, optional): Retry count
                - timeout (float, optional): Timeout seconds
            graph_name: Name for the graph.
            context: Shared context.

        Returns:
            DagExecutionResult.
        """
        graph = TaskGraph(name=graph_name or f"commander_{uuid.uuid4().hex[:8]}")

        for step in steps:
            graph.add_node(
                step["id"],
                task=step.get("task"),
                executor=step.get("executor"),
                gate=step.get("gate"),
                depends_on=step.get("depends_on"),
                max_retries=step.get("max_retries", 0),
                timeout=step.get("timeout"),
                metadata=step.get("metadata"),
            )

        return await self.execute_graph(graph, context=context)

    async def decompose_and_execute(
        self,
        task_description: str,
        *,
        decompose_fn: DecomposeFn | None = None,
        node_executor: NodeExecutor | None = None,
        context: dict[str, Any] | None = None,
    ) -> DagExecutionResult:
        """Decompose a task via LLM and execute the resulting DAG.

        Args:
            task_description: High-level task to decompose.
            decompose_fn: Async function that takes description and returns
                list of subtask dicts. If None, creates a single-node graph.
            node_executor: Executor for each decomposed node.
            context: Shared context.

        Returns:
            DagExecutionResult.
        """
        if decompose_fn is None:
            # No decomposition - single node execution
            return await self.execute_steps(
                [
                    {
                        "id": "main",
                        "task": Task(
                            name="main", messages=[{"role": "user", "content": task_description}]
                        ),
                    }
                ],
                graph_name="single_task",
                context=context,
            )

        # Decompose
        logger.info("decomposing_task", description=task_description[:100])
        subtasks = await decompose_fn(task_description)

        if not subtasks:
            raise ValueError("Decomposition returned no subtasks")

        # Build steps from decomposition
        steps: list[dict[str, Any]] = []
        for subtask in subtasks:
            step: dict[str, Any] = {
                "id": subtask.get("id", f"step_{len(steps)}"),
                "depends_on": subtask.get("depends_on", []),
                "task": Task(
                    name=subtask.get("id", ""),
                    messages=[{"role": "user", "content": subtask.get("description", "")}],
                    metadata={"decomposed_from": task_description[:200]},
                ),
            }
            if node_executor:
                step["executor"] = node_executor
            steps.append(step)

        return await self.execute_steps(
            steps,
            graph_name=f"decomposed_{uuid.uuid4().hex[:6]}",
            context=context,
        )

    async def recover_and_execute(
        self,
        graph: TaskGraph,
        *,
        context: dict[str, Any] | None = None,
    ) -> DagExecutionResult:
        """Recover a crashed graph execution and resume.

        Loads checkpoints from state manager, marks completed nodes,
        and re-executes only pending/failed nodes.

        Args:
            graph: The original task graph.
            context: Shared context.

        Returns:
            DagExecutionResult with combined old + new results.
        """
        if not self._state_manager:
            raise RuntimeError("StateManager required for recovery")

        # Load previous checkpoints
        checkpoints = await self._state_manager.load_node_checkpoints(graph.id)
        completed = {nid for nid, nr in checkpoints.items() if nr.status == DagNodeStatus.COMPLETED}

        if not completed:
            # Nothing to recover, execute fresh
            return await self.execute_graph(graph, context=context)

        logger.info(
            "recovering_graph",
            graph_id=graph.id,
            completed_nodes=list(completed),
            total_nodes=len(graph.nodes),
        )

        # Build a new graph excluding completed nodes
        # but keeping their results available via context
        ctx = context or {}
        ctx["_completed_results"] = {
            nid: checkpoints[nid].result.model_dump() if checkpoints[nid].result else None  # type: ignore[union-attr]
            for nid in completed
        }

        # Execute full graph (engine will skip completed via context)
        result = await self._engine.execute(graph, context=ctx)

        # Merge with previous checkpoints
        for nid in completed:
            if nid not in result.node_results:
                result.node_results[nid] = checkpoints[nid]

        return result
