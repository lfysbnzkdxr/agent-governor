"""Governor - main entry point for Agent Governor.

Provides the primary API for governed LLM execution with
budget control, audit logging, and model routing.
"""

from __future__ import annotations

import asyncio
from typing import Any

import structlog

from agent_governor.config import GovernorConfig
from agent_governor.control.dag import DagExecutionEngine, TaskGraph
from agent_governor.data.dispatcher import Dispatcher
from agent_governor.models import DagExecutionResult, Task, TaskResult
from agent_governor.observability.audit import AuditLogger
from agent_governor.observability.cost import CostTracker

logger = structlog.get_logger(__name__)


class Governor:
    """Main entry point for Agent Governor.

    Provides governed LLM execution with:
        - Budget control (pre-call check + post-call tracking)
        - Model whitelist enforcement
        - Automatic fallback routing
        - Structured audit logging
        - Cost tracking with per-model breakdown

    Example:
        governor = Governor(
            budget_usd=10.0,
            models=["gpt-4o", "qwen-plus"],
            fallback_model="qwen-turbo",
        )

        result = await governor.execute(
            messages=[{"role": "user", "content": "Hello!"}],
            model="gpt-4o",
        )
        print(result.content)
    """

    def __init__(
        self,
        *,
        budget_usd: float = 0.0,
        models: list[str] | None = None,
        fallback_model: str = "",
        default_model: str = "gpt-4o",
        max_concurrency: int = 10,
        default_timeout: float = 60.0,
        max_retries: int = 2,
        audit_enabled: bool = True,
        audit_log_file: str = "",
        config: GovernorConfig | None = None,
    ) -> None:
        """Initialize Governor.

        Can be configured via keyword arguments or a GovernorConfig instance.
        If both are provided, the config instance takes precedence.
        """
        if config is not None:
            self._config = config
        else:
            self._config = GovernorConfig(
                budget_usd=budget_usd,
                models=models or [],
                fallback_model=fallback_model,
                default_model=default_model,
                max_concurrency=max_concurrency,
                default_timeout=default_timeout,
                max_retries=max_retries,
                audit_enabled=audit_enabled,
                audit_log_file=audit_log_file,
            )

        self._cost_tracker = CostTracker()
        self._audit_logger = AuditLogger(
            enabled=self._config.audit_enabled,
            log_file=self._config.audit_log_file,
        )
        self._dispatcher = Dispatcher(
            config=self._config,
            cost_tracker=self._cost_tracker,
            audit_logger=self._audit_logger,
        )

        logger.info(
            "governor_initialized",
            budget_usd=self._config.budget_usd or "unlimited",
            models=self._config.models or "all",
            fallback_model=self._config.fallback_model or "none",
        )

    @property
    def config(self) -> GovernorConfig:
        """Get the governor configuration."""
        return self._config

    @property
    def cost_tracker(self) -> CostTracker:
        """Get the cost tracker instance."""
        return self._cost_tracker

    @property
    def audit_logger(self) -> AuditLogger:
        """Get the audit logger instance."""
        return self._audit_logger

    @property
    def dispatcher(self) -> Dispatcher:
        """Get the dispatcher instance."""
        return self._dispatcher

    async def execute(
        self,
        messages: list[dict[str, str]],
        *,
        model: str = "",
        name: str = "",
        timeout: float | None = None,
        **params: Any,
    ) -> str:
        """Execute an LLM call with governance controls.

        This is the primary API - a governed replacement for litellm.completion().

        Args:
            messages: Chat messages in OpenAI format.
            model: Model to use. Falls back to config.default_model.
            name: Caller name for audit logging.
            timeout: Override default timeout.
            **params: Additional parameters passed to litellm (temperature, etc.)

        Returns:
            The LLM response content as a string.

        Raises:
            BudgetExceededError: If budget is exceeded.
            ModelNotAllowedError: If model is not whitelisted.
            ExecutionTimeoutError: If the call times out.
        """
        task = Task(
            name=name,
            model=model or self._config.default_model,
            messages=messages,
            params=params,
            timeout=timeout,
        )

        result = await self._dispatcher.dispatch(task)

        if result.error:
            raise RuntimeError(f"Execution failed: {result.error}")

        return result.content

    async def execute_task(self, task: Task) -> TaskResult:
        """Execute a Task object and return the full TaskResult.

        For advanced usage when you need access to token usage, cost, etc.

        Args:
            task: The task to execute.

        Returns:
            Full TaskResult with metadata.
        """
        return await self._dispatcher.dispatch(task)

    def execute_sync(
        self,
        messages: list[dict[str, str]],
        *,
        model: str = "",
        name: str = "",
        timeout: float | None = None,
        **params: Any,
    ) -> str:
        """Synchronous convenience wrapper for execute().

        Creates or uses an event loop to run the async execute method.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Already in an async context - create a new thread
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    asyncio.run,
                    self.execute(messages, model=model, name=name, timeout=timeout, **params),
                )
                return future.result()
        else:
            return asyncio.run(
                self.execute(messages, model=model, name=name, timeout=timeout, **params)
            )

    def get_stats(self) -> dict[str, Any]:
        """Get governor statistics including cost and usage."""
        return {
            **self._dispatcher.get_stats(),
            "audit_records": self._audit_logger.total_records,
            "config": {
                "budget_usd": self._config.budget_usd,
                "models": self._config.models,
                "fallback_model": self._config.fallback_model,
                "max_concurrency": self._config.max_concurrency,
            },
        }

    def reset(self) -> None:
        """Reset all tracking data (cost, audit records)."""
        self._cost_tracker.reset()
        self._audit_logger.clear()
        logger.info("governor_reset")

    async def run_graph(
        self,
        graph: TaskGraph,
        *,
        context: dict[str, Any] | None = None,
        default_timeout: float = 120.0,
        fail_fast: bool = True,
    ) -> DagExecutionResult:
        """Execute a TaskGraph (DAG) with governance controls.

        This is the Level 2 API for multi-step agent workflows.
        Each node in the graph is executed through the governor's dispatcher,
        inheriting budget control, model whitelist, and audit logging.

        Args:
            graph: The task graph to execute.
            context: Shared context accessible by all nodes.
            default_timeout: Default timeout per node.
            fail_fast: If True, skip remaining nodes on first failure.

        Returns:
            DagExecutionResult with per-node results.

        Example:
            graph = TaskGraph("my_workflow")
            graph.add_node("step1", task=task1)
            graph.add_node("step2", task=task2, depends_on=["step1"])
            result = await governor.run_graph(graph)
        """

        async def _default_executor(task: Task) -> TaskResult:
            return await self._dispatcher.dispatch(task)

        engine = DagExecutionEngine(
            default_executor=_default_executor,
            default_timeout=default_timeout,
            fail_fast=fail_fast,
        )
        return await engine.execute(graph, context=context)
