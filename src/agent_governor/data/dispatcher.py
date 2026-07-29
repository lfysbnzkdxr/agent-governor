"""Dispatcher - model routing with budget control and whitelist enforcement.

Wraps litellm for unified LLM access with governance controls.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import litellm
import structlog

from agent_governor.config import GovernorConfig
from agent_governor.exceptions import (
    BudgetExceededError,
    ExecutionTimeoutError,
    GovernorError,
    ModelNotAllowedError,
)
from agent_governor.models import CallRecord, Task, TaskResult, TaskStatus, TokenUsage
from agent_governor.observability.audit import AuditLogger
from agent_governor.observability.cost import CostTracker
from agent_governor.observability.metrics import MetricsCollector
from agent_governor.observability.tracing import TracingManager

logger = structlog.get_logger(__name__)


class Dispatcher:
    """Core dispatcher that routes LLM calls with governance controls.

    Features:
        - Model whitelist enforcement
        - Budget interception (pre-call check)
        - Automatic fallback model routing
        - Concurrency limiting via asyncio.Semaphore
        - Retry with configurable delay
        - Cost tracking and audit logging
    """

    def __init__(
        self,
        config: GovernorConfig,
        cost_tracker: CostTracker | None = None,
        audit_logger: AuditLogger | None = None,
        tracing: TracingManager | None = None,
        metrics: MetricsCollector | None = None,
    ) -> None:
        self._config = config
        self._cost_tracker = cost_tracker or CostTracker()
        self._audit_logger = audit_logger or AuditLogger(enabled=config.audit_enabled)
        self._tracing = tracing
        self._metrics = metrics
        self._semaphore = asyncio.Semaphore(config.max_concurrency)

    @property
    def cost_tracker(self) -> CostTracker:
        return self._cost_tracker

    @property
    def audit_logger(self) -> AuditLogger:
        return self._audit_logger

    async def dispatch(self, task: Task) -> TaskResult:
        """Dispatch a task to the appropriate model with governance controls.

        Args:
            task: The task to execute.

        Returns:
            TaskResult with the LLM response.

        Raises:
            ModelNotAllowedError: If the model is not in the whitelist.
            BudgetExceededError: If the budget has been exceeded.
            ExecutionTimeoutError: If the call exceeds the timeout.
        """
        model = task.model or self._config.default_model

        # 1. Model whitelist check
        if not self._config.is_model_allowed(model):
            raise ModelNotAllowedError(model, self._config.models)

        # 2. Budget pre-check
        self._check_budget()

        # 3. Execute with concurrency control and retry (optionally traced)
        if self._tracing:
            tracer = self._tracing.get_tracer()
            if tracer:
                with tracer.start_as_current_span("dispatcher.dispatch") as span:
                    span.set_attribute("agent_governor.model", model)
                    span.set_attribute("agent_governor.task.name", task.name or "")
                    async with self._semaphore:
                        result = await self._execute_with_retry(task, model)
                    span.set_attribute("agent_governor.status", str(result.status.value))
                    return result

        async with self._semaphore:
            return await self._execute_with_retry(task, model)

    async def _execute_with_retry(self, task: Task, model: str) -> TaskResult:
        """Execute with retry logic and fallback."""
        last_error: Exception | None = None
        attempts = self._config.max_retries + 1

        for attempt in range(attempts):
            try:
                return await self._execute_single(task, model)
            except (BudgetExceededError, ModelNotAllowedError):
                raise  # Don't retry governance errors
            except Exception as e:
                last_error = e
                logger.warning(
                    "dispatch_attempt_failed",
                    task_id=task.id,
                    model=model,
                    attempt=attempt + 1,
                    error=str(e),
                )
                if attempt < attempts - 1:
                    await asyncio.sleep(self._config.retry_delay)

        # Try fallback model if configured
        if self._config.fallback_model and self._config.fallback_model != model:
            if not self._config.is_model_allowed(self._config.fallback_model):
                logger.warning(
                    "fallback_model_not_allowed",
                    task_id=task.id,
                    model=self._config.fallback_model,
                )
            else:
                logger.info(
                    "trying_fallback_model",
                    task_id=task.id,
                    fallback=self._config.fallback_model,
                )
                try:
                    return await self._execute_single(task, self._config.fallback_model)
                except Exception as fallback_err:
                    last_error = fallback_err

        # All attempts failed
        if last_error:
            if self._metrics:
                self._metrics.record_error(model, type(last_error).__name__)
            raise last_error
        return TaskResult(
            task_id=task.id,
            status=TaskStatus.FAILED,
            model=model,
            error=str(last_error) if last_error else "All execution attempts failed",
        )

    async def _execute_single(self, task: Task, model: str) -> TaskResult:
        """Execute a single LLM call via litellm."""
        timeout = task.timeout or self._config.default_timeout
        start_time = time.perf_counter()

        try:
            response = await asyncio.wait_for(
                litellm.acompletion(
                    model=model,
                    messages=task.messages,
                    **task.params,
                ),
                timeout=timeout,
            )
        except asyncio.TimeoutError as e:
            raise ExecutionTimeoutError(task.id, timeout) from e

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Extract usage info
        usage = response.usage
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0

        # Record cost
        cost = self._cost_tracker.record_usage(model, prompt_tokens, completion_tokens)

        # Extract content
        content = ""
        if response.choices:
            content = response.choices[0].message.content or ""

        result = TaskResult(
            task_id=task.id,
            status=TaskStatus.COMPLETED,
            content=content,
            model=model,
            usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            ),
            latency_ms=elapsed_ms,
            cost_usd=cost,
            metadata=task.metadata,
        )

        # Audit log
        await self._audit_logger.log(
            CallRecord(
                caller=task.name or task.id,
                tenant_id=task.tenant_id,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                cost_usd=cost,
                latency_ms=elapsed_ms,
                status="success",
            )
        )

        # Record metrics
        if self._metrics:
            self._metrics.record_call(model, "success")
            self._metrics.record_latency(model, elapsed_ms)
            self._metrics.record_tokens(model, prompt_tokens + completion_tokens)

        logger.info(
            "dispatch_success",
            task_id=task.id,
            model=model,
            tokens=prompt_tokens + completion_tokens,
            cost_usd=f"${cost:.6f}",
            latency_ms=f"{elapsed_ms:.1f}",
        )

        return result

    def _check_budget(self) -> None:
        """Check if budget allows further calls."""
        budget = self._config.budget_usd
        if budget <= 0:
            return  # No budget limit

        if not self._cost_tracker.check_budget(budget):
            if self._metrics:
                self._metrics.record_budget_exceeded()
            raise BudgetExceededError(
                current_cost=self._cost_tracker.total_cost,
                budget=budget,
            )

    def get_stats(self) -> dict[str, Any]:
        """Get dispatcher statistics."""
        return {
            "total_cost_usd": self._cost_tracker.total_cost,
            "total_calls": self._cost_tracker.total_calls,
            "budget_usd": self._config.budget_usd,
            "budget_remaining": max(
                0, self._config.budget_usd - self._cost_tracker.total_cost
            )
            if self._config.budget_usd > 0
            else float("inf"),
        }
