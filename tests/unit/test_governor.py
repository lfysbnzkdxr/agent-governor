"""Unit tests for Governor and Dispatcher."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_governor.config import GovernorConfig
from agent_governor.control.dag import TaskGraph
from agent_governor.exceptions import (
    BudgetExceededError,
    ExecutionTimeoutError,
    ModelNotAllowedError,
)
from agent_governor.governor import Governor
from agent_governor.models import Task, TaskResult, TaskStatus
from agent_governor.tenant.config import TenantPolicy


class TestGovernorConfig:
    """Tests for GovernorConfig."""

    def test_default_config(self) -> None:
        config = GovernorConfig()
        assert config.budget_usd == 0.0
        assert config.models == []
        assert config.default_model == "gpt-4o"
        assert config.max_concurrency == 10

    def test_model_whitelist_allowed(self) -> None:
        config = GovernorConfig(models=["gpt-4o", "qwen-plus"])
        assert config.is_model_allowed("gpt-4o") is True
        assert config.is_model_allowed("qwen-plus") is True

    def test_model_whitelist_denied(self) -> None:
        config = GovernorConfig(models=["gpt-4o"])
        assert config.is_model_allowed("claude-3-opus") is False

    def test_empty_whitelist_allows_all(self) -> None:
        config = GovernorConfig(models=[])
        assert config.is_model_allowed("any-model") is True


class TestGovernor:
    """Tests for Governor main class."""

    def test_init_with_kwargs(self) -> None:
        gov = Governor(budget_usd=5.0, models=["gpt-4o"], audit_enabled=False)
        assert gov.config.budget_usd == 5.0
        assert gov.config.models == ["gpt-4o"]

    def test_init_with_config(self, config: GovernorConfig) -> None:
        gov = Governor(config=config)
        assert gov.config is config

    @pytest.mark.asyncio
    async def test_execute_success(self, governor: Governor, mock_acompletion: AsyncMock) -> None:
        result = await governor.execute(
            messages=[{"role": "user", "content": "Hello"}],
            model="gpt-4o",
            name="test_call",
        )
        assert result == "Hello! How can I help you?"
        mock_acompletion.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_model_not_allowed(self, governor: Governor) -> None:
        with pytest.raises(ModelNotAllowedError) as exc_info:
            await governor.execute(
                messages=[{"role": "user", "content": "Hello"}],
                model="claude-3-opus",
            )
        assert "claude-3-opus" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_budget_exceeded(self, config: GovernorConfig) -> None:
        config.budget_usd = 0.000001  # Very tiny budget
        gov = Governor(config=config)

        # Simulate existing cost by recording usage
        gov.cost_tracker.record_usage("gpt-4", 1_000_000, 500_000)

        with pytest.raises(BudgetExceededError):
            await gov.execute(
                messages=[{"role": "user", "content": "Hello"}],
                model="gpt-4o",
            )

    @pytest.mark.asyncio
    async def test_execute_task_returns_full_result(
        self, governor: Governor, mock_acompletion: AsyncMock
    ) -> None:
        task = Task(
            name="test",
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
        )
        result = await governor.execute_task(task)
        assert isinstance(result, TaskResult)
        assert result.status == TaskStatus.COMPLETED
        assert result.content == "Hello! How can I help you?"
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 20

    @pytest.mark.asyncio
    async def test_execute_task_propagates_exceptions(self, config: GovernorConfig) -> None:
        """Test that execute_task propagates Dispatcher exceptions."""
        config.models = []  # allow all models
        gov = Governor(config=config)

        async def mock_fail(*args, **kwargs):
            raise ExecutionTimeoutError(task_id="test", timeout=1.0)

        with patch("litellm.acompletion", side_effect=mock_fail):
            with pytest.raises(ExecutionTimeoutError) as exc_info:
                await gov.execute(
                    messages=[{"role": "user", "content": "Hi"}],
                    model="gpt-4o",
                )
            assert exc_info.value.task_id == "test"

    def test_execute_sync_calls_execute(
        self, governor: Governor, mock_acompletion: AsyncMock
    ) -> None:
        """Test that execute_sync calls the async execute and returns content."""
        result = governor.execute_sync(
            messages=[{"role": "user", "content": "Hello"}],
            model="gpt-4o",
            name="sync_test",
        )
        assert result == "Hello! How can I help you?"
        mock_acompletion.assert_called_once()

    def test_get_stats(self, governor: Governor) -> None:
        stats = governor.get_stats()
        assert "total_cost_usd" in stats
        assert "config" in stats
        assert stats["config"]["budget_usd"] == 10.0

    def test_get_stats_with_tenants(self) -> None:
        """get_stats should include tenant summaries when multi-tenant is enabled."""
        config = GovernorConfig(
            budget_usd=10.0,
            models=["gpt-4o"],
            tenant_policies={
                "team-a": TenantPolicy(budget_usd=5.0, models=["gpt-4o"]),
            },
        )
        gov = Governor(config=config)
        stats = gov.get_stats()
        assert "tenants" in stats

    def test_reset(self, governor: Governor) -> None:
        governor.cost_tracker.record_usage("gpt-4o", 100, 50)
        governor.reset()
        assert governor.cost_tracker.total_cost == 0.0

    def test_get_tenant_stats_without_tenants(self, governor: Governor) -> None:
        """get_tenant_stats should return error when multi-tenant is not enabled."""
        result = governor.get_tenant_stats("team-a")
        assert "error" in result

    def test_get_billing_report_without_tenants(self, governor: Governor) -> None:
        """get_billing_report should return error when multi-tenant is not enabled."""
        result = governor.get_billing_report()
        assert "error" in result

    @pytest.mark.asyncio
    async def test_execute_error_result_raises(self, config: GovernorConfig) -> None:
        """execute should raise RuntimeError when dispatch returns error result."""
        config.models = []
        gov = Governor(config=config)

        error_result = TaskResult(
            task_id="t1",
            status=TaskStatus.FAILED,
            model="gpt-4o",
            error="something went wrong",
        )
        with (
            patch.object(gov.dispatcher, "dispatch", return_value=error_result),
            pytest.raises(RuntimeError, match="Execution failed"),
        ):
            await gov.execute(
                messages=[{"role": "user", "content": "hi"}],
                model="gpt-4o",
            )

    @pytest.mark.asyncio
    async def test_context_manager(self, config: GovernorConfig) -> None:
        """Governor should work as async context manager."""
        async with Governor(config=config) as gov:
            assert gov is not None

    @pytest.mark.asyncio
    async def test_run_graph_basic(self, governor: Governor, mock_acompletion: AsyncMock) -> None:
        """run_graph should execute a simple DAG."""
        graph = TaskGraph("test_graph")
        graph.add_node(
            "step1",
            task=Task(
                name="step1",
                model="gpt-4o",
                messages=[{"role": "user", "content": "hello"}],
            ),
        )
        result = await governor.run_graph(graph)
        assert result.status.value == "completed"
        assert "step1" in result.node_results

    def test_shutdown_without_observability(self, governor: Governor) -> None:
        """shutdown should be safe when tracing/metrics are not enabled."""
        governor.shutdown()  # Should not raise

    def test_execute_sync_in_async_context(
        self, governor: Governor, mock_acompletion: AsyncMock
    ) -> None:
        """execute_sync should work when called from within an async context."""
        import asyncio

        async def _inner() -> str:
            return governor.execute_sync(
                messages=[{"role": "user", "content": "Hello"}],
                model="gpt-4o",
            )

        result = asyncio.run(_inner())
        assert result == "Hello! How can I help you?"


class TestDispatcher:
    """Tests for Dispatcher."""

    @pytest.mark.asyncio
    async def test_fallback_on_failure(self, config: GovernorConfig) -> None:
        """Test that fallback model is used when primary fails."""
        config.max_retries = 0
        gov = Governor(config=config)

        call_count = 0

        async def mock_completion(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            model = kwargs.get("model", "")
            if model == "gpt-4o":
                raise RuntimeError("Primary model failed")
            # Fallback succeeds
            response = MagicMock()
            response.choices = [MagicMock()]
            response.choices[0].message.content = "Fallback response"
            response.usage = MagicMock()
            response.usage.prompt_tokens = 5
            response.usage.completion_tokens = 10
            return response

        with patch("litellm.acompletion", side_effect=mock_completion):
            result = await gov.execute(
                messages=[{"role": "user", "content": "test"}],
                model="gpt-4o",
            )
            assert result == "Fallback response"
            assert call_count == 2  # primary + fallback

    @pytest.mark.asyncio
    async def test_fallback_skipped_if_not_allowed(self, config: GovernorConfig) -> None:
        """Test that fallback is skipped if not in whitelist."""
        config.models = ["gpt-4o"]  # only gpt-4o allowed
        config.fallback_model = "claude-3-opus"  # not in whitelist
        config.max_retries = 0
        gov = Governor(config=config)

        call_count = 0

        async def mock_completion(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise RuntimeError("Always fails")

        with patch("litellm.acompletion", side_effect=mock_completion):
            with pytest.raises(RuntimeError, match="Always fails"):
                await gov.execute(
                    messages=[{"role": "user", "content": "test"}],
                    model="gpt-4o",
                )
            assert call_count == 1  # only primary, no fallback

    @pytest.mark.asyncio
    async def test_task_metadata_passed_to_result(
        self, config: GovernorConfig, mock_acompletion: AsyncMock
    ) -> None:
        """Test that Task metadata is passed through to TaskResult."""
        config.models = []
        gov = Governor(config=config)
        task = Task(
            name="meta_test",
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            metadata={"user_id": "123", "session": "abc"},
        )
        result = await gov.execute_task(task)
        assert result.metadata == {"user_id": "123", "session": "abc"}

    @pytest.mark.asyncio
    async def test_retry_with_delay(self, config: GovernorConfig) -> None:
        """Dispatcher should retry with delay before fallback."""
        config.max_retries = 1
        config.retry_delay = 0.01
        config.fallback_model = ""
        gov = Governor(config=config)

        call_count = 0

        async def mock_fail(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise RuntimeError("fail")

        with (
            patch("litellm.acompletion", side_effect=mock_fail),
            pytest.raises(RuntimeError, match="fail"),
        ):
            await gov.execute(
                messages=[{"role": "user", "content": "test"}],
                model="gpt-4o",
            )
        assert call_count == 2  # initial + 1 retry

    @pytest.mark.asyncio
    async def test_fallback_also_fails(self, config: GovernorConfig) -> None:
        """When both primary and fallback fail, should raise last error."""
        config.max_retries = 0
        gov = Governor(config=config)

        async def mock_fail(*args, **kwargs):
            raise RuntimeError("all models down")

        with (
            patch("litellm.acompletion", side_effect=mock_fail),
            pytest.raises(RuntimeError, match="all models down"),
        ):
            await gov.execute(
                messages=[{"role": "user", "content": "test"}],
                model="gpt-4o",
            )

    @pytest.mark.asyncio
    async def test_budget_check_with_metrics(self, config: GovernorConfig) -> None:
        """Budget exceeded should record metrics when metrics enabled."""
        config.budget_usd = 0.000001
        config.tracing_enabled = True
        gov = Governor(config=config)
        gov.cost_tracker.record_usage("gpt-4", 1_000_000, 500_000)

        with pytest.raises(BudgetExceededError):
            await gov.execute(
                messages=[{"role": "user", "content": "Hello"}],
                model="gpt-4o",
            )

    def test_get_stats_budget_remaining(self, config: GovernorConfig) -> None:
        """get_stats should compute budget_remaining correctly."""
        gov = Governor(config=config)
        gov.cost_tracker.record_usage("gpt-4o", 100, 50)
        stats = gov.dispatcher.get_stats()
        assert stats["budget_remaining"] < 10.0
        assert stats["total_calls"] == 1
