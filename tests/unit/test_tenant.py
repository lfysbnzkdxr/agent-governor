"""Tests for multi-tenant governance: TenantManager, TenantPolicy, BillingReport."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from agent_governor.exceptions import RateLimitExceededError, TenantBudgetExceededError
from agent_governor.governor import Governor
from agent_governor.tenant.billing import BillingReport
from agent_governor.tenant.config import MultiTenantConfig, TenantPolicy
from agent_governor.tenant.manager import TenantManager

# ============================================================
# TenantPolicy / MultiTenantConfig
# ============================================================


class TestTenantConfig:
    def test_tenant_policy_defaults(self):
        policy = TenantPolicy()
        assert policy.budget_usd == 0.0
        assert policy.rate_limit_rps == 0.0
        assert policy.max_concurrency == 5
        assert policy.allowed_models == []

    def test_multi_tenant_config_get_policy(self):
        config = MultiTenantConfig(
            tenants={"team-a": TenantPolicy(budget_usd=100)},
            default_policy=TenantPolicy(budget_usd=10),
        )
        assert config.get_policy("team-a") is not None
        assert config.get_policy("team-a").budget_usd == 100  # type: ignore[union-attr]
        assert config.get_policy("unknown") is not None
        assert config.get_policy("unknown").budget_usd == 10  # type: ignore[union-attr]

    def test_get_policy_no_default_returns_none(self):
        config = MultiTenantConfig(tenants={"team-a": TenantPolicy(budget_usd=50)})
        assert config.get_policy("nonexistent") is None


# ============================================================
# TenantManager
# ============================================================


class TestTenantManager:
    def _make_manager(self) -> TenantManager:
        config = MultiTenantConfig(
            tenants={
                "team-a": TenantPolicy(budget_usd=1.0, rate_limit_rps=0, max_concurrency=2),
                "team-b": TenantPolicy(budget_usd=0.5, rate_limit_rps=100, max_concurrency=5),
            },
            default_policy=TenantPolicy(budget_usd=0.1),
        )
        return TenantManager(config)

    def test_record_usage_isolates_tenants(self):
        mgr = self._make_manager()
        mgr.record_usage("team-a", "gpt-4o", 1000, 500)
        mgr.record_usage("team-b", "gpt-4o", 200, 100)

        summary_a = mgr.get_tenant_summary("team-a")
        summary_b = mgr.get_tenant_summary("team-b")

        assert summary_a["total_calls"] == 1
        assert summary_b["total_calls"] == 1
        assert summary_a["total_cost_usd"] > summary_b["total_cost_usd"]

    def test_check_budget_passes_within_limit(self):
        mgr = self._make_manager()
        mgr.record_usage("team-a", "gpt-4o", 100, 50)  # Small cost
        mgr.check_budget("team-a")  # Should not raise

    def test_check_budget_raises_when_exceeded(self):
        mgr = self._make_manager()
        # Record massive usage to exceed $1.0 budget
        mgr.record_usage("team-a", "gpt-4o", 500_000, 200_000)
        with pytest.raises(TenantBudgetExceededError) as exc_info:
            mgr.check_budget("team-a")
        assert exc_info.value.tenant_id == "team-a"

    def test_check_budget_unlimited_when_zero(self):
        config = MultiTenantConfig(tenants={"free": TenantPolicy(budget_usd=0)})
        mgr = TenantManager(config)
        mgr.record_usage("free", "gpt-4o", 999_999, 999_999)
        mgr.check_budget("free")  # Should not raise (0 = unlimited)

    def test_check_model_allowed(self):
        config = MultiTenantConfig(tenants={"restricted": TenantPolicy(allowed_models=["gpt-4o"])})
        mgr = TenantManager(config)
        assert mgr.check_model_allowed("restricted", "gpt-4o") is True
        assert mgr.check_model_allowed("restricted", "claude-3-opus") is False

    def test_check_model_allowed_empty_means_all(self):
        mgr = self._make_manager()
        assert mgr.check_model_allowed("team-a", "any-model") is True

    async def test_acquire_concurrency_control(self):
        mgr = self._make_manager()
        # team-a has max_concurrency=2
        async with mgr.acquire("team-a"), mgr.acquire("team-a"):
            pass  # 2 concurrent OK

    async def test_acquire_rate_limit_exceeded(self):
        config = MultiTenantConfig(
            tenants={"limited": TenantPolicy(rate_limit_rps=1.0, max_concurrency=10)}
        )
        mgr = TenantManager(config)
        # First call should succeed
        async with mgr.acquire("limited"):
            pass
        # Rapid second call should fail (bucket depleted)
        with pytest.raises(RateLimitExceededError):
            async with mgr.acquire("limited"):
                pass

    def test_get_all_summaries(self):
        mgr = self._make_manager()
        mgr.record_usage("team-a", "gpt-4o", 100, 50)
        mgr.record_usage("team-b", "qwen-plus", 200, 100)
        summaries = mgr.get_all_summaries()
        assert "team-a" in summaries
        assert "team-b" in summaries

    def test_reset_single_tenant(self):
        mgr = self._make_manager()
        mgr.record_usage("team-a", "gpt-4o", 100, 50)
        mgr.record_usage("team-b", "gpt-4o", 100, 50)
        mgr.reset("team-a")
        assert mgr.get_tenant_summary("team-a")["total_calls"] == 0
        assert mgr.get_tenant_summary("team-b")["total_calls"] == 1

    def test_reset_all(self):
        mgr = self._make_manager()
        mgr.record_usage("team-a", "gpt-4o", 100, 50)
        mgr.record_usage("team-b", "gpt-4o", 100, 50)
        mgr.reset()
        assert mgr.get_tenant_summary("team-a")["total_calls"] == 0
        assert mgr.get_tenant_summary("team-b")["total_calls"] == 0


# ============================================================
# BillingReport
# ============================================================


class TestBillingReport:
    def test_generate_report(self):
        config = MultiTenantConfig(tenants={"team-a": TenantPolicy(budget_usd=10.0)})
        mgr = TenantManager(config)
        mgr.record_usage("team-a", "gpt-4o", 1000, 500)

        report = BillingReport(mgr)
        data = report.generate(period="2026-07")

        assert data["period"] == "2026-07"
        assert data["tenant_count"] == 1
        assert "team-a" in data["tenants"]
        assert data["tenants"]["team-a"]["calls"] == 1
        assert data["total_cost_usd"] > 0

    def test_to_json(self):
        config = MultiTenantConfig(tenants={"t": TenantPolicy(budget_usd=5.0)})
        mgr = TenantManager(config)
        mgr.record_usage("t", "gpt-4o", 100, 50)

        report = BillingReport(mgr)
        json_str = report.to_json(period="2026-07")
        assert '"period": "2026-07"' in json_str

    def test_to_table(self):
        config = MultiTenantConfig(tenants={"t": TenantPolicy(budget_usd=5.0)})
        mgr = TenantManager(config)
        mgr.record_usage("t", "gpt-4o", 100, 50)

        report = BillingReport(mgr)
        table = report.to_table()
        assert "Billing Report" in table
        assert "TOTAL" in table


# ============================================================
# Governor multi-tenant integration
# ============================================================


class TestGovernorMultiTenant:
    async def test_execute_with_tenant_id(self):
        governor = Governor(
            budget_usd=100.0,
            models=["gpt-4o"],
            config=None,
        )
        # Manually set up tenant manager
        from agent_governor.tenant.config import MultiTenantConfig, TenantPolicy
        from agent_governor.tenant.manager import TenantManager

        governor._tenant_manager = TenantManager(
            MultiTenantConfig(tenants={"team-x": TenantPolicy(budget_usd=50.0)})
        )

        mock_response = AsyncMock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_response.choices = [AsyncMock()]
        mock_response.choices[0].message.content = "Hello from tenant"

        with patch("litellm.acompletion", return_value=mock_response):
            result = await governor.execute(
                messages=[{"role": "user", "content": "Hi"}],
                model="gpt-4o",
                tenant_id="team-x",
            )

        assert result == "Hello from tenant"
        stats = governor.get_tenant_stats("team-x")
        assert stats["total_calls"] == 1

    async def test_tenant_budget_exceeded_blocks_call(self):
        governor = Governor(budget_usd=100.0, models=["gpt-4o"])
        from agent_governor.tenant.config import MultiTenantConfig, TenantPolicy
        from agent_governor.tenant.manager import TenantManager

        governor._tenant_manager = TenantManager(
            MultiTenantConfig(tenants={"poor": TenantPolicy(budget_usd=0.001)})
        )
        # Exhaust budget
        governor._tenant_manager.record_usage("poor", "gpt-4o", 5000, 2000)

        with pytest.raises(TenantBudgetExceededError):
            await governor.execute(
                messages=[{"role": "user", "content": "Hi"}],
                model="gpt-4o",
                tenant_id="poor",
            )

    def test_get_billing_report(self):
        governor = Governor(budget_usd=100.0)
        from agent_governor.tenant.config import MultiTenantConfig, TenantPolicy
        from agent_governor.tenant.manager import TenantManager

        governor._tenant_manager = TenantManager(
            MultiTenantConfig(tenants={"a": TenantPolicy(budget_usd=10.0)})
        )
        governor._tenant_manager.record_usage("a", "gpt-4o", 100, 50)

        report = governor.get_billing_report(period="2026-07")
        assert report["tenant_count"] == 1
        assert "a" in report["tenants"]

    def test_get_billing_report_not_enabled(self):
        governor = Governor(budget_usd=100.0)
        report = governor.get_billing_report()
        assert "error" in report
