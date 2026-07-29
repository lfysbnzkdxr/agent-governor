"""Unit tests for CostTracker."""

from __future__ import annotations

from agent_governor.observability.cost import CostTracker


class TestCostTracker:
    """Tests for CostTracker."""

    def test_record_usage_basic(self, cost_tracker: CostTracker) -> None:
        """Test basic usage recording."""
        cost = cost_tracker.record_usage("gpt-4o", prompt_tokens=1000, completion_tokens=500)

        # gpt-4o: input $2.50/1M, output $10.00/1M
        expected = (1000 * 2.50 + 500 * 10.00) / 1_000_000
        assert abs(cost - expected) < 1e-10
        assert cost_tracker.total_cost == cost
        assert cost_tracker.total_calls == 1

    def test_record_usage_accumulates(self, cost_tracker: CostTracker) -> None:
        """Test that multiple calls accumulate."""
        cost1 = cost_tracker.record_usage("gpt-4o", 100, 50)
        cost2 = cost_tracker.record_usage("gpt-4o", 200, 100)

        assert cost_tracker.total_calls == 2
        assert abs(cost_tracker.total_cost - (cost1 + cost2)) < 1e-10

    def test_record_usage_multiple_models(self, cost_tracker: CostTracker) -> None:
        """Test tracking across multiple models."""
        cost_tracker.record_usage("gpt-4o", 100, 50)
        cost_tracker.record_usage("qwen-plus", 100, 50)

        assert cost_tracker.total_calls == 2
        assert cost_tracker.get_model_usage("gpt-4o") is not None
        assert cost_tracker.get_model_usage("qwen-plus") is not None
        assert cost_tracker.get_model_usage("gpt-4o").call_count == 1

    def test_check_budget_within(self, cost_tracker: CostTracker) -> None:
        """Test budget check when within budget."""
        cost_tracker.record_usage("gpt-4o", 100, 50)
        assert cost_tracker.check_budget(10.0) is True

    def test_check_budget_exceeded(self, cost_tracker: CostTracker) -> None:
        """Test budget check when exceeded."""
        # Record a large usage to exceed budget
        cost_tracker.record_usage("gpt-4", 1_000_000, 500_000)
        assert cost_tracker.check_budget(1.0) is False

    def test_check_budget_warning(self, cost_tracker: CostTracker) -> None:
        """Test budget warning threshold."""
        # Record usage that crosses 80% of $1 budget
        cost_tracker.record_usage("gpt-4o", 300_000, 10_000)
        # Cost ~ $0.85, which is > 80% of $1
        result = cost_tracker.check_budget_warning(1.0, warning_ratio=0.8)
        assert result is True

        # Second call should not re-trigger
        result2 = cost_tracker.check_budget_warning(1.0, warning_ratio=0.8)
        assert result2 is False

    def test_prefix_matching(self, cost_tracker: CostTracker) -> None:
        """Test model prefix matching for pricing."""
        # "gpt-4o-2024-08-06" should match "gpt-4o" pricing
        cost = cost_tracker.record_usage("gpt-4o-2024-08-06", 1000, 0)
        expected = (1000 * 2.50) / 1_000_000
        assert abs(cost - expected) < 1e-10

    def test_unknown_model_default_pricing(self, cost_tracker: CostTracker) -> None:
        """Test unknown model uses default pricing."""
        cost = cost_tracker.record_usage("unknown-model", 1_000_000, 0)
        # Default: $1.0/1M input
        assert abs(cost - 1.0) < 1e-10

    def test_get_summary(self, cost_tracker: CostTracker) -> None:
        """Test summary generation."""
        cost_tracker.record_usage("gpt-4o", 100, 50)
        cost_tracker.record_usage("qwen-plus", 200, 100)

        summary = cost_tracker.get_summary()
        assert summary["total_calls"] == 2
        assert summary["total_cost_usd"] > 0
        assert "gpt-4o" in summary["models"]
        assert "qwen-plus" in summary["models"]

    def test_reset(self, cost_tracker: CostTracker) -> None:
        """Test reset clears all data."""
        cost_tracker.record_usage("gpt-4o", 100, 50)
        cost_tracker.reset()

        assert cost_tracker.total_cost == 0.0
        assert cost_tracker.total_calls == 0
        assert cost_tracker.get_model_usage("gpt-4o") is None

    def test_custom_pricing(self) -> None:
        """Test custom pricing table."""
        tracker = CostTracker(pricing={"my-model": (5.0, 15.0)})
        cost = tracker.record_usage("my-model", 1_000_000, 1_000_000)
        assert abs(cost - 20.0) < 1e-10
