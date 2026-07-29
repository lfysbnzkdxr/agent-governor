"""Tests for the declarative policy engine."""

from __future__ import annotations

import tempfile

import pytest
from pydantic import ValidationError

from agent_governor.policy.engine import PolicyEngine, PolicyRule
from agent_governor.policy.rules import budget_cap, max_cost_per_call, model_whitelist, rate_limit

# ============================================================
# PolicyRule model
# ============================================================


class TestPolicyRule:
    def test_basic_creation(self):
        rule = PolicyRule(name="test", type="budget", params={"max_usd": 10.0})
        assert rule.name == "test"
        assert rule.type == "budget"
        assert rule.action == "deny"
        assert rule.priority == 0

    def test_invalid_type_rejected(self):
        with pytest.raises(ValidationError):
            PolicyRule(name="bad", type="invalid_type", params={})  # type: ignore[arg-type]


# ============================================================
# PolicyEngine evaluation
# ============================================================


class TestPolicyEngine:
    def _engine_with_rules(self, rules: list[PolicyRule]) -> PolicyEngine:
        engine = PolicyEngine()
        engine.load_rules(rules)
        return engine

    def test_empty_engine_allows_all(self):
        engine = PolicyEngine()
        decision = engine.evaluate({"model": "anything", "current_cost": 999})
        assert decision.allowed is True

    def test_budget_rule_passes(self):
        engine = self._engine_with_rules(
            [
                PolicyRule(name="cap", type="budget", params={"max_usd": 50.0}),
            ]
        )
        decision = engine.evaluate({"current_cost": 10.0})
        assert decision.allowed is True

    def test_budget_rule_denies(self):
        engine = self._engine_with_rules(
            [
                PolicyRule(name="cap", type="budget", params={"max_usd": 50.0}),
            ]
        )
        decision = engine.evaluate({"current_cost": 60.0})
        assert decision.allowed is False
        assert "cap" in decision.violated_rules

    def test_budget_rule_tenant_scoped(self):
        engine = self._engine_with_rules(
            [
                PolicyRule(
                    name="team-cap", type="budget", params={"max_usd": 10.0, "tenant_id": "team-a"}
                ),
            ]
        )
        # Different tenant — rule doesn't apply
        decision = engine.evaluate({"current_cost": 100.0, "tenant_id": "team-b"})
        assert decision.allowed is True
        # Same tenant — rule applies
        decision = engine.evaluate({"current_cost": 100.0, "tenant_id": "team-a"})
        assert decision.allowed is False

    def test_model_whitelist_passes(self):
        engine = self._engine_with_rules(
            [
                PolicyRule(
                    name="wl", type="model_whitelist", params={"models": ["gpt-4o", "qwen-plus"]}
                ),
            ]
        )
        decision = engine.evaluate({"model": "gpt-4o"})
        assert decision.allowed is True

    def test_model_whitelist_denies(self):
        engine = self._engine_with_rules(
            [
                PolicyRule(name="wl", type="model_whitelist", params={"models": ["gpt-4o"]}),
            ]
        )
        decision = engine.evaluate({"model": "claude-3-opus"})
        assert decision.allowed is False
        assert "wl" in decision.violated_rules

    def test_rate_limit_passes(self):
        engine = self._engine_with_rules(
            [
                PolicyRule(name="rl", type="rate_limit", params={"rps": 10.0}),
            ]
        )
        decision = engine.evaluate({"current_rps": 5.0})
        assert decision.allowed is True

    def test_rate_limit_denies(self):
        engine = self._engine_with_rules(
            [
                PolicyRule(name="rl", type="rate_limit", params={"rps": 10.0}),
            ]
        )
        decision = engine.evaluate({"current_rps": 15.0})
        assert decision.allowed is False

    def test_max_cost_per_call(self):
        engine = self._engine_with_rules(
            [
                PolicyRule(name="per-call", type="max_cost_per_call", params={"max_usd": 0.5}),
            ]
        )
        assert engine.evaluate({"call_cost": 0.3}).allowed is True
        assert engine.evaluate({"call_cost": 0.8}).allowed is False

    def test_warn_action_does_not_block(self):
        engine = self._engine_with_rules(
            [
                PolicyRule(name="soft-cap", type="budget", params={"max_usd": 5.0}, action="warn"),
            ]
        )
        decision = engine.evaluate({"current_cost": 10.0})
        assert decision.allowed is True
        assert "soft-cap" in decision.warnings

    def test_priority_ordering(self):
        engine = self._engine_with_rules(
            [
                PolicyRule(name="low", type="budget", params={"max_usd": 100.0}, priority=1),
                PolicyRule(
                    name="high", type="model_whitelist", params={"models": ["gpt-4o"]}, priority=10
                ),
            ]
        )
        # High priority rule evaluated first
        rules = engine.rules
        assert rules[0].name == "high"

    def test_short_circuit_on_deny(self):
        engine = self._engine_with_rules(
            [
                PolicyRule(name="first", type="budget", params={"max_usd": 1.0}, priority=10),
                PolicyRule(
                    name="second", type="model_whitelist", params={"models": ["x"]}, priority=1
                ),
            ]
        )
        decision = engine.evaluate({"current_cost": 5.0, "model": "y"})
        # Only first rule violated (short-circuit)
        assert decision.violated_rules == ["first"]


# ============================================================
# Loading from dict / YAML
# ============================================================


class TestPolicyLoading:
    def test_load_dict(self):
        engine = PolicyEngine()
        engine.load_dict(
            {
                "version": 1,
                "rules": [
                    {"name": "r1", "type": "budget", "params": {"max_usd": 10}},
                    {"name": "r2", "type": "model_whitelist", "params": {"models": ["gpt-4o"]}},
                ],
            }
        )
        assert len(engine.rules) == 2

    def test_load_yaml(self):
        yaml_content = """\
version: 1
rules:
  - name: budget-rule
    type: budget
    params:
      max_usd: 25.0
  - name: model-rule
    type: model_whitelist
    params:
      models:
        - gpt-4o
        - qwen-plus
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            engine = PolicyEngine()
            engine.load_yaml(f.name)

        assert len(engine.rules) == 2
        decision = engine.evaluate({"model": "gpt-4o", "current_cost": 5.0})
        assert decision.allowed is True

    def test_load_yaml_file_not_found(self):
        engine = PolicyEngine()
        with pytest.raises(FileNotFoundError):
            engine.load_yaml("/nonexistent/path.yaml")

    def test_load_dict_empty_rules(self):
        engine = PolicyEngine()
        engine.load_dict({"version": 1, "rules": []})
        assert len(engine.rules) == 0
        assert engine.evaluate({}).allowed is True


# ============================================================
# Rule factory functions
# ============================================================


class TestRuleFactories:
    def test_budget_cap_factory(self):
        rule = budget_cap(100.0, tenant_id="team-a")
        assert rule.type == "budget"
        assert rule.params["max_usd"] == 100.0
        assert rule.params["tenant_id"] == "team-a"

    def test_model_whitelist_factory(self):
        rule = model_whitelist(["gpt-4o", "qwen-plus"])
        assert rule.type == "model_whitelist"
        assert rule.params["models"] == ["gpt-4o", "qwen-plus"]

    def test_rate_limit_factory(self):
        rule = rate_limit(5.0, tenant_id="trial")
        assert rule.type == "rate_limit"
        assert rule.params["rps"] == 5.0
        assert rule.params["tenant_id"] == "trial"

    def test_max_cost_per_call_factory(self):
        rule = max_cost_per_call(0.5)
        assert rule.type == "max_cost_per_call"
        assert rule.params["max_usd"] == 0.5

    def test_factories_integrate_with_engine(self):
        engine = PolicyEngine()
        engine.load_rules(
            [
                budget_cap(50.0),
                model_whitelist(["gpt-4o"]),
                rate_limit(10.0),
            ]
        )
        decision = engine.evaluate({"model": "gpt-4o", "current_cost": 5.0, "current_rps": 3.0})
        assert decision.allowed is True

        decision = engine.evaluate({"model": "bad-model", "current_cost": 5.0, "current_rps": 3.0})
        assert decision.allowed is False
