"""PolicyEngine - declarative policy rules compiled to runtime predicates.

Supports loading rules from Python dicts or YAML files, evaluating them
against call contexts to produce allow/deny decisions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class PolicyRule(BaseModel):
    """A single declarative policy rule.

    Example:
        rule = PolicyRule(
            name="team-budget-cap",
            type="budget",
            params={"max_usd": 100.0, "tenant_id": "team-a"},
            action="deny",
        )
    """

    name: str = Field(description="Human-readable rule identifier")
    type: Literal["budget", "rate_limit", "model_whitelist", "max_cost_per_call"] = Field(
        description="Rule type determining evaluation logic"
    )
    params: dict[str, Any] = Field(default_factory=dict, description="Rule parameters")
    action: Literal["deny", "warn"] = Field(default="deny", description="Action on violation")
    priority: int = Field(default=0, description="Higher priority rules evaluated first")


class PolicyDecision(BaseModel):
    """Result of policy evaluation."""

    allowed: bool = True
    violated_rules: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class PolicyEngine:
    """Evaluates declarative policy rules against call contexts.

    Rules are compiled once at load time into efficient predicates.
    Evaluation is O(n) over rules with short-circuit on first deny.

    Example:
        engine = PolicyEngine()
        engine.load_rules([
            PolicyRule(name="budget", type="budget", params={"max_usd": 50}),
            PolicyRule(name="whitelist", type="model_whitelist", params={"models": ["gpt-4o"]}),
        ])

        decision = engine.evaluate({"model": "gpt-4o", "current_cost": 10.0})
        assert decision.allowed
    """

    def __init__(self) -> None:
        self._rules: list[PolicyRule] = []

    @property
    def rules(self) -> list[PolicyRule]:
        """Get loaded rules sorted by priority."""
        return sorted(self._rules, key=lambda r: r.priority, reverse=True)

    def load_rules(self, rules: list[PolicyRule]) -> None:
        """Load rules from a list of PolicyRule objects."""
        self._rules = list(rules)
        logger.info("policy_rules_loaded", count=len(self._rules))

    def load_dict(self, data: dict[str, Any]) -> None:
        """Load rules from a dictionary (e.g., parsed YAML/JSON).

        Expected format:
            {"version": 1, "rules": [{"name": ..., "type": ..., "params": ...}]}
        """
        raw_rules = data.get("rules", [])
        self._rules = [PolicyRule(**r) for r in raw_rules]
        logger.info("policy_rules_loaded_from_dict", count=len(self._rules))

    def load_yaml(self, path: str | Path) -> None:
        """Load rules from a YAML file.

        Requires PyYAML (install via: pip install agent-governor[policy]).
        """
        try:
            import yaml
        except ImportError as e:
            raise ImportError(
                "PyYAML is required for YAML policy loading. "
                "Install with: pip install agent-governor[policy]"
            ) from e

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Policy file not found: {path}")

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(
                f"Invalid policy file format: expected dict, got {type(data).__name__}"
            )

        self.load_dict(data)

    def evaluate(self, context: dict[str, Any]) -> PolicyDecision:
        """Evaluate all rules against a call context.

        Context keys (all optional):
            - model: str — model being called
            - tenant_id: str — tenant identifier
            - current_cost: float — accumulated cost so far
            - call_cost: float — estimated cost of this call
            - tokens: int — total tokens in this call

        Returns:
            PolicyDecision with allowed flag and violation details.
        """
        violated: list[str] = []
        warnings: list[str] = []

        for rule in self.rules:
            result = self._evaluate_rule(rule, context)
            if not result:
                if rule.action == "deny":
                    violated.append(rule.name)
                    # Short-circuit on first deny
                    return PolicyDecision(allowed=False, violated_rules=violated, warnings=warnings)
                else:
                    warnings.append(rule.name)

        return PolicyDecision(allowed=True, violated_rules=violated, warnings=warnings)

    def _evaluate_rule(self, rule: PolicyRule, context: dict[str, Any]) -> bool:
        """Evaluate a single rule. Returns True if the rule passes (no violation)."""
        if rule.type == "budget":
            return self._eval_budget(rule, context)
        elif rule.type == "rate_limit":
            return self._eval_rate_limit(rule, context)
        elif rule.type == "model_whitelist":
            return self._eval_model_whitelist(rule, context)
        elif rule.type == "max_cost_per_call":
            return self._eval_max_cost_per_call(rule, context)
        else:
            logger.warning("unknown_policy_rule_type", type=rule.type, name=rule.name)
            return True  # Unknown rules pass by default

    def _eval_budget(self, rule: PolicyRule, context: dict[str, Any]) -> bool:
        """Check if current cost is within budget."""
        max_usd = rule.params.get("max_usd", 0.0)
        if max_usd <= 0:
            return True
        # If tenant_id specified in params, only apply to that tenant
        rule_tenant = rule.params.get("tenant_id", "")
        ctx_tenant = context.get("tenant_id", "")
        if rule_tenant and rule_tenant != ctx_tenant:
            return True  # Rule doesn't apply to this tenant
        current_cost = context.get("current_cost", 0.0)
        return bool(current_cost < max_usd)

    def _eval_rate_limit(self, rule: PolicyRule, context: dict[str, Any]) -> bool:
        """Check rate limit (context should provide current_rps)."""
        max_rps = rule.params.get("rps", 0.0)
        if max_rps <= 0:
            return True
        rule_tenant = rule.params.get("tenant_id", "")
        ctx_tenant = context.get("tenant_id", "")
        if rule_tenant and rule_tenant != ctx_tenant:
            return True
        current_rps = context.get("current_rps", 0.0)
        return bool(current_rps <= max_rps)

    def _eval_model_whitelist(self, rule: PolicyRule, context: dict[str, Any]) -> bool:
        """Check if model is in the whitelist."""
        allowed_models = rule.params.get("models", [])
        if not allowed_models:
            return True
        rule_tenant = rule.params.get("tenant_id", "")
        ctx_tenant = context.get("tenant_id", "")
        if rule_tenant and rule_tenant != ctx_tenant:
            return True
        model = context.get("model", "")
        return model in allowed_models

    def _eval_max_cost_per_call(self, rule: PolicyRule, context: dict[str, Any]) -> bool:
        """Check if a single call's cost is within limit."""
        max_cost = rule.params.get("max_usd", 0.0)
        if max_cost <= 0:
            return True
        call_cost = context.get("call_cost", 0.0)
        return bool(call_cost <= max_cost)
