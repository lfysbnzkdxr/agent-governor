"""Built-in policy rule factory functions.

Convenience constructors for common policy patterns.
"""

from __future__ import annotations

from agent_governor.policy.engine import PolicyRule


def budget_cap(
    max_usd: float, *, tenant_id: str = "", action: str = "deny", priority: int = 0
) -> PolicyRule:
    """Create a budget cap rule.

    Args:
        max_usd: Maximum accumulated cost allowed.
        tenant_id: If set, only applies to this tenant.
        action: "deny" to block, "warn" to log only.
        priority: Higher values evaluated first.
    """
    params: dict = {"max_usd": max_usd}
    if tenant_id:
        params["tenant_id"] = tenant_id
    return PolicyRule(
        name=f"budget_cap_{tenant_id or 'global'}",
        type="budget",
        params=params,
        action=action,  # type: ignore[arg-type]
        priority=priority,
    )


def model_whitelist(
    models: list[str], *, tenant_id: str = "", action: str = "deny", priority: int = 0
) -> PolicyRule:
    """Create a model whitelist rule.

    Args:
        models: List of allowed model names.
        tenant_id: If set, only applies to this tenant.
        action: "deny" to block, "warn" to log only.
        priority: Higher values evaluated first.
    """
    params: dict = {"models": models}
    if tenant_id:
        params["tenant_id"] = tenant_id
    return PolicyRule(
        name=f"model_whitelist_{tenant_id or 'global'}",
        type="model_whitelist",
        params=params,
        action=action,  # type: ignore[arg-type]
        priority=priority,
    )


def rate_limit(
    rps: float, *, tenant_id: str = "", action: str = "deny", priority: int = 0
) -> PolicyRule:
    """Create a rate limit rule.

    Args:
        rps: Maximum requests per second.
        tenant_id: If set, only applies to this tenant.
        action: "deny" to block, "warn" to log only.
        priority: Higher values evaluated first.
    """
    params: dict = {"rps": rps}
    if tenant_id:
        params["tenant_id"] = tenant_id
    return PolicyRule(
        name=f"rate_limit_{tenant_id or 'global'}",
        type="rate_limit",
        params=params,
        action=action,  # type: ignore[arg-type]
        priority=priority,
    )


def max_cost_per_call(max_usd: float, *, action: str = "deny", priority: int = 0) -> PolicyRule:
    """Create a per-call cost limit rule.

    Args:
        max_usd: Maximum cost allowed for a single call.
        action: "deny" to block, "warn" to log only.
        priority: Higher values evaluated first.
    """
    return PolicyRule(
        name="max_cost_per_call",
        type="max_cost_per_call",
        params={"max_usd": max_usd},
        action=action,  # type: ignore[arg-type]
        priority=priority,
    )
