"""Billing report generation for multi-tenant cost attribution."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from agent_governor.tenant.manager import TenantManager

logger = structlog.get_logger(__name__)


class BillingReport:
    """Generates per-tenant billing reports from TenantManager data.

    Example:
        report = BillingReport(manager)
        data = report.generate(period="2026-07")
        print(report.to_table())
    """

    def __init__(self, manager: TenantManager) -> None:
        self._manager = manager

    def generate(self, *, period: str = "") -> dict[str, Any]:
        """Generate a billing report for all tenants.

        Args:
            period: Optional period label (e.g., "2026-07") for report metadata.

        Returns:
            Dict with per-tenant cost breakdown and totals.
        """
        summaries = self._manager.get_all_summaries()

        tenants_report: dict[str, Any] = {}
        total_cost = 0.0
        total_calls = 0

        for tenant_id, summary in summaries.items():
            cost = summary.get("total_cost_usd", 0.0)
            calls = summary.get("total_calls", 0)
            total_cost += cost
            total_calls += calls

            tenants_report[tenant_id] = {
                "cost_usd": cost,
                "calls": calls,
                "budget_usd": summary.get("budget_usd", 0.0),
                "budget_remaining": summary.get("budget_remaining", 0.0),
                "models": summary.get("models", {}),
            }

        return {
            "period": period,
            "total_cost_usd": total_cost,
            "total_calls": total_calls,
            "tenant_count": len(tenants_report),
            "tenants": tenants_report,
        }

    def to_json(self, *, period: str = "", indent: int = 2) -> str:
        """Generate report as formatted JSON string."""
        data = self.generate(period=period)
        return json.dumps(data, indent=indent, ensure_ascii=False, default=str)

    def to_table(self, *, period: str = "") -> str:
        """Generate report as a human-readable table."""
        data = self.generate(period=period)
        lines: list[str] = []

        header = "Billing Report"
        if period:
            header += f" ({period})"
        lines.append(header)
        lines.append("=" * 60)
        lines.append(f"{'Tenant':<20} {'Cost':>10} {'Calls':>8} {'Budget':>10} {'Remaining':>10}")
        lines.append("-" * 60)

        for tenant_id, info in data["tenants"].items():
            budget_str = f"${info['budget_usd']:.2f}" if info["budget_usd"] > 0 else "unlimited"
            remain_str = (
                f"${info['budget_remaining']:.2f}" if info["budget_usd"] > 0 else "∞"
            )
            lines.append(
                f"{tenant_id:<20} ${info['cost_usd']:>8.4f} {info['calls']:>8} "
                f"{budget_str:>10} {remain_str:>10}"
            )

        lines.append("-" * 60)
        lines.append(
            f"{'TOTAL':<20} ${data['total_cost_usd']:>8.4f} {data['total_calls']:>8}"
        )
        return "\n".join(lines)
