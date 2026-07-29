"""Cost report CLI - reads audit JSON logs and produces summaries.

Usage:
    agent-governor-report audit.log
    agent-governor-report audit.log --format json --group-by model
    agent-governor-report audit.log --limit 100
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import suppress
from pathlib import Path
from typing import Any


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="agent-governor-report",
        description="Generate cost reports from Agent Governor audit logs.",
    )
    parser.add_argument("input", type=str, help="Path to audit log file (JSON lines)")
    parser.add_argument(
        "--format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    parser.add_argument(
        "--group-by",
        choices=["model", "caller", "day", "tenant"],
        default="model",
        help="Group records by field (default: model)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max records to process (0 = unlimited)",
    )
    return parser.parse_args(argv)


def load_records(path: str, limit: int = 0) -> list[dict[str, Any]]:
    """Load audit log records from a JSON lines file.

    Args:
        path: Path to the audit log file.
        limit: Maximum number of records to load (0 = unlimited).

    Returns:
        List of parsed JSON records.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If a line is not valid JSON.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Audit log file not found: {path}")

    records: list[dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit > 0 and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {i + 1}: {e}") from e

    return records


def aggregate(records: list[dict[str, Any]], group_by: str) -> dict[str, Any]:
    """Aggregate records by a field.

    Args:
        records: List of audit records.
        group_by: Field to group by (model, caller, or day).

    Returns:
        Nested dict with aggregated stats.
    """
    groups: dict[str, dict[str, Any]] = {}

    for rec in records:
        key = str(rec.get(group_by, "unknown"))

        # Extract day from timestamp if grouping by day
        if group_by == "day":
            ts = rec.get("timestamp", "")
            if ts:
                with suppress(IndexError, TypeError):
                    key = ts[:10]  # YYYY-MM-DD

        if key not in groups:
            groups[key] = {
                "calls": 0,
                "total_cost_usd": 0.0,
                "total_tokens": 0,
                "total_latency_ms": 0.0,
            }

        g = groups[key]
        g["calls"] += 1
        g["total_cost_usd"] += float(rec.get("cost_usd", 0))
        g["total_tokens"] += int(rec.get("total_tokens", 0))
        g["total_latency_ms"] += float(rec.get("latency_ms", 0))

    # Compute averages
    result: dict[str, Any] = {"groups": {}, "totals": {}}
    total_calls = 0
    total_cost = 0.0
    total_tokens = 0

    for key, g in sorted(groups.items()):
        g["avg_cost_usd"] = round(g["total_cost_usd"] / g["calls"], 6) if g["calls"] else 0
        g["avg_latency_ms"] = round(g["total_latency_ms"] / g["calls"], 2) if g["calls"] else 0
        g["avg_tokens"] = round(g["total_tokens"] / g["calls"], 1) if g["calls"] else 0
        g["total_cost_usd"] = round(g["total_cost_usd"], 6)
        g["total_latency_ms"] = round(g["total_latency_ms"], 2)
        result["groups"][key] = g
        total_calls += g["calls"]
        total_cost += g["total_cost_usd"]
        total_tokens += g["total_tokens"]

    result["totals"] = {
        "total_calls": total_calls,
        "total_cost_usd": round(total_cost, 6),
        "total_tokens": total_tokens,
    }
    return result


def format_table(data: dict[str, Any]) -> str:
    """Format aggregated data as a table string."""
    lines: list[str] = []
    groups = data["groups"]
    totals = data["totals"]

    if not groups:
        return "No records found."

    # Header
    lines.append(
        f"{'Group':<20} {'Calls':>8} {'Cost (USD)':>14} {'Avg Cost':>12} {'Tokens':>10} {'Avg Latency':>14}"
    )
    lines.append("-" * 80)

    for key, g in groups.items():
        lines.append(
            f"{key:<20} {g['calls']:>8} {g['total_cost_usd']:>14.6f} "
            f"{g['avg_cost_usd']:>12.6f} {g['total_tokens']:>10} "
            f"{g['avg_latency_ms']:>14.2f}"
        )

    lines.append("-" * 80)
    lines.append(
        f"{'TOTAL':<20} {totals['total_calls']:>8} {totals['total_cost_usd']:>14.6f} "
        f"{'':>12} {totals['total_tokens']:>10} {'':>14}"
    )

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    """Main entry point for the cost report CLI."""
    args = parse_args(argv)

    try:
        records = load_records(args.input, args.limit)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if not records:
        print("No records found in audit log.")
        return

    # Map CLI name to record field name
    group_field = "tenant_id" if args.group_by == "tenant" else args.group_by
    data = aggregate(records, group_field)

    if args.format == "json":
        print(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        print(format_table(data))


if __name__ == "__main__":
    main()
