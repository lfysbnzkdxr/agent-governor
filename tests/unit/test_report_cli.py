"""Tests for Cost Report CLI."""

from __future__ import annotations

import json

import pytest

from agent_governor.cli.report import aggregate, format_table, load_records, parse_args


class TestParseArgs:
    """Tests for CLI argument parsing."""

    def test_parse_defaults(self):
        """Should parse input path and set defaults."""
        args = parse_args(["audit.log"])
        assert args.input == "audit.log"
        assert args.format == "table"
        assert args.group_by == "model"
        assert args.limit == 0

    def test_parse_with_options(self):
        """Should parse all options."""
        args = parse_args(["--format", "json", "--group-by", "caller", "--limit", "50", "audit.log"])
        assert args.input == "audit.log"
        assert args.format == "json"
        assert args.group_by == "caller"
        assert args.limit == 50


class TestLoadRecords:
    """Tests for loading audit log records."""

    def test_load_invalid_path(self):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError, match="not found"):
            load_records("/nonexistent/file.log")

    def test_load_invalid_json(self, tmp_path):
        """Should raise ValueError for malformed JSON."""
        f = tmp_path / "bad.log"
        f.write_text("not json\n")
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_records(str(f))

    def test_load_valid_records(self, tmp_path):
        """Should load valid JSON lines."""
        f = tmp_path / "audit.log"
        f.write_text(
            '{"model": "gpt-4", "cost_usd": 0.01, "total_tokens": 100, "latency_ms": 500}\n'
            '{"model": "gpt-3.5", "cost_usd": 0.002, "total_tokens": 50, "latency_ms": 300}\n'
        )
        records = load_records(str(f))
        assert len(records) == 2
        assert records[0]["model"] == "gpt-4"

    def test_load_with_limit(self, tmp_path):
        """Should respect the limit parameter."""
        f = tmp_path / "audit.log"
        lines = '\n'.join([f'{{"model": "m{i}", "cost_usd": 0.01}}' for i in range(10)])
        f.write_text(lines)
        records = load_records(str(f), limit=3)
        assert len(records) == 3


class TestAggregate:
    """Tests for aggregation logic."""

    def test_aggregate_by_model(self):
        """Should group records by model."""
        records = [
            {"model": "gpt-4", "cost_usd": 0.01, "total_tokens": 100, "latency_ms": 500},
            {"model": "gpt-4", "cost_usd": 0.02, "total_tokens": 200, "latency_ms": 300},
            {"model": "gpt-3.5", "cost_usd": 0.005, "total_tokens": 50, "latency_ms": 100},
        ]
        data = aggregate(records, "model")
        assert data["totals"]["total_calls"] == 3
        assert data["totals"]["total_cost_usd"] == 0.035
        assert len(data["groups"]) == 2
        assert data["groups"]["gpt-4"]["calls"] == 2

    def test_aggregate_by_caller(self):
        """Should group records by caller."""
        records = [
            {"caller": "alice", "cost_usd": 0.01, "total_tokens": 100, "latency_ms": 500},
            {"caller": "bob", "cost_usd": 0.02, "total_tokens": 200, "latency_ms": 300},
        ]
        data = aggregate(records, "caller")
        assert len(data["groups"]) == 2
        assert data["groups"]["alice"]["calls"] == 1

    def test_aggregate_empty_records(self):
        """Should handle empty records."""
        data = aggregate([], "model")
        assert data["totals"]["total_calls"] == 0
        assert data["groups"] == {}

    def test_aggregate_missing_field(self):
        """Should handle records missing the group-by field."""
        records = [
            {"cost_usd": 0.01, "total_tokens": 100, "latency_ms": 500},
        ]
        data = aggregate(records, "model")
        assert "unknown" in data["groups"]


class TestFormatTable:
    """Tests for table formatting."""

    def test_format_table_with_data(self):
        """Should produce formatted table string."""
        data = {
            "groups": {
                "gpt-4": {
                    "calls": 2, "total_cost_usd": 0.03, "avg_cost_usd": 0.015,
                    "total_tokens": 300, "avg_latency_ms": 400.0, "total_latency_ms": 800.0,
                    "avg_tokens": 150.0,
                },
            },
            "totals": {"total_calls": 2, "total_cost_usd": 0.03, "total_tokens": 300},
        }
        table = format_table(data)
        assert "gpt-4" in table
        assert "TOTAL" in table
        assert "0.030000" in table

    def test_format_table_no_data(self):
        """Should handle empty data."""
        table = format_table({"groups": {}, "totals": {}})
        assert "No records found." in table
