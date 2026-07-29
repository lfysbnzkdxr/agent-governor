"""Tests for Cost Report CLI."""

from __future__ import annotations

import pytest

from agent_governor.cli.report import aggregate, format_table, load_records, main, parse_args


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
        args = parse_args(
            ["--format", "json", "--group-by", "caller", "--limit", "50", "audit.log"]
        )
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
        lines = "\n".join([f'{{"model": "m{i}", "cost_usd": 0.01}}' for i in range(10)])
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
                    "calls": 2,
                    "total_cost_usd": 0.03,
                    "avg_cost_usd": 0.015,
                    "total_tokens": 300,
                    "avg_latency_ms": 400.0,
                    "total_latency_ms": 800.0,
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


class TestMain:
    """Tests for the main() CLI entry point."""

    def test_main_table_output(self, tmp_path, capsys):
        """main should print table format by default."""
        f = tmp_path / "audit.log"
        f.write_text(
            '{"model": "gpt-4", "cost_usd": 0.01, "total_tokens": 100, "latency_ms": 500}\n'
        )
        main([str(f)])
        captured = capsys.readouterr()
        assert "gpt-4" in captured.out
        assert "TOTAL" in captured.out

    def test_main_json_output(self, tmp_path, capsys):
        """main should print JSON when --format json."""
        import json

        f = tmp_path / "audit.log"
        f.write_text(
            '{"model": "gpt-4", "cost_usd": 0.01, "total_tokens": 100, "latency_ms": 500}\n'
        )
        main(["--format", "json", str(f)])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "groups" in data
        assert "totals" in data

    def test_main_file_not_found(self, capsys):
        """main should exit with code 1 for missing file."""
        with pytest.raises(SystemExit) as exc_info:
            main(["/nonexistent/file.log"])
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Error" in captured.err

    def test_main_empty_records(self, tmp_path, capsys):
        """main should handle empty log file."""
        f = tmp_path / "empty.log"
        f.write_text("")
        main([str(f)])
        captured = capsys.readouterr()
        assert "No records found" in captured.out

    def test_main_group_by_tenant(self, tmp_path, capsys):
        """main should map tenant to tenant_id field."""
        f = tmp_path / "audit.log"
        f.write_text(
            '{"tenant_id": "team-a", "cost_usd": 0.01, "total_tokens": 100, "latency_ms": 500}\n'
            '{"tenant_id": "team-b", "cost_usd": 0.02, "total_tokens": 200, "latency_ms": 300}\n'
        )
        main(["--group-by", "tenant", str(f)])
        captured = capsys.readouterr()
        assert "team-a" in captured.out
        assert "team-b" in captured.out


class TestAggregateExtended:
    """Extended tests for aggregation."""

    def test_aggregate_by_day(self):
        """Should group records by day from timestamp."""
        records = [
            {
                "timestamp": "2026-07-01T10:00:00Z",
                "cost_usd": 0.01,
                "total_tokens": 100,
                "latency_ms": 500,
            },
            {
                "timestamp": "2026-07-01T14:00:00Z",
                "cost_usd": 0.02,
                "total_tokens": 200,
                "latency_ms": 300,
            },
            {
                "timestamp": "2026-07-02T09:00:00Z",
                "cost_usd": 0.03,
                "total_tokens": 300,
                "latency_ms": 100,
            },
        ]
        data = aggregate(records, "day")
        assert len(data["groups"]) == 2
        assert data["groups"]["2026-07-01"]["calls"] == 2
        assert data["groups"]["2026-07-02"]["calls"] == 1

    def test_aggregate_by_tenant(self):
        """Should group records by tenant_id."""
        records = [
            {"tenant_id": "team-a", "cost_usd": 0.01, "total_tokens": 100, "latency_ms": 500},
            {"tenant_id": "team-a", "cost_usd": 0.02, "total_tokens": 200, "latency_ms": 300},
            {"tenant_id": "team-b", "cost_usd": 0.005, "total_tokens": 50, "latency_ms": 100},
        ]
        data = aggregate(records, "tenant_id")
        assert len(data["groups"]) == 2
        assert data["groups"]["team-a"]["calls"] == 2
        assert data["groups"]["team-b"]["calls"] == 1
