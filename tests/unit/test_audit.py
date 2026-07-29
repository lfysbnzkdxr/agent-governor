"""Unit tests for AuditLogger."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from agent_governor.models import CallRecord
from agent_governor.observability.audit import AuditLogger


class TestAuditLogger:
    """Tests for AuditLogger."""

    def test_init_disabled(self) -> None:
        logger = AuditLogger(enabled=False)
        assert logger.total_records == 0
        assert logger.records == []

    def test_log_adds_record(self) -> None:
        logger = AuditLogger(enabled=False)
        record = CallRecord(model="gpt-4o", prompt_tokens=10, completion_tokens=5)
        logger._lock = __import__("threading").Lock()  # ensure fresh lock for test

        import asyncio

        asyncio.run(logger.log(record))

        assert logger.total_records == 1
        assert logger.records[0].model == "gpt-4o"

    def test_log_multiple_records(self) -> None:
        logger = AuditLogger(enabled=False)
        import asyncio

        for i in range(3):
            record = CallRecord(model=f"model-{i}")
            asyncio.run(logger.log(record))

        assert logger.total_records == 3

    def test_clear(self) -> None:
        logger = AuditLogger(enabled=False)
        import asyncio

        asyncio.run(logger.log(CallRecord(model="gpt-4o")))
        assert logger.total_records == 1
        logger.clear()
        assert logger.total_records == 0

    def test_records_is_copy(self) -> None:
        """Verify that .records returns a copy, not the internal list."""
        logger = AuditLogger(enabled=False)
        import asyncio

        asyncio.run(logger.log(CallRecord(model="gpt-4o")))

        records_copy = logger.records
        records_copy.append(CallRecord(model="other"))
        assert logger.total_records == 1  # Internal list unchanged

    def test_get_summary_empty(self) -> None:
        logger = AuditLogger(enabled=False)
        summary = logger.get_summary()
        assert summary["total_calls"] == 0
        assert summary["total_cost_usd"] == 0.0

    def test_get_summary_with_records(self) -> None:
        logger = AuditLogger(enabled=False)
        import asyncio

        asyncio.run(
            logger.log(
                CallRecord(
                    model="gpt-4o",
                    prompt_tokens=100,
                    completion_tokens=50,
                    cost_usd=0.001,
                    latency_ms=150.0,
                    status="success",
                )
            )
        )
        asyncio.run(
            logger.log(
                CallRecord(
                    model="qwen-plus",
                    prompt_tokens=200,
                    completion_tokens=100,
                    cost_usd=0.002,
                    latency_ms=200.0,
                    status="success",
                )
            )
        )

        summary = logger.get_summary()
        assert summary["total_calls"] == 2
        assert summary["total_cost_usd"] == 0.003
        assert summary["models_used"] == {"gpt-4o": 1, "qwen-plus": 1}

    def test_get_summary_includes_errors(self) -> None:
        logger = AuditLogger(enabled=False)
        import asyncio

        asyncio.run(logger.log(CallRecord(model="gpt-4o", status="success")))
        asyncio.run(logger.log(CallRecord(model="gpt-4o", status="success")))
        asyncio.run(logger.log(CallRecord(model="gpt-4o", status="error", error="timeout")))

        summary = logger.get_summary()
        assert summary["total_calls"] == 3
        assert summary["errors"] == 1

    def test_enabled_logger_processes(self) -> None:
        """Test that enabled logger actually processes the event without error."""
        logger = AuditLogger(enabled=True)
        import asyncio

        record = CallRecord(model="gpt-4o", prompt_tokens=10, completion_tokens=5)
        # Should not raise
        asyncio.run(logger.log(record))
        assert logger.total_records == 1

    def test_file_output_writes_json(self) -> None:
        """Test that file output mode writes valid JSON lines."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "audit.jsonl"
            logger = AuditLogger(enabled=True, log_file=str(log_path))
            try:
                import asyncio

                asyncio.run(
                    logger.log(
                        CallRecord(
                            model="gpt-4o",
                            prompt_tokens=10,
                            completion_tokens=5,
                            cost_usd=0.00005,
                            latency_ms=100.0,
                            status="success",
                        )
                    )
                )

                # Verify file was written
                assert log_path.exists()
                content = log_path.read_text(encoding="utf-8")
                assert len(content.strip()) > 0

                # Verify valid JSON
                json_line = json.loads(content.strip().split("\n")[0])
                assert json_line["event"] == "llm_call"
                assert json_line["model"] == "gpt-4o"
            finally:
                logger.close()

    def test_close_releases_file_handle(self) -> None:
        """Test that close() releases the file handle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "audit.jsonl"
            logger = AuditLogger(enabled=True, log_file=str(log_path))

            assert logger._file_handle is not None
            assert not logger._file_handle.closed

            logger.close()
            assert logger._file_handle is None

    def test_stdout_logger_close_is_safe(self) -> None:
        """Test that close() on stdout logger does not fail."""
        logger = AuditLogger(enabled=True)
        logger.close()  # Should not raise
        assert logger._file_handle is None
