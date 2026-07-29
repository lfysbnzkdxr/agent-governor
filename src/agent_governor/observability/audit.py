"""Audit logger - structured JSON logging via structlog."""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from agent_governor.models import CallRecord


class AuditLogger:
    """Structured audit logger for LLM call records.

    Logs structured JSON with: timestamp, model, tokens, cost, latency, caller.
    Supports stdout or file output.
    """

    def __init__(
        self,
        enabled: bool = True,
        log_file: str = "",
        logger_name: str = "agent_governor.audit",
    ) -> None:
        self._enabled = enabled
        self._log_file = log_file
        self._records: list[CallRecord] = []
        self._lock = threading.Lock()
        self._writer: Any = None
        self._file_handle: Any = None

        if enabled:
            self._logger = self._build_logger(logger_name)
        else:
            self._logger = None

    def _build_logger(self, name: str) -> Any:
        """Build a structlog logger with appropriate processors."""
        processors: list[Any] = [
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
        ]

        if self._log_file:
            # File output - JSON lines
            Path(self._log_file).parent.mkdir(parents=True, exist_ok=True)
            processors.append(structlog.processors.JSONRenderer())
            self._file_handle = open(self._log_file, "a", encoding="utf-8")  # noqa: SIM115
            self._writer = self._file_handle
        else:
            # Console output - colored for dev
            processors.append(structlog.dev.ConsoleRenderer(colors=True))
            self._writer = sys.stdout

        return structlog.wrap_logger(
            structlog.PrintLogger(file=self._writer),
            processors=processors,
        )

    async def log(self, record: CallRecord) -> None:
        """Log a call record.

        Args:
            record: The call record to log.
        """
        # Always store in memory for programmatic access
        with self._lock:
            self._records.append(record)

        if not self._enabled or self._logger is None:
            return

        self._logger.info(
            "llm_call",
            call_id=record.id,
            caller=record.caller,
            model=record.model,
            prompt_tokens=record.prompt_tokens,
            completion_tokens=record.completion_tokens,
            total_tokens=record.total_tokens,
            cost_usd=f"${record.cost_usd:.6f}",
            latency_ms=f"{record.latency_ms:.1f}",
            status=record.status,
            error=record.error,
        )

    @property
    def records(self) -> list[CallRecord]:
        """Get all in-memory records."""
        with self._lock:
            return list(self._records)

    @property
    def total_records(self) -> int:
        """Get total number of logged records."""
        with self._lock:
            return len(self._records)

    def clear(self) -> None:
        """Clear in-memory records."""
        with self._lock:
            self._records.clear()

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of all logged records."""
        with self._lock:
            if not self._records:
                return {"total_calls": 0, "total_cost_usd": 0.0}

            total_cost = sum(r.cost_usd for r in self._records)
            total_tokens = sum(r.total_tokens for r in self._records)
            avg_latency = sum(r.latency_ms for r in self._records) / len(self._records)

            models_used: dict[str, int] = {}
            for r in self._records:
                models_used[r.model] = models_used.get(r.model, 0) + 1

            return {
                "total_calls": len(self._records),
                "total_cost_usd": total_cost,
                "total_tokens": total_tokens,
                "avg_latency_ms": avg_latency,
                "models_used": models_used,
                "errors": sum(1 for r in self._records if r.status != "success"),
            }

    def close(self) -> None:
        """Close the audit logger and release file resources."""
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None
            self._writer = None
            self._logger = None
