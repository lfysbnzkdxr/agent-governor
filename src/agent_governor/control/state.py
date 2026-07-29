"""State Manager: SQLite-based checkpoint persistence with WAL mode.

Provides crash-recoverable state persistence for DAG executions.
Uses stdlib sqlite3 with asyncio.to_thread for non-blocking I/O.
WAL (Write-Ahead Logging) mode enables concurrent reads during writes.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

import structlog

from agent_governor.models import DagNodeResult, DagNodeStatus

logger = structlog.get_logger(__name__)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS checkpoints (
    graph_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    status TEXT NOT NULL,
    data TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (graph_id, node_id)
);

CREATE TABLE IF NOT EXISTS graph_state (
    graph_id TEXT PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'pending',
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_checkpoints_graph ON checkpoints(graph_id);
"""


class StateManager:
    """SQLite-backed state manager for DAG checkpoint and recovery.

    Implements the StateBackend protocol with additional DAG-specific
    checkpoint operations.

    Example:
        state = StateManager("checkpoints.db")
        await state.initialize()

        # Save node checkpoint
        await state.save_node(graph_id, "node_1", node_result)

        # Recover after crash
        recovered = await state.load_graph_state(graph_id)
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        """Initialize StateManager.

        Args:
            db_path: Path to SQLite database file. ":memory:" for in-memory.
        """
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None

    async def initialize(self) -> None:
        """Initialize the database connection and schema."""

        def _init() -> sqlite3.Connection:
            if self._db_path != ":memory:":
                Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.executescript(_SCHEMA_SQL)
            conn.commit()
            return conn

        self._conn = await asyncio.to_thread(_init)
        logger.info("state_manager_initialized", db_path=self._db_path)

    async def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            conn = self._conn
            self._conn = None
            await asyncio.to_thread(conn.close)

    def _ensure_conn(self) -> sqlite3.Connection:
        """Get connection, raising if not initialized."""
        if self._conn is None:
            raise RuntimeError("StateManager not initialized. Call initialize() first.")
        return self._conn

    # === StateBackend Protocol ===

    async def save(self, key: str, state: dict[str, Any]) -> None:
        """Persist arbitrary state under a key (StateBackend protocol)."""
        conn = self._ensure_conn()
        now = time.time()

        def _save() -> None:
            conn.execute(
                """INSERT INTO graph_state (graph_id, status, metadata, created_at, updated_at)
                   VALUES (?, 'active', ?, ?, ?)
                   ON CONFLICT(graph_id) DO UPDATE SET
                       metadata = excluded.metadata,
                       updated_at = excluded.updated_at""",
                (key, json.dumps(state), now, now),
            )
            conn.commit()

        await asyncio.to_thread(_save)

    async def load(self, key: str) -> dict[str, Any] | None:
        """Load state by key (StateBackend protocol)."""
        conn = self._ensure_conn()

        def _load() -> dict[str, Any] | None:
            row = conn.execute(
                "SELECT metadata FROM graph_state WHERE graph_id = ?", (key,)
            ).fetchone()
            if row:
                return json.loads(row[0])  # type: ignore[no-any-return]
            return None

        return await asyncio.to_thread(_load)

    async def delete(self, key: str) -> None:
        """Delete state by key (StateBackend protocol)."""
        conn = self._ensure_conn()

        def _delete() -> None:
            conn.execute("DELETE FROM graph_state WHERE graph_id = ?", (key,))
            conn.execute("DELETE FROM checkpoints WHERE graph_id = ?", (key,))
            conn.commit()

        await asyncio.to_thread(_delete)

    # === DAG Checkpoint Operations ===

    async def save_graph_state(
        self,
        graph_id: str,
        status: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save or update overall graph execution state."""
        conn = self._ensure_conn()
        now = time.time()

        def _save() -> None:
            conn.execute(
                """INSERT INTO graph_state (graph_id, status, metadata, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(graph_id) DO UPDATE SET
                       status = excluded.status,
                       metadata = excluded.metadata,
                       updated_at = excluded.updated_at""",
                (graph_id, status, json.dumps(metadata or {}), now, now),
            )
            conn.commit()

        await asyncio.to_thread(_save)

    async def save_node(
        self,
        graph_id: str,
        node_id: str,
        node_result: DagNodeResult,
    ) -> None:
        """Save a node checkpoint."""
        conn = self._ensure_conn()
        now = time.time()
        data = node_result.model_dump_json()

        def _save() -> None:
            conn.execute(
                """INSERT INTO checkpoints (graph_id, node_id, status, data, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(graph_id, node_id) DO UPDATE SET
                       status = excluded.status,
                       data = excluded.data,
                       updated_at = excluded.updated_at""",
                (graph_id, node_id, node_result.status.value, data, now, now),
            )
            conn.commit()

        await asyncio.to_thread(_save)

    async def load_graph_state(self, graph_id: str) -> dict[str, Any] | None:
        """Load overall graph state for recovery."""
        conn = self._ensure_conn()

        def _load() -> dict[str, Any] | None:
            row = conn.execute(
                "SELECT status, metadata, created_at, updated_at FROM graph_state WHERE graph_id = ?",
                (graph_id,),
            ).fetchone()
            if row:
                return {
                    "graph_id": graph_id,
                    "status": row[0],
                    "metadata": json.loads(row[1]),
                    "created_at": row[2],
                    "updated_at": row[3],
                }
            return None

        return await asyncio.to_thread(_load)

    async def load_node_checkpoints(self, graph_id: str) -> dict[str, DagNodeResult]:
        """Load all node checkpoints for a graph.

        Returns:
            Dict mapping node_id to DagNodeResult.
        """
        conn = self._ensure_conn()

        def _load() -> dict[str, DagNodeResult]:
            rows = conn.execute(
                "SELECT node_id, data FROM checkpoints WHERE graph_id = ?",
                (graph_id,),
            ).fetchall()
            results: dict[str, DagNodeResult] = {}
            for node_id, data_json in rows:
                results[node_id] = DagNodeResult.model_validate_json(data_json)
            return results

        return await asyncio.to_thread(_load)

    async def get_completed_nodes(self, graph_id: str) -> set[str]:
        """Get set of completed node IDs for a graph (for recovery skip)."""
        conn = self._ensure_conn()

        def _load() -> set[str]:
            rows = conn.execute(
                "SELECT node_id FROM checkpoints WHERE graph_id = ? AND status = ?",
                (graph_id, DagNodeStatus.COMPLETED.value),
            ).fetchall()
            return {row[0] for row in rows}

        return await asyncio.to_thread(_load)

    async def cleanup(self, graph_id: str) -> None:
        """Remove all checkpoints for a completed graph."""
        await self.delete(graph_id)
