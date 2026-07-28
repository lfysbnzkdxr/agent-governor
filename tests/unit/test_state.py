"""Tests for StateManager (SQLite checkpoint persistence)."""

import pytest

from agent_governor.control.state import StateManager
from agent_governor.models import DagNodeResult, DagNodeStatus, TaskResult, TaskStatus


@pytest.fixture
async def state_mgr():
    """Create an in-memory StateManager for testing."""
    mgr = StateManager(":memory:")
    await mgr.initialize()
    yield mgr
    await mgr.close()


class TestStateManagerProtocol:
    async def test_save_and_load(self, state_mgr: StateManager):
        await state_mgr.save("key1", {"foo": "bar", "count": 42})
        loaded = await state_mgr.load("key1")
        assert loaded == {"foo": "bar", "count": 42}

    async def test_load_nonexistent_returns_none(self, state_mgr: StateManager):
        result = await state_mgr.load("nonexistent")
        assert result is None

    async def test_save_overwrites(self, state_mgr: StateManager):
        await state_mgr.save("key1", {"v": 1})
        await state_mgr.save("key1", {"v": 2})
        loaded = await state_mgr.load("key1")
        assert loaded == {"v": 2}

    async def test_delete(self, state_mgr: StateManager):
        await state_mgr.save("key1", {"data": True})
        await state_mgr.delete("key1")
        assert await state_mgr.load("key1") is None

    async def test_not_initialized_raises(self):
        mgr = StateManager(":memory:")
        with pytest.raises(RuntimeError, match="not initialized"):
            await mgr.save("k", {})


class TestStateManagerDagCheckpoint:
    async def test_save_and_load_graph_state(self, state_mgr: StateManager):
        await state_mgr.save_graph_state("g1", "running", {"name": "test_graph"})
        state = await state_mgr.load_graph_state("g1")
        assert state is not None
        assert state["status"] == "running"
        assert state["metadata"]["name"] == "test_graph"

    async def test_save_node_checkpoint(self, state_mgr: StateManager):
        node_result = DagNodeResult(
            node_id="n1",
            status=DagNodeStatus.COMPLETED,
            result=TaskResult(task_id="t1", status=TaskStatus.COMPLETED, content="done"),
            attempts=1,
        )
        await state_mgr.save_node("g1", "n1", node_result)

        checkpoints = await state_mgr.load_node_checkpoints("g1")
        assert "n1" in checkpoints
        assert checkpoints["n1"].status == DagNodeStatus.COMPLETED
        assert checkpoints["n1"].result.content == "done"

    async def test_multiple_node_checkpoints(self, state_mgr: StateManager):
        for i in range(3):
            nr = DagNodeResult(
                node_id=f"n{i}",
                status=DagNodeStatus.COMPLETED,
                attempts=1,
            )
            await state_mgr.save_node("g1", f"n{i}", nr)

        checkpoints = await state_mgr.load_node_checkpoints("g1")
        assert len(checkpoints) == 3

    async def test_get_completed_nodes(self, state_mgr: StateManager):
        await state_mgr.save_node(
            "g1", "done_node",
            DagNodeResult(node_id="done_node", status=DagNodeStatus.COMPLETED),
        )
        await state_mgr.save_node(
            "g1", "failed_node",
            DagNodeResult(node_id="failed_node", status=DagNodeStatus.FAILED),
        )

        completed = await state_mgr.get_completed_nodes("g1")
        assert completed == {"done_node"}

    async def test_node_checkpoint_update(self, state_mgr: StateManager):
        nr1 = DagNodeResult(node_id="n1", status=DagNodeStatus.RUNNING, attempts=1)
        await state_mgr.save_node("g1", "n1", nr1)

        nr2 = DagNodeResult(node_id="n1", status=DagNodeStatus.COMPLETED, attempts=2)
        await state_mgr.save_node("g1", "n1", nr2)

        checkpoints = await state_mgr.load_node_checkpoints("g1")
        assert checkpoints["n1"].status == DagNodeStatus.COMPLETED
        assert checkpoints["n1"].attempts == 2

    async def test_cleanup_removes_all(self, state_mgr: StateManager):
        await state_mgr.save_graph_state("g1", "completed")
        await state_mgr.save_node(
            "g1", "n1", DagNodeResult(node_id="n1", status=DagNodeStatus.COMPLETED)
        )

        await state_mgr.cleanup("g1")

        assert await state_mgr.load_graph_state("g1") is None
        assert await state_mgr.load_node_checkpoints("g1") == {}


class TestStateManagerFileBased:
    async def test_file_persistence(self, tmp_path):
        db_file = str(tmp_path / "test.db")

        # Write
        mgr1 = StateManager(db_file)
        await mgr1.initialize()
        await mgr1.save("persist_key", {"value": 123})
        await mgr1.close()

        # Read back
        mgr2 = StateManager(db_file)
        await mgr2.initialize()
        loaded = await mgr2.load("persist_key")
        assert loaded == {"value": 123}
        await mgr2.close()
