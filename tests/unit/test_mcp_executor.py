"""Tests for MCPExecutor."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_governor.models import Task, TaskStatus


@pytest.mark.asyncio
class TestMCPExecutor:
    """Tests for MCPExecutor."""

    async def test_raises_import_error_when_mcp_not_available(self):
        """MCPExecutor should raise ImportError when mcp is not installed."""
        with patch("agent_governor.data.executors.mcp_executor.MCP_AVAILABLE", False):
            from agent_governor.data.executors.mcp_executor import MCPExecutor

            with pytest.raises(ImportError, match="mcp package is required"):
                MCPExecutor(session=MagicMock())

    async def _make_executor(self, session):
        from agent_governor.data.executors.mcp_executor import MCP_AVAILABLE, MCPExecutor

        if not MCP_AVAILABLE:
            with patch("agent_governor.data.executors.mcp_executor.MCP_AVAILABLE", True):
                return MCPExecutor(session=session)
        return MCPExecutor(session=session)

    async def test_execute_calls_mcp_tool(self):
        """MCPExecutor.execute should call session.call_tool."""
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.content = [MagicMock(text="search result")]
        mock_session.call_tool.return_value = mock_result

        executor = await self._make_executor(mock_session)
        task = Task(
            name="search_tool",
            metadata={"tool_name": "search", "tool_arguments": {"q": "test"}},
        )

        result = await executor.execute(task)

        assert result.status == TaskStatus.COMPLETED
        assert "search result" in result.content
        mock_session.call_tool.assert_awaited_once_with(
            "search",
            arguments={"q": "test"},
        )

    async def test_execute_handles_error(self):
        """MCPExecutor.execute should handle tool errors."""
        mock_session = AsyncMock()
        mock_session.call_tool.side_effect = ValueError("Tool execution failed")

        executor = await self._make_executor(mock_session)
        task = Task(
            name="failing_tool",
            metadata={"tool_name": "fail", "tool_arguments": {}},
        )

        result = await executor.execute(task)

        assert result.status == TaskStatus.FAILED
        assert "Tool execution failed" in result.error

    async def test_execute_defaults_to_task_name(self):
        """MCPExecutor should use task.name when tool_name is not in metadata."""
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.content = [MagicMock(text="ok")]
        mock_session.call_tool.return_value = mock_result

        executor = await self._make_executor(mock_session)
        task = Task(name="default_tool")

        result = await executor.execute(task)

        assert result.status == TaskStatus.COMPLETED
        mock_session.call_tool.assert_awaited_once_with(
            "default_tool",
            arguments={},
        )
