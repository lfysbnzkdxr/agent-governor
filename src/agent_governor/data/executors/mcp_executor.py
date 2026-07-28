"""MCP (Model Context Protocol) tool executor.

Implements the Executor Protocol for MCP tool calls.
Requires: pip install agent-governor[mcp]
"""

from __future__ import annotations

import time
from typing import Any

try:
    from mcp import ClientSession

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

from agent_governor.models import Task, TaskResult, TaskStatus


class MCPExecutor:
    """Executor protocol implementation for MCP tool calls.

    Wraps an MCP ClientSession to call tools via the Model Context Protocol.

    Usage:
        executor = MCPExecutor(session=mcp_session)
        result = await executor.execute(Task(
            name="call_tool",
            metadata={"tool_name": "search", "tool_arguments": {"query": "..."}},
        ))
    """

    def __init__(self, session: Any, *, default_timeout: float = 30.0) -> None:
        if not MCP_AVAILABLE:
            raise ImportError(
                "mcp package is required: pip install agent-governor[mcp]"
            )
        self._session = session
        self._timeout = default_timeout

    async def execute(self, task: Task) -> TaskResult:
        """Execute an MCP tool call.

        Extracts tool_name and tool_arguments from task.metadata.

        Args:
            task: Task with metadata containing tool_name and tool_arguments.

        Returns:
            TaskResult with tool output as content.
        """
        tool_name = task.metadata.get("tool_name", task.name)
        tool_args = task.metadata.get("tool_arguments", {})
        start = time.perf_counter()

        try:
            result = await self._session.call_tool(
                tool_name,
                arguments=tool_args,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000

            # Extract content from MCP result
            content = ""
            if hasattr(result, "content") and result.content:
                if isinstance(result.content, list):
                    content = "\n".join(
                        str(item.text) if hasattr(item, "text") else str(item)
                        for item in result.content
                    )
                else:
                    content = str(result.content)

            return TaskResult(
                task_id=task.id,
                status=TaskStatus.COMPLETED,
                content=content,
                model="mcp",
                latency_ms=elapsed_ms,
                metadata={"tool_name": tool_name, "mcp_result": str(result)},
            )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return TaskResult(
                task_id=task.id,
                status=TaskStatus.FAILED,
                model="mcp",
                error=str(e),
                latency_ms=elapsed_ms,
                metadata={"tool_name": tool_name},
            )
