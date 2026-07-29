"""Executors - MCP and HTTP execution backends.

DEPRECATED: This module will be removed in v1.0.
These executors are orthogonal to the core governance mission.
Consider using dedicated libraries (mcp SDK, httpx) directly.
"""

import warnings

warnings.warn(
    "agent_governor.data.executors is deprecated and will be removed in v1.0. "
    "Use dedicated libraries (mcp SDK, httpx) directly instead.",
    FutureWarning,
    stacklevel=2,
)

__all__: list[str] = []

try:
    from agent_governor.data.executors.http_executor import (
        HTTPExecutor as _HTTPExecutor,  # noqa: F401
    )

    __all__.append("HTTPExecutor")
except ImportError:
    pass

try:
    from agent_governor.data.executors.mcp_executor import MCPExecutor as _MCPExecutor  # noqa: F401

    __all__.append("MCPExecutor")
except ImportError:
    pass
