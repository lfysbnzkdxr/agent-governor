"""Tests for HTTPExecutor."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_governor.models import Task, TaskStatus


@pytest.mark.asyncio
class TestHTTPExecutor:
    """Tests for HTTPExecutor."""

    async def test_raises_import_error_when_httpx_not_available(self):
        """HTTPExecutor should raise ImportError when httpx is not installed."""
        with patch("agent_governor.data.executors.http_executor.HTTPX_AVAILABLE", False):
            from agent_governor.data.executors.http_executor import HTTPExecutor

            with pytest.raises(ImportError, match="httpx package is required"):
                HTTPExecutor(base_url="https://api.example.com")

    async def test_execute_get_success(self):
        """HTTPExecutor should execute a GET request and return result."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = '{"result": "ok"}'
        mock_client.request.return_value = mock_response

        from agent_governor.data.executors.http_executor import HTTPExecutor

        with patch.object(HTTPExecutor, "_get_client", return_value=mock_client):
            executor = HTTPExecutor(base_url="https://api.example.com")
            task = Task(
                name="get_data",
                metadata={"method": "GET", "path": "/v1/data"},
            )

            result = await executor.execute(task)

        assert result.status == TaskStatus.COMPLETED
        assert result.content == '{"result": "ok"}'
        mock_client.request.assert_awaited_once_with(
            method="GET",
            url="/v1/data",
            json=None,
            params=None,
            headers=None,
        )

    async def test_execute_post_with_body(self):
        """HTTPExecutor should execute a POST request with json body."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.text = '{"id": 42}'
        mock_client.request.return_value = mock_response

        from agent_governor.data.executors.http_executor import HTTPExecutor

        with patch.object(HTTPExecutor, "_get_client", return_value=mock_client):
            executor = HTTPExecutor(base_url="https://api.example.com")
            task = Task(
                name="create",
                metadata={
                    "method": "POST",
                    "path": "/v1/items",
                    "json_body": {"name": "test"},
                },
            )

            result = await executor.execute(task)

        assert result.status == TaskStatus.COMPLETED
        assert result.content == '{"id": 42}'
        mock_client.request.assert_awaited_once_with(
            method="POST",
            url="/v1/items",
            json={"name": "test"},
            params=None,
            headers=None,
        )

    async def test_execute_handles_client_error(self):
        """HTTPExecutor should handle HTTP client errors (4xx)."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.text = "Not Found"
        mock_client.request.return_value = mock_response

        from agent_governor.data.executors.http_executor import HTTPExecutor

        with patch.object(HTTPExecutor, "_get_client", return_value=mock_client):
            executor = HTTPExecutor(base_url="https://api.example.com")
            task = Task(
                name="get_data",
                metadata={"method": "GET", "path": "/v1/notfound"},
            )

            result = await executor.execute(task)

        assert result.status == TaskStatus.FAILED

    async def test_execute_handles_connection_error(self):
        """HTTPExecutor should handle connection errors."""
        mock_client = AsyncMock()
        mock_client.request.side_effect = ConnectionError("Connection refused")

        from agent_governor.data.executors.http_executor import HTTPExecutor

        with patch.object(HTTPExecutor, "_get_client", return_value=mock_client):
            executor = HTTPExecutor(base_url="https://api.example.com")
            task = Task(
                name="failing",
                metadata={"method": "GET", "path": "/fail"},
            )

            result = await executor.execute(task)

        assert result.status == TaskStatus.FAILED
        assert "Connection refused" in result.error

    async def test_lazy_client_initialization(self):
        """HTTPExecutor should lazily initialize the httpx client."""
        from agent_governor.data.executors.http_executor import HTTPExecutor

        executor = HTTPExecutor(base_url="https://api.example.com")
        assert executor._client is None  # Not initialized yet
