"""HTTP generic executor using httpx (already a transitive dep via litellm)."""

from __future__ import annotations

import time

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

from agent_governor.models import Task, TaskResult, TaskStatus


class HTTPExecutor:
    """Executor protocol implementation for HTTP API calls.

    Uses httpx.AsyncClient with connection pooling (lazy init on first execute).

    Usage:
        executor = HTTPExecutor(base_url="https://api.example.com")
        result = await executor.execute(Task(
            name="get_data",
            metadata={"method": "GET", "path": "/v1/data"},
        ))
    """

    def __init__(
        self,
        base_url: str,
        *,
        headers: dict[str, str] | None = None,
        timeout: float = 30.0,
        max_connections: int = 100,
    ) -> None:
        if not HTTPX_AVAILABLE:
            raise ImportError("httpx package is required: pip install agent-governor[http]")
        self._base_url = base_url.rstrip("/")
        self._headers = headers or {}
        self._timeout = timeout
        self._max_connections = max_connections
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Lazy initialize the HTTP client with connection pooling."""
        if self._client is None:
            limits = httpx.Limits(
                max_connections=self._max_connections,
                max_keepalive_connections=min(20, self._max_connections),
            )
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers=self._headers,
                timeout=self._timeout,
                limits=limits,
            )
        return self._client

    async def execute(self, task: Task) -> TaskResult:
        """Execute an HTTP request.

        Extracts method, path, json_body, query_params, and headers
        from task.metadata.

        Args:
            task: Task with HTTP request metadata.

        Returns:
            TaskResult with response body as content.
        """
        method = task.metadata.get("method", "GET").upper()
        path = task.metadata.get("path", "/")
        json_body = task.metadata.get("json_body")
        query_params = task.metadata.get("query_params")
        extra_headers = task.metadata.get("headers", {})

        start = time.perf_counter()

        try:
            client = await self._get_client()
            response = await client.request(
                method=method,
                url=path,
                json=json_body,
                params=query_params,
                headers=extra_headers or None,
            )

            elapsed_ms = (time.perf_counter() - start) * 1000

            # Map HTTP status to TaskStatus
            if 200 <= response.status_code < 300:
                status = TaskStatus.COMPLETED
            elif response.status_code == 429:
                status = TaskStatus.FAILED  # Rate limited
            elif 400 <= response.status_code < 500:
                status = TaskStatus.FAILED
            else:
                status = TaskStatus.FAILED

            try:
                content = response.text
            except Exception:
                content = ""

            return TaskResult(
                task_id=task.id,
                status=status,
                content=content,
                model="http",
                latency_ms=elapsed_ms,
                metadata={
                    "method": method,
                    "path": path,
                    "status_code": response.status_code,
                },
            )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return TaskResult(
                task_id=task.id,
                status=TaskStatus.FAILED,
                model="http",
                error=str(e),
                latency_ms=elapsed_ms,
                metadata={"method": method, "path": path},
            )

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
