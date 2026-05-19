"""Nexus-LLM API Call Tool.

Provides the APITool for making HTTP requests to external APIs,
with support for various HTTP methods, headers, and body formats.
"""

import json
import logging
from typing import Any, Dict, List
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from nexus_llm.tools.base_tool import BaseTool, ToolParameter, ToolResult, ParameterType

logger = logging.getLogger(__name__)


class APITool(BaseTool):
    """HTTP API call tool for making requests to external services.

    Supports GET, POST, PUT, PATCH, DELETE methods with configurable
    headers, request body, and timeout.

    Example::

        api = APITool()
        result = api.execute(method="GET", url="https://api.example.com/data")
    """

    def __init__(self, default_timeout: int = 30) -> None:
        super().__init__(name="api", description="Make HTTP requests to external APIs")
        self._default_timeout = default_timeout

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="method", type=ParameterType.STRING, description="HTTP method", required=True,
                          choices=["GET", "POST", "PUT", "PATCH", "DELETE"]),
            ToolParameter(name="url", type=ParameterType.STRING, description="Request URL", required=True),
            ToolParameter(name="headers", type=ParameterType.STRING, description="JSON-encoded headers", required=False),
            ToolParameter(name="body", type=ParameterType.STRING, description="Request body (JSON string)", required=False),
            ToolParameter(name="timeout", type=ParameterType.INTEGER, description="Request timeout in seconds", required=False, default=30),
        ]

    def execute(
        self,
        method: str = "GET",
        url: str = "",
        headers: str = "",
        body: str = "",
        timeout: int = 0,
        **kwargs: Any,
    ) -> ToolResult:
        """Make an HTTP request.

        Args:
            method: HTTP method.
            url: Target URL.
            headers: JSON-encoded headers dict.
            body: Request body (JSON string).
            timeout: Request timeout in seconds.

        Returns:
            ToolResult with response data.
        """
        if not url:
            return ToolResult(tool_name=self.name, success=False, error="No URL provided")

        try:
            parsed_headers: Dict[str, str] = {}
            if headers:
                parsed_headers = json.loads(headers)

            parsed_body: Optional[bytes] = None
            if body:
                parsed_body = body.encode("utf-8")
                parsed_headers.setdefault("Content-Type", "application/json")

            request = Request(url, data=parsed_body, method=method.upper())
            for key, value in parsed_headers.items():
                request.add_header(key, value)

            actual_timeout = timeout or self._default_timeout
            with urlopen(request, timeout=actual_timeout) as response:
                response_body = response.read().decode("utf-8", errors="replace")
                status = response.status
                response_headers = dict(response.headers)

            # Try to parse as JSON
            try:
                output = json.loads(response_body)
            except (json.JSONDecodeError, TypeError):
                output = response_body

            return ToolResult(
                tool_name=self.name,
                success=200 <= status < 300,
                output=output,
                metadata={
                    "status_code": status,
                    "method": method,
                    "url": url,
                    "response_headers": response_headers,
                },
            )
        except HTTPError as exc:
            body_text = ""
            try:
                body_text = exc.read().decode("utf-8", errors="replace")[:1000]
            except Exception:
                pass
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"HTTP {exc.code}: {exc.reason}",
                output=body_text,
                metadata={"status_code": exc.code, "url": url},
            )
        except URLError as exc:
            return ToolResult(tool_name=self.name, success=False, error=f"URL error: {exc.reason}")
        except Exception as exc:
            return ToolResult(tool_name=self.name, success=False, error=f"Request error: {exc}")
