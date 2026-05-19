"""Nexus-LLM Async Client.

Provides the AsyncClient for making asynchronous HTTP requests to
the Nexus-LLM API server using asyncio and aiohttp (or fallback).
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AsyncClientConfig:
    """Configuration for the async client.

    Attributes:
        base_url: Base URL of the Nexus-LLM server.
        api_key: API key for authentication.
        timeout: Request timeout in seconds.
        max_connections: Maximum concurrent connections.
        headers: Default headers.
    """

    base_url: str = "http://localhost:8000"
    api_key: Optional[str] = None
    timeout: int = 60
    max_connections: int = 10
    headers: Dict[str, str] = field(default_factory=dict)


class AsyncClient:
    """Asynchronous client for the Nexus-LLM API.

    Provides async methods for all API endpoints, supporting
    concurrent requests and streaming responses.

    Example::

        client = AsyncClient(AsyncClientConfig(base_url="http://localhost:8000"))
        response = await client.chat(messages=[{"role": "user", "content": "Hello!"}])
    """

    def __init__(self, config: Optional[AsyncClientConfig] = None) -> None:
        self._config = config or AsyncClientConfig()
        self._session: Any = None
        self._headers = self._build_headers()
        logger.debug("AsyncClient initialized: %s", self._config.base_url)

    @property
    def config(self) -> AsyncClientConfig:
        return self._config

    async def _get_session(self) -> Any:
        """Get or create an aiohttp ClientSession."""
        if self._session is None or self._session.closed:
            try:
                import aiohttp
                self._session = aiohttp.ClientSession(
                    headers=self._headers,
                    timeout=aiohttp.ClientTimeout(total=self._config.timeout),
                )
            except ImportError:
                raise RuntimeError("aiohttp is required for AsyncClient. Install with: pip install aiohttp")
        return self._session

    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Send an async chat completion request.

        Args:
            messages: List of message dicts.
            model: Model name.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens.
            **kwargs: Additional parameters.

        Returns:
            API response dictionary.
        """
        payload = {
            "messages": messages,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }
        return await self._post("/v1/chat/completions", payload)

    async def complete(
        self,
        prompt: str,
        model: str = "",
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Send an async text completion request."""
        payload = {
            "prompt": prompt,
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs,
        }
        return await self._post("/v1/completions", payload)

    async def embed(
        self,
        input_text: Any,
        model: str = "",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Send an async embedding request."""
        payload = {"input": input_text, "model": model, **kwargs}
        return await self._post("/v1/embeddings", payload)

    async def list_models(self) -> Dict[str, Any]:
        """List available models asynchronously."""
        return await self._get("/v1/models")

    async def health(self) -> Dict[str, Any]:
        """Check server health asynchronously."""
        return await self._get("/health")

    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        model: str = "",
        **kwargs: Any,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream chat completion responses.

        Args:
            messages: List of message dicts.
            model: Model name.
            **kwargs: Additional parameters.

        Yields:
            Stream chunks as dictionaries.
        """
        payload = {"messages": messages, "model": model, "stream": True, **kwargs}
        session = await self._get_session()
        url = f"{self._config.base_url}/v1/chat/completions"

        async with session.post(url, json=payload) as response:
            async for line in response.content:
                line_str = line.decode("utf-8").strip()
                if line_str.startswith("data: "):
                    data_str = line_str[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        yield json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

    async def close(self) -> None:
        """Close the client session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def _get(self, path: str) -> Dict[str, Any]:
        """Make an async GET request."""
        session = await self._get_session()
        url = f"{self._config.base_url}{path}"
        async with session.get(url) as response:
            return await response.json()

    async def _post(self, path: str, data: Any) -> Dict[str, Any]:
        """Make an async POST request."""
        session = await self._get_session()
        url = f"{self._config.base_url}{path}"
        async with session.post(url, json=data) as response:
            return await response.json()

    def _build_headers(self) -> Dict[str, str]:
        """Build default request headers."""
        headers = dict(self._config.headers)
        if self._config.api_key:
            headers["Authorization"] = f"Bearer {self._config.api_key}"
        headers["User-Agent"] = "Nexus-LLM-AsyncClient/1.0"
        return headers

    async def __aenter__(self) -> "AsyncClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()
