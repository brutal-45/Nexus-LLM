"""Nexus-LLM HTTP Client.

Provides the HttpClient for making synchronous HTTP requests to
the Nexus-LLM API server.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode

logger = logging.getLogger(__name__)


@dataclass
class HttpClientConfig:
    """Configuration for the HTTP client.

    Attributes:
        base_url: Base URL of the Nexus-LLM server.
        api_key: API key for authentication.
        timeout: Request timeout in seconds.
        headers: Default headers to include in every request.
        verify_ssl: Whether to verify SSL certificates.
    """

    base_url: str = "http://localhost:8000"
    api_key: Optional[str] = None
    timeout: int = 60
    headers: Dict[str, str] = field(default_factory=dict)
    verify_ssl: bool = True


class HttpClient:
    """Synchronous HTTP client for the Nexus-LLM API.

    Provides methods for chat completions, text completions,
    embeddings, and model management.

    Example::

        client = HttpClient(HttpClientConfig(base_url="http://localhost:8000", api_key="key"))
        response = client.chat(messages=[{"role": "user", "content": "Hello!"}])
    """

    def __init__(self, config: Optional[HttpClientConfig] = None) -> None:
        self._config = config or HttpClientConfig()
        self._session_headers = self._build_headers()
        logger.debug("HttpClient initialized: %s", self._config.base_url)

    @property
    def config(self) -> HttpClientConfig:
        return self._config

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Send a chat completion request.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            model: Model name.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
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
        return self._post("/v1/chat/completions", payload)

    def complete(
        self,
        prompt: str,
        model: str = "",
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Send a text completion request.

        Args:
            prompt: Input prompt.
            model: Model name.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            **kwargs: Additional parameters.

        Returns:
            API response dictionary.
        """
        payload = {
            "prompt": prompt,
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs,
        }
        return self._post("/v1/completions", payload)

    def embed(
        self,
        input_text: Any,
        model: str = "",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Send an embedding request.

        Args:
            input_text: Text or list of texts to embed.
            model: Model name.
            **kwargs: Additional parameters.

        Returns:
            API response dictionary.
        """
        payload = {"input": input_text, "model": model, **kwargs}
        return self._post("/v1/embeddings", payload)

    def list_models(self) -> Dict[str, Any]:
        """List available models.

        Returns:
            API response with model list.
        """
        return self._get("/v1/models")

    def get_model(self, model_id: str) -> Dict[str, Any]:
        """Get details for a specific model.

        Args:
            model_id: Model identifier.

        Returns:
            API response with model details.
        """
        return self._get(f"/v1/models/{model_id}")

    def health(self) -> Dict[str, Any]:
        """Check server health.

        Returns:
            Health check response.
        """
        return self._get("/health")

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make a GET request."""
        url = f"{self._config.base_url}{path}"
        if params:
            url += "?" + urlencode(params)
        return self._request("GET", url)

    def _post(self, path: str, data: Any) -> Dict[str, Any]:
        """Make a POST request."""
        url = f"{self._config.base_url}{path}"
        return self._request("POST", url, data=data)

    def _request(self, method: str, url: str, data: Any = None) -> Dict[str, Any]:
        """Make an HTTP request."""
        body = None
        if data is not None:
            body = json.dumps(data).encode("utf-8")

        headers = dict(self._session_headers)
        if body is not None:
            headers["Content-Type"] = "application/json"

        request = Request(url, data=body, headers=headers, method=method)

        try:
            with urlopen(request, timeout=self._config.timeout) as response:
                response_body = response.read().decode("utf-8")
                return json.loads(response_body)
        except HTTPError as exc:
            error_body = ""
            try:
                error_body = exc.read().decode("utf-8")
            except Exception:
                pass
            return {"error": f"HTTP {exc.code}: {exc.reason}", "detail": error_body}
        except URLError as exc:
            return {"error": f"URL error: {exc.reason}"}
        except Exception as exc:
            return {"error": str(exc)}

    def _build_headers(self) -> Dict[str, str]:
        """Build default request headers."""
        headers = dict(self._config.headers)
        if self._config.api_key:
            headers["Authorization"] = f"Bearer {self._config.api_key}"
        headers["User-Agent"] = "Nexus-LLM-Client/1.0"
        return headers
