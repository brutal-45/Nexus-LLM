"""API client for connecting to a Nexus-LLM server.

Provides both synchronous REST and asynchronous WebSocket interfaces with
automatic reconnection, error handling, and streaming support.
"""

import json
import logging
import time
from enum import Enum
from typing import Any, Dict, Generator, List, Optional

from nexus_llm.core.exceptions import ServerError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Connection state
# ---------------------------------------------------------------------------

class ConnectionState(str, Enum):
    """Client connection state."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# NexusClient
# ---------------------------------------------------------------------------

class NexusClient:
    """REST and WebSocket client for the Nexus-LLM server.

    Usage::

        client = NexusClient(base_url="http://localhost:8000")
        client.connect()

        # Synchronous chat
        response = client.chat("Hello, how are you?")

        # Streaming chat
        for token in client.chat_stream("Tell me a story"):
            print(token, end="", flush=True)

        # Generate text
        result = client.generate("Once upon a time", max_length=200)

        client.disconnect()
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._state = ConnectionState.DISCONNECTED
        self._session: Optional[Any] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def base_url(self) -> str:
        """The server base URL."""
        return self._base_url

    @property
    def state(self) -> ConnectionState:
        """Current connection state."""
        return self._state

    @property
    def is_connected(self) -> bool:
        """Whether the client is connected to the server."""
        return self._state == ConnectionState.CONNECTED

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Establish a connection to the server.

        Creates an HTTP session and validates connectivity with a
        health-check request.
        """
        try:
            import requests
        except ImportError as exc:
            raise ServerError(
                "The 'requests' package is required for the API client. "
                "Install it with: pip install requests"
            ) from exc

        if self.is_connected:
            logger.debug("Already connected to %s", self._base_url)
            return

        self._state = ConnectionState.CONNECTING
        logger.info("Connecting to %s …", self._base_url)

        self._session = requests.Session()
        self._session.headers.update(self._build_headers())

        # Health check
        try:
            resp = self._request("GET", "/health", retry=False)
            if resp.get("status") != "ok":
                raise ServerError(f"Server health check failed: {resp}")
        except Exception as exc:
            self._state = ConnectionState.FAILED
            self._session = None
            raise ServerError(f"Cannot connect to {self._base_url}: {exc}") from exc

        self._state = ConnectionState.CONNECTED
        logger.info("Connected to %s", self._base_url)

    def disconnect(self) -> None:
        """Close the connection to the server."""
        if self._session is not None:
            self._session.close()
            self._session = None
        self._state = ConnectionState.DISCONNECTED
        logger.info("Disconnected from %s", self._base_url)

    # ------------------------------------------------------------------
    # Chat endpoint
    # ------------------------------------------------------------------

    def chat(
        self,
        message: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Send a chat message and return the full response.

        Args:
            message: The user message.
            model: Optional model ID override.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            history: Optional conversation history as list of
                     ``{"role": ..., "content": ...}`` dicts.

        Returns:
            The server's JSON response.
        """
        payload: Dict[str, Any] = {
            "message": message,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if model:
            payload["model"] = model
        if history:
            payload["history"] = history

        return self._request("POST", "/chat", json_data=payload)

    def chat_stream(
        self,
        message: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Generator[str, None, None]:
        """Send a chat message and yield streamed tokens.

        Uses server-sent events (SSE) over the ``/chat/stream`` endpoint.

        Args:
            message: The user message.
            model: Optional model ID override.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            history: Optional conversation history.

        Yields:
            Individual token strings as they arrive.
        """
        payload: Dict[str, Any] = {
            "message": message,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if model:
            payload["model"] = model
        if history:
            payload["history"] = history

        url = f"{self._base_url}/chat/stream"
        headers = self._build_headers()

        try:
            import requests
        except ImportError as exc:
            raise ServerError("'requests' is required for streaming.") from exc

        with requests.post(url, json=payload, headers=headers, stream=True, timeout=self._timeout) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if line.startswith("data: "):
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        break
                    try:
                        parsed = json.loads(data)
                        token = parsed.get("token", "")
                        if token:
                            yield token
                    except json.JSONDecodeError:
                        yield data

    # ------------------------------------------------------------------
    # Generate endpoint
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_length: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
    ) -> Dict[str, Any]:
        """Generate text from a prompt.

        Args:
            prompt: Input text prompt.
            model: Optional model ID override.
            max_length: Maximum length of generated text.
            temperature: Sampling temperature.
            top_p: Nucleus sampling probability.
            top_k: Top-k sampling parameter.

        Returns:
            The server's JSON response with generated text.
        """
        payload: Dict[str, Any] = {
            "prompt": prompt,
            "max_length": max_length,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        }
        if model:
            payload["model"] = model

        return self._request("POST", "/generate", json_data=payload)

    # ------------------------------------------------------------------
    # Model management endpoints
    # ------------------------------------------------------------------

    def load_model(
        self,
        model_id: str,
        device: str = "auto",
        precision: str = "fp32",
    ) -> Dict[str, Any]:
        """Request the server to load a model.

        Args:
            model_id: Short model ID from the catalog.
            device: Target device ("auto", "cuda", "cpu").
            precision: Precision mode.

        Returns:
            Server response confirming the load operation.
        """
        payload = {"model_id": model_id, "device": device, "precision": precision}
        return self._request("POST", "/model/load", json_data=payload)

    def unload_model(self) -> Dict[str, Any]:
        """Request the server to unload the current model.

        Returns:
            Server response confirming the unload operation.
        """
        return self._request("POST", "/model/unload")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the currently loaded model on the server.

        Returns:
            Dict with model information.
        """
        return self._request("GET", "/model/info")

    def list_models(self) -> List[Dict[str, Any]]:
        """List available models on the server.

        Returns:
            List of model information dicts.
        """
        resp = self._request("GET", "/models")
        return resp.get("models", [])

    # ------------------------------------------------------------------
    # Server info
    # ------------------------------------------------------------------

    def health(self) -> Dict[str, Any]:
        """Check the server health.

        Returns:
            Dict with status information.
        """
        return self._request("GET", "/health", retry=False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_headers(self) -> Dict[str, str]:
        """Build HTTP headers including optional API key."""
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _request(
        self,
        method: str,
        endpoint: str,
        json_data: Optional[Dict[str, Any]] = None,
        retry: bool = True,
    ) -> Dict[str, Any]:
        """Execute an HTTP request with retry and error handling.

        Args:
            method: HTTP method (GET, POST, etc.).
            endpoint: API endpoint path (e.g. "/chat").
            json_data: Optional JSON body.
            retry: Whether to retry on transient failures.

        Returns:
            Parsed JSON response dict.

        Raises:
            ServerError: On request failure after all retries.
        """
        if self._session is None:
            raise ServerError("Not connected. Call connect() first.")

        url = f"{self._base_url}{endpoint}"
        last_error: Optional[Exception] = None
        attempts = self._max_retries if retry else 1

        for attempt in range(1, attempts + 1):
            try:
                resp = self._session.request(
                    method=method,
                    url=url,
                    json=json_data,
                    timeout=self._timeout,
                )
                resp.raise_for_status()
                return resp.json()

            except Exception as exc:
                last_error = exc
                if attempt < attempts:
                    delay = self._retry_delay * (2 ** (attempt - 1))
                    logger.warning(
                        "Request %s %s failed (attempt %d/%d): %s — retrying in %.1fs",
                        method, endpoint, attempt, attempts, exc, delay,
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        "Request %s %s failed after %d attempts: %s",
                        method, endpoint, attempts, exc,
                    )

        raise ServerError(
            f"Request {method} {endpoint} failed: {last_error}"
        ) from last_error

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "NexusClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.disconnect()
