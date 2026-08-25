"""Nexus-LLM Python SDK.

Provides the NexusSDK class as a high-level, user-friendly interface
for interacting with the Nexus-LLM platform, combining synchronous
and asynchronous client features.
"""

from __future__ import annotations

import logging
from typing import Any

from nexus_llm.client.http_client import HttpClient, HttpClientConfig

logger = logging.getLogger(__name__)


class NexusSDK:
    """High-level Python SDK for the Nexus-LLM platform.

    The NexusSDK provides a simplified, user-friendly interface that
    wraps the lower-level client classes. It handles configuration,
    authentication, and provides convenience methods for common tasks.

    Example::

        sdk = NexusSDK(base_url="http://localhost:8000", api_key="my-key")

        # Chat
        response = sdk.chat("Hello, how are you?")

        # Completion
        result = sdk.complete("Once upon a time")

        # Embeddings
        vectors = sdk.embed(["Hello world", "Goodbye world"])

        # Models
        models = sdk.models()
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: str | None = None,
        timeout: int = 60,
        default_model: str = "",
    ) -> None:
        """Initialize the SDK.

        Args:
            base_url: Nexus-LLM server URL.
            api_key: API key for authentication.
            timeout: Request timeout in seconds.
            default_model: Default model to use for requests.
        """
        self._default_model = default_model
        self._client = HttpClient(
            HttpClientConfig(
                base_url=base_url,
                api_key=api_key,
                timeout=timeout,
            )
        )
        logger.info("NexusSDK initialized: base_url=%s, model=%s", base_url, default_model)

    @property
    def default_model(self) -> str:
        """The default model used for requests."""
        return self._default_model

    @default_model.setter
    def default_model(self, value: str) -> None:
        self._default_model = value

    def chat(
        self,
        message: str,
        model: str | None = None,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Send a chat message and get a response.

        Args:
            message: The user message.
            model: Model name (uses default if not specified).
            system_prompt: Optional system prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            history: Optional conversation history.

        Returns:
            API response dictionary.
        """
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": message})

        return self._client.chat(
            messages=messages,
            model=model or self._default_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def complete(
        self,
        prompt: str,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> dict[str, Any]:
        """Generate a text completion.

        Args:
            prompt: Input prompt.
            model: Model name (uses default if not specified).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens.

        Returns:
            API response dictionary.
        """
        return self._client.complete(
            prompt=prompt,
            model=model or self._default_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def embed(
        self,
        texts: Any,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Generate embeddings for text.

        Args:
            texts: A string or list of strings to embed.
            model: Model name.

        Returns:
            API response with embedding vectors.
        """
        return self._client.embed(
            input_text=texts,
            model=model or self._default_model,
        )

    def models(self) -> dict[str, Any]:
        """List available models.

        Returns:
            Dictionary with model list.
        """
        return self._client.list_models()

    def model_info(self, model_id: str) -> dict[str, Any]:
        """Get information about a specific model.

        Args:
            model_id: Model identifier.

        Returns:
            Model details dictionary.
        """
        return self._client.get_model(model_id)

    def health(self) -> dict[str, Any]:
        """Check server health.

        Returns:
            Health status dictionary.
        """
        return self._client.health()

    def extract_text(self, response: dict[str, Any]) -> str:
        """Extract the assistant's text from a chat response.

        Args:
            response: API response dictionary.

        Returns:
            The assistant's message text, or empty string.
        """
        try:
            choices = response.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "")
        except (AttributeError, IndexError, TypeError):
            pass
        return ""

    def extract_embedding(self, response: dict[str, Any], index: int = 0) -> list[float]:
        """Extract an embedding vector from a response.

        Args:
            response: API response dictionary.
            index: Embedding index.

        Returns:
            The embedding vector.
        """
        try:
            data = response.get("data", [])
            if data and index < len(data):
                return data[index].get("embedding", [])
        except (AttributeError, IndexError, TypeError):
            pass
        return []