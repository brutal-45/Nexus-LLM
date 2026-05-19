"""Tests for nexus_llm.client.async_client module."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from nexus_llm.client.async_client import AsyncClient, AsyncClientConfig


class TestAsyncClientConfig:
    def test_default(self):
        config = AsyncClientConfig()
        assert config.base_url == "http://localhost:8000"

    def test_custom(self):
        config = AsyncClientConfig(base_url="https://api.example.com")
        assert config.base_url == "https://api.example.com"


class TestAsyncClient:
    def test_init(self):
        client = AsyncClient(AsyncClientConfig())
        assert client is not None

    @pytest.mark.asyncio
    async def test_chat_async(self):
        client = AsyncClient(AsyncClientConfig())
        with patch.object(client, "_async_request", new_callable=AsyncMock, return_value={"choices": []}):
            result = await client.chat(messages=[], model="test")
            assert "choices" in result

    @pytest.mark.asyncio
    async def test_complete_async(self):
        client = AsyncClient(AsyncClientConfig())
        with patch.object(client, "_async_request", new_callable=AsyncMock, return_value={"choices": []}):
            result = await client.complete(prompt="Hello", model="test")
            assert "choices" in result
