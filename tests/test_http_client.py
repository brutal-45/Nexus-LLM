"""Tests for nexus_llm.client.http_client module."""

import pytest
from unittest.mock import MagicMock, patch
from nexus_llm.client.http_client import HttpClient, HttpClientConfig


class TestHttpClientConfig:
    """Tests for HttpClientConfig."""

    def test_default(self):
        config = HttpClientConfig()
        assert config.base_url == "http://localhost:8000"
        assert config.timeout == 60

    def test_custom(self):
        config = HttpClientConfig(base_url="https://api.example.com", timeout=30)
        assert config.base_url == "https://api.example.com"
        assert config.timeout == 30


class TestHttpClient:
    """Tests for the HttpClient class."""

    def test_init(self):
        client = HttpClient(HttpClientConfig())
        assert client is not None

    def test_chat(self):
        client = HttpClient(HttpClientConfig())
        with patch.object(client, "_request", return_value={"choices": []}):
            result = client.chat(messages=[], model="test")
            assert "choices" in result

    def test_complete(self):
        client = HttpClient(HttpClientConfig())
        with patch.object(client, "_request", return_value={"choices": []}):
            result = client.complete(prompt="Hello", model="test")
            assert "choices" in result

    def test_embed(self):
        client = HttpClient(HttpClientConfig())
        with patch.object(client, "_request", return_value={"data": []}):
            result = client.embed(input_text="Hello", model="test")
            assert "data" in result

    def test_list_models(self):
        client = HttpClient(HttpClientConfig())
        with patch.object(client, "_request", return_value={"data": []}):
            result = client.list_models()
            assert "data" in result

    def test_health(self):
        client = HttpClient(HttpClientConfig())
        with patch.object(client, "_request", return_value={"status": "ok"}):
            result = client.health()
            assert result["status"] == "ok"
