"""Tests for nexus_llm.client.ws_client module."""

import pytest
from unittest.mock import MagicMock, patch
from nexus_llm.client.ws_client import WebSocketClient, WebSocketConfig


class TestWebSocketConfig:
    def test_default(self):
        config = WebSocketConfig()
        assert config.url == "ws://localhost:8000/ws"

    def test_custom(self):
        config = WebSocketConfig(url="wss://api.example.com/ws")
        assert config.url == "wss://api.example.com/ws"


class TestWebSocketClient:
    def test_init(self):
        client = WebSocketClient(WebSocketConfig())
        assert client is not None

    def test_connect(self):
        client = WebSocketClient(WebSocketConfig())
        with patch.object(client, "_connect_internal", return_value=True):
            result = client.connect()
            assert result is True

    def test_disconnect(self):
        client = WebSocketClient(WebSocketConfig())
        client.disconnect()
        # Should not raise

    def test_send(self):
        client = WebSocketClient(WebSocketConfig())
        with patch.object(client, "_send_internal", return_value=True):
            result = client.send("hello")
            assert result is True

    def test_is_connected(self):
        client = WebSocketClient(WebSocketConfig())
        assert client.is_connected is False
