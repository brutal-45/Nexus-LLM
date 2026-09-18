"""Tests for API integration."""
import pytest

from nexus_llm.api import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    AuthManager,
    KeyStore,
    RateLimiter,
    TokenBucket,
    setup_middleware,
    CORSConfig,
    ConnectionManager,
    WebSocketMessageHandler,
)


class TestAPIModuleImports:
    """Test that all API module components can be imported."""

    def test_schema_imports(self):
        assert ChatMessage is not None
        assert ChatRequest is not None
        assert ChatResponse is not None
        assert GenerateRequest is not None
        assert GenerateResponse is not None
        assert HealthResponse is not None

    def test_auth_imports(self):
        assert AuthManager is not None
        assert KeyStore is not None

    def test_rate_limit_imports(self):
        assert RateLimiter is not None
        assert TokenBucket is not None

    def test_middleware_import(self):
        assert setup_middleware is not None

    def test_cors_import(self):
        assert CORSConfig is not None

    def test_websocket_import(self):
        assert ConnectionManager is not None


class TestModelServer:
    """The package exposes the server as nexus_llm.serving.ModelServer."""

    def test_server_starts_without_an_app(self):
        from nexus_llm.serving.server import ModelServer

        server = ModelServer()
        assert server.app is None

    def test_server_status_starts_stopped(self):
        from nexus_llm.serving.server import ModelServer

        assert ModelServer()._status == "stopped"


class TestSchemasIntegration:
    """Test API schema classes."""

    def test_chat_message(self):
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
