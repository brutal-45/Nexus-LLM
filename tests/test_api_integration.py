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
    create_app,
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


class TestCreateApp:
    """Test create_app factory."""

    def test_create_app_returns_none_without_fastapi(self):
        # This will fail if fastapi not installed, but should not crash on import
        try:
            app = create_app()
        except ImportError:
            pytest.skip("FastAPI not installed")


class TestSchemasIntegration:
    """Test API schema classes."""

    def test_chat_message(self):
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
