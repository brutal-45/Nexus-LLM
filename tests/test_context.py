"""Tests for application context."""
import pytest
from unittest.mock import patch

from nexus_llm.context import ApplicationContext, app_context


class TestApplicationContextInit:
    """Test ApplicationContext initialization."""

    @pytest.fixture(autouse=True)
    def reset_context(self):
        ApplicationContext.reset()
        yield
        ApplicationContext.reset()

    def test_create_context(self):
        ctx = ApplicationContext()
        assert ctx.is_initialized is False
        assert ctx.is_shutting_down is False

    def test_singleton(self):
        c1 = ApplicationContext.get_instance()
        c2 = ApplicationContext.get_instance()
        assert c1 is c2

    def test_reset(self):
        c1 = ApplicationContext.get_instance()
        ApplicationContext.reset()
        c2 = ApplicationContext.get_instance()
        assert c1 is not c2


class TestApplicationContextLifecycle:
    """Test lifecycle management."""

    @pytest.fixture(autouse=True)
    def reset_context(self):
        ApplicationContext.reset()
        yield
        ApplicationContext.reset()

    def test_initialize(self):
        with patch("nexus_llm.context.PluginManager"):
            ctx = ApplicationContext()
            ctx.initialize()
        assert ctx.is_initialized is True

    def test_double_initialize_warning(self):
        with patch("nexus_llm.context.PluginManager"):
            ctx = ApplicationContext()
            ctx.initialize()
            ctx.initialize()  # Should log warning, not crash
        assert ctx.is_initialized is True

    def test_shutdown(self):
        with patch("nexus_llm.context.PluginManager"):
            ctx = ApplicationContext()
            ctx.initialize()
            ctx.shutdown()
        assert ctx.is_initialized is False
        assert ctx.is_shutting_down is True

    def test_double_shutdown(self):
        with patch("nexus_llm.context.PluginManager"):
            ctx = ApplicationContext()
            ctx.initialize()
            ctx.shutdown()
            ctx.shutdown()  # Should not crash


class TestApplicationContextProperties:
    """Test lazy property access."""

    @pytest.fixture(autouse=True)
    def reset_context(self):
        ApplicationContext.reset()
        yield
        ApplicationContext.reset()

    def test_event_bus_lazy(self):
        ctx = ApplicationContext()
        bus = ctx.event_bus
        assert bus is not None

    def test_registry_lazy(self):
        ctx = ApplicationContext()
        reg = ctx.registry
        assert reg is not None

    def test_state_manager_lazy(self):
        ctx = ApplicationContext()
        sm = ctx.state_manager
        assert sm is not None

    def test_config_property(self):
        ctx = ApplicationContext()
        config = ctx.config
        assert isinstance(config, dict)


class TestApplicationContextModelManagement:
    """Test model registration and retrieval."""

    @pytest.fixture(autouse=True)
    def reset_context(self):
        ApplicationContext.reset()
        yield
        ApplicationContext.reset()

    def test_register_and_get_model(self):
        ctx = ApplicationContext()
        mock_model = object()
        mock_tokenizer = object()
        ctx.register_model("test_model", mock_model, mock_tokenizer)
        assert ctx.get_model("test_model") is mock_model
        assert ctx.get_tokenizer("test_model") is mock_tokenizer

    def test_get_nonexistent_model_raises(self):
        ctx = ApplicationContext()
        with pytest.raises(KeyError, match="not found"):
            ctx.get_model("nonexistent")

    def test_get_nonexistent_tokenizer_raises(self):
        ctx = ApplicationContext()
        with pytest.raises(KeyError, match="not found"):
            ctx.get_tokenizer("nonexistent")

    def test_list_models(self):
        ctx = ApplicationContext()
        ctx.register_model("m1", object())
        ctx.register_model("m2", object())
        models = ctx.list_models()
        assert "m1" in models
        assert "m2" in models


class TestApplicationContextManager:
    """Test context manager usage."""

    @pytest.fixture(autouse=True)
    def reset_context(self):
        ApplicationContext.reset()
        yield
        ApplicationContext.reset()

    def test_with_statement(self):
        with patch("nexus_llm.context.PluginManager"):
            with ApplicationContext() as ctx:
                assert ctx.is_initialized is True

    def test_app_context_function(self):
        with patch("nexus_llm.context.PluginManager"):
            with app_context() as ctx:
                assert ctx.is_initialized is True

    def test_repr(self):
        ctx = ApplicationContext()
        r = repr(ctx)
        assert "ApplicationContext" in r
        assert "initialized=False" in r
