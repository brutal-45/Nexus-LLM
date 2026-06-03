"""Tests for signal handling."""
import signal
import threading

import pytest

from nexus_llm.signals import SignalHandler, GracefulContextManager


class TestSignalHandler:
    """Test SignalHandler."""

    def test_create_handler(self):
        handler = SignalHandler()
        assert handler.is_shutting_down is False

    def test_register_shutdown_callback(self):
        handler = SignalHandler()
        called = []
        handler.register_shutdown_callback(lambda: called.append(True))
        assert len(handler._shutdown_callbacks) == 1

    def test_register_pre_shutdown_callback(self):
        handler = SignalHandler()
        called = []
        handler.register_pre_shutdown_callback(lambda: called.append(True))
        assert len(handler._pre_shutdown_callbacks) == 1

    def test_install_and_uninstall(self):
        handler = SignalHandler()
        handler.install()
        assert handler._installed is True
        handler.uninstall()
        assert handler._installed is False

    def test_double_install_warning(self):
        handler = SignalHandler()
        handler.install()
        handler.install()  # Should log warning
        handler.uninstall()

    def test_uninstall_without_install(self):
        handler = SignalHandler()
        handler.uninstall()  # Should not crash

    def test_trigger_shutdown(self):
        handler = SignalHandler()
        called = []
        handler.register_shutdown_callback(lambda: called.append("shutdown"))
        handler.trigger_shutdown("test")
        assert handler.is_shutting_down is True
        assert "shutdown" in called

    def test_double_trigger_shutdown(self):
        handler = SignalHandler()
        called = []
        handler.register_shutdown_callback(lambda: called.append(1))
        handler.trigger_shutdown("first")
        handler.trigger_shutdown("second")  # Should be no-op
        assert len(called) == 1

    def test_shutdown_callback_priority(self):
        handler = SignalHandler()
        order = []

        def low_cb():
            order.append("low")

        def high_cb():
            order.append("high")

        handler.register_shutdown_callback(low_cb, priority=0)
        handler.register_shutdown_callback(high_cb, priority=10)
        handler.trigger_shutdown("test")
        assert order == ["high", "low"]

    def test_wait_for_shutdown_timeout(self):
        handler = SignalHandler()
        result = handler.wait_for_shutdown(timeout=0.01)
        assert result is False

    def test_repr(self):
        handler = SignalHandler()
        r = repr(handler)
        assert "SignalHandler" in r
        assert "installed=False" in r


class TestGracefulContextManager:
    """Test GracefulContextManager."""

    def test_context_manager(self):
        with GracefulContextManager() as handler:
            assert isinstance(handler, SignalHandler)
            assert handler._installed is True
        assert handler._installed is False

    def test_handler_property(self):
        gcm = GracefulContextManager()
        assert isinstance(gcm.handler, SignalHandler)

    def test_custom_timeout(self):
        gcm = GracefulContextManager(timeout=60.0)
        assert gcm.handler._graceful_timeout == 60.0
