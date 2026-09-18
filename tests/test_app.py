"""Tests for Application class."""
import pytest
from unittest.mock import MagicMock, patch

from nexus_llm.__version__ import __version__


class TestNexusLLMAppInit:
    """Test NexusLLMApp initialization."""

    def test_default_init(self):
        from nexus_llm.app import NexusLLMApp
        with patch("nexus_llm.app.PluginManager"):
            app = NexusLLMApp()
        assert app._config_path is None
        assert app._verbose is False

    def test_init_with_verbose(self):
        from nexus_llm.app import NexusLLMApp
        with patch("nexus_llm.app.PluginManager"):
            app = NexusLLMApp(verbose=True)
        assert app._verbose is True

    def test_init_with_log_level(self):
        from nexus_llm.app import NexusLLMApp
        with patch("nexus_llm.app.PluginManager"):
            app = NexusLLMApp(log_level="DEBUG")
        assert app._log_level == "DEBUG"

    def test_core_components_initialized(self):
        from nexus_llm.app import NexusLLMApp
        with patch("nexus_llm.app.PluginManager"):
            app = NexusLLMApp()
        assert app._event_bus is not None
        assert app._registry is not None
        assert app._state_manager is not None
        assert app._signal_handler is not None


class TestNexusLLMAppDevice:
    """Test device resolution."""

    def test_explicit_device(self):
        from nexus_llm.app import NexusLLMApp
        with patch("nexus_llm.app.PluginManager"):
            app = NexusLLMApp()
        assert app._get_device("cpu") == "cpu"
        assert app._get_device("cuda") == "cuda"

    def test_auto_device_resolves(self):
        from nexus_llm.app import NexusLLMApp
        with patch("nexus_llm.app.PluginManager"):
            app = NexusLLMApp()
        device = app._get_device("auto")
        assert device in ("cpu", "cuda", "mps")


class TestNexusLLMAppVersion:
    """Test version access."""

    def test_version_available(self):
        assert __version__ is not None
        assert isinstance(__version__, str)


class TestNexusLLMAppConfig:
    """Test configuration loading."""

    def test_nonexistent_config_does_not_crash(self):
        from nexus_llm.app import NexusLLMApp
        with patch("nexus_llm.app.PluginManager"):
            app = NexusLLMApp(config_path="/nonexistent/config.yaml")

    def test_config_dict_default(self):
        from nexus_llm.app import NexusLLMApp
        with patch("nexus_llm.app.PluginManager"):
            app = NexusLLMApp()
        assert isinstance(app._config, dict)
