"""Tests for nexus_llm.nexus.core module."""

import pytest
from unittest.mock import MagicMock, patch
from nexus_llm.nexus.core import NexusCore, create_nexus, get_nexus_instance, shutdown_nexus
from nexus_llm.exceptions import NexusLLMError, ConfigError


class TestNexusCore:
    """Tests for the NexusCore class."""

    def test_init_default(self):
        core = NexusCore()
        assert core.config == {}
        assert core.is_running is False

    def test_init_with_config(self):
        config = {"key": "value", "nested": {"a": 1}}
        core = NexusCore(config=config)
        assert core.config == config

    def test_config_is_copy(self):
        core = NexusCore(config={"a": 1})
        cfg = core.config
        cfg["a"] = 2
        assert core.config["a"] == 1

    def test_start(self):
        core = NexusCore()
        core.start()
        assert core.is_running is True

    def test_start_already_running(self):
        core = NexusCore()
        core.start()
        with pytest.raises(NexusLLMError):
            core.start()

    def test_stop(self):
        core = NexusCore()
        core.start()
        core.stop()
        assert core.is_running is False

    def test_stop_when_not_running(self):
        core = NexusCore()
        core.stop()  # Should not raise
        assert core.is_running is False

    def test_register_component(self):
        core = NexusCore()
        comp = MagicMock()
        core.register_component("engine", comp)
        assert core.get_component("engine") is comp

    def test_register_duplicate_component(self):
        core = NexusCore()
        core.register_component("engine", MagicMock())
        with pytest.raises(ConfigError):
            core.register_component("engine", MagicMock())

    def test_unregister_component(self):
        core = NexusCore()
        comp = MagicMock()
        core.register_component("engine", comp)
        result = core.unregister_component("engine")
        assert result is comp
        assert core.get_component("engine") is None

    def test_unregister_nonexistent(self):
        core = NexusCore()
        result = core.unregister_component("missing")
        assert result is None

    def test_get_component_nonexistent(self):
        core = NexusCore()
        assert core.get_component("missing") is None

    def test_start_starts_components(self):
        core = NexusCore()
        comp = MagicMock()
        core.register_component("engine", comp)
        core.start()
        comp.start.assert_called_once()

    def test_stop_stops_components_in_reverse(self):
        core = NexusCore()
        comp1 = MagicMock()
        comp2 = MagicMock()
        core.register_component("a", comp1)
        core.register_component("b", comp2)
        core.start()
        core.stop()
        # Reverse order: b then a
        assert comp2.stop.call_count == 1
        assert comp1.stop.call_count == 1

    def test_stop_handles_component_error(self):
        core = NexusCore()
        comp = MagicMock()
        comp.stop.side_effect = RuntimeError("fail")
        core.register_component("engine", comp)
        core.start()
        core.stop()  # Should not raise
        assert core.is_running is False

    def test_health_check_stopped(self):
        core = NexusCore()
        result = core.health_check()
        assert result["nexus_core"] == "stopped"
        assert result["components"] == {}

    def test_health_check_running(self):
        core = NexusCore()
        core.start()
        result = core.health_check()
        assert result["nexus_core"] == "healthy"

    def test_health_check_with_components(self):
        core = NexusCore()
        comp = MagicMock()
        comp.health_check.return_value = {"status": "ok"}
        core.register_component("engine", comp)
        core.start()
        result = core.health_check()
        assert result["components"]["engine"]["status"] == "ok"

    def test_health_check_component_error(self):
        core = NexusCore()
        comp = MagicMock()
        comp.health_check.side_effect = RuntimeError("fail")
        core.register_component("engine", comp)
        core.start()
        result = core.health_check()
        assert result["components"]["engine"]["status"] == "error"

    def test_update_config(self):
        core = NexusCore()
        core.update_config({"new_key": "new_value"})
        assert core.config["new_key"] == "new_value"

    def test_context_manager(self):
        with NexusCore() as core:
            assert core.is_running is True
        assert core.is_running is False


class TestCreateNexus:
    """Tests for module-level nexus functions."""

    def setup_method(self):
        shutdown_nexus()

    def teardown_method(self):
        shutdown_nexus()

    def test_create_nexus(self):
        nexus = create_nexus()
        assert isinstance(nexus, NexusCore)

    def test_create_nexus_with_config(self):
        nexus = create_nexus(config={"test": True})
        assert nexus.config == {"test": True}

    def test_get_nexus_instance(self):
        nexus = create_nexus()
        assert get_nexus_instance() is nexus

    def test_get_nexus_instance_none(self):
        assert get_nexus_instance() is None

    def test_shutdown_nexus(self):
        nexus = create_nexus()
        nexus.start()
        shutdown_nexus()
        assert get_nexus_instance() is None
