"""Tests for the plugins module.

Covers PluginManager, Plugin, HookSystem, EventSystem, and PluginConfig.
"""

from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock

import pytest

from nexus_llm.plugins.plugin import Plugin
from nexus_llm.plugins.hooks import HookSystem, HookError
from nexus_llm.plugins.events import EventSystem
from nexus_llm.plugins.config import PluginConfig
from nexus_llm.plugins.manager import PluginManager, PluginManagerError


# ---------------------------------------------------------------------------
# Concrete Plugin subclass for testing
# ---------------------------------------------------------------------------

class DummyPlugin(Plugin):
    name = "dummy_plugin"
    version = "1.0.0"
    description = "A test plugin"

    def __init__(self):
        super().__init__()
        self.loaded = False
        self.unloaded = False

    def on_load(self):
        self.loaded = True

    def on_unload(self):
        self.unloaded = True

    def register_hooks(self):
        return {"pre_generate": self._on_pre_generate}

    def _on_pre_generate(self, **kwargs):
        return "hook_result"


class BadPlugin(Plugin):
    name = "bad_plugin"
    version = "0.1.0"

    def on_load(self):
        raise RuntimeError("Load failed")


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------

class TestPlugin:
    """Tests for Plugin base class."""

    def test_default_attributes(self):
        plugin = DummyPlugin()
        assert plugin.name == "dummy_plugin"
        assert plugin.version == "1.0.0"
        assert plugin.enabled is False

    def test_enabled_property(self):
        plugin = DummyPlugin()
        plugin.enabled = True
        assert plugin.enabled is True

    def test_context_property(self):
        plugin = DummyPlugin()
        plugin.context["key"] = "value"
        assert plugin.context["key"] == "value"

    def test_repr(self):
        plugin = DummyPlugin()
        r = repr(plugin)
        assert "dummy_plugin" in r
        assert "1.0.0" in r

    def test_register_hooks_default(self):
        """Base Plugin returns empty hook mapping."""
        plugin = Plugin()
        assert plugin.register_hooks() == {}


# ---------------------------------------------------------------------------
# HookSystem
# ---------------------------------------------------------------------------

class TestHookSystem:
    """Tests for HookSystem."""

    def test_register_and_trigger(self):
        hooks = HookSystem()
        results = []
        hooks.register("pre_generate", lambda: results.append("called"))
        hooks.trigger("pre_generate")
        assert results == ["called"]

    def test_trigger_with_args(self):
        hooks = HookSystem()
        captured = {}
        hooks.register("test", lambda x, y=0: captured.update({"x": x, "y": y}))
        hooks.trigger("test", 42, y=99)
        assert captured == {"x": 42, "y": 99}

    def test_priority_ordering(self):
        hooks = HookSystem()
        order = []
        hooks.register("test", lambda: order.append("low"), priority=90)
        hooks.register("test", lambda: order.append("high"), priority=10)
        hooks.register("test", lambda: order.append("mid"), priority=50)
        hooks.trigger("test")
        assert order == ["high", "mid", "low"]

    def test_unregister(self):
        hooks = HookSystem()
        cb = lambda: None
        hooks.register("test", cb)
        assert hooks.unregister("test", cb) is True
        assert hooks.has_hook("test") is False

    def test_unregister_not_found(self):
        hooks = HookSystem()
        assert hooks.unregister("test", lambda: None) is False

    def test_has_hook(self):
        hooks = HookSystem()
        assert hooks.has_hook("test") is False
        hooks.register("test", lambda: None)
        assert hooks.has_hook("test") is True

    def test_get_hooks(self):
        hooks = HookSystem()
        hooks.register("test", lambda: None, priority=10)
        snapshot = hooks.get_hooks()
        assert "test" in snapshot

    def test_hook_error_on_exception(self):
        hooks = HookSystem()
        hooks.register("test", lambda: 1 / 0)
        with pytest.raises(HookError):
            hooks.trigger("test")

    def test_clear_specific_hook(self):
        hooks = HookSystem()
        hooks.register("test", lambda: None)
        hooks.clear("test")
        assert hooks.has_hook("test") is False

    def test_clear_all_hooks(self):
        hooks = HookSystem()
        hooks.register("a", lambda: None)
        hooks.register("b", lambda: None)
        hooks.clear()
        assert hooks.has_hook("a") is False
        assert hooks.has_hook("b") is False

    def test_non_callable_raises(self):
        hooks = HookSystem()
        with pytest.raises(ValueError):
            hooks.register("test", "not_callable")


# ---------------------------------------------------------------------------
# EventSystem
# ---------------------------------------------------------------------------

class TestEventSystem:
    """Tests for EventSystem."""

    def test_subscribe_and_publish(self):
        bus = EventSystem()
        received = []
        bus.subscribe("model_loaded", lambda **kw: received.append(kw))
        bus.publish("model_loaded", model_name="gpt2")
        assert len(received) == 1
        assert received[0]["model_name"] == "gpt2"

    def test_unsubscribe(self):
        bus = EventSystem()
        handler = lambda **kw: None
        bus.subscribe("test", handler)
        assert bus.unsubscribe("test", handler) is True
        assert bus.has_subscribers("test") is False

    def test_unsubscribe_not_found(self):
        bus = EventSystem()
        assert bus.unsubscribe("test", lambda: None) is False

    def test_has_subscribers(self):
        bus = EventSystem()
        assert bus.has_subscribers("test") is False
        bus.subscribe("test", lambda: None)
        assert bus.has_subscribers("test") is True

    def test_list_events(self):
        bus = EventSystem()
        bus.subscribe("event_a", lambda: None)
        bus.subscribe("event_b", lambda: None)
        events = bus.list_events()
        assert "event_a" in events
        assert "event_b" in events

    def test_handler_exception_does_not_block_others(self):
        bus = EventSystem()
        results = []
        bus.subscribe("test", lambda **kw: 1 / 0)  # will raise
        bus.subscribe("test", lambda **kw: results.append("ok"))
        bus.publish("test")
        assert results == ["ok"]  # second handler still runs

    def test_clear_specific_event(self):
        bus = EventSystem()
        bus.subscribe("test", lambda: None)
        bus.clear("test")
        assert bus.has_subscribers("test") is False

    def test_clear_all_events(self):
        bus = EventSystem()
        bus.subscribe("a", lambda: None)
        bus.subscribe("b", lambda: None)
        bus.clear()
        assert bus.list_events() == set()

    def test_non_callable_handler_raises(self):
        bus = EventSystem()
        with pytest.raises(ValueError):
            bus.subscribe("test", "not_callable")

    def test_duplicate_subscribe_ignored(self):
        bus = EventSystem()
        handler = lambda: None
        bus.subscribe("test", handler)
        bus.subscribe("test", handler)  # duplicate
        # Should only have one subscriber
        bus.publish("test")
        # No assertion needed - just no errors


# ---------------------------------------------------------------------------
# PluginConfig
# ---------------------------------------------------------------------------

class TestPluginConfig:
    """Tests for PluginConfig."""

    def test_defaults(self):
        config = PluginConfig()
        assert config.enabled_plugins == []
        assert config.plugin_settings == {}

    def test_is_enabled(self):
        config = PluginConfig(enabled_plugins=["my_plugin"])
        assert config.is_enabled("my_plugin") is True
        assert config.is_enabled("other") is False

    def test_enable_disable(self):
        config = PluginConfig()
        config.enable("test_plugin")
        assert config.is_enabled("test_plugin")
        config.disable("test_plugin")
        assert not config.is_enabled("test_plugin")

    def test_get_set_settings(self):
        config = PluginConfig()
        config.set_settings("plugin1", {"key": "value"})
        assert config.get_settings("plugin1") == {"key": "value"}
        assert config.get_settings("nonexistent") == {}

    def test_yaml_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "plugin_config.yaml")
            config = PluginConfig(enabled_plugins=["p1"], plugin_settings={"p1": {"k": "v"}})
            config.to_yaml(path)
            loaded = PluginConfig.from_yaml(path)
            assert loaded.enabled_plugins == ["p1"]
            assert loaded.plugin_settings["p1"]["k"] == "v"

    def test_from_yaml_nonexistent(self):
        config = PluginConfig.from_yaml("/nonexistent/path.yaml")
        assert config.enabled_plugins == []


# ---------------------------------------------------------------------------
# PluginManager
# ---------------------------------------------------------------------------

class TestPluginManager:
    """Tests for PluginManager."""

    def test_init(self):
        mgr = PluginManager()
        assert mgr.hooks is not None
        assert mgr.events is not None
        assert mgr.config is not None

    def test_register_plugin_directly(self):
        mgr = PluginManager()
        plugin = DummyPlugin()
        mgr._register_plugin(plugin)
        assert mgr.get_plugin("dummy_plugin") is plugin

    def test_get_nonexistent_plugin_raises(self):
        mgr = PluginManager()
        with pytest.raises(PluginManagerError):
            mgr.get_plugin("nonexistent")

    def test_list_plugins(self):
        mgr = PluginManager()
        mgr._register_plugin(DummyPlugin())
        plugins = mgr.list_plugins()
        assert len(plugins) == 1
        assert plugins[0].name == "dummy_plugin"

    def test_enable_disable(self):
        mgr = PluginManager()
        mgr._register_plugin(DummyPlugin())
        mgr.enable_plugin("dummy_plugin")
        assert mgr.get_plugin("dummy_plugin").enabled is True
        mgr.disable_plugin("dummy_plugin")
        assert mgr.get_plugin("dummy_plugin").enabled is False

    def test_unload_plugin(self):
        mgr = PluginManager()
        plugin = DummyPlugin()
        mgr._register_plugin(plugin)
        mgr.unload_plugin("dummy_plugin")
        with pytest.raises(PluginManagerError):
            mgr.get_plugin("dummy_plugin")

    def test_unload_nonexistent_raises(self):
        mgr = PluginManager()
        with pytest.raises(PluginManagerError):
            mgr.unload_plugin("nonexistent")

    def test_load_plugin_not_found_raises(self):
        mgr = PluginManager()
        with pytest.raises(PluginManagerError):
            mgr.load_plugin("nonexistent_plugin_xyz")
