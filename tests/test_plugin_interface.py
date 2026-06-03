"""Tests for plugin interface and manager."""
import pytest
from unittest.mock import MagicMock

from nexus_llm.plugins import PluginManager as PkgPluginManager
from nexus_llm.events import EventBus
from nexus_llm.exceptions import PluginError

# Import PluginInterface from the top-level plugins.py module
import importlib.util
import os
_spec = importlib.util.spec_from_file_location(
    "nexus_llm._plugins_standalone",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "nexus_llm", "plugins.py")
)
_plugins_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_plugins_mod)
PluginInterface = _plugins_mod.PluginInterface
AppPluginManager = _plugins_mod.PluginManager


class TestPluginInterface:
    """Test PluginInterface ABC."""

    def test_concrete_plugin(self):
        class MyPlugin(PluginInterface):
            name = "my_plugin"
            version = "1.0.0"

            def on_load(self):
                pass

            def on_unload(self):
                pass

        plugin = MyPlugin()
        assert plugin.name == "my_plugin"
        assert plugin.version == "1.0.0"
        assert plugin.is_enabled is False
        assert plugin.is_initialized is False

    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            PluginInterface()

    def test_plugin_lifecycle(self):
        class TestPlugin(PluginInterface):
            name = "test"
            version = "0.1"

            def on_load(self):
                pass

            def on_unload(self):
                pass

        plugin = TestPlugin()
        plugin.on_load()
        plugin._initialized = True
        plugin.on_enable()
        plugin._enabled = True
        assert plugin.is_enabled is True
        plugin.on_disable()
        plugin._enabled = False
        plugin.on_unload()

    def test_get_info(self):
        class InfoPlugin(PluginInterface):
            name = "info"
            version = "2.0"
            description = "A test plugin"

            def on_load(self):
                pass

            def on_unload(self):
                pass

        plugin = InfoPlugin()
        info = plugin.get_info()
        assert info["name"] == "info"
        assert info["version"] == "2.0"
        assert info["description"] == "A test plugin"
        assert info["enabled"] is False

    def test_emit_event(self):
        class EmitterPlugin(PluginInterface):
            name = "emitter"
            version = "1.0"

            def on_load(self):
                pass

            def on_unload(self):
                pass

        bus = EventBus()
        plugin = EmitterPlugin()
        plugin.set_event_bus(bus)
        plugin.emit_event("custom.event", data={"key": "val"})

    def test_on_config_update(self):
        class ConfigPlugin(PluginInterface):
            name = "config_test"
            version = "1.0"

            def on_load(self):
                pass

            def on_unload(self):
                pass

        plugin = ConfigPlugin()
        plugin.on_config_update({"setting": "value"})
        assert plugin.config["setting"] == "value"

    def test_repr(self):
        class RPlugin(PluginInterface):
            name = "repr_test"
            version = "3.0"

            def on_load(self):
                pass

            def on_unload(self):
                pass

        plugin = RPlugin()
        r = repr(plugin)
        assert "repr_test" in r
        assert "3.0" in r


class TestAppPluginManager:
    """Test the top-level PluginManager from nexus_llm.plugins."""

    def test_create_manager(self):
        manager = AppPluginManager()
        assert manager.plugin_count == 0

    def test_register_and_load_plugin(self):
        class DemoPlugin(PluginInterface):
            name = "demo"
            version = "1.0"

            def on_load(self):
                pass

            def on_unload(self):
                pass

        manager = AppPluginManager()
        manager.register_plugin(DemoPlugin)
        plugin = manager.load_plugin("demo")
        assert plugin.name == "demo"
        assert plugin.is_initialized is True
        assert manager.plugin_count == 1

    def test_load_nonexistent_raises(self):
        manager = AppPluginManager()
        with pytest.raises(PluginError, match="not found"):
            manager.load_plugin("nonexistent")

    def test_unload_plugin(self):
        class DemoPlugin(PluginInterface):
            name = "demo"
            version = "1.0"

            def on_load(self):
                pass

            def on_unload(self):
                pass

        manager = AppPluginManager()
        manager.register_plugin(DemoPlugin)
        manager.load_plugin("demo")
        manager.unload_plugin("demo")
        assert manager.plugin_count == 0

    def test_enable_plugin(self):
        class DemoPlugin(PluginInterface):
            name = "demo"
            version = "1.0"

            def on_load(self):
                pass

            def on_unload(self):
                pass

        manager = AppPluginManager()
        manager.register_plugin(DemoPlugin)
        manager.load_plugin("demo")
        manager.enable_plugin("demo")
        assert manager.enabled_count == 1

    def test_disable_plugin(self):
        class DemoPlugin(PluginInterface):
            name = "demo"
            version = "1.0"

            def on_load(self):
                pass

            def on_unload(self):
                pass

        manager = AppPluginManager()
        manager.register_plugin(DemoPlugin)
        manager.load_plugin("demo")
        manager.enable_plugin("demo")
        manager.disable_plugin("demo")
        assert manager.enabled_count == 0

    def test_list_plugins(self):
        class P1(PluginInterface):
            name = "p1"
            version = "1.0"
            def on_load(self): pass
            def on_unload(self): pass

        class P2(PluginInterface):
            name = "p2"
            version = "2.0"
            def on_load(self): pass
            def on_unload(self): pass

        manager = AppPluginManager()
        manager.register_plugin(P1)
        manager.register_plugin(P2)
        plugins = manager.list_plugins()
        assert len(plugins) == 2

    def test_repr(self):
        manager = AppPluginManager()
        r = repr(manager)
        assert "PluginManager" in r
