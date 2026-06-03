"""Test plugin manager for Nexus-LLM."""
import pytest
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable


class PluginError(Exception):
    pass


@dataclass
class PluginInfo:
    name: str
    version: str = "1.0.0"
    description: str = ""
    author: str = ""
    dependencies: List[str] = field(default_factory=list)
    enabled: bool = True

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "enabled": self.enabled,
        }


class Plugin:
    def __init__(self, info: PluginInfo, initialize: Callable = None, teardown: Callable = None):
        self._info = info
        self._initialize = initialize
        self._teardown = teardown
        self._initialized = False

    @property
    def info(self):
        return self._info

    @property
    def is_initialized(self):
        return self._initialized

    def initialize(self):
        if self._initialized:
            return
        if self._initialize:
            self._initialize()
        self._initialized = True

    def teardown(self):
        if not self._initialized:
            return
        if self._teardown:
            self._teardown()
        self._initialized = False


class PluginManager:
    def __init__(self):
        self._plugins: Dict[str, Plugin] = {}

    def register(self, plugin: Plugin):
        if plugin.info.name in self._plugins:
            raise PluginError(f"Plugin '{plugin.info.name}' already registered")
        for dep in plugin.info.dependencies:
            if dep not in self._plugins:
                raise PluginError(f"Missing dependency: {dep}")
        self._plugins[plugin.info.name] = plugin

    def unregister(self, name: str):
        if name not in self._plugins:
            raise PluginError(f"Plugin '{name}' not found")
        # Check if any other plugin depends on this one
        for p in self._plugins.values():
            if name in p.info.dependencies:
                raise PluginError(f"Cannot unregister: '{p.info.name}' depends on '{name}'")
        plugin = self._plugins[name]
        if plugin.is_initialized:
            plugin.teardown()
        del self._plugins[name]

    def initialize(self, name: str):
        if name not in self._plugins:
            raise PluginError(f"Plugin '{name}' not found")
        plugin = self._plugins[name]
        if not plugin.info.enabled:
            raise PluginError(f"Plugin '{name}' is disabled")
        for dep in plugin.info.dependencies:
            dep_plugin = self._plugins.get(dep)
            if dep_plugin and not dep_plugin.is_initialized:
                self.initialize(dep)
        plugin.initialize()

    def initialize_all(self):
        for name in self._plugins:
            try:
                self.initialize(name)
            except PluginError:
                pass

    def teardown(self, name: str):
        if name not in self._plugins:
            raise PluginError(f"Plugin '{name}' not found")
        self._plugins[name].teardown()

    def get_plugin(self, name: str) -> Optional[Plugin]:
        return self._plugins.get(name)

    def list_plugins(self) -> List[PluginInfo]:
        return [p.info for p in self._plugins.values()]

    def enable_plugin(self, name: str):
        if name not in self._plugins:
            raise PluginError(f"Plugin '{name}' not found")
        self._plugins[name].info.enabled = True

    def disable_plugin(self, name: str):
        if name not in self._plugins:
            raise PluginError(f"Plugin '{name}' not found")
        self._plugins[name].info.enabled = False


class TestPluginInfo:
    def test_defaults(self):
        info = PluginInfo(name="test")
        assert info.version == "1.0.0"
        assert info.enabled is True

    def test_to_dict(self):
        info = PluginInfo(name="test", version="2.0", description="A plugin")
        d = info.to_dict()
        assert d["name"] == "test"
        assert d["version"] == "2.0"


class TestPlugin:
    def test_initialize(self):
        initialized = [False]
        def on_init():
            initialized[0] = True
        plugin = Plugin(PluginInfo(name="test"), initialize=on_init)
        plugin.initialize()
        assert plugin.is_initialized is True
        assert initialized[0] is True

    def test_initialize_idempotent(self):
        count = [0]
        def on_init():
            count[0] += 1
        plugin = Plugin(PluginInfo(name="test"), initialize=on_init)
        plugin.initialize()
        plugin.initialize()
        assert count[0] == 1

    def test_teardown(self):
        torn_down = [False]
        def on_teardown():
            torn_down[0] = True
        plugin = Plugin(PluginInfo(name="test"), teardown=on_teardown)
        plugin.initialize()
        plugin.teardown()
        assert plugin.is_initialized is False
        assert torn_down[0] is True


class TestPluginManager:
    def test_register(self):
        pm = PluginManager()
        pm.register(Plugin(PluginInfo(name="test")))
        assert pm.get_plugin("test") is not None

    def test_register_duplicate(self):
        pm = PluginManager()
        pm.register(Plugin(PluginInfo(name="test")))
        with pytest.raises(PluginError, match="already"):
            pm.register(Plugin(PluginInfo(name="test")))

    def test_register_with_missing_dependency(self):
        pm = PluginManager()
        with pytest.raises(PluginError, match="Missing dependency"):
            pm.register(Plugin(PluginInfo(name="test", dependencies=["nonexistent"])))

    def test_register_with_dependency(self):
        pm = PluginManager()
        pm.register(Plugin(PluginInfo(name="base")))
        pm.register(Plugin(PluginInfo(name="advanced", dependencies=["base"])))
        assert pm.get_plugin("advanced") is not None

    def test_unregister(self):
        pm = PluginManager()
        pm.register(Plugin(PluginInfo(name="test")))
        pm.unregister("test")
        assert pm.get_plugin("test") is None

    def test_unregister_with_dependent(self):
        pm = PluginManager()
        pm.register(Plugin(PluginInfo(name="base")))
        pm.register(Plugin(PluginInfo(name="adv", dependencies=["base"])))
        with pytest.raises(PluginError, match="depends on"):
            pm.unregister("base")

    def test_initialize(self):
        pm = PluginManager()
        pm.register(Plugin(PluginInfo(name="test")))
        pm.initialize("test")
        assert pm.get_plugin("test").is_initialized is True

    def test_initialize_disabled(self):
        pm = PluginManager()
        pm.register(Plugin(PluginInfo(name="test", enabled=False)))
        with pytest.raises(PluginError, match="disabled"):
            pm.initialize("test")

    def test_enable_disable(self):
        pm = PluginManager()
        pm.register(Plugin(PluginInfo(name="test")))
        pm.disable_plugin("test")
        assert pm.get_plugin("test").info.enabled is False
        pm.enable_plugin("test")
        assert pm.get_plugin("test").info.enabled is True

    def test_list_plugins(self):
        pm = PluginManager()
        pm.register(Plugin(PluginInfo(name="a")))
        pm.register(Plugin(PluginInfo(name="b")))
        assert len(pm.list_plugins()) == 2

    def test_initialize_all(self):
        pm = PluginManager()
        pm.register(Plugin(PluginInfo(name="a")))
        pm.register(Plugin(PluginInfo(name="b")))
        pm.initialize_all()
