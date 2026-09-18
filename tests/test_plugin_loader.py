"""Test plugin loader for Nexus-LLM."""
import os
import sys
import importlib
import pytest
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from pathlib import Path


class PluginLoaderError(Exception):
    pass


@dataclass
class PluginMetadata:
    name: str
    module_path: str
    version: str = "1.0.0"
    description: str = ""
    author: str = ""


class PluginLoader:
    def __init__(self, plugin_dirs: List[str] = None):
        self._plugin_dirs = plugin_dirs or []
        self._loaded: Dict[str, Any] = {}

    def add_plugin_dir(self, path: str):
        if not os.path.isdir(path):
            raise PluginLoaderError(f"Plugin directory not found: {path}")
        self._plugin_dirs.append(path)

    def discover_plugins(self) -> List[str]:
        found = []
        for plugin_dir in self._plugin_dirs:
            if not os.path.isdir(plugin_dir):
                continue
            for item in os.listdir(plugin_dir):
                item_path = os.path.join(plugin_dir, item)
                if os.path.isdir(item_path) and os.path.isfile(os.path.join(item_path, "__init__.py")):
                    found.append(item)
                elif item.endswith(".py") and not item.startswith("_"):
                    found.append(item[:-3])
        return found

    def load_plugin(self, name: str) -> Any:
        if name in self._loaded:
            return self._loaded[name]

        for plugin_dir in self._plugin_dirs:
            module_path = os.path.join(plugin_dir, name)
            if os.path.isfile(module_path + ".py"):
                full_path = module_path + ".py"
                spec = importlib.util.spec_from_file_location(name, full_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    self._loaded[name] = module
                    return module

        raise PluginLoaderError(f"Plugin '{name}' not found in plugin directories")

    def unload_plugin(self, name: str):
        if name not in self._loaded:
            raise PluginLoaderError(f"Plugin '{name}' not loaded")
        del self._loaded[name]

    def is_loaded(self, name: str) -> bool:
        return name in self._loaded

    def get_loaded_plugins(self) -> List[str]:
        return list(self._loaded.keys())

    def validate_plugin(self, name: str) -> bool:
        required_attrs = ["name", "version"]
        if name not in self._loaded:
            return False
        module = self._loaded[name]
        for attr in required_attrs:
            if not hasattr(module, attr):
                return False
        return True


class TestPluginLoader:
    def test_init(self):
        loader = PluginLoader()
        assert loader.get_loaded_plugins() == []

    def test_add_plugin_dir(self, tmp_dir):
        loader = PluginLoader()
        loader.add_plugin_dir(str(tmp_dir))
        assert str(tmp_dir) in loader._plugin_dirs

    def test_add_invalid_dir(self):
        loader = PluginLoader()
        with pytest.raises(PluginLoaderError, match="not found"):
            loader.add_plugin_dir("/nonexistent/path")

    def test_discover_plugins(self, tmp_dir):
        (tmp_dir / "plugin1.py").write_text("# plugin 1")
        (tmp_dir / "plugin2.py").write_text("# plugin 2")
        (tmp_dir / "_private.py").write_text("# private")
        loader = PluginLoader([str(tmp_dir)])
        found = loader.discover_plugins()
        assert "plugin1" in found
        assert "plugin2" in found
        assert "_private" not in found

    def test_discover_package_plugins(self, tmp_dir):
        pkg_dir = tmp_dir / "my_plugin"
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").write_text("name='my_plugin'\nversion='1.0'")
        loader = PluginLoader([str(tmp_dir)])
        found = loader.discover_plugins()
        assert "my_plugin" in found

    def test_load_plugin(self, tmp_dir):
        plugin_file = tmp_dir / "hello_plugin.py"
        plugin_file.write_text("name = 'hello'\nversion = '1.0'\ndef run(): return 'hello world'")
        loader = PluginLoader([str(tmp_dir)])
        module = loader.load_plugin("hello_plugin")
        assert module.name == "hello"
        assert module.run() == "hello world"

    def test_load_nonexistent(self):
        loader = PluginLoader(["/tmp"])
        with pytest.raises(PluginLoaderError, match="not found"):
            loader.load_plugin("nonexistent")

    def test_unload_plugin(self, tmp_dir):
        (tmp_dir / "test_plug.py").write_text("name='test'")
        loader = PluginLoader([str(tmp_dir)])
        loader.load_plugin("test_plug")
        loader.unload_plugin("test_plug")
        assert loader.is_loaded("test_plug") is False

    def test_unload_not_loaded(self):
        loader = PluginLoader()
        with pytest.raises(PluginLoaderError, match="not loaded"):
            loader.unload_plugin("nonexistent")

    def test_is_loaded(self, tmp_dir):
        (tmp_dir / "test_plug.py").write_text("name='test'")
        loader = PluginLoader([str(tmp_dir)])
        assert loader.is_loaded("test_plug") is False
        loader.load_plugin("test_plug")
        assert loader.is_loaded("test_plug") is True

    def test_validate_plugin(self, tmp_dir):
        (tmp_dir / "valid.py").write_text("name='valid'\nversion='1.0'")
        (tmp_dir / "invalid.py").write_text("# missing name and version")
        loader = PluginLoader([str(tmp_dir)])
        loader.load_plugin("valid")
        assert loader.validate_plugin("valid") is True
        loader.load_plugin("invalid")
        assert loader.validate_plugin("invalid") is False
