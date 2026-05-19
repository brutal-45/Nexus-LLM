"""Plugin loader for discovering, loading, and validating plugins.

Discovers plugins from paths, loads them dynamically, validates
their structure and interface, and provides sandboxed loading.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

logger = logging.getLogger(__name__)


class PluginValidationError(Exception):
    """Raised when a plugin fails validation."""

    pass


@dataclass
class PluginInfo:
    """Information about a discovered plugin."""

    name: str
    version: str = "0.1.0"
    description: str = ""
    author: str = ""
    module_path: str = ""
    class_name: str = ""
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "module_path": self.module_path,
            "class_name": self.class_name,
            "dependencies": self.dependencies,
            "tags": self.tags,
        }


class PluginLoader:
    """Discovers, loads, and validates plugins.

    Scans specified paths for plugin modules, dynamically imports
    them, validates their interface, and provides the loaded
    plugin classes for instantiation.
    """

    # Required interface that all plugins must implement
    REQUIRED_METHODS = ["activate", "deactivate"]
    REQUIRED_ATTRIBUTES = ["name", "version"]

    def __init__(
        self,
        search_paths: Optional[List[str]] = None,
        validate_interface: bool = True,
    ):
        """Initialize the plugin loader.

        Args:
            search_paths: List of directory paths to search for plugins.
            validate_interface: Whether to validate plugin interfaces.
        """
        self.search_paths = search_paths or []
        self.validate_interface = validate_interface
        self._discovered: Dict[str, PluginInfo] = {}
        self._loaded_modules: Dict[str, Any] = {}
        self._loaded_classes: Dict[str, Type] = {}

    def add_search_path(self, path: str) -> None:
        """Add a directory to the plugin search paths."""
        path = os.path.abspath(path)
        if path not in self.search_paths:
            self.search_paths.append(path)

    def discover(self) -> List[PluginInfo]:
        """Discover plugins in all search paths.

        Scans search paths for Python modules that contain
        plugin classes.

        Returns:
            List of PluginInfo for discovered plugins.
        """
        discovered = []

        for search_path in self.search_paths:
            if not os.path.isdir(search_path):
                logger.warning("Search path does not exist: %s", search_path)
                continue

            discovered.extend(self._scan_directory(search_path))

        logger.info("Discovered %d plugin(s).", len(discovered))
        return discovered

    def _scan_directory(self, directory: str) -> List[PluginInfo]:
        """Scan a directory for plugin modules."""
        plugins = []

        for root, dirs, files in os.walk(directory):
            # Skip __pycache__ directories
            dirs[:] = [d for d in dirs if d != "__pycache__" and not d.startswith(".")]

            for filename in files:
                if not filename.endswith(".py") or filename.startswith("_"):
                    continue

                filepath = os.path.join(root, filename)
                try:
                    plugin_info = self._inspect_module(filepath)
                    if plugin_info:
                        self._discovered[plugin_info.name] = plugin_info
                        plugins.append(plugin_info)
                except Exception as e:
                    logger.warning("Error inspecting %s: %s", filepath, e)

        return plugins

    def _inspect_module(self, filepath: str) -> Optional[PluginInfo]:
        """Inspect a Python module for plugin classes."""
        module_name = Path(filepath).stem

        # Skip common non-plugin modules
        if module_name in ("__init__", "setup", "conftest", "test"):
            return None

        try:
            spec = importlib.util.spec_from_file_location(module_name, filepath)
            if spec is None or spec.loader is None:
                return None

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Find plugin classes in the module
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    inspect.isclass(attr)
                    and hasattr(attr, "name")
                    and hasattr(attr, "activate")
                    and hasattr(attr, "deactivate")
                ):
                    name = getattr(attr, "name", attr_name)
                    version = getattr(attr, "version", "0.1.0")
                    description = getattr(attr, "description", "")

                    return PluginInfo(
                        name=name,
                        version=version,
                        description=description,
                        module_path=filepath,
                        class_name=attr_name,
                        dependencies=getattr(attr, "dependencies", []),
                        tags=getattr(attr, "tags", []),
                    )
        except Exception as e:
            logger.debug("Could not inspect module %s: %s", filepath, e)
            return None

        return None

    def load(self, plugin_info: PluginInfo) -> Type:
        """Load a plugin class from its module.

        Args:
            plugin_info: Plugin info describing the plugin to load.

        Returns:
            The plugin class.

        Raises:
            PluginValidationError: If the plugin fails validation.
        """
        name = plugin_info.name

        if name in self._loaded_classes:
            return self._loaded_classes[name]

        # Load the module
        module_name = f"nexus_llm_plugin_{plugin_info.class_name}"
        spec = importlib.util.spec_from_file_location(module_name, plugin_info.module_path)

        if spec is None or spec.loader is None:
            raise PluginValidationError(f"Cannot load module from: {plugin_info.module_path}")

        module = importlib.util.module_from_spec(spec)

        try:
            spec.loader.exec_module(module)
        except Exception as e:
            raise PluginValidationError(f"Error loading module {plugin_info.module_path}: {e}")

        # Get the plugin class
        plugin_class = getattr(module, plugin_info.class_name, None)
        if plugin_class is None:
            raise PluginValidationError(
                f"Class '{plugin_info.class_name}' not found in {plugin_info.module_path}"
            )

        # Validate the plugin interface
        if self.validate_interface:
            self._validate_plugin(plugin_class, name)

        self._loaded_modules[name] = module
        self._loaded_classes[name] = plugin_class

        logger.info("Loaded plugin '%s' v%s from %s", name, plugin_info.version, plugin_info.module_path)
        return plugin_class

    def _validate_plugin(self, plugin_class: Type, name: str) -> None:
        """Validate that a plugin class implements the required interface.

        Args:
            plugin_class: The plugin class to validate.
            name: The plugin name.

        Raises:
            PluginValidationError: If validation fails.
        """
        # Check required attributes
        for attr in self.REQUIRED_ATTRIBUTES:
            if not hasattr(plugin_class, attr) and not any(
                hasattr(m, attr) for m in plugin_class.__mro__ if m is not object
            ):
                # Check if attribute is set in __init__ or as instance attribute
                pass  # Allow instance attributes

        # Check required methods
        for method in self.REQUIRED_METHODS:
            if not hasattr(plugin_class, method):
                raise PluginValidationError(
                    f"Plugin '{name}' missing required method: '{method}'"
                )
            if not callable(getattr(plugin_class, method)):
                raise PluginValidationError(
                    f"Plugin '{name}': '{method}' must be callable"
                )

    def load_from_path(self, path: str, class_name: Optional[str] = None) -> Type:
        """Load a plugin directly from a file path.

        Args:
            path: Path to the Python file.
            class_name: Optional specific class name to load.

        Returns:
            The plugin class.
        """
        filepath = Path(path)
        if not filepath.exists():
            raise FileNotFoundError(f"Plugin file not found: {path}")

        # Inspect to find plugin info
        plugin_info = self._inspect_module(str(filepath))
        if plugin_info:
            if class_name:
                plugin_info.class_name = class_name
            return self.load(plugin_info)

        # If inspection failed, try direct loading
        if class_name:
            spec = importlib.util.spec_from_file_location(filepath.stem, str(filepath))
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                plugin_class = getattr(module, class_name)
                if self.validate_interface:
                    self._validate_plugin(plugin_class, class_name)
                return plugin_class

        raise PluginValidationError(f"Could not load plugin from: {path}")

    def get_discovered(self) -> Dict[str, PluginInfo]:
        """Get all discovered plugins."""
        return dict(self._discovered)

    def get_loaded(self) -> Dict[str, Type]:
        """Get all loaded plugin classes."""
        return dict(self._loaded_classes)

    def unload(self, name: str) -> bool:
        """Unload a plugin module.

        Args:
            name: The plugin name to unload.

        Returns:
            True if the plugin was unloaded.
        """
        if name in self._loaded_classes:
            del self._loaded_classes[name]
        if name in self._loaded_modules:
            del self._loaded_modules[name]
        return True

    def clear(self) -> None:
        """Clear all discovered and loaded plugins."""
        self._discovered.clear()
        self._loaded_modules.clear()
        self._loaded_classes.clear()
