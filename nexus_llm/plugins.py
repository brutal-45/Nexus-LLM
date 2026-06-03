"""Nexus-LLM Plugin System Module.

Provides the plugin interface and manager for extending Nexus-LLM
functionality through dynamically loaded plugins. Supports plugin
discovery, lifecycle management, and dependency resolution.
"""

import importlib
import inspect
import logging
import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Type

from nexus_llm.enums import TrainingStage
from nexus_llm.events import Event, EventBus, get_event_bus
from nexus_llm.exceptions import PluginError
from nexus_llm.registry import Registry

logger = logging.getLogger(__name__)


class PluginInterface(ABC):
    """Abstract base class for all Nexus-LLM plugins.

    All plugins must inherit from this class and implement the required
    lifecycle methods. Plugins can hook into various stages of the
    application lifecycle to extend functionality.

    Attributes:
        name: Unique name of the plugin.
        version: Plugin version string.
        description: Human-readable description.
        author: Plugin author.
        dependencies: Set of plugin names this plugin depends on.
    """

    # Plugin metadata - override in subclasses
    name: str = "unnamed_plugin"
    version: str = "0.0.1"
    description: str = ""
    author: str = ""
    dependencies: Set[str] = set()

    def __init__(self) -> None:
        """Initialize the plugin."""
        self._enabled: bool = False
        self._initialized: bool = False
        self._config: Dict[str, Any] = {}
        self._event_bus: Optional[EventBus] = None

    @abstractmethod
    def on_load(self) -> None:
        """Called when the plugin is loaded.

        Perform any setup required before the plugin is activated,
        such as registering event handlers or initializing resources.
        """
        ...

    @abstractmethod
    def on_unload(self) -> None:
        """Called when the plugin is unloaded.

        Clean up any resources, unregister handlers, and perform
        any necessary shutdown operations.
        """
        ...

    def on_enable(self) -> None:
        """Called when the plugin is enabled.

        Called after on_load when the plugin is explicitly enabled.
        Default implementation does nothing.
        """
        pass

    def on_disable(self) -> None:
        """Called when the plugin is disabled.

        Called before on_unload when the plugin is explicitly disabled.
        Default implementation does nothing.
        """
        pass

    def on_config_update(self, config: Dict[str, Any]) -> None:
        """Called when plugin configuration is updated.

        Args:
            config: The updated configuration dictionary.
        """
        self._config.update(config)

    def on_event(self, event: Event) -> None:
        """Handle an event from the event bus.

        Override this to handle specific event types. The default
        implementation does nothing.

        Args:
            event: The event to handle.
        """
        pass

    @property
    def is_enabled(self) -> bool:
        """Check if the plugin is currently enabled."""
        return self._enabled

    @property
    def is_initialized(self) -> bool:
        """Check if the plugin has been initialized."""
        return self._initialized

    @property
    def config(self) -> Dict[str, Any]:
        """Get the plugin configuration."""
        return self._config

    def set_event_bus(self, bus: EventBus) -> None:
        """Set the event bus for this plugin.

        Args:
            bus: The EventBus instance.
        """
        self._event_bus = bus
        bus.subscribe(handler=self.on_event)

    def emit_event(self, event_type: str, data: Optional[Dict[str, Any]] = None) -> None:
        """Emit an event through the event bus.

        Args:
            event_type: Type of event to emit.
            data: Event data payload.
        """
        if self._event_bus is not None:
            event = Event(
                event_type=event_type,
                data=data or {},
                source=f"plugin:{self.name}",
            )
            self._event_bus.publish(event)

    def get_info(self) -> Dict[str, Any]:
        """Get information about this plugin.

        Returns:
            Dictionary with plugin metadata and status.
        """
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "dependencies": list(self.dependencies),
            "enabled": self._enabled,
            "initialized": self._initialized,
        }

    def __repr__(self) -> str:
        return f"Plugin(name={self.name!r}, version={self.version!r}, enabled={self._enabled})"


class PluginManager:
    """Manager for loading, enabling, and managing plugins.

    Handles plugin discovery, lifecycle management, dependency resolution,
    and provides a unified interface for interacting with all plugins.

    Example:
        >>> manager = PluginManager()
        >>> manager.discover_plugins()
        >>> manager.load_plugin("my_plugin")
        >>> manager.enable_plugin("my_plugin")
    """

    def __init__(
        self,
        plugin_dirs: Optional[List[str]] = None,
        event_bus: Optional[EventBus] = None,
    ) -> None:
        """Initialize the plugin manager.

        Args:
            plugin_dirs: Directories to search for plugins.
            event_bus: Event bus for inter-plugin communication.
        """
        self._plugins: Dict[str, PluginInterface] = {}
        self._plugin_classes: Dict[str, Type[PluginInterface]] = {}
        self._plugin_dirs: List[str] = plugin_dirs or []
        self._event_bus = event_bus or get_event_bus()
        self._registry = Registry[PluginInterface](name="plugins")
        self._load_order: List[str] = []

    def add_plugin_dir(self, directory: str) -> None:
        """Add a directory to search for plugins.

        Args:
            directory: Path to the plugin directory.
        """
        if directory not in self._plugin_dirs:
            self._plugin_dirs.append(directory)

    def discover_plugins(self) -> List[str]:
        """Discover plugins in all registered plugin directories.

        Scans the plugin directories for Python modules containing
        PluginInterface subclasses.

        Returns:
            List of discovered plugin names.
        """
        discovered: List[str] = []

        for plugin_dir in self._plugin_dirs:
            plugin_path = Path(plugin_dir)
            if not plugin_path.exists():
                logger.warning("Plugin directory does not exist: %s", plugin_dir)
                continue

            for item in plugin_path.iterdir():
                if item.is_file() and item.suffix == ".py" and not item.name.startswith("_"):
                    module_name = item.stem
                    try:
                        spec = importlib.util.spec_from_file_location(
                            f"nexus_llm_plugin_{module_name}",
                            str(item),
                        )
                        if spec is not None and spec.loader is not None:
                            module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(module)
                            self._find_plugin_classes(module, module_name)
                            discovered.append(module_name)
                            logger.info("Discovered plugin module: %s", module_name)
                    except Exception as exc:
                        logger.error("Error discovering plugin in %s: %s", item, exc)

                elif item.is_dir() and (item / "__init__.py").exists():
                    module_name = item.name
                    try:
                        sys.path.insert(0, str(plugin_path))
                        module = importlib.import_module(module_name)
                        self._find_plugin_classes(module, module_name)
                        discovered.append(module_name)
                        logger.info("Discovered plugin package: %s", module_name)
                    except Exception as exc:
                        logger.error("Error discovering plugin package %s: %s", module_name, exc)

        return discovered

    def _find_plugin_classes(self, module: Any, module_name: str) -> None:
        """Find PluginInterface subclasses in a module.

        Args:
            module: The module to search.
            module_name: Name of the module for logging.
        """
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (
                inspect.isclass(attr)
                and issubclass(attr, PluginInterface)
                and attr is not PluginInterface
                and not inspect.isabstract(attr)
            ):
                plugin_name = getattr(attr, "name", attr_name)
                self._plugin_classes[plugin_name] = attr
                logger.debug("Found plugin class: %s (%s)", plugin_name, attr.__name__)

    def register_plugin(self, plugin_class: Type[PluginInterface]) -> str:
        """Register a plugin class directly.

        Args:
            plugin_class: The plugin class to register.

        Returns:
            The registered plugin name.
        """
        plugin_name = getattr(plugin_class, "name", plugin_class.__name__)
        self._plugin_classes[plugin_name] = plugin_class
        return plugin_name

    def load_plugin(self, name: str) -> PluginInterface:
        """Load a plugin by name.

        Instantiates the plugin class and calls on_load().

        Args:
            name: Name of the plugin to load.

        Returns:
            The loaded plugin instance.

        Raises:
            PluginError: If the plugin is not found or fails to load.
        """
        if name in self._plugins:
            return self._plugins[name]

        if name not in self._plugin_classes:
            raise PluginError(
                plugin_name=name,
                message=f"Plugin '{name}' not found. Available: {list(self._plugin_classes.keys())}",
            )

        plugin_class = self._plugin_classes[name]

        # Check dependencies
        deps = getattr(plugin_class, "dependencies", set())
        missing_deps = [d for d in deps if d not in self._plugins and d not in self._plugin_classes]
        if missing_deps:
            raise PluginError(
                plugin_name=name,
                message=f"Missing dependencies: {missing_deps}",
            )

        try:
            plugin = plugin_class()
            plugin.set_event_bus(self._event_bus)
            plugin.on_load()
            plugin._initialized = True

            self._plugins[name] = plugin
            self._registry.register(name, plugin)
            self._load_order.append(name)

            self._event_bus.publish(Event(
                event_type="plugin.loaded",
                data={"plugin_name": name, "version": plugin.version},
                source="PluginManager",
            ))

            logger.info("Loaded plugin: %s v%s", name, plugin.version)
            return plugin

        except Exception as exc:
            raise PluginError(
                plugin_name=name,
                message=f"Failed to load: {exc}",
            ) from exc

    def unload_plugin(self, name: str) -> None:
        """Unload a plugin by name.

        Disables and unloads the plugin, calling on_disable() and on_unload().

        Args:
            name: Name of the plugin to unload.

        Raises:
            PluginError: If the plugin is not loaded.
        """
        if name not in self._plugins:
            raise PluginError(plugin_name=name, message="Plugin is not loaded.")

        plugin = self._plugins[name]

        try:
            if plugin.is_enabled:
                plugin.on_disable()
                plugin._enabled = False

            plugin.on_unload()
            plugin._initialized = False

            del self._plugins[name]
            self._registry.unregister(name)
            if name in self._load_order:
                self._load_order.remove(name)

            self._event_bus.publish(Event(
                event_type="plugin.unloaded",
                data={"plugin_name": name},
                source="PluginManager",
            ))

            logger.info("Unloaded plugin: %s", name)

        except Exception as exc:
            raise PluginError(
                plugin_name=name,
                message=f"Failed to unload: {exc}",
            ) from exc

    def enable_plugin(self, name: str) -> None:
        """Enable a loaded plugin.

        Args:
            name: Name of the plugin to enable.

        Raises:
            PluginError: If the plugin is not loaded.
        """
        if name not in self._plugins:
            raise PluginError(plugin_name=name, message="Plugin is not loaded.")

        plugin = self._plugins[name]
        if plugin.is_enabled:
            logger.warning("Plugin '%s' is already enabled.", name)
            return

        try:
            # Load dependencies first
            for dep_name in plugin.dependencies:
                if dep_name in self._plugins and not self._plugins[dep_name].is_enabled:
                    self.enable_plugin(dep_name)

            plugin.on_enable()
            plugin._enabled = True

            self._event_bus.publish(Event(
                event_type="plugin.enabled",
                data={"plugin_name": name},
                source="PluginManager",
            ))

            logger.info("Enabled plugin: %s", name)

        except Exception as exc:
            raise PluginError(
                plugin_name=name,
                message=f"Failed to enable: {exc}",
            ) from exc

    def disable_plugin(self, name: str) -> None:
        """Disable a loaded plugin.

        Args:
            name: Name of the plugin to disable.

        Raises:
            PluginError: If the plugin is not loaded or not enabled.
        """
        if name not in self._plugins:
            raise PluginError(plugin_name=name, message="Plugin is not loaded.")

        plugin = self._plugins[name]
        if not plugin.is_enabled:
            logger.warning("Plugin '%s' is not enabled.", name)
            return

        # Check if any enabled plugins depend on this one
        dependents = [
            p_name for p_name, p in self._plugins.items()
            if p.is_enabled and name in p.dependencies
        ]
        if dependents:
            raise PluginError(
                plugin_name=name,
                message=f"Cannot disable: plugins {dependents} depend on it.",
            )

        try:
            plugin.on_disable()
            plugin._enabled = False

            self._event_bus.publish(Event(
                event_type="plugin.disabled",
                data={"plugin_name": name},
                source="PluginManager",
            ))

            logger.info("Disabled plugin: %s", name)

        except Exception as exc:
            raise PluginError(
                plugin_name=name,
                message=f"Failed to disable: {exc}",
            ) from exc

    def get_plugin(self, name: str) -> Optional[PluginInterface]:
        """Get a loaded plugin by name.

        Args:
            name: Plugin name.

        Returns:
            The plugin instance, or None if not loaded.
        """
        return self._plugins.get(name)

    def list_plugins(self) -> List[Dict[str, Any]]:
        """List all known plugins with their status.

        Returns:
            List of plugin info dictionaries.
        """
        result = []
        all_names = set(list(self._plugin_classes.keys()) + list(self._plugins.keys()))
        for name in sorted(all_names):
            if name in self._plugins:
                result.append(self._plugins[name].get_info())
            else:
                plugin_class = self._plugin_classes.get(name)
                if plugin_class:
                    result.append({
                        "name": name,
                        "version": getattr(plugin_class, "version", "unknown"),
                        "description": getattr(plugin_class, "description", ""),
                        "enabled": False,
                        "initialized": False,
                        "loaded": False,
                    })
        return result

    def load_all(self) -> List[str]:
        """Load all discovered plugins.

        Respects dependency order.

        Returns:
            List of loaded plugin names.
        """
        loaded: List[str] = []
        for name in self._resolve_load_order():
            try:
                self.load_plugin(name)
                loaded.append(name)
            except PluginError as exc:
                logger.error("Failed to load plugin %s: %s", name, exc)
        return loaded

    def unload_all(self) -> None:
        """Unload all loaded plugins in reverse order."""
        for name in reversed(self._load_order[:]):
            try:
                self.unload_plugin(name)
            except PluginError as exc:
                logger.error("Failed to unload plugin %s: %s", name, exc)

    def enable_all(self) -> List[str]:
        """Enable all loaded plugins.

        Returns:
            List of enabled plugin names.
        """
        enabled: List[str] = []
        for name in self._load_order:
            try:
                self.enable_plugin(name)
                enabled.append(name)
            except PluginError as exc:
                logger.error("Failed to enable plugin %s: %s", name, exc)
        return enabled

    def disable_all(self) -> None:
        """Disable all enabled plugins in reverse order."""
        for name in reversed(self._load_order[:]):
            try:
                if name in self._plugins and self._plugins[name].is_enabled:
                    self.disable_plugin(name)
            except PluginError as exc:
                logger.error("Failed to disable plugin %s: %s", name, exc)

    def _resolve_load_order(self) -> List[str]:
        """Resolve plugin load order based on dependencies.

        Returns:
            List of plugin names in dependency order.

        Raises:
            PluginError: If circular dependencies are detected.
        """
        order: List[str] = []
        visited: Set[str] = set()
        visiting: Set[str] = set()

        def visit(name: str) -> None:
            if name in visited:
                return
            if name in visiting:
                raise PluginError(plugin_name=name, message="Circular dependency detected")
            visiting.add(name)
            plugin_class = self._plugin_classes.get(name)
            if plugin_class:
                for dep in getattr(plugin_class, "dependencies", set()):
                    if dep in self._plugin_classes:
                        visit(dep)
            visiting.discard(name)
            visited.add(name)
            order.append(name)

        for name in self._plugin_classes:
            visit(name)

        return order

    @property
    def plugin_count(self) -> int:
        """Get the number of loaded plugins."""
        return len(self._plugins)

    @property
    def enabled_count(self) -> int:
        """Get the number of enabled plugins."""
        return sum(1 for p in self._plugins.values() if p.is_enabled)

    def __repr__(self) -> str:
        return f"PluginManager(loaded={self.plugin_count}, enabled={self.enabled_count})"
