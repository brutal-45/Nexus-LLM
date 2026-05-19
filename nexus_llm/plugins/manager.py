"""Plugin manager for lifecycle management and dependency resolution.

Manages plugin loading, activation, deactivation, and dependency
resolution with support for plugin ordering and conflict detection.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Type

from nexus_llm.plugins.hook import HookManager
from nexus_llm.plugins.loader import PluginInfo, PluginLoader, PluginValidationError

logger = logging.getLogger(__name__)


class PluginState(Enum):
    """States in the plugin lifecycle."""

    DISCOVERED = "discovered"
    LOADED = "loaded"
    ACTIVATING = "activating"
    ACTIVE = "active"
    DEACTIVATING = "deactivating"
    INACTIVE = "inactive"
    ERROR = "error"


@dataclass
class PluginRecord:
    """Tracks a plugin's state and instance."""

    name: str
    info: PluginInfo
    state: PluginState = PluginState.DISCOVERED
    instance: Optional[Any] = None
    error: Optional[str] = None
    dependencies_satisfied: bool = False

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "state": self.state.value,
            "version": self.info.version,
            "description": self.info.description,
            "error": self.error,
            "dependencies_satisfied": self.dependencies_satisfied,
        }


class PluginManager:
    """Manages the full lifecycle of plugins.

    Handles discovery, loading, dependency resolution, activation,
    deactivation, and error handling for plugins. Integrates with
    the hook system for plugin event handling.
    """

    def __init__(
        self,
        hook_manager: Optional[HookManager] = None,
        loader: Optional[PluginLoader] = None,
        auto_discover: bool = False,
        search_paths: Optional[List[str]] = None,
    ):
        """Initialize the plugin manager.

        Args:
            hook_manager: Optional hook manager for plugin events.
            loader: Optional plugin loader.
            auto_discover: Whether to auto-discover plugins on init.
            search_paths: Optional search paths for auto-discovery.
        """
        self.hook_manager = hook_manager or HookManager()
        self.loader = loader or PluginLoader(search_paths=search_paths)
        self._plugins: Dict[str, PluginRecord] = {}
        self._activation_order: List[str] = []

        if auto_discover and search_paths:
            self.discover()

    def add_search_path(self, path: str) -> None:
        """Add a search path for plugin discovery."""
        self.loader.add_search_path(path)

    def discover(self) -> List[PluginInfo]:
        """Discover plugins in all search paths.

        Returns:
            List of discovered plugin info objects.
        """
        discovered = self.loader.discover()
        for info in discovered:
            if info.name not in self._plugins:
                self._plugins[info.name] = PluginRecord(
                    name=info.name,
                    info=info,
                    state=PluginState.DISCOVERED,
                )
            else:
                # Update info for already known plugins
                self._plugins[info.name].info = info

        logger.info("Discovered %d plugin(s). Total known: %d", len(discovered), len(self._plugins))
        return discovered

    def load_plugin(self, name: str) -> bool:
        """Load a discovered plugin.

        Args:
            name: The plugin name to load.

        Returns:
            True if the plugin was loaded successfully.
        """
        record = self._plugins.get(name)
        if record is None:
            logger.error("Plugin '%s' not found.", name)
            return False

        if record.state in (PluginState.LOADED, PluginState.ACTIVE, PluginState.INACTIVE):
            logger.info("Plugin '%s' already loaded.", name)
            return True

        try:
            plugin_class = self.loader.load(record.info)
            record.state = PluginState.LOADED
            record.instance = None  # Not instantiated until activation
            logger.info("Loaded plugin '%s'.", name)
            return True
        except PluginValidationError as e:
            record.state = PluginState.ERROR
            record.error = str(e)
            logger.error("Failed to load plugin '%s': %s", name, e)
            return False
        except Exception as e:
            record.state = PluginState.ERROR
            record.error = str(e)
            logger.error("Unexpected error loading plugin '%s': %s", name, e)
            return False

    def load_all(self) -> Dict[str, bool]:
        """Load all discovered plugins.

        Returns:
            Dict mapping plugin names to load success status.
        """
        results = {}
        for name in list(self._plugins.keys()):
            results[name] = self.load_plugin(name)
        return results

    def activate_plugin(self, name: str, **kwargs) -> bool:
        """Activate a loaded plugin.

        Checks dependencies, instantiates the plugin, calls activate(),
        and registers its hooks.

        Args:
            name: The plugin name to activate.
            **kwargs: Additional arguments for plugin activation.

        Returns:
            True if the plugin was activated successfully.
        """
        record = self._plugins.get(name)
        if record is None:
            logger.error("Plugin '%s' not found.", name)
            return False

        if record.state == PluginState.ACTIVE:
            logger.info("Plugin '%s' is already active.", name)
            return True

        if record.state == PluginState.DISCOVERED:
            if not self.load_plugin(name):
                return False
            record = self._plugins[name]

        # Check dependencies
        if not self._check_dependencies(name):
            record.dependencies_satisfied = False
            logger.error("Plugin '%s' has unsatisfied dependencies.", name)
            return False
        record.dependencies_satisfied = True

        try:
            record.state = PluginState.ACTIVATING

            # Get the plugin class
            plugin_class = self.loader.get_loaded().get(name)
            if plugin_class is None:
                raise RuntimeError(f"Plugin class for '{name}' not found in loader")

            # Instantiate the plugin
            record.instance = plugin_class(hook_manager=self.hook_manager, **kwargs)

            # Call activate
            record.instance.activate()

            record.state = PluginState.ACTIVE
            self._activation_order.append(name)

            # Trigger hook
            self.hook_manager.trigger("plugin_activated", result=None, plugin_name=name)

            logger.info("Activated plugin '%s'.", name)
            return True

        except Exception as e:
            record.state = PluginState.ERROR
            record.error = str(e)
            logger.error("Failed to activate plugin '%s': %s", name, e)
            return False

    def deactivate_plugin(self, name: str) -> bool:
        """Deactivate an active plugin.

        Args:
            name: The plugin name to deactivate.

        Returns:
            True if the plugin was deactivated successfully.
        """
        record = self._plugins.get(name)
        if record is None:
            logger.error("Plugin '%s' not found.", name)
            return False

        if record.state != PluginState.ACTIVE:
            logger.info("Plugin '%s' is not active (state: %s).", name, record.state.value)
            return True

        try:
            record.state = PluginState.DEACTIVATING

            if record.instance:
                record.instance.deactivate()

            # Remove hooks owned by this plugin
            self.hook_manager.unregister_by_owner(name)

            record.state = PluginState.INACTIVE
            if name in self._activation_order:
                self._activation_order.remove(name)

            # Trigger hook
            self.hook_manager.trigger("plugin_deactivated", result=None, plugin_name=name)

            logger.info("Deactivated plugin '%s'.", name)
            return True

        except Exception as e:
            record.state = PluginState.ERROR
            record.error = str(e)
            logger.error("Failed to deactivate plugin '%s': %s", name, e)
            return False

    def activate_all(self) -> Dict[str, bool]:
        """Activate all loaded plugins in dependency order.

        Returns:
            Dict mapping plugin names to activation success status.
        """
        ordered = self._resolve_activation_order()
        results = {}
        for name in ordered:
            results[name] = self.activate_plugin(name)
        return results

    def deactivate_all(self) -> Dict[str, bool]:
        """Deactivate all active plugins in reverse order.

        Returns:
            Dict mapping plugin names to deactivation success status.
        """
        results = {}
        for name in reversed(self._activation_order):
            results[name] = self.deactivate_plugin(name)
        return results

    def _check_dependencies(self, name: str) -> bool:
        """Check if a plugin's dependencies are satisfied."""
        record = self._plugins.get(name)
        if not record:
            return False

        for dep in record.info.dependencies:
            dep_record = self._plugins.get(dep)
            if dep_record is None or dep_record.state not in (PluginState.ACTIVE, PluginState.INACTIVE):
                return False
        return True

    def _resolve_activation_order(self) -> List[str]:
        """Resolve activation order based on dependencies.

        Uses topological sort to determine the order in which
        plugins should be activated.
        """
        # Build dependency graph
        graph: Dict[str, Set[str]] = defaultdict(set)
        all_plugins = set(self._plugins.keys())

        for name, record in self._plugins.items():
            for dep in record.info.dependencies:
                if dep in all_plugins:
                    graph[name].add(dep)

        # Topological sort (Kahn's algorithm)
        in_degree = {name: 0 for name in all_plugins}
        for name, deps in graph.items():
            for dep in deps:
                in_degree[name] = in_degree.get(name, 0)

        # Actually compute in-degrees
        in_degree = {name: len(deps) for name, deps in graph.items()}
        for name in all_plugins - set(graph.keys()):
            in_degree[name] = 0

        queue = [name for name, deg in in_degree.items() if deg == 0]
        order = []

        while queue:
            node = queue.pop(0)
            order.append(node)
            for name, deps in graph.items():
                if node in deps:
                    in_degree[name] -= 1
                    if in_degree[name] == 0:
                        queue.append(name)

        # Check for cycles
        if len(order) != len(all_plugins):
            cycle_plugins = all_plugins - set(order)
            logger.warning("Dependency cycle detected among: %s", cycle_plugins)
            # Add remaining plugins in arbitrary order
            order.extend(cycle_plugins)

        return order

    def get_plugin(self, name: str) -> Optional[Any]:
        """Get an active plugin instance by name."""
        record = self._plugins.get(name)
        if record and record.state == PluginState.ACTIVE:
            return record.instance
        return None

    def get_plugin_info(self, name: str) -> Optional[PluginInfo]:
        """Get plugin info by name."""
        record = self._plugins.get(name)
        return record.info if record else None

    def list_plugins(self, state: Optional[PluginState] = None) -> List[PluginRecord]:
        """List plugins, optionally filtered by state."""
        records = list(self._plugins.values())
        if state:
            records = [r for r in records if r.state == state]
        return records

    def get_status(self) -> Dict[str, Any]:
        """Get overall plugin system status."""
        state_counts = defaultdict(int)
        for record in self._plugins.values():
            state_counts[record.state.value] += 1

        return {
            "total_plugins": len(self._plugins),
            "active_plugins": len([r for r in self._plugins.values() if r.state == PluginState.ACTIVE]),
            "state_counts": dict(state_counts),
            "activation_order": list(self._activation_order),
            "hook_count": self.hook_manager.count(),
        }

    def shutdown(self) -> None:
        """Gracefully shut down all plugins."""
        logger.info("Shutting down plugin manager...")
        self.deactivate_all()
        self.hook_manager.clear()
        logger.info("Plugin manager shut down complete.")
