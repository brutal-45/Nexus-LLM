"""Plugin manager for Nexus-LLM.

Coordinates loading, enabling/disabling, and lifecycle of plugins
alongside the hook and event subsystems.
"""

import logging
from typing import Dict, List, Optional

from nexus_llm.plugins.config import PluginConfig
from nexus_llm.plugins.events import EventSystem
from nexus_llm.plugins.hooks import HookSystem
from nexus_llm.plugins.loader import PluginLoader, PluginLoadError, PluginValidationError
from nexus_llm.plugins.plugin import Plugin

logger = logging.getLogger(__name__)


class PluginManagerError(Exception):
    """General error raised by :class:`PluginManager`."""


class PluginManager:
    """Central registry and coordinator for all loaded plugins.

    Responsibilities:
      * Load / unload plugins via a :class:`PluginLoader`.
      * Enable or disable individual plugins.
      * Bridge plugins to the :class:`HookSystem` and :class:`EventSystem`.

    Example::

        mgr = PluginManager()
        plugin = mgr.load_plugin("my_plugin")   # discovers via loader
        mgr.enable_plugin("my_plugin")
        results = mgr.hooks.trigger("pre_generate", prompt="Hi")
    """

    def __init__(
        self,
        config: Optional[PluginConfig] = None,
        hooks: Optional[HookSystem] = None,
        events: Optional[EventSystem] = None,
    ) -> None:
        self._config: PluginConfig = config or PluginConfig()
        self._hooks: HookSystem = hooks or HookSystem()
        self._events: EventSystem = events or EventSystem()
        self._loader: PluginLoader = PluginLoader()

        # name -> Plugin instance
        self._plugins: Dict[str, Plugin] = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def hooks(self) -> HookSystem:
        """The hook system shared by all plugins."""
        return self._hooks

    @property
    def events(self) -> EventSystem:
        """The event system shared by all plugins."""
        return self._events

    @property
    def config(self) -> PluginConfig:
        """The active plugin configuration."""
        return self._config

    # ------------------------------------------------------------------
    # Loading / unloading
    # ------------------------------------------------------------------

    def load_plugin(self, name: str) -> Plugin:
        """Load a plugin identified by *name*.

        The loader searches the plugin directories configured on the
        :class:`PluginLoader`.  If a plugin with the same *name* is
        already loaded, the existing instance is returned.

        Args:
            name: Plugin name (must match ``Plugin.name``).

        Returns:
            The loaded (and possibly already-cached) Plugin instance.

        Raises:
            PluginManagerError: If loading or validation fails.
        """
        if name in self._plugins:
            logger.debug("Plugin %r already loaded, returning cached instance", name)
            return self._plugins[name]

        # Attempt to find and load from standard plugin directories
        plugin = self._try_load_from_directories(name)
        if plugin is None:
            raise PluginManagerError(f"Could not find or load plugin {name!r}")

        self._register_plugin(plugin)
        return plugin

    def load_plugin_from_file(self, path: str) -> Plugin:
        """Load a plugin directly from a filesystem path.

        Args:
            path: Path to a Python file containing a Plugin subclass.

        Returns:
            The loaded Plugin instance.

        Raises:
            PluginManagerError: If loading or validation fails.
        """
        try:
            plugin = self._loader.load_from_file(path)
        except PluginLoadError as exc:
            raise PluginManagerError(str(exc)) from exc

        if plugin.name in self._plugins:
            logger.warning(
                "Plugin %r is already loaded; skipping duplicate from %s",
                plugin.name,
                path,
            )
            return self._plugins[plugin.name]

        self._register_plugin(plugin)
        return plugin

    def unload_plugin(self, name: str) -> None:
        """Unload a previously loaded plugin.

        The plugin's ``on_unload`` lifecycle hook is called and its
        registered hooks are removed from the :class:`HookSystem`.

        Args:
            name: Plugin name.

        Raises:
            PluginManagerError: If the plugin is not loaded.
        """
        plugin = self._plugins.pop(name, None)
        if plugin is None:
            raise PluginManagerError(f"Plugin {name!r} is not loaded")

        # Call lifecycle hook
        try:
            plugin.on_unload()
        except Exception:
            logger.exception("Plugin %r raised during on_unload", name)

        # Unregister hooks
        for hook_name, cb in list(plugin.register_hooks().items()):
            self._hooks.unregister(hook_name, cb)

        plugin.enabled = False
        self._config.disable(name)

        # Publish event
        self._events.publish("plugin_unloaded", plugin_name=name)

        logger.info("Unloaded plugin %r", name)

    # ------------------------------------------------------------------
    # Enable / disable
    # ------------------------------------------------------------------

    def enable_plugin(self, name: str) -> None:
        """Enable a loaded plugin.

        Args:
            name: Plugin name.

        Raises:
            PluginManagerError: If the plugin is not loaded.
        """
        plugin = self.get_plugin(name)
        if plugin.enabled:
            logger.debug("Plugin %r is already enabled", name)
            return
        plugin.enabled = True
        self._config.enable(name)
        self._events.publish("plugin_enabled", plugin_name=name)
        logger.info("Enabled plugin %r", name)

    def disable_plugin(self, name: str) -> None:
        """Disable a loaded plugin without unloading it.

        Args:
            name: Plugin name.

        Raises:
            PluginManagerError: If the plugin is not loaded.
        """
        plugin = self.get_plugin(name)
        if not plugin.enabled:
            logger.debug("Plugin %r is already disabled", name)
            return
        plugin.enabled = False
        self._config.disable(name)
        self._events.publish("plugin_disabled", plugin_name=name)
        logger.info("Disabled plugin %r", name)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def list_plugins(self) -> List[Plugin]:
        """Return a list of all loaded plugins."""
        return list(self._plugins.values())

    def get_plugin(self, name: str) -> Plugin:
        """Retrieve a loaded plugin by name.

        Args:
            name: Plugin name.

        Returns:
            The Plugin instance.

        Raises:
            PluginManagerError: If no plugin with that name is loaded.
        """
        if name not in self._plugins:
            raise PluginManagerError(f"Plugin {name!r} is not loaded")
        return self._plugins[name]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _register_plugin(self, plugin: Plugin) -> None:
        """Validate, initialise, and register a plugin."""
        try:
            self._loader.validate_plugin(plugin)
        except PluginValidationError as exc:
            raise PluginManagerError(str(exc)) from exc

        # Lifecycle: on_load
        try:
            plugin.on_load()
        except Exception as exc:
            raise PluginManagerError(
                f"Plugin {plugin.name!r} raised during on_load: {exc}"
            ) from exc

        # Register hooks
        hooks_mapping = plugin.register_hooks()
        for hook_name, callback in hooks_mapping.items():
            self._hooks.register(hook_name, callback)

        self._plugins[plugin.name] = plugin
        self._config.enable(plugin.name)

        # Apply any stored settings
        stored = self._config.get_settings(plugin.name)
        if stored:
            plugin.context.update(stored)

        self._events.publish("plugin_loaded", plugin_name=plugin.name)
        logger.info("Registered plugin %r v%s", plugin.name, plugin.version)

    def _try_load_from_directories(self, name: str) -> Optional[Plugin]:
        """Attempt to load a plugin by scanning known directories."""
        from nexus_llm.core.config import PROJECT_ROOT

        search_dirs = [
            str(PROJECT_ROOT / "plugins"),
            str(PROJECT_ROOT / "nexus_llm" / "plugins" / "builtin"),
        ]

        for directory in search_dirs:
            candidates = self._loader.discover_plugins(directory)
            for path in candidates:
                try:
                    plugin = self._loader.load_from_file(path)
                    if plugin.name == name:
                        return plugin
                except (PluginLoadError, PluginValidationError):
                    continue
        return None
