"""Plugin configuration for Nexus-LLM."""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List

import yaml


@dataclass
class PluginConfig:
    """Configuration container for the plugin system.

    Attributes:
        enabled_plugins: Ordered list of plugin names that should be active.
        plugin_settings: Per-plugin settings keyed by plugin name.
    """

    enabled_plugins: List[str] = field(default_factory=list)
    plugin_settings: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # YAML serialisation
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str) -> "PluginConfig":
        """Load a PluginConfig from a YAML file.

        If the file does not exist a default (empty) config is returned.

        Args:
            path: Filesystem path to the YAML file.

        Returns:
            A populated PluginConfig instance.
        """
        if not os.path.exists(path):
            return cls()

        with open(path, "r") as fh:
            data = yaml.safe_load(fh) or {}

        enabled = data.get("enabled_plugins", [])
        settings = data.get("plugin_settings", {})
        return cls(enabled_plugins=list(enabled), plugin_settings=dict(settings))

    def to_yaml(self, path: str) -> None:
        """Persist the config to a YAML file.

        Parent directories are created automatically if they do not exist.

        Args:
            path: Filesystem path for the output file.
        """
        import dataclasses

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = dataclasses.asdict(self)
        with open(path, "w") as fh:
            yaml.dump(payload, fh, default_flow_style=False)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def is_enabled(self, plugin_name: str) -> bool:
        """Return True if *plugin_name* is in the enabled list."""
        return plugin_name in self.enabled_plugins

    def get_settings(self, plugin_name: str) -> Dict[str, Any]:
        """Return the settings dict for *plugin_name* (empty dict if none)."""
        return dict(self.plugin_settings.get(plugin_name, {}))

    def set_settings(self, plugin_name: str, settings: Dict[str, Any]) -> None:
        """Overwrite the settings dict for *plugin_name*."""
        self.plugin_settings[plugin_name] = dict(settings)

    def enable(self, plugin_name: str) -> None:
        """Add *plugin_name* to the enabled list (idempotent)."""
        if plugin_name not in self.enabled_plugins:
            self.enabled_plugins.append(plugin_name)

    def disable(self, plugin_name: str) -> None:
        """Remove *plugin_name* from the enabled list (idempotent)."""
        if plugin_name in self.enabled_plugins:
            self.enabled_plugins.remove(plugin_name)
