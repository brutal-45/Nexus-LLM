"""Base Plugin class for Nexus-LLM plugin system."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class Plugin(ABC):
    """Abstract base class for all Nexus-LLM plugins.

    Subclass this to create a custom plugin. At minimum, override
    ``name``, ``version``, and ``on_load`` / ``on_unload`` lifecycle hooks.
    """

    # ------------------------------------------------------------------
    # Metadata — subclasses should override these as class-level attrs
    # ------------------------------------------------------------------
    name: str = "unnamed_plugin"
    version: str = "0.1.0"
    description: str = ""
    author: str = ""

    def __init__(self) -> None:
        self._enabled: bool = False
        self._hooks_registered: bool = False
        self._context: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def on_load(self) -> None:
        """Called when the plugin is loaded into the manager.

        Override to perform initialisation such as allocating resources,
        reading config, or registering hooks.
        """

    def on_unload(self) -> None:
        """Called when the plugin is unloaded from the manager.

        Override to clean up resources, close connections, etc.
        """

    # ------------------------------------------------------------------
    # Hook registration
    # ------------------------------------------------------------------

    def register_hooks(self) -> Dict[str, Any]:
        """Return a mapping of hook names to callback callables.

        The default implementation returns an empty dict. Override to
        register plugin-specific hooks, e.g.::

            def register_hooks(self):
                return {
                    "pre_generate": self._before_generate,
                    "post_generate": self._after_generate,
                }
        """
        return {}

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """Whether the plugin is currently enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    @property
    def context(self) -> Dict[str, Any]:
        """Shared context dictionary available to the plugin at runtime."""
        return self._context

    @context.setter
    def context(self, value: Dict[str, Any]) -> None:
        self._context = value

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"<Plugin name={self.name!r} version={self.version!r} "
            f"enabled={self._enabled}>"
        )
