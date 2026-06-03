"""Base CLI extension class for the plugin architecture."""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:
    from nexus_llm.cli_ext.registry import CommandRegistry


class CLIExtension(abc.ABC):
    """Abstract base class for CLI extensions.

    Extensions add new commands and behaviour to the Nexus-LLM CLI.
    Subclasses must implement :meth:`register_commands` and may
    override the lifecycle hooks :meth:`on_register` and
    :meth:`on_unregister`.

    Attributes:
        name: Unique identifier for the extension.
        version: Semantic version string.
    """

    def __init__(self, name: str, version: str = "0.1.0") -> None:
        self._name = name
        self._version = version
        self._registered = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Return the extension's unique name."""
        return self._name

    @property
    def version(self) -> str:
        """Return the extension's version string."""
        return self._version

    @property
    def is_registered(self) -> bool:
        """Return whether the extension has been registered."""
        return self._registered

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def register_commands(self, group: CommandRegistry) -> None:
        """Register commands with the CLI *group*.

        This method is called when the extension is loaded.  Use the
        provided registry to add commands, sub-commands, and help text.

        Args:
            group: The command registry / group to attach commands to.
        """
        ...

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def on_register(self) -> None:
        """Lifecycle hook called after the extension is registered.

        Override to perform initialization (e.g. loading config,
        starting background tasks).
        """
        self._registered = True

    def on_unregister(self) -> None:
        """Lifecycle hook called before the extension is removed.

        Override to perform cleanup (e.g. closing connections,
        flushing state).
        """
        self._registered = False

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # noqa: D105
        return f"<CLIExtension name={self._name!r} version={self._version!r}>"
