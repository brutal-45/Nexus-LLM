"""CLI plugin with subcommand and hook support."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from nexus_llm.cli_ext.extension import CLIExtension

if TYPE_CHECKING:
    from nexus_llm.cli_ext.registry import CommandRegistry


class CLIPlugin(CLIExtension):
    """Extended CLI extension that supports subcommands and execution hooks.

    A :class:`CLIPlugin` can:

    - Add **subcommands** to existing top-level commands.
    - Register **hooks** that run before or after a command executes.

    Example::

        class MyPlugin(CLIPlugin):
            def register_commands(self, group):
                group.register("greet", self._greet, "Say hello")
                self.add_subcommand("model", "benchmark", self._bench, "Benchmark a model")

            def register_hooks(self):
                self.add_hook("before", "model.load", self._before_load)
                self.add_hook("after", "model.load", self._after_load)
    """

    def __init__(self, name: str, version: str = "0.1.0") -> None:
        super().__init__(name, version)
        self._subcommands: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._hooks: Dict[str, Dict[str, List[Callable]]] = {
            "before": {},
            "after": {},
        }

    # ------------------------------------------------------------------
    # Subcommand management
    # ------------------------------------------------------------------

    def add_subcommand(
        self,
        parent: str,
        name: str,
        func: Callable,
        help_text: str = "",
    ) -> None:
        """Register a subcommand under an existing parent command.

        Args:
            parent: The parent command name (e.g. ``"model"``).
            name: The subcommand name (e.g. ``"benchmark"``).
            func: Callable to execute when the subcommand is invoked.
            help_text: Short description for help output.
        """
        self._subcommands.setdefault(parent, {})[name] = {
            "func": func,
            "help_text": help_text,
        }

    def get_subcommands(self, parent: str) -> Dict[str, Dict[str, Any]]:
        """Return all subcommands registered under *parent*."""
        return dict(self._subcommands.get(parent, {}))

    def execute_subcommand(
        self, parent: str, name: str, *args: Any, **kwargs: Any
    ) -> Any:
        """Execute a registered subcommand.

        Args:
            parent: Parent command name.
            name: Subcommand name.
            *args, **kwargs: Passed through to the subcommand function.

        Returns:
            The return value of the subcommand function.

        Raises:
            KeyError: If the subcommand does not exist.
        """
        parent_cmds = self._subcommands.get(parent)
        if parent_cmds is None:
            raise KeyError(f"No subcommands registered under '{parent}'.")
        sub = parent_cmds.get(name)
        if sub is None:
            raise KeyError(
                f"Subcommand '{name}' not found under '{parent}'."
            )
        return sub["func"](*args, **kwargs)

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------

    def add_hook(
        self,
        phase: str,
        command: str,
        callback: Callable,
    ) -> None:
        """Register a hook that runs before or after a command.

        Args:
            phase: ``"before"`` or ``"after"``.
            command: The command name to hook into.
            callback: Callable invoked with ``(command, *args, **kwargs)``.

        Raises:
            ValueError: If *phase* is not ``"before"`` or ``"after"``.
        """
        if phase not in ("before", "after"):
            raise ValueError(
                f"Invalid hook phase '{phase}'. Must be 'before' or 'after'."
            )
        self._hooks[phase].setdefault(command, []).append(callback)

    def get_hooks(self, phase: str, command: str) -> List[Callable]:
        """Return the hooks registered for a given phase and command."""
        if phase not in ("before", "after"):
            raise ValueError(f"Invalid phase '{phase}'.")
        return list(self._hooks[phase].get(command, []))

    def run_hooks(
        self, phase: str, command: str, *args: Any, **kwargs: Any
    ) -> None:
        """Execute all hooks for *phase* and *command*.

        Hooks are executed in registration order.  If a hook raises an
        exception, execution stops and the exception propagates.
        """
        for callback in self._hooks.get(phase, {}).get(command, []):
            callback(command, *args, **kwargs)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_register(self) -> None:
        """Lifecycle hook — also registers hooks if overridden."""
        super().on_register()

    def on_unregister(self) -> None:
        """Lifecycle hook — clears subcommands and hooks."""
        self._subcommands.clear()
        self._hooks = {"before": {}, "after": {}}
        super().on_unregister()
