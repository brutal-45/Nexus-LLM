"""Command registry for CLI extensions."""

from __future__ import annotations

import threading
from typing import Any, Callable, Dict, List, Optional, Tuple


class _Command:
    """Internal representation of a registered command."""

    __slots__ = ("name", "func", "help_text")

    def __init__(self, name: str, func: Callable, help_text: str) -> None:
        self.name = name
        self.func = func
        self.help_text = help_text

    def __repr__(self) -> str:  # noqa: D105
        return f"<Command {self.name!r}>"


class CommandRegistry:
    """Central registry for CLI commands.

    Provides a simple dictionary-based command registry where each
    command is identified by name and backed by a callable.

    Example::

        registry = CommandRegistry()
        registry.register("greet", lambda name: f"Hello, {name}!", "Greet someone")
        registry.execute("greet", "World")  # "Hello, World!"
    """

    def __init__(self, name: str = "root") -> None:
        self._name = name
        self._commands: Dict[str, _Command] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        command_func: Callable,
        help_text: str = "",
    ) -> None:
        """Register a new command.

        Args:
            name: Unique command name.
            command_func: Callable to invoke when the command is executed.
            help_text: Short description for help output.

        Raises:
            ValueError: If a command with *name* already exists.
        """
        with self._lock:
            if name in self._commands:
                raise ValueError(
                    f"Command '{name}' is already registered in "
                    f"group '{self._name}'."
                )
            self._commands[name] = _Command(
                name=name, func=command_func, help_text=help_text
            )

    def unregister(self, name: str) -> None:
        """Remove a registered command.

        Args:
            name: Command name to remove.

        Raises:
            KeyError: If the command does not exist.
        """
        with self._lock:
            if name not in self._commands:
                raise KeyError(
                    f"Command '{name}' not found in group '{self._name}'."
                )
            del self._commands[name]

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_command(self, name: str) -> Optional[Callable]:
        """Return the callable for *name*, or ``None`` if not found."""
        with self._lock:
            cmd = self._commands.get(name)
            return cmd.func if cmd else None

    def get_help(self, name: str) -> Optional[str]:
        """Return the help text for *name*, or ``None``."""
        with self._lock:
            cmd = self._commands.get(name)
            return cmd.help_text if cmd else None

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------

    def list_commands(self) -> List[str]:
        """Return a sorted list of registered command names."""
        with self._lock:
            return sorted(self._commands.keys())

    def list_commands_with_help(self) -> List[Tuple[str, str]]:
        """Return ``[(name, help_text), ...]`` sorted by name."""
        with self._lock:
            return [
                (cmd.name, cmd.help_text)
                for cmd in sorted(
                    self._commands.values(), key=lambda c: c.name
                )
            ]

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Execute a registered command.

        Args:
            name: Command name.
            *args, **kwargs: Passed to the command callable.

        Returns:
            The return value of the command.

        Raises:
            KeyError: If the command does not exist.
        """
        cmd = self._commands.get(name)
        if cmd is None:
            raise KeyError(
                f"Command '{name}' not found in group '{self._name}'."
            )
        return cmd.func(*args, **kwargs)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Return the registry / group name."""
        return self._name

    def __contains__(self, name: str) -> bool:  # type: ignore[override]
        return name in self._commands

    def __len__(self) -> int:  # noqa: D105
        return len(self._commands)

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"<CommandRegistry name={self._name!r} "
            f"commands={len(self._commands)}>"
        )
