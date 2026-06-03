"""
Nexus-LLM Status Bar Module

Provides a configurable status bar displaying model info, token count,
latency, memory usage, and other session metrics.
"""

from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class StatusField:
    """A single field in the status bar."""
    name: str
    value: str = ""
    style: str = "white"
    icon: str = ""
    separator: str = " │ "
    width: int | None = None
    update_func: Callable[[], str] | None = None

    def get_display(self) -> str:
        """Get the formatted display string for this field.

        Returns:
            ANSI-styled field string.
        """
        if self.update_func:
            try:
                self.value = self.update_func()
            except Exception:
                self.value = "err"

        color_map = {
            "white": "\033[38;5;231m",
            "cyan": "\033[38;5;117m",
            "green": "\033[38;5;84m",
            "yellow": "\033[38;5;186m",
            "red": "\033[38;5;198m",
            "blue": "\033[38;5;117m",
            "magenta": "\033[38;5;198m",
            "dim": "\033[2m",
            "bold": "\033[1m",
        }
        reset = "\033[0m"

        # Parse compound styles
        style_codes = []
        for part in self.style.split():
            if part in color_map:
                style_codes.append(color_map[part])
            elif part == "bold":
                style_codes.append("\033[1m")
            elif part == "dim":
                style_codes.append("\033[2m")

        prefix = "".join(style_codes)
        icon_str = f"{self.icon} " if self.icon else ""

        if self.width:
            value = self.value[:self.width]
            padding = self.width - len(value)
            return f"{prefix}{icon_str}{value}{' ' * max(0, padding)}{reset}"
        return f"{prefix}{icon_str}{self.value}{reset}"


class StatusBar:
    """Configurable terminal status bar.

    Displays a row of status fields at the bottom of the terminal
    showing model info, token count, latency, memory usage, and
    other real-time metrics. Fields can be dynamically added,
    updated, and removed.
    """

    def __init__(self, position: str = "bottom", style: str = "dim") -> None:
        self._fields: list[StatusField] = []
        self._position = position  # "top" or "bottom"
        self._style = style
        self._last_render = ""
        self._visible = True

    @property
    def visible(self) -> bool:
        """Get whether the status bar is visible."""
        return self._visible

    @visible.setter
    def visible(self, value: bool) -> None:
        """Set status bar visibility."""
        self._visible = value

    def add_field(
        self,
        field: StatusField | None = None,
        name: str = "",
        value: str = "",
        style: str = "white",
        icon: str = "",
        width: int | None = None,
        update_func: Callable[[], str] | None = None,
    ) -> StatusField:
        """Add a field to the status bar.

        Args:
            field: Pre-built StatusField (overrides other args).
            name: Field name.
            value: Initial value.
            style: Display style.
            icon: Optional icon character.
            width: Optional fixed width.
            update_func: Optional callback for dynamic values.

        Returns:
            The added StatusField.
        """
        if field:
            self._fields.append(field)
            return field

        f = StatusField(
            name=name,
            value=value,
            style=style,
            icon=icon,
            width=width,
            update_func=update_func,
        )
        self._fields.append(f)
        return f

    def remove_field(self, name: str) -> bool:
        """Remove a field by name.

        Args:
            name: Field name to remove.

        Returns:
            True if the field was found and removed.
        """
        for i, f in enumerate(self._fields):
            if f.name == name:
                self._fields.pop(i)
                return True
        return False

    def update_field(self, name: str, value: str) -> bool:
        """Update a field's value.

        Args:
            name: Field name.
            value: New value.

        Returns:
            True if the field was found and updated.
        """
        for f in self._fields:
            if f.name == name:
                f.value = value
                return True
        return False

    def get_field(self, name: str) -> StatusField | None:
        """Get a field by name.

        Args:
            name: Field name.

        Returns:
            The StatusField, or None if not found.
        """
        for f in self._fields:
            if f.name == name:
                return f
        return None

    def render(self) -> str:
        """Render the status bar.

        Returns:
            ANSI-formatted status bar string.
        """
        if not self._visible or not self._fields:
            return ""

        terminal_width = shutil.get_terminal_size().columns

        # Build field displays
        field_displays = []
        for f in self._fields:
            display = f.get_display()
            field_displays.append(display)

        # Join with separators
        content = f"{' │ '.join(field_displays)}"

        # Calculate visible content length
        import re
        visible_len = len(re.sub(r"\033\[[0-9;]*m", "", content))

        # Pad or truncate to terminal width
        if visible_len < terminal_width:
            padding = terminal_width - visible_len
            content += " " * padding
        elif visible_len > terminal_width:
            # Truncate - this is approximate for ANSI strings
            content = content[:terminal_width + (len(content) - visible_len)]

        # Apply status bar background
        bg_color = "\033[48;5;236m"
        reset = "\033[0m"

        bar = f"{bg_color}{content}{reset}"
        self._last_render = bar
        return bar

    def display(self) -> None:
        """Render and write the status bar to stdout."""
        bar = self.render()
        if bar:
            # Move cursor to appropriate line and write
            if self._position == "bottom":
                terminal_height = shutil.get_terminal_size().lines
                sys.stdout.write(f"\033[{terminal_height};1H")
            sys.stdout.write(bar)
            sys.stdout.flush()

    def clear(self) -> None:
        """Clear the status bar from the terminal."""
        terminal_width = shutil.get_terminal_size().columns
        if self._position == "bottom":
            terminal_height = shutil.get_terminal_size().lines
            sys.stdout.write(f"\033[{terminal_height};1H")
        sys.stdout.write(" " * terminal_width)
        sys.stdout.flush()

    def refresh(self) -> None:
        """Refresh the status bar by re-rendering and displaying."""
        self.display()


def get_memory_usage() -> str:
    """Get the current process memory usage as a formatted string.

    Returns:
        Memory usage string like "128.5MB".
    """
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / (1024 * 1024)
        if mem_mb >= 1024:
            return f"{mem_mb / 1024:.1f}GB"
        return f"{mem_mb:.1f}MB"
    except ImportError:
        # Fallback using /proc on Linux
        try:
            with open(f"/proc/{os.getpid()}/status", "r") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        kb = int(line.split()[1])
                        mb = kb / 1024
                        if mb >= 1024:
                            return f"{mb / 1024:.1f}GB"
                        return f"{mb:.1f}MB"
        except (OSError, ValueError):
            pass
        return "N/A"


def create_default_status_bar() -> StatusBar:
    """Create a status bar with default fields for a chat session.

    Returns:
        A StatusBar pre-configured with model, tokens, latency, and memory fields.
    """
    bar = StatusBar(position="bottom")
    bar.add_field(StatusField(name="model", value="gpt2-medium", style="bold cyan", icon="🤖"))
    bar.add_field(StatusField(name="tokens", value="0", style="green", icon="📝"))
    bar.add_field(StatusField(name="latency", value="-", style="yellow", icon="⏱"))
    bar.add_field(StatusField(name="memory", value="", style="dim", icon="💾", update_func=get_memory_usage))
    bar.add_field(StatusField(name="mode", value="chat", style="dim cyan", icon="💬"))
    return bar
