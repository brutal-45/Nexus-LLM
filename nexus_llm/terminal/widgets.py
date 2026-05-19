"""
Nexus-LLM Terminal Widgets Module

Provides interactive UI widgets for the terminal:
TextBox, SelectBox, ProgressBar, ConfirmDialog, InputBox.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, Callable

from nexus_llm.terminal.ansi import AnsiFormatter


@dataclass
class WidgetStyle:
    """Styling configuration for widgets."""
    border_char: str = "─"
    border_style: str = "cyan"
    title_style: str = "bold cyan"
    text_style: str = "white"
    highlight_style: str = "bold green"
    error_style: str = "bold red"
    prompt_char: str = "❯"
    cursor_char: str = "▏"
    selected_char: str = "●"
    unselected_char: str = "○"
    check_char: str = "✓"
    cross_char: str = "✗"


class TextBox:
    """A multi-line text input widget with editing support.

    Supports cursor movement, insertion, deletion, and scrolling
    for editing text content within a bounded terminal area.
    """

    def __init__(
        self,
        title: str = "",
        initial_text: str = "",
        width: int = 60,
        height: int = 10,
        style: WidgetStyle | None = None,
        placeholder: str = "Type here...",
    ) -> None:
        self._lines = initial_text.split("\n") if initial_text else [""]
        self._cursor_row = 0
        self._cursor_col = 0
        self._scroll_offset = 0
        self._title = title
        self._width = width
        self._height = height
        self._style = style or WidgetStyle()
        self._placeholder = placeholder
        self._ansi = AnsiFormatter()
        self._history: list[list[str]] = [list(self._lines)]
        self._history_index = 0

    @property
    def text(self) -> str:
        """Get the current text content."""
        return "\n".join(self._lines)

    @text.setter
    def text(self, value: str) -> None:
        """Set the text content."""
        self._lines = value.split("\n") if value else [""]
        self._cursor_row = 0
        self._cursor_col = 0

    @property
    def cursor_position(self) -> tuple[int, int]:
        """Get the current (row, col) cursor position."""
        return (self._cursor_row, self._cursor_col)

    def insert_char(self, char: str) -> None:
        """Insert a character at the current cursor position.

        Args:
            char: The character to insert.
        """
        if not char:
            return
        line = self._lines[self._cursor_row]
        self._lines[self._cursor_row] = (
            line[:self._cursor_col] + char + line[self._cursor_col:]
        )
        self._cursor_col += len(char)

    def insert_newline(self) -> None:
        """Insert a newline at the current cursor position, splitting the line."""
        line = self._lines[self._cursor_row]
        before = line[:self._cursor_col]
        after = line[self._cursor_col:]
        self._lines[self._cursor_row] = before
        self._lines.insert(self._cursor_row + 1, after)
        self._cursor_row += 1
        self._cursor_col = 0

    def delete_char(self) -> None:
        """Delete the character before the cursor (backspace)."""
        if self._cursor_col > 0:
            line = self._lines[self._cursor_row]
            self._lines[self._cursor_row] = line[:self._cursor_col - 1] + line[self._cursor_col:]
            self._cursor_col -= 1
        elif self._cursor_row > 0:
            # Merge with previous line
            prev_line = self._lines[self._cursor_row - 1]
            self._cursor_col = len(prev_line)
            self._lines[self._cursor_row - 1] = prev_line + self._lines[self._cursor_row]
            self._lines.pop(self._cursor_row)
            self._cursor_row -= 1

    def delete_char_forward(self) -> None:
        """Delete the character at the cursor position."""
        line = self._lines[self._cursor_row]
        if self._cursor_col < len(line):
            self._lines[self._cursor_row] = line[:self._cursor_col] + line[self._cursor_col + 1:]
        elif self._cursor_row < len(self._lines) - 1:
            # Merge with next line
            self._lines[self._cursor_row] += self._lines[self._cursor_row + 1]
            self._lines.pop(self._cursor_row + 1)

    def move_cursor(self, direction: str) -> None:
        """Move the cursor in the specified direction.

        Args:
            direction: One of 'left', 'right', 'up', 'down', 'home', 'end'.
        """
        if direction == "left":
            if self._cursor_col > 0:
                self._cursor_col -= 1
            elif self._cursor_row > 0:
                self._cursor_row -= 1
                self._cursor_col = len(self._lines[self._cursor_row])
        elif direction == "right":
            if self._cursor_col < len(self._lines[self._cursor_row]):
                self._cursor_col += 1
            elif self._cursor_row < len(self._lines) - 1:
                self._cursor_row += 1
                self._cursor_col = 0
        elif direction == "up":
            if self._cursor_row > 0:
                self._cursor_row -= 1
                self._cursor_col = min(self._cursor_col, len(self._lines[self._cursor_row]))
        elif direction == "down":
            if self._cursor_row < len(self._lines) - 1:
                self._cursor_row += 1
                self._cursor_col = min(self._cursor_col, len(self._lines[self._cursor_row]))
        elif direction == "home":
            self._cursor_col = 0
        elif direction == "end":
            self._cursor_col = len(self._lines[self._cursor_row])

    def render(self) -> str:
        """Render the text box for display.

        Returns:
            ANSI-formatted string representation of the text box.
        """
        ansi = self._ansi
        lines = []

        # Top border
        title_str = f" {self._title} " if self._title else ""
        top = ansi.color(f"┌{title_str:{'─'}^{self._width - 2}}┐", self._style.border_style)
        lines.append(top)

        # Content lines
        visible_start = self._scroll_offset
        visible_end = min(visible_start + self._height, len(self._lines))

        for i in range(self._height):
            row = visible_start + i
            if row < len(self._lines):
                line_text = self._lines[row]
                # Truncate to width
                display_text = line_text[: self._width - 4]
                if row == self._cursor_row and i < self._height:
                    # Show cursor
                    before = display_text[:self._cursor_col]
                    after = display_text[self._cursor_col:]
                    cursor_char = self._style.cursor_char
                    content = f"{before}{ansi.color(cursor_char, 'reverse')}{after}"
                else:
                    content = ansi.color(display_text, self._style.text_style) if display_text else ""
                if not display_text and row == self._cursor_row:
                    content = ansi.color(self._placeholder[:self._width - 4], "dim")
            else:
                content = ansi.color("˜", "dim")

            padding = self._width - 4 - len(content)
            lines.append(f"│ {content}{' ' * max(0, padding)} │")

        # Bottom border
        bottom = ansi.color(f"└{'─' * (self._width - 2)}┘", self._style.border_style)
        lines.append(bottom)

        return "\n".join(lines)


class SelectBox:
    """A selection widget allowing the user to choose from a list of options.

    Supports single and multiple selection, scrolling, and keyboard navigation.
    """

    def __init__(
        self,
        options: list[str],
        title: str = "Select",
        multi_select: bool = False,
        style: WidgetStyle | None = None,
    ) -> None:
        self._options = options
        self._title = title
        self._multi_select = multi_select
        self._style = style or WidgetStyle()
        self._selected_index = 0
        self._selected_indices: set[int] = set()
        self._scroll_offset = 0
        self._ansi = AnsiFormatter()

    @property
    def selected(self) -> str | None:
        """Get the currently selected option."""
        if 0 <= self._selected_index < len(self._options):
            return self._options[self._selected_index]
        return None

    @property
    def selected_all(self) -> list[str]:
        """Get all selected options (for multi-select mode)."""
        return [self._options[i] for i in sorted(self._selected_indices) if i < len(self._options)]

    def move_up(self) -> None:
        """Move selection up."""
        if self._selected_index > 0:
            self._selected_index -= 1

    def move_down(self) -> None:
        """Move selection down."""
        if self._selected_index < len(self._options) - 1:
            self._selected_index += 1

    def toggle_selection(self) -> None:
        """Toggle the current item's selection (for multi-select)."""
        if self._multi_select:
            if self._selected_index in self._selected_indices:
                self._selected_indices.discard(self._selected_index)
            else:
                self._selected_indices.add(self._selected_index)

    def confirm(self) -> str | list[str]:
        """Confirm the selection.

        Returns:
            Selected option string (single-select) or list of strings (multi-select).
        """
        if self._multi_select:
            return self.selected_all
        return self.selected

    def render(self, visible_height: int = 10) -> str:
        """Render the select box for display.

        Args:
            visible_height: Maximum number of visible options.

        Returns:
            ANSI-formatted string representation.
        """
        ansi = self._ansi
        lines = []

        # Title
        lines.append(ansi.color(f"  {self._title}", self._style.title_style))
        lines.append(ansi.color(f"  {'─' * 40}", "dim"))

        # Adjust scroll offset
        if self._selected_index < self._scroll_offset:
            self._scroll_offset = self._selected_index
        elif self._selected_index >= self._scroll_offset + visible_height:
            self._scroll_offset = self._selected_index - visible_height + 1

        # Render options
        visible_end = min(self._scroll_offset + visible_height, len(self._options))
        for i in range(self._scroll_offset, visible_end):
            option = self._options[i]
            is_current = i == self._selected_index
            is_selected = i in self._selected_indices

            if self._multi_select:
                checkbox = self._style.check_char if is_selected else " "
                prefix = f"[{checkbox}]"
            else:
                prefix = self._style.selected_char if is_current else self._style.unselected_char

            if is_current:
                line = ansi.color(f"  {prefix} {option}", self._style.highlight_style)
            else:
                line = f"  {prefix} {option}"
            lines.append(line)

        # Navigation hint
        if len(self._options) > visible_height:
            lines.append(ansi.color(f"  (showing {self._scroll_offset + 1}-{visible_end} of {len(self._options)})", "dim"))

        return "\n".join(lines)


class ProgressBarWidget:
    """A visual progress bar widget.

    Displays a horizontal bar with optional percentage, label, and ETA.
    """

    def __init__(
        self,
        total: float = 100.0,
        width: int = 40,
        label: str = "",
        show_percentage: bool = True,
        show_eta: bool = True,
        style: WidgetStyle | None = None,
    ) -> None:
        self._total = total
        self._current = 0.0
        self._width = width
        self._label = label
        self._show_percentage = show_percentage
        self._show_eta = show_eta
        self._style = style or WidgetStyle()
        self._ansi = AnsiFormatter()
        self._start_time: float | None = None

    @property
    def current(self) -> float:
        """Get the current progress value."""
        return self._current

    @property
    def percentage(self) -> float:
        """Get the current progress as a percentage."""
        if self._total <= 0:
            return 0.0
        return min(100.0, (self._current / self._total) * 100)

    def update(self, value: float) -> None:
        """Update the progress value.

        Args:
            value: New current value.
        """
        if self._start_time is None:
            import time
            self._start_time = time.time()
        self._current = min(value, self._total)

    def increment(self, amount: float = 1.0) -> None:
        """Increment the progress by a given amount.

        Args:
            amount: Amount to increment.
        """
        self.update(self._current + amount)

    @property
    def eta_seconds(self) -> float | None:
        """Estimate the remaining time in seconds."""
        if self._start_time is None or self._current <= 0:
            return None
        import time
        elapsed = time.time() - self._start_time
        rate = self._current / elapsed
        remaining = self._total - self._current
        if rate <= 0:
            return None
        return remaining / rate

    def render(self) -> str:
        """Render the progress bar for display.

        Returns:
            ANSI-formatted string representation of the progress bar.
        """
        ansi = self._ansi
        pct = self.percentage / 100.0
        filled = int(self._width * pct)
        empty = self._width - filled

        bar_filled = ansi.color("█" * filled, "green")
        bar_empty = ansi.color("░" * empty, "dim")

        parts = []
        if self._label:
            parts.append(ansi.color(self._label, self._style.title_style))

        parts.append(f"[{bar_filled}{bar_empty}]")

        if self._show_percentage:
            parts.append(f"{self.percentage:5.1f}%")

        if self._show_eta and self.eta_seconds is not None:
            eta = self.eta_seconds
            if eta < 60:
                eta_str = f"{eta:.0f}s"
            elif eta < 3600:
                eta_str = f"{eta / 60:.1f}m"
            else:
                eta_str = f"{eta / 3600:.1f}h"
            parts.append(f"ETA: {eta_str}")

        return " ".join(parts)


class ConfirmDialog:
    """A confirmation dialog widget with yes/no options.

    Displays a message and waits for user confirmation.
    """

    def __init__(
        self,
        message: str,
        title: str = "Confirm",
        default: bool = False,
        style: WidgetStyle | None = None,
    ) -> None:
        self._message = message
        self._title = title
        self._default = default
        self._style = style or WidgetStyle()
        self._result: bool | None = None
        self._ansi = AnsiFormatter()

    @property
    def result(self) -> bool | None:
        """Get the dialog result."""
        return self._result

    def confirm(self) -> bool:
        """Record a 'yes' response."""
        self._result = True
        return True

    def deny(self) -> bool:
        """Record a 'no' response."""
        self._result = False
        return False

    def render(self) -> str:
        """Render the confirm dialog for display.

        Returns:
            ANSI-formatted string representation.
        """
        ansi = self._ansi
        lines = []

        # Border top
        width = max(50, len(self._message) + 6)
        lines.append(ansi.color(f"╔{'═' * (width - 2)}╗", self._style.border_style))

        # Title
        title_line = f"║ {self._title:^{width - 4}} ║"
        lines.append(ansi.color(title_line, self._style.title_style))

        # Separator
        lines.append(ansi.color(f"╟{'─' * (width - 2)}╢", self._style.border_style))

        # Message
        msg_line = f"║ {self._message:<{width - 4}} ║"
        lines.append(msg_line)

        # Options
        default_marker = "Y" if self._default else "N"
        options = f"[Y/n]" if self._default else f"[y/N]"
        opt_line = f"║ {options:^{width - 4}} ║"
        lines.append(ansi.color(opt_line, self._style.highlight_style))

        # Border bottom
        lines.append(ansi.color(f"╚{'═' * (width - 2)}╝", self._style.border_style))

        return "\n".join(lines)

    def ask(self) -> bool:
        """Display the dialog and wait for user input.

        Returns:
            True for yes, False for no.
        """
        print(self.render())
        try:
            response = input().strip().lower()
            if not response:
                return self._default
            return response in ("y", "yes", "1", "true")
        except (EOFError, KeyboardInterrupt):
            return self._default


class InputBox:
    """A single-line input widget with validation and default value support.

    Provides a styled input field with optional validation, default values,
    and error message display.
    """

    def __init__(
        self,
        prompt: str = "Input",
        default: str = "",
        placeholder: str = "",
        validator: Callable[[str], str | None] | None = None,
        is_password: bool = False,
        style: WidgetStyle | None = None,
    ) -> None:
        self._prompt = prompt
        self._default = default
        self._placeholder = placeholder
        self._validator = validator
        self._is_password = is_password
        self._style = style or WidgetStyle()
        self._value = ""
        self._error: str | None = None
        self._ansi = AnsiFormatter()

    @property
    def value(self) -> str:
        """Get the current input value."""
        return self._value

    @property
    def error(self) -> str | None:
        """Get the current validation error, if any."""
        return self._error

    def validate(self) -> bool:
        """Validate the current value.

        Returns:
            True if valid, False otherwise.
        """
        if self._validator:
            error = self._validator(self._value)
            if error:
                self._error = error
                return False
        self._error = None
        return True

    def render(self) -> str:
        """Render the input box for display.

        Returns:
            ANSI-formatted string representation.
        """
        ansi = self._ansi
        lines = []

        # Prompt
        prompt_line = f"{ansi.color(self._style.prompt_char, self._style.highlight_style)} {self._prompt}"
        if self._default:
            prompt_line += f" {ansi.color(f'[{self._default}]', 'dim')}"
        lines.append(prompt_line)

        # Input field
        if self._value:
            display = "*" * len(self._value) if self._is_password else self._value
            lines.append(f"  {display}")
        elif self._placeholder:
            lines.append(ansi.color(f"  {self._placeholder}", "dim"))

        # Error message
        if self._error:
            lines.append(ansi.color(f"  {self._style.cross_char} {self._error}", self._style.error_style))

        return "\n".join(lines)

    def ask(self) -> str:
        """Display the input box and wait for user input.

        Returns:
            The validated input value.
        """
        while True:
            print(self.render())
            try:
                if self._is_password:
                    import getpass
                    self._value = getpass.getpass("  ")
                else:
                    self._value = input("  ")

                if not self._value and self._default:
                    self._value = self._default

                if self.validate():
                    return self._value
            except (EOFError, KeyboardInterrupt):
                return self._default
