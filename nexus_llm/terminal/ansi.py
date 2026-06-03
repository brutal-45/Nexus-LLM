"""
Nexus-LLM ANSI Utilities Module

Provides ANSI escape code utilities for colors, styles, cursor control,
and screen clearing for terminal output.
"""

from __future__ import annotations

import os
import sys
from enum import Enum


class AnsiColor(str, Enum):
    """Standard ANSI color names."""
    BLACK = "black"
    RED = "red"
    GREEN = "green"
    YELLOW = "yellow"
    BLUE = "blue"
    MAGENTA = "magenta"
    CYAN = "cyan"
    WHITE = "white"
    BRIGHT_BLACK = "bright_black"
    BRIGHT_RED = "bright_red"
    BRIGHT_GREEN = "bright_green"
    BRIGHT_YELLOW = "bright_yellow"
    BRIGHT_BLUE = "bright_blue"
    BRIGHT_MAGENTA = "bright_magenta"
    BRIGHT_CYAN = "bright_cyan"
    BRIGHT_WHITE = "bright_white"


class AnsiStyle(str, Enum):
    """ANSI text styles."""
    RESET = "reset"
    BOLD = "bold"
    DIM = "dim"
    ITALIC = "italic"
    UNDERLINE = "underline"
    BLINK = "blink"
    REVERSE = "reverse"
    HIDDEN = "hidden"
    STRIKETHROUGH = "strikethrough"
    BOLD_OFF = "bold_off"
    DIM_OFF = "dim_off"
    ITALIC_OFF = "italic_off"
    UNDERLINE_OFF = "underline_off"
    BLINK_OFF = "blink_off"
    REVERSE_OFF = "reverse_off"
    HIDDEN_OFF = "hidden_off"
    STRIKETHROUGH_OFF = "strikethrough_off"


# ANSI escape code mappings
FOREGROUND_CODES: dict[str, str] = {
    "black": "30",
    "red": "31",
    "green": "32",
    "yellow": "33",
    "blue": "34",
    "magenta": "35",
    "cyan": "36",
    "white": "37",
    "bright_black": "90",
    "bright_red": "91",
    "bright_green": "92",
    "bright_yellow": "93",
    "bright_blue": "94",
    "bright_magenta": "95",
    "bright_cyan": "96",
    "bright_white": "97",
}

BACKGROUND_CODES: dict[str, str] = {
    "black": "40",
    "red": "41",
    "green": "42",
    "yellow": "43",
    "blue": "44",
    "magenta": "45",
    "cyan": "46",
    "white": "47",
    "bright_black": "100",
    "bright_red": "101",
    "bright_green": "102",
    "bright_yellow": "103",
    "bright_blue": "104",
    "bright_magenta": "105",
    "bright_cyan": "106",
    "bright_white": "107",
}

STYLE_CODES: dict[str, str] = {
    "reset": "0",
    "bold": "1",
    "dim": "2",
    "italic": "3",
    "underline": "4",
    "blink": "5",
    "reverse": "7",
    "hidden": "8",
    "strikethrough": "9",
    "bold_off": "21",
    "dim_off": "22",
    "italic_off": "23",
    "underline_off": "24",
    "blink_off": "25",
    "reverse_off": "27",
    "hidden_off": "28",
    "strikethrough_off": "29",
}

# Extended 256-color support
def color256(index: int) -> str:
    """Generate an ANSI 256-color foreground escape code.

    Args:
        index: Color index (0-255).

    Returns:
        ANSI escape code string.
    """
    return f"\033[38;5;{index}m"


def bg_color256(index: int) -> str:
    """Generate an ANSI 256-color background escape code.

    Args:
        index: Color index (0-255).

    Returns:
        ANSI escape code string.
    """
    return f"\033[48;5;{index}m"


# 24-bit true color support
def rgb(r: int, g: int, b: int) -> str:
    """Generate an ANSI 24-bit true color foreground escape code.

    Args:
        r: Red component (0-255).
        g: Green component (0-255).
        b: Blue component (0-255).

    Returns:
        ANSI escape code string.
    """
    return f"\033[38;2;{r};{g};{b}m"


def bg_rgb(r: int, g: int, b: int) -> str:
    """Generate an ANSI 24-bit true color background escape code.

    Args:
        r: Red component (0-255).
        g: Green component (0-255).
        b: Blue component (0-255).

    Returns:
        ANSI escape code string.
    """
    return f"\033[48;2;{r};{g};{b}m"


class AnsiFormatter:
    """Utility class for building ANSI-formatted terminal output.

    Provides methods for coloring, styling, cursor control, and
    screen manipulation using ANSI escape sequences.
    """

    RESET = "\033[0m"

    def __init__(self, support_color: bool | None = None) -> None:
        if support_color is None:
            self._supports_color = self._detect_color_support()
        else:
            self._supports_color = support_color

    @staticmethod
    def _detect_color_support() -> bool:
        """Detect whether the terminal supports ANSI colors.

        Returns:
            True if colors are supported.
        """
        # Check NO_COLOR environment variable
        if os.environ.get("NO_COLOR"):
            return False

        # Check FORCE_COLOR
        if os.environ.get("FORCE_COLOR"):
            return True

        # Check if stdout is a TTY
        if not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
            return False

        # Check TERM
        term = os.environ.get("TERM", "")
        if term in ("dumb", "unknown"):
            return False

        # Check COLORTERM for true color support
        colorterm = os.environ.get("COLORTERM", "")
        if colorterm in ("truecolor", "24bit"):
            return True

        # Most modern terminals support at least 256 colors
        if "color" in term or "xterm" in term:
            return True

        return True  # Default to supporting color

    @property
    def supports_color(self) -> bool:
        """Check if the terminal supports color output."""
        return self._supports_color

    def color(self, text: str, fg: str | None = None, bg: str | None = None) -> str:
        """Apply foreground and background colors to text.

        Args:
            text: The text to color.
            fg: Foreground color name (e.g., 'red', 'bright_green').
            bg: Background color name.

        Returns:
            ANSI-formatted string.
        """
        if not self._supports_color:
            return text

        codes = []
        if fg and fg in FOREGROUND_CODES:
            codes.append(FOREGROUND_CODES[fg])
        if bg and bg in BACKGROUND_CODES:
            codes.append(BACKGROUND_CODES[bg])

        if not codes:
            return text

        return f"\033[{';'.join(codes)}m{text}{self.RESET}"

    def style(self, text: str, *styles: str) -> str:
        """Apply text styles to text.

        Args:
            text: The text to style.
            *styles: Style names (e.g., 'bold', 'italic', 'underline').

        Returns:
            ANSI-formatted string.
        """
        if not self._supports_color:
            return text

        codes = []
        for s in styles:
            if s in STYLE_CODES:
                codes.append(STYLE_CODES[s])

        if not codes:
            return text

        return f"\033[{';'.join(codes)}m{text}{self.RESET}"

    def fg(self, color: str | int) -> str:
        """Get an ANSI foreground color escape code.

        Args:
            color: Color name (string) or 256-color index (int).

        Returns:
            ANSI escape code string.
        """
        if not self._supports_color:
            return ""

        if isinstance(color, int):
            return color256(color)
        if color in FOREGROUND_CODES:
            return f"\033[{FOREGROUND_CODES[color]}m"
        return ""

    def bg(self, color: str | int) -> str:
        """Get an ANSI background color escape code.

        Args:
            color: Color name (string) or 256-color index (int).

        Returns:
            ANSI escape code string.
        """
        if not self._supports_color:
            return ""

        if isinstance(color, int):
            return bg_color256(color)
        if color in BACKGROUND_CODES:
            return f"\033[{BACKGROUND_CODES[color]}m"
        return ""

    # Cursor control

    @staticmethod
    def cursor_up(n: int = 1) -> str:
        """Move cursor up n lines."""
        return f"\033[{n}A"

    @staticmethod
    def cursor_down(n: int = 1) -> str:
        """Move cursor down n lines."""
        return f"\033[{n}B"

    @staticmethod
    def cursor_right(n: int = 1) -> str:
        """Move cursor right n columns."""
        return f"\033[{n}C"

    @staticmethod
    def cursor_left(n: int = 1) -> str:
        """Move cursor left n columns."""
        return f"\033[{n}D"

    @staticmethod
    def cursor_home() -> str:
        """Move cursor to top-left corner."""
        return "\033[H"

    @staticmethod
    def cursor_pos(row: int, col: int) -> str:
        """Move cursor to specific position.

        Args:
            row: Row number (1-based).
            col: Column number (1-based).

        Returns:
            ANSI escape code string.
        """
        return f"\033[{row};{col}H"

    @staticmethod
    def save_cursor() -> str:
        """Save cursor position."""
        return "\033[s"

    @staticmethod
    def restore_cursor() -> str:
        """Restore saved cursor position."""
        return "\033[u"

    @staticmethod
    def hide_cursor() -> str:
        """Hide the cursor."""
        return "\033[?25l"

    @staticmethod
    def show_cursor() -> str:
        """Show the cursor."""
        return "\033[?25h"

    # Screen clearing

    @staticmethod
    def clear_screen() -> str:
        """Clear the entire screen."""
        return "\033[2J"

    @staticmethod
    def clear_line() -> str:
        """Clear the current line."""
        return "\033[2K"

    @staticmethod
    def clear_line_from_cursor() -> str:
        """Clear from cursor to end of line."""
        return "\033[K"

    @staticmethod
    def clear_line_to_cursor() -> str:
        """Clear from beginning of line to cursor."""
        return "\033[1K"

    @staticmethod
    def clear_screen_from_cursor() -> str:
        """Clear from cursor to end of screen."""
        return "\033[J"

    @staticmethod
    def clear_screen_to_cursor() -> str:
        """Clear from beginning of screen to cursor."""
        return "\033[1J"

    # Scrolling

    @staticmethod
    def scroll_up(n: int = 1) -> str:
        """Scroll the screen up n lines."""
        return f"\033[{n}S"

    @staticmethod
    def scroll_down(n: int = 1) -> str:
        """Scroll the screen down n lines."""
        return f"\033[{n}T"

    # Alternative screen buffer

    @staticmethod
    def enter_alt_screen() -> str:
        """Switch to the alternate screen buffer."""
        return "\033[?1049h"

    @staticmethod
    def exit_alt_screen() -> str:
        """Switch back to the main screen buffer."""
        return "\033[?1049l"

    # Utility methods

    @staticmethod
    def strip_ansi(text: str) -> str:
        """Remove all ANSI escape sequences from text.

        Args:
            text: Text possibly containing ANSI codes.

        Returns:
            Plain text without any ANSI sequences.
        """
        import re
        return re.sub(r"\033\[[0-9;]*[a-zA-Z]", "", text)

    @staticmethod
    def visible_length(text: str) -> int:
        """Calculate the visible length of text (excluding ANSI codes).

        Args:
            text: Text possibly containing ANSI codes.

        Returns:
            Visible character count.
        """
        return len(AnsiFormatter.strip_ansi(text))

    def hyperlink(self, url: str, text: str | None = None) -> str:
        """Create a terminal hyperlink (OSC 8).

        Args:
            url: The URL to link to.
            text: Display text (defaults to the URL).

        Returns:
            ANSI-formatted hyperlink string.
        """
        display = text or url
        return f"\033]8;;{url}\033\\{display}\033]8;;\033\\"

    @staticmethod
    def bell() -> str:
        """Ring the terminal bell."""
        return "\a"

    @staticmethod
    def set_title(title: str) -> str:
        """Set the terminal window title.

        Args:
            title: New window title.

        Returns:
            ANSI escape code string.
        """
        return f"\033]0;{title}\007"
