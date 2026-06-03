"""
Nexus-LLM Panel Display Module

Provides bordered panels, titled panels, and collapsible panels
for structured terminal output.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from typing import Any

try:
    from rich.console import Console
    from rich.panel import Panel as RichPanel
    from rich.box import ROUNDED, HEAVY, MINIMAL, DOUBLE, ASCII
    from rich.text import Text
    from rich.padding import Padding

    HAS_RICH = True
except ImportError:
    HAS_RICH = False


@dataclass
class PanelStyle:
    """Styling options for a panel."""
    border_color: str = "cyan"
    title_color: str = "bold cyan"
    subtitle_color: str = "dim"
    content_color: str = "white"
    border_style: str = "rounded"  # rounded, heavy, double, minimal, ascii
    padding: int = 1
    expand: bool = False


class PanelRenderer:
    """Renders styled panels for terminal output.

    Provides bordered panels with optional titles, subtitles,
    and various border styles. Uses Rich when available, with
    a plain text fallback.
    """

    # Box drawing characters for different styles
    BOX_CHARS: dict[str, dict[str, str]] = {
        "rounded": {"tl": "╭", "tr": "╮", "bl": "╰", "br": "╯", "h": "─", "v": "│"},
        "heavy": {"tl": "┏", "tr": "┓", "bl": "┗", "br": "┛", "h": "━", "v": "┃"},
        "double": {"tl": "╔", "tr": "╗", "bl": "╚", "br": "╝", "h": "═", "v": "║"},
        "ascii": {"tl": "+", "tr": "+", "bl": "+", "br": "+", "h": "-", "v": "|"},
        "minimal": {"tl": " ", "tr": " ", "bl": " ", "br": " ", "h": "─", "v": "│"},
    }

    def __init__(self, default_style: PanelStyle | None = None) -> None:
        self._default_style = default_style or PanelStyle()

    def render(
        self,
        content: str,
        title: str | None = None,
        subtitle: str | None = None,
        style: PanelStyle | None = None,
        width: int | None = None,
    ) -> str:
        """Render a bordered panel.

        Args:
            content: The panel body content.
            title: Optional panel title.
            subtitle: Optional panel subtitle.
            style: Panel style (uses default if None).
            width: Panel width (auto-detected if None).

        Returns:
            ANSI-formatted panel string.
        """
        s = style or self._default_style

        if HAS_RICH:
            return self._render_rich(content, title, subtitle, s, width)
        return self._render_plain(content, title, subtitle, s, width)

    def print(
        self,
        content: str,
        title: str | None = None,
        subtitle: str | None = None,
        style: PanelStyle | None = None,
    ) -> None:
        """Render and print a panel.

        Args:
            content: The panel body content.
            title: Optional panel title.
            subtitle: Optional panel subtitle.
            style: Panel style.
        """
        output = self.render(content, title, subtitle, style)
        if HAS_RICH:
            console = Console()
            s = style or self._default_style
            box_map = {
                "rounded": ROUNDED,
                "heavy": HEAVY,
                "double": DOUBLE,
                "minimal": MINIMAL,
                "ascii": ASCII,
            }
            box = box_map.get(s.border_style, ROUNDED)
            panel = RichPanel(
                content,
                title=title,
                subtitle=subtitle,
                style=s.border_color,
                border_style=s.border_color,
                box=box,
                padding=s.padding,
                expand=s.expand,
            )
            console.print(panel)
        else:
            print(output)

    def _render_rich(
        self,
        content: str,
        title: str | None,
        subtitle: str | None,
        style: PanelStyle,
        width: int | None,
    ) -> str:
        """Render using Rich library."""
        box_map = {
            "rounded": ROUNDED,
            "heavy": HEAVY,
            "double": DOUBLE,
            "minimal": MINIMAL,
            "ascii": ASCII,
        }
        box = box_map.get(style.border_style, ROUNDED)

        from io import StringIO
        console = Console(file=StringIO(), force_terminal=True, width=width)
        panel = RichPanel(
            content,
            title=title,
            subtitle=subtitle,
            style=style.border_color,
            border_style=style.border_color,
            box=box,
            padding=style.padding,
            expand=style.expand,
        )
        console.print(panel)
        return console.file.getvalue() if hasattr(console.file, 'getvalue') else ""

    def _render_plain(
        self,
        content: str,
        title: str | None,
        subtitle: str | None,
        style: PanelStyle,
        width: int | None,
    ) -> str:
        """Render as plain text with box-drawing characters."""
        term_width = width or shutil.get_terminal_size().columns
        padding = style.padding
        inner_width = term_width - 2 - (padding * 2)
        chars = self.BOX_CHARS.get(style.border_style, self.BOX_CHARS["rounded"])

        # ANSI colors
        border_color = self._get_ansi_color(style.border_color)
        title_color = self._get_ansi_color(style.title_color)
        reset = "\033[0m"

        lines: list[str] = []

        # Top border with optional title
        if title:
            title_str = f" {title} "
            title_len = len(title_str)
            left_len = (inner_width - title_len) // 2
            right_len = inner_width - title_len - left_len
            top = (
                f"{border_color}{chars['tl']}{chars['h'] * left_len}"
                f"{title_color}{title_str}"
                f"{border_color}{chars['h'] * right_len}{chars['tr']}{reset}"
            )
        else:
            top = f"{border_color}{chars['tl']}{chars['h'] * inner_width}{chars['tr']}{reset}"
        lines.append(top)

        # Content lines
        content_lines = content.split("\n")
        pad_str = " " * padding
        for line in content_lines:
            # Truncate or pad line to fit
            visible_len = len(re.sub(r"\033\[[0-9;]*m", "", line))
            if visible_len > inner_width:
                # Truncate
                line = line[:inner_width]
                visible_len = inner_width
            extra_padding = inner_width - visible_len
            lines.append(
                f"{border_color}{chars['v']}{reset}{pad_str}{line}{' ' * extra_padding}{pad_str}{border_color}{chars['v']}{reset}"
            )

        # Bottom border with optional subtitle
        if subtitle:
            sub_str = f" {subtitle} "
            sub_len = len(sub_str)
            left_len = (inner_width - sub_len) // 2
            right_len = inner_width - sub_len - left_len
            bottom = (
                f"{border_color}{chars['bl']}{chars['h'] * left_len}"
                f"{sub_str}"
                f"{chars['h'] * right_len}{chars['br']}{reset}"
            )
        else:
            bottom = f"{border_color}{chars['bl']}{chars['h'] * inner_width}{chars['br']}{reset}"
        lines.append(bottom)

        return "\n".join(lines)

    def _get_ansi_color(self, color: str) -> str:
        """Convert a color name to an ANSI escape code.

        Args:
            color: Color name or Rich-style color string.

        Returns:
            ANSI escape code string.
        """
        color_map = {
            "cyan": "\033[38;5;117m",
            "green": "\033[38;5;84m",
            "yellow": "\033[38;5;186m",
            "red": "\033[38;5;198m",
            "blue": "\033[38;5;117m",
            "magenta": "\033[38;5;198m",
            "white": "\033[38;5;231m",
            "dim": "\033[2m",
            "bold": "\033[1m",
        }

        # Handle compound styles like "bold cyan"
        parts = color.split()
        codes = []
        for part in parts:
            if part == "bold":
                codes.append("\033[1m")
            elif part == "dim":
                codes.append("\033[2m")
            elif part in color_map:
                codes.append(color_map[part])
            else:
                codes.append("")

        return "".join(codes) if codes else ""


class CollapsiblePanel:
    """A panel that can be collapsed/expanded interactively.

    Displays a header that can be clicked or toggled to show
    or hide the panel's content.
    """

    def __init__(
        self,
        title: str,
        content: str = "",
        collapsed: bool = False,
        style: PanelStyle | None = None,
    ) -> None:
        self._title = title
        self._content = content
        self._collapsed = collapsed
        self._style = style or PanelStyle()
        self._renderer = PanelRenderer(self._style)

    @property
    def title(self) -> str:
        """Get the panel title."""
        return self._title

    @property
    def content(self) -> str:
        """Get the panel content."""
        return self._content

    @content.setter
    def content(self, value: str) -> None:
        """Set the panel content."""
        self._content = value

    @property
    def collapsed(self) -> bool:
        """Get whether the panel is collapsed."""
        return self._collapsed

    @collapsed.setter
    def collapsed(self, value: bool) -> None:
        """Set the collapsed state."""
        self._collapsed = value

    def toggle(self) -> None:
        """Toggle the collapsed state."""
        self._collapsed = not self._collapsed

    def expand(self) -> None:
        """Expand the panel."""
        self._collapsed = False

    def collapse(self) -> None:
        """Collapse the panel."""
        self._collapsed = True

    def render(self, width: int | None = None) -> str:
        """Render the collapsible panel.

        Args:
            width: Optional width override.

        Returns:
            ANSI-formatted string representation.
        """
        if self._collapsed:
            return self._render_collapsed(width)
        return self._render_expanded(width)

    def _render_collapsed(self, width: int | None = None) -> str:
        """Render the panel in collapsed state (header only)."""
        term_width = width or shutil.get_terminal_size().columns
        indicator = "\033[38;5;186m▶\033[0m"
        border_color = self._get_border_ansi()
        reset = "\033[0m"
        title_color = self._get_title_ansi()

        inner_width = term_width - 4
        title_display = f" {indicator} {self._title}"
        visible_len = len(re.sub(r"\033\[[0-9;]*m", "", title_display))
        padding = inner_width - visible_len
        line = f"{border_color}┌─{title_color}{title_display}{' ' * max(0, padding)}{border_color}─┐{reset}"
        bottom = f"{border_color}└{'─' * (inner_width + 2)}┘{reset}"
        return f"{line}\n{bottom}"

    def _render_expanded(self, width: int | None = None) -> str:
        """Render the panel in expanded state (header + content)."""
        indicator = "\033[38;5;84m▼\033[0m"
        title = f"{indicator} {self._title}"
        return self._renderer.render(
            self._content,
            title=title,
            style=self._style,
            width=width,
        )

    def _get_border_ansi(self) -> str:
        """Get ANSI code for the border color."""
        color = self._style.border_color
        color_map = {
            "cyan": "\033[38;5;117m",
            "green": "\033[38;5;84m",
            "yellow": "\033[38;5;186m",
            "red": "\033[38;5;198m",
            "blue": "\033[38;5;117m",
        }
        return color_map.get(color, "\033[38;5;117m")

    def _get_title_ansi(self) -> str:
        """Get ANSI code for the title color."""
        color = self._style.title_color
        codes = []
        if "bold" in color:
            codes.append("\033[1m")
        for part in color.split():
            if part == "cyan":
                codes.append("\033[38;5;117m")
            elif part == "green":
                codes.append("\033[38;5;84m")
            elif part == "yellow":
                codes.append("\033[38;5;186m")
        return "".join(codes) if codes else ""
