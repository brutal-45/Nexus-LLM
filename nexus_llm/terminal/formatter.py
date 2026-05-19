"""
Nexus-LLM Rich Formatter Module

Provides rich text formatting for terminal output including markdown
rendering, code syntax highlighting, tables, and styled panels.
"""

from __future__ import annotations

import re
import sys
from typing import Any, Optional

try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.table import Table as RichTable
    from rich.text import Text
    from rich.theme import Theme as RichTheme
    from rich.syntax import Syntax
    from rich.columns import Columns
    from rich.padding import Padding
    from rich.rule import Rule
    from rich.box import ROUNDED, MINIMAL, HEAVY

    HAS_RICH = True
except ImportError:
    HAS_RICH = False


class RichFormatter:
    """Rich terminal formatter for styled output.

    Supports markdown rendering, syntax-highlighted code blocks,
    styled tables, panels, and various text decorations. Falls back
    to plain text output when Rich is unavailable.
    """

    def __init__(
        self,
        console: Any | None = None,
        theme: dict[str, str] | None = None,
        width: int | None = None,
    ) -> None:
        if HAS_RICH:
            rich_theme = RichTheme(theme) if theme else None
            self._console = console or Console(theme=rich_theme, width=width)
        else:
            self._console = None
        self._width = width or 80

    @property
    def console(self) -> Any:
        """Get the underlying Rich console."""
        return self._console

    @property
    def width(self) -> int:
        """Get the console width."""
        if self._console and HAS_RICH:
            return self._console.width
        return self._width

    def print(self, *objects: Any, **kwargs: Any) -> None:
        """Print objects with Rich formatting."""
        if self._console and HAS_RICH:
            self._console.print(*objects, **kwargs)
        else:
            for obj in objects:
                self._plain_print(str(obj))

    def _plain_print(self, text: str) -> None:
        """Fallback plain text printer that strips Rich markup."""
        clean = re.sub(r"\[/?[^\]]*\]", "", text)
        print(clean)

    def print_markdown(self, text: str) -> None:
        """Render and print markdown text.

        Args:
            text: Markdown-formatted text to render.
        """
        if self._console and HAS_RICH:
            md = Markdown(text)
            self._console.print(md)
        else:
            self._plain_print(text)

    def print_code(self, code: str, language: str = "python", line_numbers: bool = True) -> None:
        """Render and print syntax-highlighted code.

        Args:
            code: Source code to highlight.
            language: Programming language for syntax highlighting.
            line_numbers: Whether to display line numbers.
        """
        if self._console and HAS_RICH:
            syntax = Syntax(
                code,
                language,
                line_numbers=line_numbers,
                theme="monokai",
                word_wrap=True,
            )
            self._console.print(syntax)
        else:
            self._plain_print(code)

    def print_panel(
        self,
        content: str,
        title: str | None = None,
        subtitle: str | None = None,
        style: str = "blue",
        expand: bool = False,
    ) -> None:
        """Render and print a bordered panel.

        Args:
            content: The panel body content.
            title: Optional panel title.
            subtitle: Optional panel subtitle.
            style: Border color/style.
            expand: Whether the panel should expand to fill width.
        """
        if self._console and HAS_RICH:
            panel = Panel(
                content,
                title=title,
                subtitle=subtitle,
                style=style,
                expand=expand,
                border_style=style,
                box=ROUNDED,
                padding=(1, 2),
            )
            self._console.print(panel)
        else:
            width = self._width
            border = "+" + "-" * (width - 2) + "+"
            lines = [border]
            if title:
                lines.append(f"| {title:^{width - 4}} |")
                lines.append(border)
            for line in content.split("\n"):
                lines.append(f"| {line:<{width - 4}} |")
            lines.append(border)
            print("\n".join(lines))

    def print_table(
        self,
        headers: list[str],
        rows: list[list[str]],
        title: str | None = None,
        style: str = "cyan",
        show_lines: bool = False,
    ) -> None:
        """Render and print a styled table.

        Args:
            headers: Column header strings.
            rows: List of row data lists.
            title: Optional table title.
            style: Table border style.
            show_lines: Whether to show row separator lines.
        """
        if self._console and HAS_RICH:
            table = RichTable(
                title=title,
                style=style,
                show_lines=show_lines,
                box=ROUNDED,
                header_style=f"bold {style}",
            )
            for header in headers:
                table.add_column(header)
            for row in rows:
                table.add_row(*row)
            self._console.print(table)
        else:
            col_widths = [len(h) for h in headers]
            for row in rows:
                for i, cell in enumerate(row):
                    if i < len(col_widths):
                        col_widths[i] = max(col_widths[i], len(cell))
            fmt = " | ".join(f"{{:<{w}}}" for w in col_widths)
            sep = "-+-".join("-" * w for w in col_widths)
            print(fmt.format(*headers))
            print(sep)
            for row in rows:
                padded = row + [""] * (len(headers) - len(row))
                print(fmt.format(*padded[:len(headers)]))

    def print_error(self, message: str) -> None:
        """Print an error message with red styling.

        Args:
            message: The error message.
        """
        if self._console and HAS_RICH:
            self._console.print(f"[bold red]Error:[/bold red] {message}")
        else:
            print(f"Error: {message}", file=sys.stderr)

    def print_warning(self, message: str) -> None:
        """Print a warning message with yellow styling.

        Args:
            message: The warning message.
        """
        if self._console and HAS_RICH:
            self._console.print(f"[bold yellow]Warning:[/bold yellow] {message}")
        else:
            print(f"Warning: {message}")

    def print_info(self, message: str) -> None:
        """Print an informational message with blue styling.

        Args:
            message: The informational message.
        """
        if self._console and HAS_RICH:
            self._console.print(f"[bold blue]Info:[/bold blue] {message}")
        else:
            print(f"Info: {message}")

    def print_success(self, message: str) -> None:
        """Print a success message with green styling.

        Args:
            message: The success message.
        """
        if self._console and HAS_RICH:
            self._console.print(f"[bold green]✓[/bold green] {message}")
        else:
            print(f"✓ {message}")

    def print_assistant(self, text: str) -> None:
        """Print an assistant response with markdown rendering.

        Args:
            text: The assistant's response text.
        """
        if self._console and HAS_RICH:
            self._console.print()
            md = Markdown(text)
            panel = Panel(md, style="green", box=MINIMAL, padding=(0, 1))
            self._console.print(panel)
            self._console.print()
        else:
            print(f"\n{text}\n")

    def print_stream_chunk(self, chunk: str) -> None:
        """Print a streaming response chunk.

        Args:
            chunk: A chunk of text from a streaming response.
        """
        if self._console and HAS_RICH:
            self._console.print(chunk, end="", highlight=False)
        else:
            print(chunk, end="", flush=True)

    def print_rule(self, title: str = "", style: str = "dim") -> None:
        """Print a horizontal rule with an optional title.

        Args:
            title: Optional title text in the rule.
            style: Rule color/style.
        """
        if self._console and HAS_RICH:
            self._console.print(Rule(title, style=style))
        else:
            width = self._width
            if title:
                side = (width - len(title) - 2) // 2
                print("-" * side + f" {title} " + "-" * side)
            else:
                print("-" * width)

    def print_columns(self, items: list[str], column_count: int = 3) -> None:
        """Print items in a multi-column layout.

        Args:
            items: List of items to display.
            column_count: Number of columns.
        """
        if self._console and HAS_RICH:
            renderables = [Text(item) for item in items]
            self._console.print(Columns(renderables, padding=(0, 2)))
        else:
            max_width = max(len(item) for item in items) + 2 if items else 10
            cols_per_row = max(1, self._width // max_width)
            for i, item in enumerate(items):
                print(f"{item:<{max_width}}", end="")
                if (i + 1) % cols_per_row == 0:
                    print()
            if len(items) % cols_per_row != 0:
                print()

    def print_key_value(self, data: dict[str, Any], style: str = "cyan") -> None:
        """Print a dictionary as key-value pairs.

        Args:
            data: Dictionary of key-value pairs.
            style: Text style for keys.
        """
        if self._console and HAS_RICH:
            for key, value in data.items():
                self._console.print(f"  [{style}]{key}:[/{style}] {value}")
        else:
            for key, value in data.items():
                print(f"  {key}: {value}")

    def print_list(self, items: list[str], bullet: str = "•", style: str = "cyan") -> None:
        """Print a bulleted list.

        Args:
            items: List of item strings.
            bullet: Bullet character.
            style: Bullet color/style.
        """
        if self._console and HAS_RICH:
            for item in items:
                self._console.print(f"  [{style}]{bullet}[/{style}] {item}")
        else:
            for item in items:
                print(f"  {bullet} {item}")
