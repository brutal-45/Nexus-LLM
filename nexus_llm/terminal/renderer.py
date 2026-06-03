"""
Nexus-LLM Text Renderer Module

Provides text rendering utilities including word wrapping, indentation,
alignment, and width management for terminal output.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from enum import Enum
from typing import Sequence


class Alignment(str, Enum):
    """Text alignment options."""
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"
    JUSTIFY = "justify"


@dataclass
class RenderOptions:
    """Options controlling text rendering behavior."""
    width: int = 80
    indent: int = 0
    subsequent_indent: int = 0
    alignment: Alignment = Alignment.LEFT
    initial_indent: str = ""
    subsequent_indent_str: str = ""
    drop_whitespace: bool = True
    break_long_words: bool = True
    break_on_hyphens: bool = True
    max_lines: int | None = None
    placeholder: str = "..."
    tab_size: int = 4
    preserve_paragraphs: bool = False


class TextRenderer:
    """Renders text for terminal display with wrapping, indentation, and alignment.

    Handles word wrapping, indentation, alignment, and width management
    for consistent terminal output formatting. Strips Rich/ANSI markup
    for width calculations while preserving it in output.
    """

    # Regex to match ANSI escape sequences
    ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
    # Regex to match Rich markup tags
    RICH_MARKUP = re.compile(r"\[/?[^\]]*\]")

    def __init__(self, default_width: int | None = None) -> None:
        self._default_width = default_width or self._detect_terminal_width()

    @staticmethod
    def _detect_terminal_width() -> int:
        """Detect the terminal width, falling back to 80."""
        try:
            return shutil.get_terminal_size().columns
        except (AttributeError, ValueError):
            return 80

    @property
    def default_width(self) -> int:
        """Get the default rendering width."""
        return self._default_width

    def visible_length(self, text: str) -> int:
        """Calculate the visible (non-markup) length of text.

        Args:
            text: Text that may contain ANSI or Rich markup.

        Returns:
            The visible character count.
        """
        clean = self.ANSI_ESCAPE.sub("", text)
        clean = self.RICH_MARKUP.sub("", clean)
        return len(clean)

    def wrap(self, text: str, options: RenderOptions | None = None) -> str:
        """Wrap text according to rendering options.

        Args:
            text: The text to wrap.
            options: Rendering options. Uses defaults if None.

        Returns:
            The wrapped text string.
        """
        opts = options or RenderOptions(width=self._default_width)

        if opts.preserve_paragraphs:
            return self._wrap_paragraphs(text, opts)

        width = max(opts.width - opts.indent, 10)
        initial = opts.initial_indent or " " * opts.indent
        subsequent = opts.subsequent_indent_str or " " * (opts.subsequent_indent or opts.indent)

        lines = self._wrap_text(
            text,
            width=width,
            initial_indent=initial,
            subsequent_indent=subsequent,
            drop_whitespace=opts.drop_whitespace,
            break_long_words=opts.break_long_words,
            break_on_hyphens=opts.break_on_hyphens,
            max_lines=opts.max_lines,
            placeholder=opts.placeholder,
        )

        if opts.alignment != Alignment.LEFT:
            lines = self._align_lines(lines, opts.alignment, opts.width)

        return "\n".join(lines)

    def _wrap_paragraphs(self, text: str, opts: RenderOptions) -> str:
        """Wrap text while preserving paragraph breaks.

        Args:
            text: Text with paragraph breaks (blank lines).
            opts: Rendering options.

        Returns:
            Wrapped text with preserved paragraphs.
        """
        paragraphs = re.split(r"\n\s*\n", text)
        wrapped_paragraphs = []
        for para in paragraphs:
            para = para.strip()
            if para:
                wrapped = self.wrap(para, RenderOptions(
                    width=opts.width,
                    indent=opts.indent,
                    subsequent_indent=opts.subsequent_indent,
                    alignment=opts.alignment,
                    drop_whitespace=opts.drop_whitespace,
                    break_long_words=opts.break_long_words,
                    break_on_hyphens=opts.break_on_hyphens,
                    max_lines=opts.max_lines,
                    placeholder=opts.placeholder,
                    preserve_paragraphs=False,
                ))
                wrapped_paragraphs.append(wrapped)
            else:
                wrapped_paragraphs.append("")
        return "\n\n".join(wrapped_paragraphs)

    def _wrap_text(
        self,
        text: str,
        width: int,
        initial_indent: str = "",
        subsequent_indent: str = "",
        drop_whitespace: bool = True,
        break_long_words: bool = True,
        break_on_hyphens: bool = True,
        max_lines: int | None = None,
        placeholder: str = "...",
    ) -> list[str]:
        """Core text wrapping algorithm.

        Args:
            text: Raw text to wrap.
            width: Maximum line width (excluding indent).
            initial_indent: Indent for the first line.
            subsequent_indent: Indent for subsequent lines.
            drop_whitespace: Whether to drop whitespace at start/end of lines.
            break_long_words: Whether to break words longer than width.
            break_on_hyphens: Whether to break on hyphens.
            max_lines: Maximum number of lines.
            placeholder: Suffix for truncated text.

        Returns:
            List of wrapped lines.
        """
        import textwrap
        wrapper = textwrap.TextWrapper(
            width=width,
            initial_indent=initial_indent,
            subsequent_indent=subsequent_indent,
            drop_whitespace=drop_whitespace,
            break_long_words=break_long_words,
            break_on_hyphens=break_on_hyphens,
            max_lines=max_lines,
            placeholder=placeholder,
        )
        # Replace tabs with spaces first
        text = text.replace("\t", "    ")
        # Normalize newlines within paragraphs
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
        try:
            return wrapper.wrap(text)
        except Exception:
            return text.split("\n")

    def _align_lines(self, lines: list[str], alignment: Alignment, width: int) -> list[str]:
        """Apply alignment to a list of lines.

        Args:
            lines: Text lines to align.
            alignment: Desired alignment.
            width: Total width for alignment.

        Returns:
            Aligned lines.
        """
        aligned = []
        for line in lines:
            visible_len = self.visible_length(line)
            if visible_len >= width:
                aligned.append(line)
                continue

            padding = width - visible_len

            if alignment == Alignment.CENTER:
                left_pad = padding // 2
                right_pad = padding - left_pad
                aligned.append(" " * left_pad + line + " " * right_pad)
            elif alignment == Alignment.RIGHT:
                aligned.append(" " * padding + line)
            elif alignment == Alignment.JUSTIFY:
                aligned.append(self._justify_line(line, width))
            else:
                aligned.append(line)
        return aligned

    def _justify_line(self, line: str, width: int) -> str:
        """Justify a single line by distributing extra spaces.

        Args:
            line: The line to justify.
            width: Target width.

        Returns:
            The justified line.
        """
        words = line.split()
        if len(words) <= 1:
            return line

        total_word_len = sum(len(w) for w in words)
        total_spaces = width - total_word_len
        gaps = len(words) - 1

        if gaps == 0:
            return line

        base_space = total_spaces // gaps
        extra_spaces = total_spaces % gaps

        result_parts = []
        for i, word in enumerate(words):
            result_parts.append(word)
            if i < gaps:
                spaces = base_space + (1 if i < extra_spaces else 0)
                result_parts.append(" " * spaces)

        return "".join(result_parts)

    def indent(self, text: str, indent_str: str = "    ", skip_first: bool = False) -> str:
        """Indent all lines of text.

        Args:
            text: Text to indent.
            indent_str: Indentation string to prepend.
            skip_first: Whether to skip indenting the first line.

        Returns:
            Indented text.
        """
        lines = text.split("\n")
        result = []
        for i, line in enumerate(lines):
            if i == 0 and skip_first:
                result.append(line)
            elif line.strip():
                result.append(indent_str + line)
            else:
                result.append(line)
        return "\n".join(result)

    def dedent(self, text: str) -> str:
        """Remove common leading whitespace from all lines.

        Args:
            text: Text to dedent.

        Returns:
            Dedented text.
        """
        import textwrap
        return textwrap.dedent(text)

    def truncate(self, text: str, width: int, suffix: str = "...") -> str:
        """Truncate text to fit within a given width.

        Args:
            text: Text to potentially truncate.
            width: Maximum visible width.
            suffix: Suffix to append when truncating.

        Returns:
            Truncated text or original if it fits.
        """
        visible_len = self.visible_length(text)
        if visible_len <= width:
            return text

        # Calculate how many visible characters we can keep
        suffix_len = len(suffix)
        target_len = width - suffix_len

        # Strip markup for truncation, then rebuild
        clean = self.ANSI_ESCAPE.sub("", text)
        clean = self.RICH_MARKUP.sub("", clean)

        if len(clean) <= width:
            return text

        return clean[:target_len] + suffix

    def pad(self, text: str, width: int, alignment: Alignment = Alignment.LEFT, fill: str = " ") -> str:
        """Pad text to a given width.

        Args:
            text: Text to pad.
            width: Target width.
            alignment: How to align the text.
            fill: Fill character.

        Returns:
            Padded text.
        """
        visible_len = self.visible_length(text)
        if visible_len >= width:
            return text

        padding = width - visible_len

        if alignment == Alignment.LEFT:
            return text + fill * padding
        elif alignment == Alignment.RIGHT:
            return fill * padding + text
        elif alignment == Alignment.CENTER:
            left = padding // 2
            right = padding - left
            return fill * left + text + fill * right
        return text

    def columnize(self, items: Sequence[str], columns: int = 0, column_width: int = 0) -> str:
        """Arrange items into columns.

        Args:
            items: Items to arrange.
            columns: Number of columns (0 = auto).
            column_width: Width per column (0 = auto).

        Returns:
            Columnized text.
        """
        if not items:
            return ""

        terminal_width = self._default_width
        max_item_len = max(self.visible_length(item) for item in items) + 2

        if not column_width:
            column_width = max_item_len

        if not columns:
            columns = max(1, terminal_width // column_width)

        rows: list[list[str]] = []
        row: list[str] = []
        for item in items:
            row.append(self.pad(item, column_width))
            if len(row) >= columns:
                rows.append(row)
                row = []
        if row:
            rows.append(row)

        return "\n".join("".join(r) for r in rows)

    def ruler(self, char: str = "─", width: int | None = None, title: str = "") -> str:
        """Create a horizontal ruler line.

        Args:
            char: Character to use for the ruler.
            width: Ruler width (defaults to terminal width).
            title: Optional title text in the ruler.

        Returns:
            The ruler string.
        """
        w = width or self._default_width
        if not title:
            return char * w
        title = f" {title} "
        side_len = (w - len(title)) // 2
        left = char * side_len
        right = char * (w - len(title) - side_len)
        return left + title + right

    def box(
        self,
        text: str,
        width: int | None = None,
        style: str = "single",
        title: str = "",
    ) -> str:
        """Draw a box around text.

        Args:
            text: Text content for the box.
            width: Box width (defaults to terminal width).
            style: Border style - 'single', 'double', 'rounded', 'bold', 'ascii'.
            title: Optional box title.

        Returns:
            The boxed text string.
        """
        w = width or self._default_width
        borders = {
            "single": ("┌", "─", "┐", "│", "│", "└", "─", "┘"),
            "double": ("╔", "═", "╗", "║", "║", "╚", "═", "╝"),
            "rounded": ("╭", "─", "╮", "│", "│", "╰", "─", "╯"),
            "bold": ("┏", "━", "┓", "┃", "┃", "┗", "━", "┛"),
            "ascii": ("+", "-", "+", "|", "|", "+", "-", "+"),
        }
        tl, ts, tr, ls, rs, bl, bs, br = borders.get(style, borders["ascii"])

        inner_width = w - 2
        top = tl + (ts * inner_width) + tr
        if title:
            title_str = f" {title} "
            title_pos = 1
            top = tl + ts[:title_pos] + title_str + ts * (inner_width - len(title_str) - title_pos) + tr
        bottom = bl + (bs * inner_width) + br

        lines = []
        for line in text.split("\n"):
            visible = self.visible_length(line)
            if visible > inner_width:
                line = self.truncate(line, inner_width)
            padding = inner_width - self.visible_length(line)
            lines.append(ls + line + " " * padding + rs)

        return "\n".join([top] + lines + [bottom])
