"""
Nexus-LLM Custom Markdown Extensions Module

Provides extended markdown rendering with support for code blocks with
syntax highlighting, math expressions, tables, task lists, footnotes,
and other custom extensions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from nexus_llm.terminal.syntax import SyntaxHighlighter, Language


@dataclass
class MarkdownNode:
    """A node in the parsed markdown tree."""
    type: str
    content: str
    children: list[MarkdownNode] | None = None
    attrs: dict[str, str] | None = None
    language: str = ""
    level: int = 0

    def __repr__(self) -> str:
        return f"MarkdownNode(type={self.type!r}, content={self.content[:30]!r}...)"


class MarkdownRenderer:
    """Extended markdown renderer with custom extensions.

    Renders markdown to ANSI-formatted terminal output with support for:
    - Code blocks with syntax highlighting
    - Inline code
    - Math expressions (LaTeX-style)
    - Tables with alignment
    - Task lists (checkboxes)
    - Footnotes
    - Headings, lists, blockquotes, etc.
    """

    def __init__(self, syntax_highlighter: SyntaxHighlighter | None = None) -> None:
        self._highlighter = syntax_highlighter or SyntaxHighlighter()
        self._footnotes: dict[str, str] = {}
        self._footnote_refs: list[tuple[str, int]] = []

    def render(self, text: str) -> str:
        """Render markdown text to ANSI-formatted terminal output.

        Args:
            text: Markdown-formatted text.

        Returns:
            ANSI-colored string suitable for terminal display.
        """
        self._footnotes.clear()
        self._footnote_refs.clear()

        # Pre-process: extract and store footnotes
        text = self._extract_footnotes(text)

        # Process block-level elements
        blocks = self._split_blocks(text)
        rendered_blocks = []

        for block in blocks:
            rendered = self._render_block(block)
            if rendered:
                rendered_blocks.append(rendered)

        # Append footnotes section if any
        if self._footnotes:
            rendered_blocks.append(self._render_footnotes())

        return "\n\n".join(rendered_blocks)

    def _extract_footnotes(self, text: str) -> str:
        """Extract footnote definitions from text.

        Args:
            text: Markdown text with potential footnotes.

        Returns:
            Text with footnote definitions removed.
        """
        # Match footnote definitions: [^label]: content
        pattern = re.compile(r"^\[\^(\w+)\]:\s+(.+)$", re.MULTILINE)
        for match in pattern.finditer(text):
            label = match.group(1)
            content = match.group(2)
            self._footnotes[label] = content
        return pattern.sub("", text)

    def _split_blocks(self, text: str) -> list[str]:
        """Split text into block-level elements.

        Handles code blocks (which may contain blank lines) as single blocks.

        Args:
            text: Pre-processed markdown text.

        Returns:
            List of block strings.
        """
        blocks = []
        current_lines: list[str] = []
        in_code_block = False

        for line in text.split("\n"):
            if line.strip().startswith("```"):
                if in_code_block:
                    # End of code block
                    current_lines.append(line)
                    blocks.append("\n".join(current_lines))
                    current_lines = []
                    in_code_block = False
                else:
                    # Start of code block - flush current block
                    if current_lines:
                        blocks.append("\n".join(current_lines))
                        current_lines = []
                    current_lines.append(line)
                    in_code_block = True
            else:
                current_lines.append(line)

        if current_lines:
            blocks.append("\n".join(current_lines))

        return blocks

    def _render_block(self, block: str) -> str:
        """Render a single block-level element.

        Args:
            block: A block of markdown text.

        Returns:
            Rendered ANSI string for the block.
        """
        stripped = block.strip()
        if not stripped:
            return ""

        # Code block
        if stripped.startswith("```"):
            return self._render_code_block(stripped)

        # Heading
        heading_match = re.match(r"^(#{1,6})\s+(.+)$", stripped, re.MULTILINE)
        if heading_match:
            level = len(heading_match.group(1))
            content = heading_match.group(2)
            return self._render_heading(content, level)

        # Table
        if "|" in stripped and re.search(r"^\|.+\|$", stripped, re.MULTILINE):
            return self._render_table(stripped)

        # Task list
        if re.search(r"^\s*[-*+]\s+\[[ xX]\]", stripped, re.MULTILINE):
            return self._render_task_list(stripped)

        # Blockquote
        if stripped.startswith(">"):
            return self._render_blockquote(stripped)

        # Unordered list
        if re.search(r"^\s*[-*+]\s+", stripped, re.MULTILINE):
            return self._render_list(stripped, ordered=False)

        # Ordered list
        if re.search(r"^\s*\d+\.\s+", stripped, re.MULTILINE):
            return self._render_list(stripped, ordered=True)

        # Horizontal rule
        if re.match(r"^(-{3,}|\*{3,}|_{3,})$", stripped):
            return self._render_horizontal_rule()

        # Paragraph (fallback)
        return self._render_paragraph(stripped)

    def _render_heading(self, content: str, level: int) -> str:
        """Render a heading with ANSI styling."""
        styles = {
            1: "\033[1;38;5;117m",  # Bold cyan
            2: "\033[1;38;5;84m",   # Bold green
            3: "\033[1;38;5;186m",  # Bold yellow
            4: "\033[1;38;5;215m",  # Bold orange
            5: "\033[1;38;5;141m",  # Bold purple
            6: "\033[1;38;5;198m",  # Bold pink
        }
        reset = "\033[0m"
        style = styles.get(level, "\033[1m")
        inline = self._render_inline(content)
        prefix = "#" * level + " "
        return f"{style}{prefix}{inline}{reset}"

    def _render_code_block(self, block: str) -> str:
        """Render a fenced code block with syntax highlighting."""
        # Extract language and code
        match = re.match(r"```(\w*)\n([\s\S]*?)```", block)
        if not match:
            return block

        language = match.group(1) or "text"
        code = match.group(2)

        highlighted = self._highlighter.highlight(code, language, line_numbers=True)
        return highlighted

    def _render_table(self, block: str) -> str:
        """Render a markdown table with alignment."""
        lines = block.strip().split("\n")
        rows: list[list[str]] = []
        alignments: list[str] = []
        header_row: list[str] | None = None

        for line in lines:
            line = line.strip()
            if not line or not line.startswith("|"):
                continue

            cells = [c.strip() for c in line.strip("|").split("|")]

            # Check if this is a separator row
            if all(re.match(r"^:?-+:?$", c) for c in cells):
                alignments = []
                for cell in cells:
                    if cell.startswith(":") and cell.endswith(":"):
                        alignments.append("center")
                    elif cell.endswith(":"):
                        alignments.append("right")
                    else:
                        alignments.append("left")
                continue

            if header_row is None:
                header_row = cells
            else:
                rows.append(cells)

        if not header_row:
            return block

        # Calculate column widths
        col_widths = [len(h) for h in header_row]
        for row in rows:
            for i, cell in enumerate(row):
                if i < len(col_widths):
                    col_widths[i] = max(col_widths[i], len(cell))

        # Render header
        header_cells = []
        for i, cell in enumerate(header_row):
            width = col_widths[i] if i < len(col_widths) else len(cell)
            header_cells.append(f" \033[1;38;5;117m{cell:<{width}}\033[0m ")
        result = "|" + "|".join(header_cells) + "|"

        # Render separator
        sep_parts = []
        for i, width in enumerate(col_widths):
            align = alignments[i] if i < len(alignments) else "left"
            if align == "center":
                sep_parts.append(":" + "-" * (width + 1) + ":")
            elif align == "right":
                sep_parts.append("-" * (width + 1) + ":")
            else:
                sep_parts.append(":" + "-" * (width + 1))
        result += "\n|" + "|".join(sep_parts) + "|"

        # Render data rows
        for row in rows:
            row_cells = []
            for i in range(len(header_row)):
                cell = row[i] if i < len(row) else ""
                width = col_widths[i] if i < len(col_widths) else len(cell)
                rendered_cell = self._render_inline(cell)
                # Calculate padding based on visible vs rendered length
                visible_len = len(re.sub(r"\033\[[0-9;]*m", "", rendered_cell))
                padding = width - visible_len
                row_cells.append(f" {rendered_cell}{' ' * max(0, padding)} ")
            result += "\n|" + "|".join(row_cells) + "|"

        return result

    def _render_task_list(self, block: str) -> str:
        """Render a task list with checkbox styling."""
        lines = block.strip().split("\n")
        result_lines = []
        for line in lines:
            match = re.match(r"^(\s*)[-*+]\s+\[([ xX])\]\s+(.+)$", line)
            if match:
                indent = match.group(1)
                checked = match.group(2).lower() == "x"
                content = match.group(3)
                checkbox = "\033[38;5;84m✓\033[0m" if checked else "\033[38;5;67m○\033[0m"
                rendered_content = self._render_inline(content)
                if checked:
                    rendered_content = f"\033[38;5;67m{rendered_content}\033[0m"
                result_lines.append(f"{indent}{checkbox} {rendered_content}")
            else:
                result_lines.append(self._render_inline(line))
        return "\n".join(result_lines)

    def _render_blockquote(self, block: str) -> str:
        """Render a blockquote with styled prefix."""
        lines = block.strip().split("\n")
        result_lines = []
        for line in lines:
            content = re.sub(r"^>\s*", "", line)
            rendered = self._render_inline(content)
            result_lines.append(f"\033[38;5;67m│\033[0m {rendered}")
        return "\n".join(result_lines)

    def _render_list(self, block: str, ordered: bool = False) -> str:
        """Render a list (ordered or unordered)."""
        lines = block.strip().split("\n")
        result_lines = []
        counter = 1
        for line in lines:
            if ordered:
                match = re.match(r"^(\s*)\d+\.\s+(.+)$", line)
                if match:
                    indent = match.group(1)
                    content = match.group(2)
                    marker = f"\033[38;5;215m{counter}.\033[0m"
                    counter += 1
                else:
                    indent = ""
                    content = line
                    marker = "  "
            else:
                match = re.match(r"^(\s*)[-*+]\s+(.+)$", line)
                if match:
                    indent = match.group(1)
                    content = match.group(2)
                    marker = f"\033[38;5;84m•\033[0m"
                else:
                    indent = ""
                    content = line
                    marker = " "

            rendered = self._render_inline(content)
            result_lines.append(f"{indent}{marker} {rendered}")
        return "\n".join(result_lines)

    def _render_horizontal_rule(self) -> str:
        """Render a horizontal rule."""
        return "\033[38;5;67m" + "─" * 60 + "\033[0m"

    def _render_paragraph(self, block: str) -> str:
        """Render a paragraph with inline formatting."""
        return self._render_inline(block)

    def _render_inline(self, text: str) -> str:
        """Render inline markdown elements.

        Handles: bold, italic, code, links, images, math, footnote refs.
        """
        # Inline code
        text = re.sub(
            r"`([^`]+)`",
            r"\033[38;5;186m\1\033[0m",
            text,
        )

        # Bold
        text = re.sub(
            r"\*\*(.+?)\*\*",
            r"\033[1m\1\033[0m",
            text,
        )

        # Italic
        text = re.sub(
            r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)",
            r"\033[3m\1\033[0m",
            text,
        )

        # Strikethrough
        text = re.sub(
            r"~~(.+?)~~",
            r"\033[9m\1\033[0m",
            text,
        )

        # Images (render as [img: alt])
        text = re.sub(
            r"!\[([^\]]*)\]\(([^)]+)\)",
            r"\033[38;5;141m[img: \1]\033[0m",
            text,
        )

        # Links
        text = re.sub(
            r"\[([^\]]+)\]\(([^)]+)\)",
            r"\033[38;5;117m\1\033[38;5;67m (\2)\033[0m",
            text,
        )

        # Inline math (LaTeX)
        text = re.sub(
            r"\$(.+?)\$",
            r"\033[38;5;141m𝑓(\1)\033[0m",
            text,
        )

        # Display math
        text = re.sub(
            r"\$\$(.+?)\$\$",
            r"\033[38;5;141m𝑓𝑓(\1)\033[0m",
            text,
            flags=re.DOTALL,
        )

        # Footnote references
        def _replace_footnote(match: re.Match[str]) -> str:
            label = match.group(1)
            self._footnote_refs.append((label, len(self._footnote_refs) + 1))
            idx = len(self._footnote_refs)
            return f"\033[38;5;215m[{idx}]\033[0m"

        text = re.sub(r"\[\^(\w+)\]", _replace_footnote, text)

        return text

    def _render_footnotes(self) -> str:
        """Render the footnotes section."""
        lines = ["\033[1;38;5;67m─── Footnotes ───\033[0m"]
        for label, idx in self._footnote_refs:
            content = self._footnotes.get(label, "???")
            rendered = self._render_inline(content)
            lines.append(f"  \033[38;5;215m{idx}.\033[0m {rendered}")
        return "\n".join(lines)
