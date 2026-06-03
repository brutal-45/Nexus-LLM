"""Nexus-LLM Response Formatter.

Provides formatting capabilities for LLM responses, supporting
multiple output formats including Markdown, HTML, plain text,
and structured data formats.
"""

import html
import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class OutputFormat(Enum):
    """Supported output formats."""

    PLAIN = "plain"
    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"


class ResponseFormatter:
    """Formats LLM responses into various output formats.

    Example::

        formatter = ResponseFormatter()
        md = formatter.format("Hello world", output_format=OutputFormat.MARKDOWN)
        html = formatter.format("Hello world", output_format=OutputFormat.HTML)
    """

    def __init__(self, default_format: OutputFormat = OutputFormat.PLAIN) -> None:
        self._default_format = default_format
        logger.debug("ResponseFormatter initialized with default: %s", default_format.value)

    def format(
        self,
        text: str,
        output_format: Optional[OutputFormat] = None,
        title: str = "",
        **kwargs: Any,
    ) -> str:
        """Format text into the specified output format.

        Args:
            text: The text to format.
            output_format: Desired output format (uses default if not specified).
            title: Optional title/heading.
            **kwargs: Additional format-specific options.

        Returns:
            Formatted text string.
        """
        fmt = output_format or self._default_format

        if fmt == OutputFormat.PLAIN:
            return self._format_plain(text, title)
        elif fmt == OutputFormat.MARKDOWN:
            return self._format_markdown(text, title)
        elif fmt == OutputFormat.HTML:
            return self._format_html(text, title, **kwargs)
        elif fmt == OutputFormat.JSON:
            return self._format_json(text, title)
        else:
            raise ValueError(f"Unsupported format: {fmt}")

    def _format_plain(self, text: str, title: str = "") -> str:
        """Format as plain text."""
        parts = []
        if title:
            parts.append(title)
            parts.append("=" * len(title))
            parts.append("")
        parts.append(text)
        return "\n".join(parts)

    def _format_markdown(self, text: str, title: str = "") -> str:
        """Format as Markdown."""
        parts = []
        if title:
            parts.append(f"# {title}")
            parts.append("")
        parts.append(text)
        return "\n".join(parts)

    def _format_html(self, text: str, title: str = "", **kwargs: Any) -> str:
        """Format as HTML."""
        escaped = html.escape(text)
        escaped = escaped.replace("\n", "<br>\n")
        parts = ["<div>"]
        if title:
            parts.append(f"<h2>{html.escape(title)}</h2>")
        parts.append(f"<p>{escaped}</p>")
        parts.append("</div>")
        return "\n".join(parts)

    def _format_json(self, text: str, title: str = "") -> str:
        """Format as JSON."""
        data = {"text": text}
        if title:
            data["title"] = title
        return json.dumps(data, indent=2)

    def format_table(self, headers: List[str], rows: List[List[Any]], output_format: Optional[OutputFormat] = None) -> str:
        """Format data as a table.

        Args:
            headers: Column headers.
            rows: Table data rows.
            output_format: Desired output format.

        Returns:
            Formatted table string.
        """
        fmt = output_format or self._default_format

        if fmt == OutputFormat.MARKDOWN:
            return self._markdown_table(headers, rows)
        elif fmt == OutputFormat.HTML:
            return self._html_table(headers, rows)
        elif fmt == OutputFormat.PLAIN:
            return self._plain_table(headers, rows)
        else:
            return self._plain_table(headers, rows)

    def _markdown_table(self, headers: List[str], rows: List[List[Any]]) -> str:
        """Format as Markdown table."""
        header_line = "| " + " | ".join(str(h) for h in headers) + " |"
        separator = "| " + " | ".join("---" for _ in headers) + " |"
        data_lines = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
        return "\n".join([header_line, separator] + data_lines)

    def _html_table(self, headers: List[str], rows: List[List[Any]]) -> str:
        """Format as HTML table."""
        header_cells = "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
        header_row = f"<tr>{header_cells}</tr>"
        data_rows = []
        for row in rows:
            cells = "".join(f"<td>{html.escape(str(cell))}</td>" for cell in row)
            data_rows.append(f"<tr>{cells}</tr>")
        return f"<table>\n<thead>{header_row}</thead>\n<tbody>{''.join(data_rows)}</tbody>\n</table>"

    def _plain_table(self, headers: List[str], rows: List[List[Any]]) -> str:
        """Format as plain text table."""
        all_rows = [headers] + rows
        col_widths = []
        for i, header in enumerate(headers):
            max_width = len(str(header))
            for row in rows:
                if i < len(row):
                    max_width = max(max_width, len(str(row[i])))
            col_widths.append(max_width + 2)

        lines = []
        for row in all_rows:
            parts = []
            for i, cell in enumerate(row):
                parts.append(str(cell).ljust(col_widths[i] if i < len(col_widths) else 10))
            lines.append("".join(parts))
        return "\n".join(lines)

    def format_list(self, items: List[str], ordered: bool = False, output_format: Optional[OutputFormat] = None) -> str:
        """Format a list of items.

        Args:
            items: List items.
            ordered: Whether to use ordered numbering.
            output_format: Desired output format.

        Returns:
            Formatted list string.
        """
        fmt = output_format or self._default_format
        lines = []
        for i, item in enumerate(items, 1):
            if fmt == OutputFormat.MARKDOWN:
                prefix = f"{i}." if ordered else "-"
            elif fmt == OutputFormat.HTML:
                prefix = ""
            else:
                prefix = f"{i}." if ordered else "*"
            lines.append(f"{prefix} {item}")

        if fmt == OutputFormat.HTML:
            tag = "ol" if ordered else "ul"
            inner = "".join(f"<li>{html.escape(item)}</li>" for item in items)
            return f"<{tag}>{inner}</{tag}>"

        return "\n".join(lines)
