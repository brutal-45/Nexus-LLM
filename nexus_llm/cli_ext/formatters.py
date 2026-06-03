"""
CLI Output Formatters for Nexus-LLM

Provides formatters for rendering CLI output in different formats:
table, JSON, and YAML. Also includes a unified format_output()
function that selects the appropriate formatter.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Union

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class OutputFormatter(ABC):
    """Abstract base class for output formatters."""

    @abstractmethod
    def format(self, data: Any, **kwargs: Any) -> str:
        """Format data for output.

        Args:
            data: The data to format.
            **kwargs: Formatter-specific options.

        Returns:
            Formatted string.
        """

    def print(self, data: Any, **kwargs: Any) -> None:
        """Format and print data to stdout."""
        print(self.format(data, **kwargs))


# ---------------------------------------------------------------------------
# Table Formatter
# ---------------------------------------------------------------------------

class TableFormatter(OutputFormatter):
    """Formats data as an ASCII table with aligned columns.

    Supports:
    - List of dicts (each dict is a row)
    - List of lists (each list is a row)
    - Custom column headers
    - Custom column widths
    - Optional borders and separators
    """

    DEFAULT_PADDING = 2
    DEFAULT_MAX_COL_WIDTH = 50

    def format(
        self,
        data: Any,
        *,
        headers: Optional[List[str]] = None,
        max_col_width: int = DEFAULT_MAX_COL_WIDTH,
        padding: int = DEFAULT_PADDING,
        show_borders: bool = False,
        **kwargs: Any,
    ) -> str:
        """Format data as a table.

        Args:
            data: List of dicts or list of lists.
            headers: Optional column headers. Auto-detected from data.
            max_col_width: Maximum width per column.
            padding: Spaces between columns.
            show_borders: Whether to draw border lines.

        Returns:
            Formatted table string.
        """
        if not data:
            return "(no data)"

        # Normalize to rows
        if isinstance(data[0], dict):
            if headers is None:
                headers = list(data[0].keys())
            rows = [[str(row.get(h, "")) for h in headers] for row in data]
        elif isinstance(data[0], (list, tuple)):
            rows = [[str(cell) for cell in row] for row in data]
            if headers is None:
                headers = [f"Col{i}" for i in range(len(rows[0]))]
        else:
            # Simple list
            headers = headers or ["Value"]
            rows = [[str(item)] for item in data]

        # Truncate long values
        rows = [
            [
                self._truncate(cell, max_col_width) for cell in row
            ]
            for row in rows
        ]

        # Calculate column widths
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                if i < len(col_widths):
                    col_widths[i] = max(col_widths[i], len(cell))

        pad = " " * padding

        # Build output
        lines: List[str] = []

        if show_borders:
            lines.append(self._separator(col_widths, padding))

        # Header
        header_line = pad.join(
            h.ljust(col_widths[i]) for i, h in enumerate(headers)
        )
        lines.append(header_line)

        if show_borders:
            lines.append(self._separator(col_widths, padding))
        else:
            # Underline header
            lines.append(pad.join(
                "-" * col_widths[i] for i in range(len(headers))
            ))

        # Data rows
        for row in rows:
            line = pad.join(
                row[i].ljust(col_widths[i]) if i < len(row) else " " * col_widths[i]
                for i in range(len(headers))
            )
            lines.append(line)

        if show_borders:
            lines.append(self._separator(col_widths, padding))

        return "\n".join(lines)

    @staticmethod
    def _truncate(text: str, max_width: int) -> str:
        """Truncate text with ellipsis if too long."""
        if len(text) <= max_width:
            return text
        return text[: max_width - 3] + "..."

    @staticmethod
    def _separator(col_widths: List[int], padding: int) -> str:
        """Create a horizontal separator line."""
        pad_sep = "-" * padding
        return "+" + pad_sep.join(
            "-" * w for w in col_widths
        ).replace(pad_sep, "-" * padding) + "+" if padding > 0 else ""
        # Simplified separator
        inner = ("-" * padding).join("-" * w for w in col_widths)
        return f"+{inner}+"


# ---------------------------------------------------------------------------
# JSON Formatter
# ---------------------------------------------------------------------------

class JsonFormatter(OutputFormatter):
    """Formats data as pretty-printed JSON."""

    def format(
        self,
        data: Any,
        *,
        indent: int = 2,
        sort_keys: bool = False,
        ensure_ascii: bool = False,
        **kwargs: Any,
    ) -> str:
        """Format data as JSON.

        Args:
            data: Any JSON-serializable data.
            indent: JSON indentation level.
            sort_keys: Whether to sort dictionary keys.
            ensure_ascii: Whether to escape non-ASCII characters.

        Returns:
            JSON string.
        """
        return json.dumps(
            data,
            indent=indent,
            sort_keys=sort_keys,
            ensure_ascii=ensure_ascii,
            default=str,
        )


# ---------------------------------------------------------------------------
# YAML Formatter
# ---------------------------------------------------------------------------

class YamlFormatter(OutputFormatter):
    """Formats data as YAML."""

    def format(
        self,
        data: Any,
        *,
        default_flow_style: bool = False,
        allow_unicode: bool = True,
        sort_keys: bool = False,
        **kwargs: Any,
    ) -> str:
        """Format data as YAML.

        Args:
            data: Any YAML-serializable data.
            default_flow_style: Use flow style for collections.
            allow_unicode: Allow Unicode characters.
            sort_keys: Whether to sort dictionary keys.

        Returns:
            YAML string.

        Raises:
            ImportError: If PyYAML is not installed.
        """
        if not HAS_YAML:
            return "# YAML output requires PyYAML\n" + json.dumps(
                data, indent=2, default=str
            )

        return yaml.dump(
            data,
            default_flow_style=default_flow_style,
            allow_unicode=allow_unicode,
            sort_keys=sort_keys,
        )


# ---------------------------------------------------------------------------
# Unified format function
# ---------------------------------------------------------------------------

_FORMATTERS: Dict[str, OutputFormatter] = {
    "table": TableFormatter(),
    "json": JsonFormatter(),
    "yaml": YamlFormatter(),
}


def format_output(
    data: Any,
    fmt: str = "table",
    **kwargs: Any,
) -> str:
    """Format data using the specified format.

    Args:
        data: The data to format.
        fmt: Format name ('table', 'json', or 'yaml').
        **kwargs: Additional arguments passed to the formatter.

    Returns:
        Formatted string.

    Raises:
        ValueError: If the format is not supported.
    """
    formatter = _FORMATTERS.get(fmt)
    if formatter is None:
        supported = ", ".join(_FORMATTERS.keys())
        raise ValueError(
            f"Unsupported format '{fmt}'. Supported: {supported}"
        )
    return formatter.format(data, **kwargs)
