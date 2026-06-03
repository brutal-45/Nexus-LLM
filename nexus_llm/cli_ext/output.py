"""Output formatting for CLI commands."""

from __future__ import annotations

import csv
import io
import json
import os
import shutil
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union


class OutputFormat(Enum):
    """Supported output formats for CLI results."""

    table = "table"
    json = "json"
    yaml = "yaml"
    csv = "csv"
    markdown = "markdown"

    def __str__(self) -> str:  # noqa: D105
        return self.value

    @classmethod
    def from_string(cls, name: str) -> OutputFormat:
        """Look up a format by name (case-insensitive).

        Raises:
            ValueError: If the format is unknown.
        """
        try:
            return cls(name.lower())
        except ValueError:
            valid = ", ".join(f.value for f in cls)
            raise ValueError(
                f"Unknown output format '{name}'. Valid: {valid}"
            ) from None


# ------------------------------------------------------------------
# Terminal helpers
# ------------------------------------------------------------------

def get_terminal_width(default: int = 80) -> int:
    """Return the current terminal width, falling back to *default*."""
    try:
        return shutil.get_terminal_size((default, 24)).columns
    except (AttributeError, ValueError):
        return default


# ------------------------------------------------------------------
# Public formatting API
# ------------------------------------------------------------------

def format_output(
    data: Any,
    fmt: OutputFormat = OutputFormat.table,
    columns: Optional[Sequence[str]] = None,
) -> str:
    """Format *data* into the requested output format.

    Args:
        data: The data to format.  For tabular formats, this should be
            a list of dicts or a list of sequences.
        fmt: The target output format.
        columns: Optional column names (used for table / csv / markdown).

    Returns:
        The formatted string.
    """
    formatter = _FORMATTERS.get(fmt)
    if formatter is None:
        raise ValueError(f"No formatter for format '{fmt}'.")
    return formatter(data, columns=columns)


# ------------------------------------------------------------------
# Individual formatters
# ------------------------------------------------------------------

def _format_json(data: Any, **kwargs: Any) -> str:
    """Format data as pretty-printed JSON."""
    return json.dumps(data, indent=2, sort_keys=True, default=str)


def _format_yaml(data: Any, **kwargs: Any) -> str:
    """Format data as a simple YAML string.

    This is a lightweight YAML formatter that does not require the
    ``pyyaml`` package.  It handles basic scalars, lists, and dicts.
    For complex structures, install ``pyyaml`` for full support.
    """
    try:
        import yaml  # type: ignore[import-untyped]
        return yaml.dump(data, default_flow_style=False, sort_keys=True)
    except ImportError:
        return _simple_yaml(data)


def _simple_yaml(data: Any, indent: int = 0) -> str:
    """Minimal YAML-like serializer as a fallback."""
    prefix = "  " * indent
    lines: list[str] = []

    if isinstance(data, dict):
        for key, val in sorted(data.items()):
            if isinstance(val, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.append(_simple_yaml(val, indent + 1))
            else:
                lines.append(f"{prefix}{key}: {_yaml_scalar(val)}")
    elif isinstance(data, (list, tuple)):
        for item in data:
            if isinstance(item, dict):
                first = True
                for key, val in sorted(item.items()):
                    if first:
                        lines.append(
                            f"{prefix}- {key}: {_yaml_scalar(val) if not isinstance(val, (dict, list)) else ''}"
                        )
                        if isinstance(val, (dict, list)):
                            lines.append(_simple_yaml(val, indent + 2))
                        first = False
                    else:
                        if isinstance(val, (dict, list)):
                            lines.append(f"{prefix}  {key}:")
                            lines.append(_simple_yaml(val, indent + 2))
                        else:
                            lines.append(f"{prefix}  {key}: {_yaml_scalar(val)}")
            else:
                lines.append(f"{prefix}- {_yaml_scalar(item)}")
    else:
        lines.append(f"{prefix}{_yaml_scalar(data)}")

    return "\n".join(lines)


def _yaml_scalar(val: Any) -> str:
    """Format a scalar for YAML output."""
    if val is None:
        return "null"
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, (int, float)):
        return str(val)
    s = str(val)
    # Quote strings that could be misinterpreted
    if any(c in s for c in (":", "#", "{", "}", "[", "]", ",", "&", "*", "?", "|", "-", "<", ">", "=", "!", "%", "@", "`")):
        return json.dumps(s)
    if s.lower() in ("true", "false", "null", "yes", "no", "on", "off"):
        return json.dumps(s)
    return s


def _format_csv(data: Any, columns: Optional[Sequence[str]] = None, **kwargs: Any) -> str:
    """Format tabular data as CSV."""
    buf = io.StringIO()
    rows = _normalize_rows(data)
    if not rows:
        return ""

    cols = columns or list(rows[0].keys())
    writer = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: str(v) for k, v in row.items()})
    return buf.getvalue().rstrip("\n")


def _format_table(data: Any, columns: Optional[Sequence[str]] = None, **kwargs: Any) -> str:
    """Format tabular data as a terminal-friendly table."""
    rows = _normalize_rows(data)
    if not rows:
        return "(no data)"

    cols = columns or list(rows[0].keys())
    term_width = get_terminal_width()

    # Calculate column widths
    col_widths: dict[str, int] = {c: len(c) for c in cols}
    for row in rows:
        for c in cols:
            col_widths[c] = max(col_widths[c], len(str(row.get(c, ""))))

    # Scale down if total width exceeds terminal
    total = sum(col_widths.values()) + 3 * (len(cols) - 1)
    if total > term_width - 4 and term_width > 20:
        scale = (term_width - 4 - 3 * (len(cols) - 1)) / sum(col_widths.values())
        col_widths = {c: max(int(w * scale), 4) for c, w in col_widths.items()}

    # Build header
    header = " | ".join(c.ljust(col_widths[c]) for c in cols)
    sep = "-+-".join("-" * col_widths[c] for c in cols)

    lines = [header, sep]
    for row in rows:
        line = " | ".join(
            str(row.get(c, "")).ljust(col_widths[c])[:col_widths[c]]
            for c in cols
        )
        lines.append(line)

    return "\n".join(lines)


def _format_markdown(data: Any, columns: Optional[Sequence[str]] = None, **kwargs: Any) -> str:
    """Format tabular data as a Markdown table."""
    rows = _normalize_rows(data)
    if not rows:
        return ""

    cols = columns or list(rows[0].keys())

    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"

    lines = [header, sep]
    for row in rows:
        line = "| " + " | ".join(str(row.get(c, "")) for c in cols) + " |"
        lines.append(line)

    return "\n".join(lines)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _normalize_rows(data: Any) -> List[Dict[str, Any]]:
    """Convert data into a list of dicts for tabular formatters."""
    if isinstance(data, dict):
        # Single row
        return [data]
    if isinstance(data, (list, tuple)):
        result: list[Dict[str, Any]] = []
        for item in data:
            if isinstance(item, dict):
                result.append(item)
            elif isinstance(item, (list, tuple)):
                result.append({str(i): v for i, v in enumerate(item)})
            else:
                result.append({"value": item})
        return result
    return [{"value": data}]


# Formatter dispatch table
_FORMATTERS = {
    OutputFormat.table: _format_table,
    OutputFormat.json: _format_json,
    OutputFormat.yaml: _format_yaml,
    OutputFormat.csv: _format_csv,
    OutputFormat.markdown: _format_markdown,
}
