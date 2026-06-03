"""Format conversion: bytes to human readable, seconds to duration, number formatting."""

import math
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def bytes_to_human(
    num_bytes: float,
    precision: int = 2,
    binary: bool = False,
) -> str:
    """Convert bytes to a human-readable string.

    Args:
        num_bytes: Number of bytes.
        precision: Decimal places.
        binary: If True, use binary prefixes (KiB, MiB) instead of decimal (KB, MB).

    Returns:
        Human-readable string (e.g., "1.50 GB", "512.00 MiB").
    """
    if num_bytes < 0:
        return f"-{bytes_to_human(-num_bytes, precision, binary)}"

    if binary:
        units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB"]
        base = 1024.0
    else:
        units = ["B", "KB", "MB", "GB", "TB", "PB", "EB"]
        base = 1000.0

    if num_bytes == 0:
        return f"0 {units[0]}"

    magnitude = 0
    value = float(num_bytes)
    while value >= base and magnitude < len(units) - 1:
        value /= base
        magnitude += 1

    return f"{value:.{precision}f} {units[magnitude]}"


def seconds_to_duration(
    seconds: float,
    precision: int = 0,
    compact: bool = False,
) -> str:
    """Convert seconds to a human-readable duration string.

    Args:
        seconds: Number of seconds.
        precision: Decimal places for the seconds component.
        compact: If True, use compact format (1h30m instead of 1h 30m).

    Returns:
        Human-readable duration string.
    """
    if seconds < 0:
        return f"-{seconds_to_duration(-seconds, precision, compact)}"

    if math.isinf(seconds):
        return "∞"

    days = int(seconds // 86400)
    hours = int((seconds % 86400) // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60

    separator = "" if compact else " "

    if days > 0:
        if precision > 0:
            return f"{days}d{separator}{hours}h{separator}{minutes}m{separator}{secs:.{precision}f}s"
        return f"{days}d{separator}{hours}h{separator}{minutes}m"
    elif hours > 0:
        if precision > 0:
            return f"{hours}h{separator}{minutes}m{separator}{secs:.{precision}f}s"
        return f"{hours}h{separator}{minutes}m"
    elif minutes > 0:
        if precision > 0:
            return f"{minutes}m{separator}{secs:.{precision}f}s"
        return f"{minutes}m{separator}{int(secs)}s"
    else:
        if secs < 0.001:
            return f"{secs * 1_000_000:.0f}μs"
        elif secs < 1.0:
            return f"{secs * 1000:.0f}ms"
        elif precision > 0:
            return f"{secs:.{precision}f}s"
        else:
            return f"{secs:.1f}s"


def format_number(
    value: float,
    precision: int = 2,
    use_si: bool = False,
) -> str:
    """Format a number with optional SI prefixes.

    Args:
        value: Numeric value.
        precision: Decimal places.
        use_si: If True, use SI prefixes (K, M, B) for large numbers.

    Returns:
        Formatted number string.
    """
    if use_si:
        abs_value = abs(value)
        if abs_value >= 1e9:
            return f"{value / 1e9:.{precision}f}B"
        elif abs_value >= 1e6:
            return f"{value / 1e6:.{precision}f}M"
        elif abs_value >= 1e3:
            return f"{value / 1e3:.{precision}f}K"

    if value == int(value) and abs(value) < 1e15:
        return f"{int(value):,}"
    return f"{value:,.{precision}f}"


def format_percentage(
    value: float,
    total: float,
    precision: int = 1,
) -> str:
    """Format a value as a percentage of a total.

    Args:
        value: Numerator value.
        total: Denominator value.
        precision: Decimal places.

    Returns:
        Formatted percentage string (e.g., "75.3%").
    """
    if total == 0:
        return "0.0%"
    pct = (value / total) * 100
    return f"{pct:.{precision}f}%"


def format_rate(
    value: float,
    unit: str = "items",
    time_unit: str = "s",
    precision: int = 2,
) -> str:
    """Format a rate value (value per time unit).

    Args:
        value: Rate value.
        unit: Unit of the items (e.g., "tokens", "items").
        time_unit: Time unit (e.g., "s", "min").
        precision: Decimal places.

    Returns:
        Formatted rate string (e.g., "1,234.56 tokens/s").
    """
    if value >= 1e6:
        return f"{value / 1e6:.{precision}f}M {unit}/{time_unit}"
    elif value >= 1e3:
        return f"{value / 1e3:.{precision}f}K {unit}/{time_unit}"
    return f"{value:.{precision}f} {unit}/{time_unit}"


def format_ratio(
    numerator: float,
    denominator: float,
    precision: int = 2,
) -> str:
    """Format a ratio with colon notation.

    Args:
        numerator: Top of the ratio.
        denominator: Bottom of the ratio.
        precision: Decimal places for the decimal representation.

    Returns:
        Formatted ratio string (e.g., "3:1 (3.00)").
    """
    if denominator == 0:
        return "∞"
    ratio = numerator / denominator
    return f"{ratio:.{precision}f}"


def format_table(
    headers: list,
    rows: list,
    col_widths: Optional[list] = None,
    alignment: Optional[list] = None,
) -> str:
    """Format data as a simple text table.

    Args:
        headers: Column header strings.
        rows: List of row tuples/lists.
        col_widths: Optional explicit column widths.
        alignment: Optional alignment per column ('left', 'right', 'center').

    Returns:
        Formatted table string.
    """
    if not rows:
        return ""

    num_cols = len(headers)
    if col_widths is None:
        col_widths = []
        for i in range(num_cols):
            max_width = len(str(headers[i]))
            for row in rows:
                if i < len(row):
                    max_width = max(max_width, len(str(row[i])))
            col_widths.append(max_width + 2)

    alignment = alignment or ["left"] * num_cols

    def format_cell(value, width, align):
        s = str(value)
        if align == "right":
            return s.rjust(width - 1)
        elif align == "center":
            return s.center(width)
        return s.ljust(width)

    lines = []

    header_line = ""
    for i, h in enumerate(headers):
        header_line += format_cell(h, col_widths[i], alignment[i])
    lines.append(header_line)

    separator = "-" * len(header_line)
    lines.append(separator)

    for row in rows:
        line = ""
        for i in range(num_cols):
            value = row[i] if i < len(row) else ""
            line += format_cell(value, col_widths[i], alignment[i])
        lines.append(line)

    return "\n".join(lines)
