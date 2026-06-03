"""
Nexus-LLM Table Display Module

Provides rich table display with sorting, filtering, and styling
using the Rich library with a fallback to plain text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Sequence

try:
    from rich.console import Console
    from rich.table import Table as RichTable
    from rich.box import ROUNDED, MINIMAL, HEAVY, SIMPLE

    HAS_RICH = True
except ImportError:
    HAS_RICH = False


class SortOrder(str, Enum):
    """Sort order for table columns."""
    ASCENDING = "asc"
    DESCENDING = "desc"


class FilterOp(str, Enum):
    """Filter operations for table data."""
    EQ = "eq"
    NE = "ne"
    CONTAINS = "contains"
    STARTSWITH = "startswith"
    ENDSWITH = "endswith"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    REGEX = "regex"


@dataclass
class Column:
    """Definition of a table column."""
    name: str
    header: str = ""
    width: int | None = None
    min_width: int = 0
    max_width: int | None = None
    align: str = "left"
    style: str = ""
    sortable: bool = True
    filterable: bool = True
    format_func: Callable[[Any], str] | None = None

    def __post_init__(self) -> None:
        if not self.header:
            self.header = self.name.replace("_", " ").title()

    def format_value(self, value: Any) -> str:
        """Format a cell value for display.

        Args:
            value: The raw cell value.

        Returns:
            Formatted string.
        """
        if self.format_func:
            return self.format_func(value)
        if value is None:
            return ""
        return str(value)


@dataclass
class TableFilter:
    """A filter applied to a table column."""
    column: str
    op: FilterOp = FilterOp.CONTAINS
    value: Any = None

    def matches(self, cell_value: Any) -> bool:
        """Check if a cell value matches this filter.

        Args:
            cell_value: The cell value to test.

        Returns:
            True if the value matches the filter condition.
        """
        str_val = str(cell_value) if cell_value is not None else ""

        if self.op == FilterOp.EQ:
            return str_val == str(self.value)
        elif self.op == FilterOp.NE:
            return str_val != str(self.value)
        elif self.op == FilterOp.CONTAINS:
            return str(self.value).lower() in str_val.lower()
        elif self.op == FilterOp.STARTSWITH:
            return str_val.lower().startswith(str(self.value).lower())
        elif self.op == FilterOp.ENDSWITH:
            return str_val.lower().endswith(str(self.value).lower())
        elif self.op == FilterOp.GT:
            try:
                return float(str_val) > float(self.value)
            except (ValueError, TypeError):
                return False
        elif self.op == FilterOp.GTE:
            try:
                return float(str_val) >= float(self.value)
            except (ValueError, TypeError):
                return False
        elif self.op == FilterOp.LT:
            try:
                return float(str_val) < float(self.value)
            except (ValueError, TypeError):
                return False
        elif self.op == FilterOp.LTE:
            try:
                return float(str_val) <= float(self.value)
            except (ValueError, TypeError):
                return False
        elif self.op == FilterOp.REGEX:
            try:
                return bool(re.search(str(self.value), str_val))
            except re.error:
                return False
        return True


class TableBuilder:
    """Builder for rich, sortable, filterable terminal tables.

    Provides a fluent API for constructing tables with column definitions,
    data rows, sorting, filtering, and styled output.
    """

    def __init__(self, title: str = "", style: str = "cyan") -> None:
        self._title = title
        self._style = style
        self._columns: list[Column] = []
        self._rows: list[dict[str, Any]] = []
        self._filters: list[TableFilter] = []
        self._sort_column: str | None = None
        self._sort_order: SortOrder = SortOrder.ASCENDING
        self._show_lines: bool = False
        self._row_styles: list[str] = []

    def add_column(
        self,
        name: str,
        header: str = "",
        width: int | None = None,
        align: str = "left",
        style: str = "",
        sortable: bool = True,
        filterable: bool = True,
        format_func: Callable[[Any], str] | None = None,
    ) -> TableBuilder:
        """Add a column definition.

        Args:
            name: Column key (used for data access).
            header: Display header text.
            width: Fixed column width.
            align: Text alignment - 'left', 'center', 'right'.
            style: Rich style string for the column.
            sortable: Whether the column can be sorted.
            filterable: Whether the column can be filtered.
            format_func: Optional function to format cell values.

        Returns:
            Self for chaining.
        """
        self._columns.append(Column(
            name=name,
            header=header,
            width=width,
            align=align,
            style=style,
            sortable=sortable,
            filterable=filterable,
            format_func=format_func,
        ))
        return self

    def add_row(self, **kwargs: Any) -> TableBuilder:
        """Add a data row.

        Args:
            **kwargs: Column name to value mapping.

        Returns:
            Self for chaining.
        """
        self._rows.append(kwargs)
        return self

    def add_rows(self, rows: Sequence[dict[str, Any]]) -> TableBuilder:
        """Add multiple data rows.

        Args:
            rows: Sequence of row dictionaries.

        Returns:
            Self for chaining.
        """
        self._rows.extend(rows)
        return self

    def set_data(self, rows: Sequence[dict[str, Any]]) -> TableBuilder:
        """Replace all data rows.

        Args:
            rows: New row data.

        Returns:
            Self for chaining.
        """
        self._rows = list(rows)
        return self

    def sort_by(self, column: str, order: SortOrder = SortOrder.ASCENDING) -> TableBuilder:
        """Set the sort column and order.

        Args:
            column: Column name to sort by.
            order: Sort order.

        Returns:
            Self for chaining.
        """
        self._sort_column = column
        self._sort_order = order
        return self

    def filter(
        self,
        column: str,
        op: FilterOp = FilterOp.CONTAINS,
        value: Any = None,
    ) -> TableBuilder:
        """Add a filter condition.

        Args:
            column: Column name to filter on.
            op: Filter operation.
            value: Value to compare against.

        Returns:
            Self for chaining.
        """
        self._filters.append(TableFilter(column=column, op=op, value=value))
        return self

    def show_lines(self, show: bool = True) -> TableBuilder:
        """Set whether to show row separator lines.

        Args:
            show: Whether to show lines.

        Returns:
            Self for chaining.
        """
        self._show_lines = show
        return self

    def _apply_filters(self) -> list[dict[str, Any]]:
        """Apply all filters to the data rows.

        Returns:
            Filtered list of row dictionaries.
        """
        if not self._filters:
            return list(self._rows)

        result = []
        for row in self._rows:
            matches_all = True
            for f in self._filters:
                cell_value = row.get(f.column)
                if not f.matches(cell_value):
                    matches_all = False
                    break
            if matches_all:
                result.append(row)
        return result

    def _apply_sort(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Apply sorting to the data rows.

        Args:
            rows: Data rows to sort.

        Returns:
            Sorted list of row dictionaries.
        """
        if not self._sort_column:
            return rows

        reverse = self._sort_order == SortOrder.DESCENDING

        def sort_key(row: dict[str, Any]) -> Any:
            val = row.get(self._sort_column)
            if val is None:
                return ("", "")
            if isinstance(val, (int, float)):
                return (0, val)
            return (1, str(val).lower())

        return sorted(rows, key=sort_key, reverse=reverse)

    def get_processed_rows(self) -> list[dict[str, Any]]:
        """Get rows after applying filters and sorting.

        Returns:
            Processed list of row dictionaries.
        """
        filtered = self._apply_filters()
        return self._apply_sort(filtered)

    def render(self) -> str:
        """Render the table to a string.

        Returns:
            ANSI-formatted table string.
        """
        rows = self.get_processed_rows()

        if HAS_RICH:
            return self._render_rich(rows)
        return self._render_plain(rows)

    def _render_rich(self, rows: list[dict[str, Any]]) -> str:
        """Render using Rich library."""
        table = RichTable(
            title=self._title or None,
            style=self._style,
            show_lines=self._show_lines,
            box=ROUNDED,
            header_style=f"bold {self._style}",
        )

        for col in self._columns:
            table.add_column(
                col.header,
                width=col.width,
                min_width=col.min_width or None,
                max_width=col.max_width or None,
                justify=col.align,
                style=col.style or None,
            )

        for row in rows:
            cells = []
            for col in self._columns:
                value = row.get(col.name)
                cells.append(col.format_value(value))
            table.add_row(*cells)

        from io import StringIO
        console = Console(file=StringIO(), force_terminal=True)
        console.print(table)
        return console.file.getvalue() if hasattr(console.file, 'getvalue') else ""

    def _render_plain(self, rows: list[dict[str, Any]]) -> str:
        """Render as plain text table."""
        if not self._columns:
            return ""

        headers = [col.header for col in self._columns]

        # Format all cells
        formatted_rows = []
        for row in rows:
            formatted_row = []
            for col in self._columns:
                value = row.get(col.name)
                formatted_row.append(col.format_value(value))
            formatted_rows.append(formatted_row)

        # Calculate column widths
        col_widths = [len(h) for h in headers]
        for frow in formatted_rows:
            for i, cell in enumerate(frow):
                if i < len(col_widths):
                    col_widths[i] = max(col_widths[i], len(cell))

        # Apply fixed widths
        for i, col in enumerate(self._columns):
            if col.width:
                col_widths[i] = col.width

        # Build output
        lines = []

        # Header
        header_cells = []
        for i, h in enumerate(headers):
            if i < len(col_widths):
                header_cells.append(f" {h:<{col_widths[i]}} ")
        lines.append("│".join(header_cells))

        # Separator
        sep_cells = []
        for w in col_widths:
            sep_cells.append("─" * (w + 2))
        lines.append("┼".join(sep_cells))

        # Data rows
        for frow in formatted_rows:
            row_cells = []
            for i, cell in enumerate(frow):
                w = col_widths[i] if i < len(col_widths) else len(cell)
                row_cells.append(f" {cell:<{w}} ")
            lines.append("│".join(row_cells))

        # Title
        if self._title:
            total_width = sum(col_widths) + len(col_widths) * 3 - 1
            title_line = f" {self._title:^{total_width}} "
            lines.insert(0, title_line)
            lines.insert(1, "─" * (total_width + 2))

        return "\n".join(lines)

    def print(self) -> None:
        """Print the table to the terminal."""
        output = self.render()
        if HAS_RICH:
            # Use console for proper Rich rendering
            console = Console()
            rows = self.get_processed_rows()
            table = RichTable(
                title=self._title or None,
                style=self._style,
                show_lines=self._show_lines,
                box=ROUNDED,
                header_style=f"bold {self._style}",
            )
            for col in self._columns:
                table.add_column(
                    col.header,
                    width=col.width,
                    justify=col.align,
                    style=col.style or None,
                )
            for row in rows:
                cells = [col.format_value(row.get(col.name)) for col in self._columns]
                table.add_row(*cells)
            console.print(table)
        else:
            print(output)

    @property
    def row_count(self) -> int:
        """Get the total number of rows."""
        return len(self._rows)

    @property
    def filtered_count(self) -> int:
        """Get the number of rows after filtering."""
        return len(self._apply_filters())
