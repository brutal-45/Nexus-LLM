"""Nexus-LLM CSV Processing Tool.

Provides CSV file reading, writing, filtering, and transformation
capabilities for tabular data processing.
"""

import csv
import io
import logging
from typing import Any, Dict, List, Optional

from nexus_llm.tools.base_tool import BaseTool, ToolParameter, ToolResult, ParameterType

logger = logging.getLogger(__name__)


class CsvTool(BaseTool):
    """Tool for CSV file processing.

    Supports reading, writing, filtering, and transforming CSV data.

    Example::

        tool = CsvTool()
        result = tool.run(operation="read", data="name,age\nAlice,30\nBob,25")
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name="csv", description="Process CSV data", **kwargs)

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="operation", type=ParameterType.STRING, description="Operation to perform",
                         required=True, choices=["read", "write", "filter", "transform", "headers"]),
            ToolParameter(name="data", type=ParameterType.STRING, description="CSV data string", required=True),
            ToolParameter(name="delimiter", type=ParameterType.STRING, description="CSV delimiter", required=False, default=","),
            ToolParameter(name="filter_column", type=ParameterType.STRING, description="Column to filter on", required=False),
            ToolParameter(name="filter_value", type=ParameterType.STRING, description="Value to match", required=False),
            ToolParameter(name="has_header", type=ParameterType.BOOLEAN, description="Whether CSV has headers", required=False, default=True),
        ]

    def execute(self, **kwargs: Any) -> ToolResult:
        operation = kwargs.get("operation", "")
        data = kwargs.get("data", "")
        delimiter = kwargs.get("delimiter", ",")
        has_header = kwargs.get("has_header", True)

        if not operation:
            return ToolResult(success=False, error="No operation specified")
        if not data and operation != "write":
            return ToolResult(success=False, error="No data provided")

        try:
            if operation == "read":
                return self._read(data, delimiter, has_header)
            elif operation == "write":
                rows = kwargs.get("rows", [])
                return self._write(rows, delimiter, has_header)
            elif operation == "filter":
                column = kwargs.get("filter_column", "")
                value = kwargs.get("filter_value", "")
                return self._filter(data, delimiter, has_header, column, value)
            elif operation == "transform":
                return self._transform(data, delimiter, has_header)
            elif operation == "headers":
                return self._headers(data, delimiter)
            else:
                return ToolResult(success=False, error=f"Unknown operation: {operation}")
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))

    def _read(self, data: str, delimiter: str, has_header: bool) -> ToolResult:
        """Read CSV data into a list of dictionaries."""
        reader = csv.reader(io.StringIO(data), delimiter=delimiter)
        rows = list(reader)
        if not rows:
            return ToolResult(success=True, output=[])

        if has_header:
            headers = rows[0]
            result = [dict(zip(headers, row)) for row in rows[1:]]
        else:
            result = rows

        return ToolResult(success=True, output=result, metadata={"row_count": len(result) if has_header else len(rows)})

    def _write(self, rows: List, delimiter: str, has_header: bool) -> ToolResult:
        """Write rows to CSV format."""
        output = io.StringIO()
        writer = csv.writer(output, delimiter=delimiter)
        for row in rows:
            if isinstance(row, dict):
                writer.writerow(row.values())
            else:
                writer.writerow(row)
        return ToolResult(success=True, output=output.getvalue())

    def _filter(self, data: str, delimiter: str, has_header: bool, column: str, value: str) -> ToolResult:
        """Filter CSV rows by column value."""
        read_result = self._read(data, delimiter, has_header)
        if not read_result.success:
            return read_result

        rows = read_result.output
        if not has_header:
            return ToolResult(success=False, error="Filtering requires headers")

        filtered = [row for row in rows if str(row.get(column, "")) == value]
        return ToolResult(success=True, output=filtered, metadata={"filtered_count": len(filtered)})

    def _transform(self, data: str, delimiter: str, has_header: bool) -> ToolResult:
        """Transform CSV data (uppercase headers)."""
        read_result = self._read(data, delimiter, has_header)
        if not read_result.success:
            return read_result

        rows = read_result.output
        if has_header:
            transformed = [{k.upper(): v for k, v in row.items()} for row in rows]
        else:
            transformed = [[str(cell).upper() for cell in row] for row in rows]

        return ToolResult(success=True, output=transformed)

    def _headers(self, data: str, delimiter: str) -> ToolResult:
        """Extract CSV headers."""
        reader = csv.reader(io.StringIO(data), delimiter=delimiter)
        headers = next(reader, [])
        return ToolResult(success=True, output=headers)
