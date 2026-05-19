"""Nexus-LLM SQL Query Tool.

Provides the SQLTool for executing SQL queries against in-memory
SQLite databases.
"""

import logging
import sqlite3
from typing import Any, Dict, List

from nexus_llm.tools.base_tool import BaseTool, ToolParameter, ToolResult, ParameterType

logger = logging.getLogger(__name__)


class SQLTool(BaseTool):
    """SQL query tool using in-memory SQLite.

    Supports creating tables, inserting data, and running queries.
    The database persists for the lifetime of the tool instance.

    Example::

        sql = SQLTool()
        sql.execute(operation="exec", query="CREATE TABLE users (id INTEGER, name TEXT)")
        sql.execute(operation="exec", query="INSERT INTO users VALUES (1, 'Alice')")
        result = sql.execute(operation="query", query="SELECT * FROM users")
    """

    def __init__(self) -> None:
        super().__init__(name="sql", description="Execute SQL queries against an in-memory SQLite database")
        self._conn = sqlite3.connect(":memory:")
        self._conn.row_factory = sqlite3.Row
        logger.debug("SQLTool initialized with in-memory SQLite database")

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="operation", type=ParameterType.STRING, description="SQL operation", required=True,
                          choices=["query", "exec", "tables", "schema", "load_csv"]),
            ToolParameter(name="query", type=ParameterType.STRING, description="SQL query string", required=False),
            ToolParameter(name="table_name", type=ParameterType.STRING, description="Table name for schema op", required=False),
            ToolParameter(name="csv_data", type=ParameterType.STRING, description="CSV data for load_csv", required=False),
        ]

    def execute(
        self,
        operation: str = "",
        query: str = "",
        table_name: str = "",
        csv_data: str = "",
        **kwargs: Any,
    ) -> ToolResult:
        """Execute a SQL operation.

        Args:
            operation: The operation type.
            query: SQL query string.
            table_name: Table name for schema queries.
            csv_data: CSV data for bulk loading.

        Returns:
            ToolResult with query output.
        """
        try:
            if operation == "query":
                return self._query(query)
            elif operation == "exec":
                return self._exec(query)
            elif operation == "tables":
                return self._tables()
            elif operation == "schema":
                return self._schema(table_name)
            elif operation == "load_csv":
                return self._load_csv(table_name, csv_data)
            else:
                return ToolResult(tool_name=self.name, success=False, error=f"Unknown operation: {operation}")
        except Exception as exc:
            return ToolResult(tool_name=self.name, success=False, error=str(exc))

    def _query(self, query: str) -> ToolResult:
        if not query:
            return ToolResult(tool_name=self.name, success=False, error="No query provided")
        cursor = self._conn.execute(query)
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
        return ToolResult(
            tool_name=self.name,
            success=True,
            output=rows,
            metadata={"columns": columns, "row_count": len(rows)},
        )

    def _exec(self, query: str) -> ToolResult:
        if not query:
            return ToolResult(tool_name=self.name, success=False, error="No query provided")
        cursor = self._conn.execute(query)
        self._conn.commit()
        return ToolResult(
            tool_name=self.name,
            success=True,
            output=f"Affected {cursor.rowcount} rows",
            metadata={"rowcount": cursor.rowcount},
        )

    def _tables(self) -> ToolResult:
        cursor = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row[0] for row in cursor.fetchall()]
        return ToolResult(tool_name=self.name, success=True, output=tables)

    def _schema(self, table_name: str) -> ToolResult:
        if not table_name:
            return ToolResult(tool_name=self.name, success=False, error="Table name required")
        cursor = self._conn.execute(f"PRAGMA table_info({table_name})")
        columns = [
            {"name": row[1], "type": row[2], "notnull": bool(row[3]), "default": row[4], "pk": bool(row[5])}
            for row in cursor.fetchall()
        ]
        return ToolResult(tool_name=self.name, success=True, output=columns, metadata={"table": table_name})

    def _load_csv(self, table_name: str, csv_data: str) -> ToolResult:
        """Load CSV data into a table, auto-creating the table if needed."""
        if not table_name or not csv_data:
            return ToolResult(tool_name=self.name, success=False, error="table_name and csv_data required")

        import csv
        import io

        reader = csv.DictReader(io.StringIO(csv_data))
        if not reader.fieldnames:
            return ToolResult(tool_name=self.name, success=False, error="No CSV headers found")

        # Create table
        cols = ", ".join(f'"{col}" TEXT' for col in reader.fieldnames)
        self._conn.execute(f'CREATE TABLE IF NOT EXISTS "{table_name}" ({cols})')

        # Insert rows
        placeholders = ", ".join("?" for _ in reader.fieldnames)
        rows_inserted = 0
        for row in reader:
            values = [row[col] for col in reader.fieldnames]
            self._conn.execute(
                f'INSERT INTO "{table_name}" VALUES ({placeholders})', values
            )
            rows_inserted += 1

        self._conn.commit()
        return ToolResult(
            tool_name=self.name,
            success=True,
            output=f"Loaded {rows_inserted} rows into {table_name}",
            metadata={"table": table_name, "rows": rows_inserted},
        )

    def __del__(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
