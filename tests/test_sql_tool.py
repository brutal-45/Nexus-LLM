"""Tests for nexus_llm.tools.sql_tool module."""

import pytest
from nexus_llm.tools.sql_tool import SqlTool


class TestSqlTool:
    """Tests for the SqlTool class."""

    def test_init(self):
        tool = SqlTool()
        assert tool.name == "sql"

    def test_execute_query(self):
        tool = SqlTool()
        result = tool.run(query="SELECT 1 AS value")
        assert result.success is True

    def test_invalid_query(self):
        tool = SqlTool()
        result = tool.run(query="NOT VALID SQL")
        assert result.success is False

    def test_missing_query(self):
        tool = SqlTool()
        result = tool.run()
        assert result.success is False
