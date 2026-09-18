"""Tests for nexus_llm.tools.csv_tool module."""

import pytest
from nexus_llm.tools.csv_tool import CsvTool


class TestCsvTool:
    def test_init(self):
        tool = CsvTool()
        assert tool.name == "csv"

    def test_read(self):
        tool = CsvTool()
        result = tool.run(operation="read", data="name,age\nAlice,30\nBob,25")
        assert result.success is True
        assert len(result.output) == 2
        assert result.output[0]["name"] == "Alice"

    def test_read_no_header(self):
        tool = CsvTool()
        result = tool.run(operation="read", data="Alice,30\nBob,25", has_header=False)
        assert result.success is True
        assert len(result.output) == 2

    def test_headers(self):
        tool = CsvTool()
        result = tool.run(operation="headers", data="name,age,city\nAlice,30,NYC")
        assert result.success is True
        assert result.output == ["name", "age", "city"]

    def test_filter(self):
        tool = CsvTool()
        result = tool.run(
            operation="filter",
            data="name,age\nAlice,30\nBob,25\nCharlie,30",
            filter_column="age",
            filter_value="30",
        )
        assert result.success is True
        assert len(result.output) == 2

    def test_transform(self):
        tool = CsvTool()
        result = tool.run(
            operation="transform",
            data="name,age\nAlice,30",
        )
        assert result.success is True
        assert "NAME" in result.output[0]

    def test_write(self):
        tool = CsvTool()
        result = tool.run(
            operation="write",
            data="",
            rows=[{"name": "Alice", "age": "30"}],
        )
        assert result.success is True
        assert isinstance(result.output, str)
