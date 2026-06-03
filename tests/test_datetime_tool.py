"""Tests for nexus_llm.tools.datetime_tool module."""

import pytest
from datetime import datetime
from nexus_llm.tools.datetime_tool import DatetimeTool


class TestDatetimeTool:
    """Tests for the DatetimeTool class."""

    def test_init(self):
        tool = DatetimeTool()
        assert tool.name == "datetime"

    def test_now(self):
        tool = DatetimeTool()
        result = tool.run(operation="now")
        assert result.success is True
        assert isinstance(result.output, str)

    def test_format(self):
        tool = DatetimeTool()
        result = tool.run(operation="format", format_str="%Y-%m-%d")
        assert result.success is True

    def test_parse(self):
        tool = DatetimeTool()
        result = tool.run(operation="parse", date_string="2024-01-15", format_str="%Y-%m-%d")
        assert result.success is True

    def test_add_days(self):
        tool = DatetimeTool()
        result = tool.run(operation="add", days=7)
        assert result.success is True

    def test_unknown_operation(self):
        tool = DatetimeTool()
        result = tool.run(operation="unknown")
        assert result.success is False
