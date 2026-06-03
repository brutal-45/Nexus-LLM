"""Tests for nexus_llm.tools.json_tool module."""

import pytest
from nexus_llm.tools.json_tool import JsonTool


class TestJsonTool:
    """Tests for the JsonTool class."""

    def test_init(self):
        tool = JsonTool()
        assert tool.name == "json"

    def test_parse(self):
        tool = JsonTool()
        result = tool.run(operation="parse", data='{"key": "value"}')
        assert result.success is True
        assert result.output == {"key": "value"}

    def test_stringify(self):
        tool = JsonTool()
        result = tool.run(operation="stringify", data={"key": "value"})
        assert result.success is True
        assert "key" in result.output

    def test_get_key(self):
        tool = JsonTool()
        result = tool.run(operation="get", data='{"a": 1, "b": 2}', key="a")
        assert result.success is True
        assert result.output == 1

    def test_set_key(self):
        tool = JsonTool()
        result = tool.run(operation="set", data='{"a": 1}', key="b", value=2)
        assert result.success is True

    def test_invalid_json(self):
        tool = JsonTool()
        result = tool.run(operation="parse", data="not json")
        assert result.success is False

    def test_unknown_operation(self):
        tool = JsonTool()
        result = tool.run(operation="unknown", data="{}")
        assert result.success is False
