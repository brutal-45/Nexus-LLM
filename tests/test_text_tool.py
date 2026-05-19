"""Tests for nexus_llm.tools.text_tool module."""

import pytest
from nexus_llm.tools.text_tool import TextTool


class TestTextTool:
    """Tests for the TextTool class."""

    def test_init(self):
        tool = TextTool()
        assert tool.name == "text"

    def test_upper(self):
        tool = TextTool()
        result = tool.run(operation="upper", text="hello")
        assert result.success is True
        assert result.output == "HELLO"

    def test_lower(self):
        tool = TextTool()
        result = tool.run(operation="lower", text="HELLO")
        assert result.success is True
        assert result.output == "hello"

    def test_reverse(self):
        tool = TextTool()
        result = tool.run(operation="reverse", text="hello")
        assert result.success is True
        assert result.output == "olleh"

    def test_length(self):
        tool = TextTool()
        result = tool.run(operation="length", text="hello")
        assert result.success is True
        assert result.output == 5

    def test_replace(self):
        tool = TextTool()
        result = tool.run(operation="replace", text="hello world", old="world", new="earth")
        assert result.success is True
        assert result.output == "hello earth"

    def test_split(self):
        tool = TextTool()
        result = tool.run(operation="split", text="a,b,c", delimiter=",")
        assert result.success is True
        assert result.output == ["a", "b", "c"]

    def test_unknown_operation(self):
        tool = TextTool()
        result = tool.run(operation="unknown", text="hello")
        assert result.success is False
