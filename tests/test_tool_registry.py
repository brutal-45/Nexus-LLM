"""Tests for nexus_llm.tools.registry module."""

import pytest
from unittest.mock import MagicMock
from nexus_llm.tools.registry import ToolRegistry
from nexus_llm.tools.base_tool import BaseTool, ToolParameter, ToolResult, ParameterType


class DummyTool(BaseTool):
    @property
    def parameters(self):
        return [ToolParameter(name="text", type=ParameterType.STRING, required=True)]

    def execute(self, **kwargs):
        return ToolResult(output=kwargs.get("text", ""), success=True)


class TestToolRegistry:
    """Tests for the ToolRegistry class."""

    def test_register_tool(self):
        registry = ToolRegistry()
        tool = DummyTool(name="dummy")
        registry.register(tool)
        assert registry.get("dummy") is tool

    def test_register_duplicate(self):
        registry = ToolRegistry()
        registry.register(DummyTool(name="dummy"))
        with pytest.raises(ValueError):
            registry.register(DummyTool(name="dummy"))

    def test_unregister(self):
        registry = ToolRegistry()
        registry.register(DummyTool(name="dummy"))
        registry.unregister("dummy")
        assert registry.get("dummy") is None

    def test_unregister_nonexistent(self):
        registry = ToolRegistry()
        # Should not raise
        registry.unregister("nonexistent")

    def test_list_tools(self):
        registry = ToolRegistry()
        registry.register(DummyTool(name="a"))
        registry.register(DummyTool(name="b"))
        tools = registry.list_tools()
        assert len(tools) == 2

    def test_has_tool(self):
        registry = ToolRegistry()
        registry.register(DummyTool(name="dummy"))
        assert registry.has("dummy") is True
        assert registry.has("missing") is False

    def test_execute_tool(self):
        registry = ToolRegistry()
        registry.register(DummyTool(name="dummy"))
        result = registry.execute("dummy", text="hello")
        assert result.success is True
        assert result.output == "hello"

    def test_execute_missing_tool(self):
        registry = ToolRegistry()
        with pytest.raises(KeyError):
            registry.execute("missing", text="hello")

    def test_get_schemas(self):
        registry = ToolRegistry()
        registry.register(DummyTool(name="dummy"))
        schemas = registry.get_schemas()
        assert "dummy" in schemas
