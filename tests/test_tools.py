"""Tests for the tools module.

Covers Tool, BuiltinTools, ToolBuilder, and ToolManager.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from nexus_llm.tools.tool import Tool
from nexus_llm.tools.builtins import BuiltinTools
from nexus_llm.tools.builder import ToolBuilder
from nexus_llm.tools.manager import ToolManager


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

class TestTool:
    """Tests for Tool."""

    def test_create_tool(self):
        tool = Tool(name="calculator", description="Performs calculations")
        assert tool.name == "calculator"
        assert tool.description == "Performs calculations"

    def test_tool_with_function(self):
        def add(a, b):
            return a + b

        tool = Tool(name="add", description="Add two numbers", func=add)
        result = tool.run(a=2, b=3)
        assert result == 5

    def test_tool_repr(self):
        tool = Tool(name="test")
        r = repr(tool)
        assert "test" in r


# ---------------------------------------------------------------------------
# BuiltinTools
# ---------------------------------------------------------------------------

class TestBuiltinTools:
    """Tests for BuiltinTools."""

    def test_list_builtin_tools(self):
        builtins = BuiltinTools()
        tools = builtins.list_tools()
        assert isinstance(tools, list)
        assert len(tools) > 0

    def test_get_builtin_tool(self):
        builtins = BuiltinTools()
        tools = builtins.list_tools()
        if tools:
            tool = builtins.get_tool(tools[0])
            assert tool is not None

    def test_calculator_tool(self):
        builtins = BuiltinTools()
        calc = builtins.get_tool("calculator")
        if calc:
            result = calc.run(expression="2+2")
            assert result is not None


# ---------------------------------------------------------------------------
# ToolBuilder
# ---------------------------------------------------------------------------

class TestToolBuilder:
    """Tests for ToolBuilder."""

    def test_build_tool(self):
        builder = ToolBuilder()
        tool = builder.name("my_tool").description("My custom tool").build()
        assert tool.name == "my_tool"
        assert tool.description == "My custom tool"

    def test_build_with_function(self):
        def my_func(x):
            return x * 2

        builder = ToolBuilder()
        tool = builder.name("doubler").func(my_func).build()
        result = tool.run(x=5)
        assert result == 10

    def test_build_with_parameters(self):
        builder = ToolBuilder()
        tool = (
            builder.name("greet")
            .description("Greet someone")
            .add_parameter("name", type="string", description="Name to greet")
            .build()
        )
        assert tool.name == "greet"


# ---------------------------------------------------------------------------
# ToolManager
# ---------------------------------------------------------------------------

class TestToolManager:
    """Tests for ToolManager."""

    def test_register_and_get(self):
        mgr = ToolManager()
        tool = Tool(name="test_tool", description="Test")
        mgr.register(tool)
        retrieved = mgr.get("test_tool")
        assert retrieved is not None
        assert retrieved.name == "test_tool"

    def test_get_nonexistent(self):
        mgr = ToolManager()
        assert mgr.get("nonexistent") is None

    def test_list_tools(self):
        mgr = ToolManager()
        mgr.register(Tool(name="tool1", description="T1"))
        mgr.register(Tool(name="tool2", description="T2"))
        tools = mgr.list_tools()
        assert len(tools) == 2

    def test_unregister(self):
        mgr = ToolManager()
        mgr.register(Tool(name="tool1", description="T1"))
        mgr.unregister("tool1")
        assert mgr.get("tool1") is None

    def test_run_tool(self):
        def add(a, b):
            return a + b

        mgr = ToolManager()
        tool = Tool(name="add", description="Add", func=add)
        mgr.register(tool)
        result = mgr.run("add", a=3, b=4)
        assert result == 7

    def test_run_nonexistent_raises(self):
        mgr = ToolManager()
        with pytest.raises(KeyError):
            mgr.run("nonexistent")

    def test_has_tool(self):
        mgr = ToolManager()
        mgr.register(Tool(name="tool1", description="T1"))
        assert mgr.has("tool1") is True
        assert mgr.has("nonexistent") is False
