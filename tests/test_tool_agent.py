"""Test tool agent for Nexus-LLM."""
import pytest
from dataclasses import dataclass, field
from typing import Dict, Any, List, Callable, Optional


class ToolError(Exception):
    pass


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Callable = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


@dataclass
class ToolCall:
    tool_name: str
    arguments: Dict[str, Any]
    result: Any = None
    error: str = ""


class ToolAgent:
    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}
        self._history: List[ToolCall] = []

    def register_tool(self, tool: ToolDefinition):
        if tool.name in self._tools:
            raise ToolError(f"Tool '{tool.name}' already registered")
        self._tools[tool.name] = tool

    def unregister_tool(self, name: str):
        if name not in self._tools:
            raise ToolError(f"Tool '{name}' not found")
        del self._tools[name]

    def list_tools(self) -> List[str]:
        return list(self._tools.keys())

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        return self._tools.get(name)

    def call_tool(self, name: str, arguments: Dict[str, Any] = None) -> ToolCall:
        if name not in self._tools:
            call = ToolCall(tool_name=name, arguments=arguments or {}, error=f"Tool '{name}' not found")
            self._history.append(call)
            return call

        tool = self._tools[name]
        if tool.handler is None:
            call = ToolCall(tool_name=name, arguments=arguments or {}, error="No handler registered")
            self._history.append(call)
            return call

        try:
            result = tool.handler(**(arguments or {}))
            call = ToolCall(tool_name=name, arguments=arguments or {}, result=result)
        except Exception as e:
            call = ToolCall(tool_name=name, arguments=arguments or {}, error=str(e))
        self._history.append(call)
        return call

    def run(self, task: str) -> Dict[str, Any]:
        return {
            "task": task,
            "available_tools": self.list_tools(),
            "status": "planning",
        }

    def get_history(self) -> List[ToolCall]:
        return list(self._history)

    def clear_history(self):
        self._history.clear()


# Built-in tools
def calculator(expression: str) -> float:
    try:
        return float(eval(expression, {"__builtins__": {}}, {}))
    except Exception:
        raise ToolError("Invalid expression")


def string_reverse(text: str) -> str:
    return text[::-1]


def word_count(text: str) -> int:
    return len(text.split())


class TestToolDefinition:
    def test_creation(self):
        tool = ToolDefinition(name="calc", description="Calculator", parameters={"expression": {"type": "string"}})
        assert tool.name == "calc"

    def test_to_dict(self):
        tool = ToolDefinition(name="calc", description="Calculator", parameters={"x": {"type": "int"}})
        d = tool.to_dict()
        assert d["name"] == "calc"
        assert "parameters" in d


class TestToolAgent:
    def test_register_tool(self):
        agent = ToolAgent()
        agent.register_tool(ToolDefinition(name="calc", description="Calc", parameters={}))
        assert "calc" in agent.list_tools()

    def test_register_duplicate(self):
        agent = ToolAgent()
        agent.register_tool(ToolDefinition(name="calc", description="Calc", parameters={}))
        with pytest.raises(ToolError, match="already registered"):
            agent.register_tool(ToolDefinition(name="calc", description="Dup", parameters={}))

    def test_unregister_tool(self):
        agent = ToolAgent()
        agent.register_tool(ToolDefinition(name="calc", description="Calc", parameters={}))
        agent.unregister_tool("calc")
        assert "calc" not in agent.list_tools()

    def test_unregister_nonexistent(self):
        agent = ToolAgent()
        with pytest.raises(ToolError, match="not found"):
            agent.unregister_tool("nonexistent")

    def test_call_tool(self):
        agent = ToolAgent()
        agent.register_tool(ToolDefinition(
            name="reverse", description="Reverse string",
            parameters={"text": {"type": "string"}},
            handler=string_reverse,
        ))
        call = agent.call_tool("reverse", {"text": "hello"})
        assert call.result == "olleh"
        assert call.error == ""

    def test_call_nonexistent_tool(self):
        agent = ToolAgent()
        call = agent.call_tool("nonexistent", {})
        assert call.error != ""

    def test_call_tool_no_handler(self):
        agent = ToolAgent()
        agent.register_tool(ToolDefinition(name="empty", description="No handler", parameters={}))
        call = agent.call_tool("empty", {})
        assert "No handler" in call.error

    def test_call_tool_with_error(self):
        agent = ToolAgent()
        agent.register_tool(ToolDefinition(
            name="calc", description="Calc",
            parameters={"expression": {"type": "string"}},
            handler=calculator,
        ))
        call = agent.call_tool("calc", {"expression": "2 + 'a'"})
        assert call.error != ""

    def test_history(self):
        agent = ToolAgent()
        agent.register_tool(ToolDefinition(name="wc", description="Word count", parameters={}, handler=word_count))
        agent.call_tool("wc", {"text": "hello world"})
        assert len(agent.get_history()) == 1

    def test_clear_history(self):
        agent = ToolAgent()
        agent.register_tool(ToolDefinition(name="wc", description="Word count", parameters={}, handler=word_count))
        agent.call_tool("wc", {"text": "hello"})
        agent.clear_history()
        assert len(agent.get_history()) == 0

    def test_run(self):
        agent = ToolAgent()
        agent.register_tool(ToolDefinition(name="calc", description="Calc", parameters={}))
        result = agent.run("Calculate 2+2")
        assert "calc" in result["available_tools"]

    def test_list_tools(self):
        agent = ToolAgent()
        agent.register_tool(ToolDefinition(name="a", description="A", parameters={}))
        agent.register_tool(ToolDefinition(name="b", description="B", parameters={}))
        assert len(agent.list_tools()) == 2

    def test_get_tool(self):
        agent = ToolAgent()
        tool = ToolDefinition(name="calc", description="Calc", parameters={})
        agent.register_tool(tool)
        assert agent.get_tool("calc") is tool
        assert agent.get_tool("nonexistent") is None
