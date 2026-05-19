"""Test tool definitions for Nexus-LLM."""
import pytest
from dataclasses import dataclass, field
from typing import Dict, Any, List, Callable, Optional


class ToolError(Exception):
    pass


@dataclass
class ParameterSchema:
    name: str
    type: str
    description: str = ""
    required: bool = True
    default: Any = None
    enum: List[Any] = None

    def to_dict(self) -> dict:
        d = {"name": self.name, "type": self.type, "required": self.required}
        if self.description:
            d["description"] = self.description
        if self.default is not None:
            d["default"] = self.default
        if self.enum:
            d["enum"] = self.enum
        return d

    def validate(self, value: Any) -> bool:
        type_map = {"string": str, "integer": int, "float": float, "boolean": bool, "list": list, "dict": dict}
        expected = type_map.get(self.type)
        if expected and not isinstance(value, expected):
            if self.type == "integer" and isinstance(value, bool):
                return False
            if self.type == "float" and isinstance(value, int) and not isinstance(value, bool):
                return True
            return False
        if self.enum and value not in self.enum:
            return False
        return True


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: List[ParameterSchema] = field(default_factory=list)
    handler: Optional[Callable] = None
    category: str = "general"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "parameters": [p.to_dict() for p in self.parameters],
        }

    def validate_args(self, args: Dict[str, Any]) -> List[str]:
        errors = []
        for param in self.parameters:
            if param.required and param.name not in args:
                errors.append(f"Missing required parameter: {param.name}")
            elif param.name in args:
                if not param.validate(args[param.name]):
                    errors.append(f"Invalid value for parameter '{param.name}'")
        return errors

    def execute(self, **kwargs) -> Any:
        errors = self.validate_args(kwargs)
        if errors:
            raise ToolError("; ".join(errors))
        if self.handler is None:
            raise ToolError("No handler registered")
        return self.handler(**kwargs)


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition):
        if tool.name in self._tools:
            raise ToolError(f"Tool '{tool.name}' already registered")
        self._tools[tool.name] = tool

    def unregister(self, name: str):
        self._tools.pop(name, None)

    def get(self, name: str) -> Optional[ToolDefinition]:
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        return list(self._tools.keys())

    def list_by_category(self, category: str) -> List[ToolDefinition]:
        return [t for t in self._tools.values() if t.category == category]

    def get_all_schemas(self) -> List[dict]:
        return [t.to_dict() for t in self._tools.values()]


class TestParameterSchema:
    def test_to_dict(self):
        param = ParameterSchema(name="query", type="string", description="Search query")
        d = param.to_dict()
        assert d["name"] == "query"
        assert d["required"] is True

    def test_validate_string(self):
        param = ParameterSchema(name="x", type="string")
        assert param.validate("hello") is True
        assert param.validate(123) is False

    def test_validate_integer(self):
        param = ParameterSchema(name="x", type="integer")
        assert param.validate(42) is True
        assert param.validate("42") is False
        assert param.validate(True) is False

    def test_validate_float(self):
        param = ParameterSchema(name="x", type="float")
        assert param.validate(3.14) is True
        assert param.validate(5) is True  # int is ok for float

    def test_validate_enum(self):
        param = ParameterSchema(name="x", type="string", enum=["a", "b"])
        assert param.validate("a") is True
        assert param.validate("c") is False


class TestToolDefinition:
    def test_creation(self):
        tool = ToolDefinition(name="search", description="Search the web")
        assert tool.name == "search"
        assert tool.category == "general"

    def test_to_dict(self):
        tool = ToolDefinition(
            name="search", description="Search",
            parameters=[ParameterSchema(name="q", type="string")],
        )
        d = tool.to_dict()
        assert d["name"] == "search"
        assert len(d["parameters"]) == 1

    def test_validate_args_ok(self):
        tool = ToolDefinition(
            name="search", description="Search",
            parameters=[ParameterSchema(name="q", type="string")],
        )
        errors = tool.validate_args({"q": "hello"})
        assert errors == []

    def test_validate_args_missing_required(self):
        tool = ToolDefinition(
            name="search", description="Search",
            parameters=[ParameterSchema(name="q", type="string")],
        )
        errors = tool.validate_args({})
        assert len(errors) == 1
        assert "Missing" in errors[0]

    def test_validate_args_wrong_type(self):
        tool = ToolDefinition(
            name="calc", description="Calc",
            parameters=[ParameterSchema(name="n", type="integer")],
        )
        errors = tool.validate_args({"n": "not a number"})
        assert len(errors) == 1

    def test_execute(self):
        tool = ToolDefinition(
            name="add", description="Add",
            parameters=[
                ParameterSchema(name="a", type="integer"),
                ParameterSchema(name="b", type="integer"),
            ],
            handler=lambda a, b: a + b,
        )
        result = tool.execute(a=2, b=3)
        assert result == 5

    def test_execute_no_handler(self):
        tool = ToolDefinition(name="test", description="Test")
        with pytest.raises(ToolError, match="No handler"):
            tool.execute()

    def test_execute_invalid_args(self):
        tool = ToolDefinition(
            name="test", description="Test",
            parameters=[ParameterSchema(name="x", type="integer")],
            handler=lambda x: x,
        )
        with pytest.raises(ToolError):
            tool.execute(x="bad")


class TestToolRegistry:
    def test_register_and_get(self):
        registry = ToolRegistry()
        tool = ToolDefinition(name="search", description="Search")
        registry.register(tool)
        assert registry.get("search") is tool

    def test_register_duplicate(self):
        registry = ToolRegistry()
        registry.register(ToolDefinition(name="x", description="X"))
        with pytest.raises(ToolError, match="already"):
            registry.register(ToolDefinition(name="x", description="Dup"))

    def test_list_tools(self):
        registry = ToolRegistry()
        registry.register(ToolDefinition(name="a", description="A"))
        registry.register(ToolDefinition(name="b", description="B"))
        assert set(registry.list_tools()) == {"a", "b"}

    def test_list_by_category(self):
        registry = ToolRegistry()
        registry.register(ToolDefinition(name="a", description="A", category="math"))
        registry.register(ToolDefinition(name="b", description="B", category="web"))
        math_tools = registry.list_by_category("math")
        assert len(math_tools) == 1

    def test_get_all_schemas(self):
        registry = ToolRegistry()
        registry.register(ToolDefinition(name="a", description="A"))
        schemas = registry.get_all_schemas()
        assert len(schemas) == 1
        assert schemas[0]["name"] == "a"
