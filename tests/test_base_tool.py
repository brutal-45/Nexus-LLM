"""Tests for nexus_llm.tools.base_tool module."""

import pytest
from nexus_llm.tools.base_tool import BaseTool, ToolParameter, ToolResult, ParameterType


class SampleTool(BaseTool):
    """A sample tool for testing."""

    @property
    def parameters(self):
        return [
            ToolParameter(name="input", type=ParameterType.STRING, description="Input text", required=True),
            ToolParameter(name="count", type=ParameterType.INTEGER, description="Repeat count", required=False, default=1),
        ]

    def execute(self, **kwargs):
        text = kwargs["input"]
        count = kwargs.get("count", 1)
        return ToolResult(output=text * count, success=True)


class TestToolParameter:
    """Tests for ToolParameter dataclass."""

    def test_defaults(self):
        param = ToolParameter(name="test")
        assert param.type == ParameterType.STRING
        assert param.required is True
        assert param.default is None
        assert param.choices is None

    def test_with_choices(self):
        param = ToolParameter(name="mode", choices=["fast", "slow"])
        assert param.choices == ["fast", "slow"]


class TestToolResult:
    """Tests for ToolResult dataclass."""

    def test_default_result(self):
        result = ToolResult()
        assert result.success is True
        assert result.output is None
        assert result.error is None
        assert result.duration_ms == 0.0

    def test_to_dict(self):
        result = ToolResult(success=True, output="hello")
        d = result.to_dict()
        assert d["success"] is True
        assert d["output"] == "hello"


class TestBaseTool:
    """Tests for BaseTool abstract class."""

    def test_tool_name(self):
        tool = SampleTool(name="sample")
        assert tool.name == "sample"

    def test_tool_default_name(self):
        tool = SampleTool()
        assert tool.name == "sample"

    def test_tool_enabled(self):
        tool = SampleTool()
        assert tool.enabled is True

    def test_tool_disable(self):
        tool = SampleTool()
        tool.enabled = False
        assert tool.enabled is False

    def test_run_disabled(self):
        tool = SampleTool()
        tool.enabled = False
        result = tool.run(input="test")
        assert result.success is False
        assert "disabled" in result.error

    def test_run_success(self):
        tool = SampleTool()
        result = tool.run(input="hello")
        assert result.success is True
        assert result.output == "hello"

    def test_run_with_count(self):
        tool = SampleTool()
        result = tool.run(input="hi", count=3)
        assert result.success is True
        assert result.output == "hihihi"

    def test_run_missing_required(self):
        tool = SampleTool()
        result = tool.run()
        assert result.success is False
        assert "Missing required" in result.error

    def test_validate_inputs_unknown_param(self):
        tool = SampleTool()
        errors = tool.validate_inputs({"input": "test", "unknown": "value"})
        assert any("Unknown parameter" in e for e in errors)

    def test_schema(self):
        tool = SampleTool()
        schema = tool.schema()
        assert schema["name"] == "sample"
        assert len(schema["parameters"]) == 2

    def test_run_records_duration(self):
        tool = SampleTool()
        result = tool.run(input="test")
        assert result.duration_ms >= 0
