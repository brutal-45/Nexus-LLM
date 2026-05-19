"""Tests for nexus_llm.tools.math_tool module."""

import math
import pytest
from nexus_llm.tools.math_tool import MathTool


class TestMathTool:
    """Tests for the MathTool class."""

    def test_init(self):
        tool = MathTool()
        assert tool.name == "math"

    def test_sqrt(self):
        tool = MathTool()
        result = tool.run(operation="sqrt", value=16)
        assert result.success is True
        assert result.output == 4.0

    def test_abs(self):
        tool = MathTool()
        result = tool.run(operation="abs", value=-5)
        assert result.success is True
        assert result.output == 5

    def test_ceil(self):
        tool = MathTool()
        result = tool.run(operation="ceil", value=4.3)
        assert result.success is True
        assert result.output == 5

    def test_floor(self):
        tool = MathTool()
        result = tool.run(operation="floor", value=4.7)
        assert result.success is True
        assert result.output == 4

    def test_log(self):
        tool = MathTool()
        result = tool.run(operation="log", value=100)
        assert result.success is True
        assert abs(result.output - math.log(100)) < 1e-6

    def test_unknown_operation(self):
        tool = MathTool()
        result = tool.run(operation="unknown", value=1)
        assert result.success is False
