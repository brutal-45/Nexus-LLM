"""Tests for nexus_llm.tools.calculator module."""

import pytest
from nexus_llm.tools.calculator import CalculatorTool


class TestCalculatorTool:
    """Tests for the CalculatorTool class."""

    def test_init(self):
        tool = CalculatorTool()
        assert tool.name == "calculator"

    def test_add(self):
        tool = CalculatorTool()
        result = tool.run(expression="2 + 3")
        assert result.success is True
        assert result.output == 5

    def test_subtract(self):
        tool = CalculatorTool()
        result = tool.run(expression="10 - 4")
        assert result.success is True
        assert result.output == 6

    def test_multiply(self):
        tool = CalculatorTool()
        result = tool.run(expression="6 * 7")
        assert result.success is True
        assert result.output == 42

    def test_divide(self):
        tool = CalculatorTool()
        result = tool.run(expression="20 / 4")
        assert result.success is True
        assert result.output == 5.0

    def test_divide_by_zero(self):
        tool = CalculatorTool()
        result = tool.run(expression="10 / 0")
        assert result.success is False

    def test_power(self):
        tool = CalculatorTool()
        result = tool.run(expression="2 ** 3")
        assert result.success is True
        assert result.output == 8

    def test_complex_expression(self):
        tool = CalculatorTool()
        result = tool.run(expression="(2 + 3) * 4")
        assert result.success is True
        assert result.output == 20

    def test_invalid_expression(self):
        tool = CalculatorTool()
        result = tool.run(expression="not math")
        assert result.success is False

    def test_negative_numbers(self):
        tool = CalculatorTool()
        result = tool.run(expression="-5 + 3")
        assert result.success is True
        assert result.output == -2

    def test_float_result(self):
        tool = CalculatorTool()
        result = tool.run(expression="7 / 2")
        assert result.success is True
        assert result.output == 3.5
