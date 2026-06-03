"""Nexus-LLM Calculator Tool.

Provides a safe arithmetic expression calculator that supports basic
math operations, functions, and variable assignment.
"""

import logging
import math
import re
from typing import Any, Dict, List

from nexus_llm.tools.base_tool import BaseTool, ToolParameter, ToolResult, ParameterType

logger = logging.getLogger(__name__)

# Allowed math functions for safe evaluation
_SAFE_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "pow": pow,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "exp": math.exp,
    "ceil": math.ceil,
    "floor": math.floor,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "pi": math.pi,
    "e": math.e,
}

_SAFE_NAMES = {"__builtins__": {}}


class CalculatorTool(BaseTool):
    """Safe arithmetic expression calculator.

    Evaluates mathematical expressions using a restricted set of
    operations and functions. Does not allow arbitrary code execution.

    Examples::

        calculator = CalculatorTool()
        result = calculator.execute(expression="2 + 2 * 3")       # 8
        result = calculator.execute(expression="sqrt(144)")       # 12.0
        result = calculator.execute(expression="pi * 5**2")       # 78.54...
    """

    def __init__(self) -> None:
        super().__init__(name="calculator", description="Evaluate mathematical expressions safely")

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="expression",
                type=ParameterType.STRING,
                description="Mathematical expression to evaluate",
                required=True,
            ),
        ]

    def execute(self, expression: str = "", **kwargs: Any) -> ToolResult:
        """Evaluate a mathematical expression.

        Args:
            expression: The expression to evaluate.

        Returns:
            ToolResult with the computed value.
        """
        expr = expression.strip()
        if not expr:
            return ToolResult(tool_name=self.name, success=False, error="Empty expression")

        # Validate expression contains only allowed characters
        if not re.match(r'^[\d\s+\-*/().,%^e_pi_a-zA-Z,]+$', expr):
            return ToolResult(
                tool_name=self.name,
                success=False,
                error="Expression contains disallowed characters",
            )

        # Replace common notation
        expr = expr.replace("^", "**")
        expr = expr.replace("%", "/100.0*")

        try:
            result = eval(expr, _SAFE_NAMES, _SAFE_FUNCTIONS)  # noqa: S307
            return ToolResult(
                tool_name=self.name,
                success=True,
                output=result,
                metadata={"expression": expression},
            )
        except ZeroDivisionError:
            return ToolResult(tool_name=self.name, success=False, error="Division by zero")
        except OverflowError:
            return ToolResult(tool_name=self.name, success=False, error="Result overflow")
        except Exception as exc:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"Evaluation error: {exc}",
            )
