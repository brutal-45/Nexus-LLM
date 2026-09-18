"""Calculator plugin for safe mathematical expression evaluation.

A builtin plugin that provides safe arithmetic and mathematical
function evaluation using AST-based parsing to prevent code injection.
"""

from __future__ import annotations

import ast
import logging
import math
import operator
from typing import Any, Dict, List, Optional

from nexus_llm.plugins.hook import HookManager, HookPriority

logger = logging.getLogger(__name__)


class CalculatorPlugin:
    """Plugin providing safe mathematical expression evaluation.

    Evaluates mathematical expressions using a restricted AST
    parser that only allows arithmetic operations and math
    functions, preventing code injection attacks.
    """

    name = "calculator"
    version = "1.0.0"
    description = "Evaluate mathematical expressions safely"
    dependencies: List[str] = []
    tags = ["math", "calculator", "builtin"]

    SAFE_OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    SAFE_FUNCTIONS = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "sqrt": math.sqrt,
        "cbrt": lambda x: x ** (1 / 3),
        "log": math.log,
        "log10": math.log10,
        "log2": math.log2,
        "exp": math.exp,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "asin": math.asin,
        "acos": math.acos,
        "atan": math.atan,
        "atan2": math.atan2,
        "sinh": math.sinh,
        "cosh": math.cosh,
        "tanh": math.tanh,
        "ceil": math.ceil,
        "floor": math.floor,
        "degrees": math.degrees,
        "radians": math.radians,
        "gcd": math.gcd,
        "factorial": math.factorial,
        "perm": math.perm,
        "comb": math.comb,
    }

    SAFE_CONSTANTS = {
        "pi": math.pi,
        "e": math.e,
        "tau": math.tau,
        "inf": math.inf,
        "phi": (1 + math.sqrt(5)) / 2,  # Golden ratio
    }

    def __init__(self, hook_manager: Optional[HookManager] = None, **kwargs):
        self.hook_manager = hook_manager
        self._active = False
        self._history: List[Dict[str, Any]] = []

    def activate(self) -> None:
        """Activate the calculator plugin."""
        if self.hook_manager:
            self.hook_manager.register(
                "tool_request",
                self._handle_tool_request,
                name="calculator_tool_handler",
                priority=HookPriority.NORMAL,
                owner=self.name,
            )
        self._active = True
        logger.info("Calculator plugin activated.")

    def deactivate(self) -> None:
        """Deactivate the calculator plugin."""
        if self.hook_manager:
            self.hook_manager.unregister_by_owner(self.name)
        self._active = False
        logger.info("Calculator plugin deactivated.")

    def evaluate(self, expression: str) -> Dict[str, Any]:
        """Safely evaluate a mathematical expression.

        Args:
            expression: Mathematical expression string.

        Returns:
            Dict with 'success', 'result', 'expression', and optionally 'error'.
        """
        expression = expression.strip()
        if not expression:
            return {"success": False, "error": "Empty expression", "expression": expression}

        try:
            tree = ast.parse(expression, mode="eval")
            result = self._safe_eval(tree.body)

            # Record in history
            entry = {
                "expression": expression,
                "result": result,
                "success": True,
            }
            self._history.append(entry)

            return {
                "success": True,
                "result": result,
                "expression": expression,
                "result_str": str(result),
            }
        except (ValueError, SyntaxError, TypeError, ZeroDivisionError, OverflowError) as e:
            return {
                "success": False,
                "error": f"Calculation error: {type(e).__name__}: {e}",
                "expression": expression,
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Unexpected error: {type(e).__name__}: {e}",
                "expression": expression,
            }

    def _safe_eval(self, node: ast.AST) -> Any:
        """Recursively evaluate an AST node safely."""
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"Unsupported constant: {node.value!r}")

        elif isinstance(node, ast.Name):
            if node.id in self.SAFE_CONSTANTS:
                return self.SAFE_CONSTANTS[node.id]
            if node.id in self.SAFE_FUNCTIONS:
                return self.SAFE_FUNCTIONS[node.id]
            raise ValueError(f"Name '{node.id}' is not allowed")

        elif isinstance(node, ast.BinOp):
            left = self._safe_eval(node.left)
            right = self._safe_eval(node.right)
            op_type = type(node.op)
            if op_type in self.SAFE_OPERATORS:
                return self.SAFE_OPERATORS[op_type](left, right)
            raise ValueError(f"Unsupported operator: {op_type.__name__}")

        elif isinstance(node, ast.UnaryOp):
            operand = self._safe_eval(node.operand)
            op_type = type(node.op)
            if op_type in self.SAFE_OPERATORS:
                return self.SAFE_OPERATORS[op_type](operand)
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")

        elif isinstance(node, ast.Call):
            func = self._safe_eval(node.func)
            args = [self._safe_eval(arg) for arg in node.args]
            return func(*args)

        elif isinstance(node, ast.Tuple):
            return tuple(self._safe_eval(elt) for elt in node.elts)

        elif isinstance(node, ast.List):
            return [self._safe_eval(elt) for elt in node.elts]

        else:
            raise ValueError(f"Unsupported AST node: {type(node).__name__}")

    def get_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get calculation history."""
        return self._history[-limit:]

    def clear_history(self) -> None:
        """Clear calculation history."""
        self._history.clear()

    def get_help(self) -> str:
        """Get help text for available operations."""
        return (
            "Calculator Plugin - Available Operations:\n"
            "  Arithmetic: +, -, *, /, //, %, **\n"
            f"  Functions: {', '.join(sorted(self.SAFE_FUNCTIONS.keys()))}\n"
            f"  Constants: {', '.join(sorted(self.SAFE_CONSTANTS.keys()))}\n"
            "  Examples:\n"
            "    2 + 3 * 4\n"
            "    sqrt(16)\n"
            "    sin(pi / 2)\n"
            "    log(e ** 3)"
        )

    def _handle_tool_request(self, result, *args, **kwargs):
        """Handle tool requests for calculator operations."""
        tool_name = kwargs.get("tool_name", "")
        if tool_name == "calculator":
            expression = kwargs.get("expression", kwargs.get("query", ""))
            if expression:
                eval_result = self.evaluate(expression)
                if eval_result["success"]:
                    return f"Result: {eval_result['result']}"
                return f"Error: {eval_result['error']}"
        return result
