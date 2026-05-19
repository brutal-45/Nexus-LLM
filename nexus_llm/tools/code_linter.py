"""Nexus-LLM Code Linting Tool.

Provides code linting and static analysis capabilities for Python
and other supported languages, identifying syntax errors, style
issues, and potential bugs.
"""

import ast
import io
import logging
import sys
from typing import Any, Dict, List, Optional

from nexus_llm.tools.base_tool import BaseTool, ToolParameter, ToolResult, ParameterType

logger = logging.getLogger(__name__)


class CodeLinterTool(BaseTool):
    """Tool for linting and analyzing source code.

    Supports Python syntax checking, style analysis, and
    basic static analysis for common issues.

    Example::

        tool = CodeLinterTool()
        result = tool.run(code="def hello():\n    print('hello')", language="python")
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name="code_linter", description="Lint and analyze source code", **kwargs)

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="code", type=ParameterType.STRING, description="Source code to lint", required=True),
            ToolParameter(name="language", type=ParameterType.STRING, description="Programming language",
                         required=False, default="python", choices=["python", "javascript", "json"]),
            ToolParameter(name="strict", type=ParameterType.BOOLEAN, description="Enable strict mode",
                         required=False, default=False),
        ]

    def execute(self, **kwargs: Any) -> ToolResult:
        code = kwargs.get("code", "")
        language = kwargs.get("language", "python")
        strict = kwargs.get("strict", False)

        if not code:
            return ToolResult(success=False, error="No code provided")

        try:
            if language == "python":
                return self._lint_python(code, strict)
            elif language == "javascript":
                return self._lint_javascript(code, strict)
            elif language == "json":
                return self._lint_json(code)
            else:
                return ToolResult(success=False, error=f"Unsupported language: {language}")
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))

    def _lint_python(self, code: str, strict: bool) -> ToolResult:
        """Lint Python code for syntax errors and style issues."""
        issues: List[Dict[str, Any]] = []

        # Syntax check
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            issues.append({
                "type": "error",
                "message": f"SyntaxError: {e.msg}",
                "line": e.lineno,
                "column": e.offset,
            })
            return ToolResult(success=True, output={"issues": issues, "has_errors": True})

        # Style checks
        lines = code.splitlines()
        for i, line in enumerate(lines, 1):
            # Line length
            if len(line) > 120:
                issues.append({
                    "type": "style",
                    "message": f"Line too long ({len(line)} > 120)",
                    "line": i,
                })
            # Trailing whitespace
            if line.rstrip() != line:
                issues.append({
                    "type": "style",
                    "message": "Trailing whitespace",
                    "line": i,
                })
            # Mixed indentation
            if line.startswith(" ") and "\t" in line[:8]:
                issues.append({
                    "type": "style",
                    "message": "Mixed indentation",
                    "line": i,
                })

        # Check for common issues in AST
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                if len(node.id) == 1 and node.id != "_" and strict:
                    issues.append({
                        "type": "style",
                        "message": f"Single-letter variable name: {node.id}",
                        "line": getattr(node, "lineno", 0),
                    })

        error_count = sum(1 for i in issues if i["type"] == "error")
        style_count = sum(1 for i in issues if i["type"] == "style")

        return ToolResult(success=True, output={
            "issues": issues,
            "has_errors": error_count > 0,
            "error_count": error_count,
            "style_count": style_count,
            "total_lines": len(lines),
        })

    def _lint_javascript(self, code: str, strict: bool) -> ToolResult:
        """Basic JavaScript linting (simplified)."""
        issues: List[Dict[str, Any]] = []
        lines = code.splitlines()

        for i, line in enumerate(lines, 1):
            # Check for console.log (style)
            if "console.log" in line and strict:
                issues.append({
                    "type": "style",
                    "message": "console.log found",
                    "line": i,
                })
            # Check for var usage (modern JS prefers let/const)
            if "var " in line and strict:
                issues.append({
                    "type": "style",
                    "message": "Use 'let' or 'const' instead of 'var'",
                    "line": i,
                })
            # Missing semicolons (simplified check)
            stripped = line.rstrip()
            if stripped and not stripped.endswith((";", "{", "}", "//", "*/")) and not stripped.startswith(("//", "/*", "*")):
                if any(kw in stripped for kw in ["let ", "const ", "var ", "return "]):
                    issues.append({
                        "type": "style",
                        "message": "Possibly missing semicolon",
                        "line": i,
                    })

        return ToolResult(success=True, output={
            "issues": issues,
            "has_errors": False,
            "style_count": len(issues),
            "total_lines": len(lines),
        })

    def _lint_json(self, code: str) -> ToolResult:
        """Validate JSON syntax."""
        import json
        issues: List[Dict[str, Any]] = []
        try:
            json.loads(code)
        except json.JSONDecodeError as e:
            issues.append({
                "type": "error",
                "message": f"JSON error: {e.msg}",
                "line": e.lineno,
                "column": e.colno,
            })

        return ToolResult(success=True, output={
            "issues": issues,
            "has_errors": len(issues) > 0,
        })
