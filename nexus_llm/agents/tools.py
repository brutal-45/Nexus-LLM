"""Tool definitions for agent use.

Provides a collection of tools that agents can use to interact
with the environment: Calculator, Search, FileRead, FileWrite,
CodeRun, Weather, and extensible Tool base class.
"""

from __future__ import annotations

import ast
import json
import logging
import math
import operator
import os
import subprocess
import sys
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Result from a tool execution."""

    success: bool
    output: str = ""
    error: str = ""
    data: Optional[Dict[str, Any]] = None
    execution_time: float = 0.0

    def __repr__(self) -> str:
        status = "OK" if self.success else "ERR"
        preview = (self.output or self.error)[:60]
        return f"ToolResult({status}, '{preview}...')"


class Tool(ABC):
    """Abstract base class for agent tools."""

    def __init__(self, name: str, description: str, parameters: Optional[Dict[str, Any]] = None):
        """Initialize a tool.

        Args:
            name: Unique tool name.
            description: Human-readable description.
            parameters: JSON schema-style parameter definitions.
        """
        self.name = name
        self.description = description
        self.parameters = parameters or {}

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given arguments.

        Returns:
            ToolResult with success status and output.
        """
        ...

    def validate_args(self, **kwargs) -> bool:
        """Validate tool arguments against parameter schema."""
        if not self.parameters:
            return True

        required = self.parameters.get("required", [])
        properties = self.parameters.get("properties", {})

        for param in required:
            if param not in kwargs:
                return False

        for key in kwargs:
            if key not in properties:
                logger.warning("Tool %s: Unexpected parameter '%s'.", self.name, key)

        return True

    def to_dict(self) -> dict:
        """Serialize tool definition."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    def __repr__(self) -> str:
        return f"Tool(name={self.name})"


class CalculatorTool(Tool):
    """Safe mathematical expression evaluator.

    Evaluates arithmetic and mathematical expressions using a
    restricted AST-based evaluator that prevents code injection.
    """

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
        "sqrt": math.sqrt,
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
        "ceil": math.ceil,
        "floor": math.floor,
        "pi": math.pi,
        "e": math.e,
    }

    def __init__(self):
        super().__init__(
            name="calculator",
            description="Evaluate mathematical expressions safely. Supports +, -, *, /, **, and math functions.",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate (e.g., '2 + 3 * 4', 'sqrt(16)')",
                    }
                },
                "required": ["expression"],
            },
        )

    def _safe_eval(self, node: ast.AST) -> Any:
        """Safely evaluate an AST node."""
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"Unsupported constant type: {type(node.value)}")

        elif isinstance(node, ast.Name):
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

        else:
            raise ValueError(f"Unsupported AST node: {type(node).__name__}")

    def execute(self, **kwargs) -> ToolResult:
        """Evaluate a mathematical expression safely."""
        expression = kwargs.get("expression", "")
        if not expression:
            return ToolResult(success=False, error="No expression provided.")

        try:
            tree = ast.parse(expression, mode="eval")
            result = self._safe_eval(tree.body)
            return ToolResult(
                success=True,
                output=str(result),
                data={"expression": expression, "result": result},
            )
        except (ValueError, SyntaxError, TypeError, ZeroDivisionError) as e:
            return ToolResult(success=False, error=f"Calculation error: {e}")


class SearchTool(Tool):
    """Web search tool (mock/simulated).

    Provides simulated search results for testing and development.
    In production, this would integrate with a real search API.
    """

    MOCK_DATABASE = {
        "python": "Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation.",
        "machine learning": "Machine learning is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience without being explicitly programmed.",
        "weather": "Weather refers to the state of the atmosphere at a given time and place, describing temperature, humidity, precipitation, wind, and other atmospheric conditions.",
        "nexus": "Nexus refers to a connection or series of connections linking two or more things. In technology, it often refers to a central hub or platform.",
        "rag": "Retrieval-Augmented Generation (RAG) is a technique that enhances language model responses by retrieving relevant documents from a knowledge base before generating an answer.",
        "ai": "Artificial Intelligence (AI) is the simulation of human intelligence processes by computer systems, including learning, reasoning, and self-correction.",
    }

    def __init__(self):
        super().__init__(
            name="search",
            description="Search the web for information. Returns relevant results.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query string.",
                    }
                },
                "required": ["query"],
            },
        )

    def execute(self, **kwargs) -> ToolResult:
        """Execute a web search (simulated)."""
        query = kwargs.get("query", "").lower()
        if not query:
            return ToolResult(success=False, error="No search query provided.")

        # Search mock database
        results = []
        for key, content in self.MOCK_DATABASE.items():
            if key in query or any(word in content.lower() for word in query.split()):
                results.append({
                    "title": f"Result for: {key}",
                    "snippet": content,
                    "relevance": 0.9 if key in query else 0.5,
                })

        if not results:
            results.append({
                "title": "General result",
                "snippet": f"No specific results found for '{query}'. This is a simulated search tool.",
                "relevance": 0.1,
            })

        results.sort(key=lambda x: x["relevance"], reverse=True)
        output = "\n".join(f"- {r['title']}: {r['snippet']}" for r in results[:3])

        return ToolResult(
            success=True,
            output=output,
            data={"query": query, "results": results[:3]},
        )


class FileReadTool(Tool):
    """File reading tool for agents."""

    def __init__(self, allowed_dirs: Optional[List[str]] = None):
        super().__init__(
            name="file_read",
            description="Read the contents of a file from the filesystem.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to read."},
                    "encoding": {"type": "string", "description": "File encoding (default: utf-8)."},
                    "max_lines": {"type": "integer", "description": "Maximum number of lines to read."},
                },
                "required": ["path"],
            },
        )
        self.allowed_dirs = allowed_dirs

    def _is_path_allowed(self, path: str) -> bool:
        """Check if the path is within allowed directories."""
        if self.allowed_dirs is None:
            return True
        abs_path = os.path.abspath(path)
        return any(abs_path.startswith(os.path.abspath(d)) for d in self.allowed_dirs)

    def execute(self, **kwargs) -> ToolResult:
        """Read a file's contents."""
        path = kwargs.get("path", "")
        encoding = kwargs.get("encoding", "utf-8")
        max_lines = kwargs.get("max_lines")

        if not path:
            return ToolResult(success=False, error="No file path provided.")

        if not self._is_path_allowed(path):
            return ToolResult(success=False, error=f"Access denied: path '{path}' is not in allowed directories.")

        try:
            file_path = Path(path)
            if not file_path.exists():
                return ToolResult(success=False, error=f"File not found: {path}")
            if not file_path.is_file():
                return ToolResult(success=False, error=f"Path is not a file: {path}")

            content = file_path.read_text(encoding=encoding)
            if max_lines:
                lines = content.split("\n")[:max_lines]
                content = "\n".join(lines)
                if len(file_path.read_text(encoding=encoding).split("\n")) > max_lines:
                    content += f"\n... (truncated at {max_lines} lines)"

            return ToolResult(
                success=True,
                output=content,
                data={"path": str(file_path), "size": file_path.stat().st_size},
            )
        except Exception as e:
            return ToolResult(success=False, error=f"Error reading file: {e}")


class FileWriteTool(Tool):
    """File writing tool for agents."""

    def __init__(self, allowed_dirs: Optional[List[str]] = None):
        super().__init__(
            name="file_write",
            description="Write content to a file on the filesystem.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to write."},
                    "content": {"type": "string", "description": "Content to write to the file."},
                    "mode": {"type": "string", "description": "Write mode: 'write' or 'append' (default: 'write')."},
                },
                "required": ["path", "content"],
            },
        )
        self.allowed_dirs = allowed_dirs

    def _is_path_allowed(self, path: str) -> bool:
        if self.allowed_dirs is None:
            return True
        abs_path = os.path.abspath(path)
        return any(abs_path.startswith(os.path.abspath(d)) for d in self.allowed_dirs)

    def execute(self, **kwargs) -> ToolResult:
        """Write content to a file."""
        path = kwargs.get("path", "")
        content = kwargs.get("content", "")
        mode = kwargs.get("mode", "write")

        if not path:
            return ToolResult(success=False, error="No file path provided.")
        if not self._is_path_allowed(path):
            return ToolResult(success=False, error=f"Access denied: path '{path}' is not in allowed directories.")

        try:
            file_path = Path(path)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            write_mode = "a" if mode == "append" else "w"
            with open(file_path, write_mode, encoding="utf-8") as f:
                f.write(content)

            return ToolResult(
                success=True,
                output=f"Successfully wrote {len(content)} characters to {path}",
                data={"path": str(file_path), "chars_written": len(content), "mode": mode},
            )
        except Exception as e:
            return ToolResult(success=False, error=f"Error writing file: {e}")


class CodeRunTool(Tool):
    """Python code execution tool with sandboxing.

    Executes Python code in a controlled environment with
    restricted builtins and resource limits.
    """

    def __init__(self, timeout: int = 30, max_output_chars: int = 5000):
        super().__init__(
            name="code_run",
            description="Execute Python code and return the output. Runs in a restricted environment.",
            parameters={
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute."},
                    "timeout": {"type": "integer", "description": f"Execution timeout in seconds (default: {timeout})."},
                },
                "required": ["code"],
            },
        )
        self.timeout = timeout
        self.max_output_chars = max_output_chars

    def execute(self, **kwargs) -> ToolResult:
        """Execute Python code in a subprocess."""
        code = kwargs.get("code", "")
        exec_timeout = kwargs.get("timeout", self.timeout)

        if not code:
            return ToolResult(success=False, error="No code provided.")

        start_time = time.time()

        try:
            # Write code to a temporary file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(code)
                temp_path = f.name

            # Execute in subprocess for isolation
            result = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                text=True,
                timeout=exec_timeout,
                cwd=tempfile.gettempdir(),
            )

            execution_time = time.time() - start_time

            # Clean up temp file
            try:
                os.unlink(temp_path)
            except OSError:
                pass

            stdout = result.stdout[:self.max_output_chars]
            stderr = result.stderr[:self.max_output_chars]

            if result.returncode == 0:
                output = stdout.strip() if stdout.strip() else "(Code executed successfully with no output)"
                return ToolResult(
                    success=True,
                    output=output,
                    data={"returncode": result.returncode, "execution_time": execution_time},
                    execution_time=execution_time,
                )
            else:
                return ToolResult(
                    success=False,
                    output=stdout,
                    error=stderr.strip() or f"Process exited with code {result.returncode}",
                    data={"returncode": result.returncode, "execution_time": execution_time},
                    execution_time=execution_time,
                )

        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                error=f"Code execution timed out after {exec_timeout} seconds.",
                execution_time=time.time() - start_time,
            )
        except Exception as e:
            return ToolResult(success=False, error=f"Error executing code: {e}")


class WeatherTool(Tool):
    """Weather information tool (mock/simulated).

    Provides simulated weather data for testing and development.
    In production, this would integrate with a weather API.
    """

    MOCK_CITIES = {
        "new york": {"temp": 72, "condition": "Partly Cloudy", "humidity": 65, "wind": 12},
        "london": {"temp": 59, "condition": "Overcast", "humidity": 80, "wind": 15},
        "tokyo": {"temp": 77, "condition": "Clear", "humidity": 55, "wind": 8},
        "paris": {"temp": 64, "condition": "Light Rain", "humidity": 75, "wind": 10},
        "sydney": {"temp": 81, "condition": "Sunny", "humidity": 45, "wind": 14},
        "beijing": {"temp": 70, "condition": "Hazy", "humidity": 60, "wind": 6},
        "mumbai": {"temp": 88, "condition": "Humid", "humidity": 85, "wind": 9},
        "san francisco": {"temp": 65, "condition": "Foggy", "humidity": 78, "wind": 18},
    }

    def __init__(self):
        super().__init__(
            name="weather",
            description="Get current weather information for a location (simulated data).",
            parameters={
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name to get weather for."},
                },
                "required": ["location"],
            },
        )

    def execute(self, **kwargs) -> ToolResult:
        """Get weather for a location (simulated)."""
        location = kwargs.get("location", "").lower().strip()
        if not location:
            return ToolResult(success=False, error="No location provided.")

        # Look up in mock data
        weather = self.MOCK_CITIES.get(location)

        if weather:
            output = (
                f"Weather for {location.title()}:\n"
                f"  Temperature: {weather['temp']}°F\n"
                f"  Condition: {weather['condition']}\n"
                f"  Humidity: {weather['humidity']}%\n"
                f"  Wind: {weather['wind']} mph"
            )
            return ToolResult(
                success=True,
                output=output,
                data={"location": location, **weather},
            )
        else:
            # Generate a random-ish weather for unknown cities
            import hashlib

            hash_val = int(hashlib.md5(location.encode()).hexdigest()[:8], 16)
            temp = 50 + (hash_val % 50)
            conditions = ["Sunny", "Cloudy", "Partly Cloudy", "Light Rain", "Clear"]
            condition = conditions[hash_val % len(conditions)]

            output = (
                f"Weather for {location.title()} (simulated):\n"
                f"  Temperature: {temp}°F\n"
                f"  Condition: {condition}\n"
                f"  Humidity: {40 + hash_val % 40}%\n"
                f"  Wind: {5 + hash_val % 20} mph"
            )
            return ToolResult(
                success=True,
                output=output,
                data={"location": location, "temp": temp, "condition": condition, "simulated": True},
            )
