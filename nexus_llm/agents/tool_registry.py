"""Tool registry for Nexus-LLM agents.

Provides a central registry for named tools with descriptions, and
includes built-in tools: calculator, web_search (mock), file_read,
and file_write.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from nexus_llm.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Tool descriptor
# ---------------------------------------------------------------------------

@dataclass
class ToolInfo:
    """Metadata about a registered tool.

    Attributes:
        name: The tool's unique name.
        description: What the tool does.
        parameters: Expected parameter names with descriptions.
    """

    name: str
    description: str
    parameters: Dict[str, str]


# ---------------------------------------------------------------------------
# Built-in tools
# ---------------------------------------------------------------------------

def _calculator(expression: str) -> str:
    """Evaluate a mathematical expression safely.

    Supports basic arithmetic and math-module functions.
    """
    # Allow only safe characters and function names
    allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
    allowed_names.update({"abs": abs, "round": round, "min": min, "max": max})
    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)  # noqa: S307
        return str(result)
    except Exception as exc:
        return f"Error evaluating expression: {exc}"


def _web_search(query: str) -> str:
    """Mock web search – returns placeholder results."""
    return json.dumps({
        "query": query,
        "results": [
            {
                "title": f"Mock result for '{query}'",
                "url": "https://example.com/mock",
                "snippet": f"This is a mock search result for the query '{query}'. "
                           f"In production, this would connect to a real search API.",
            }
        ],
    })


def _file_read(path: str) -> str:
    """Read a text file and return its contents."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: File not found: {path}"
    except Exception as exc:
        return f"Error reading file: {exc}"


def _file_write(path: str, content: str) -> str:
    """Write content to a text file."""
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote {len(content)} characters to {path}"
    except Exception as exc:
        return f"Error writing file: {exc}"


_BUILTIN_TOOLS: Dict[str, Dict[str, Any]] = {
    "calculator": {
        "func": _calculator,
        "description": "Evaluate a mathematical expression and return the result.",
        "parameters": {"expression": "A mathematical expression string (e.g. '2 + 3 * 4')."},
    },
    "web_search": {
        "func": _web_search,
        "description": "Search the web for information (mock – returns placeholder results).",
        "parameters": {"query": "The search query string."},
    },
    "file_read": {
        "func": _file_read,
        "description": "Read the contents of a text file.",
        "parameters": {"path": "Absolute or relative path to the file."},
    },
    "file_write": {
        "func": _file_write,
        "description": "Write content to a text file.",
        "parameters": {"path": "File path.", "content": "Text content to write."},
    },
}


# ---------------------------------------------------------------------------
# Tool Registry
# ---------------------------------------------------------------------------

class ToolRegistry:
    """Central registry for agent tools.

    Comes pre-loaded with four built-in tools (calculator, web_search,
    file_read, file_write).  Custom tools can be registered at runtime.
    """

    def __init__(self) -> None:
        self._tools: Dict[str, Callable] = {}
        self._info: Dict[str, ToolInfo] = {}

        # Register built-ins
        for name, entry in _BUILTIN_TOOLS.items():
            self.register(
                name=name,
                func=entry["func"],
                description=entry["description"],
                parameters=entry["parameters"],
            )

        logger.info("ToolRegistry initialised with %d built-in tool(s)", len(self._tools))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        func: Callable,
        description: str = "",
        parameters: Optional[Dict[str, str]] = None,
    ) -> None:
        """Register a new tool.

        Args:
            name: Unique tool name.
            func: The callable implementing the tool.
            description: What the tool does.
            parameters: Mapping of parameter name → description.
        """
        if name in self._tools:
            logger.warning("Overwriting existing tool: %s", name)
        self._tools[name] = func
        self._info[name] = ToolInfo(
            name=name,
            description=description,
            parameters=parameters or {},
        )
        logger.debug("Registered tool: %s", name)

    def get_tool(self, name: str) -> Optional[Callable]:
        """Return the tool function by name, or ``None``."""
        return self._tools.get(name)

    def list_tools(self) -> List[ToolInfo]:
        """Return metadata for all registered tools."""
        return list(self._info.values())

    def execute(self, tool_name: str, **kwargs: Any) -> str:
        """Execute a tool by name with the given keyword arguments.

        Returns:
            The string result of the tool execution.

        Raises:
            KeyError: If the tool is not registered.
        """
        tool = self._tools.get(tool_name)
        if tool is None:
            raise KeyError(f"Tool not found: {tool_name}")
        try:
            result = tool(**kwargs)
            logger.debug("Executed tool %s → %s", tool_name, str(result)[:100])
            return str(result)
        except Exception as exc:
            logger.error("Tool %s execution failed: %s", tool_name, exc)
            return f"Tool execution error: {exc}"

    def has_tool(self, name: str) -> bool:
        """Check whether a tool is registered."""
        return name in self._tools

    def unregister(self, name: str) -> bool:
        """Remove a tool from the registry. Returns ``True`` if found."""
        if name in self._tools:
            del self._tools[name]
            del self._info[name]
            logger.debug("Unregistered tool: %s", name)
            return True
        return False
