"""ToolManager — registry for tool registration, lookup, and execution."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from nexus_llm.tools.tool import Tool
from nexus_llm.tools.builtins import BuiltinTools
from nexus_llm.tools.builder import ToolBuilder

logger = logging.getLogger(__name__)


class ToolNotFoundError(KeyError):
    """Raised when a requested tool does not exist."""


class DuplicateToolError(ValueError):
    """Raised when attempting to register a tool with a name already in use."""


class ToolManager:
    """Central registry for :class:`Tool` instances.

    Supports registration, lookup, listing, execution, and automatic
    registration of all built-in tools.

    Example
    -------
    >>> mgr = ToolManager()
    >>> mgr.register_builtin_tools()
    >>> mgr.execute("calculator", expression="2 + 3 * 4")
    14
    """

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, tool: Tool) -> None:
        """Register a :class:`Tool`.

        Raises
        ------
        DuplicateToolError
            If a tool with the same name already exists.
        TypeError
            If *tool* is not a :class:`Tool` instance.
        """
        if not isinstance(tool, Tool):
            raise TypeError(f"Expected Tool instance, got {type(tool)!r}")
        if tool.name in self._tools:
            raise DuplicateToolError(f"Tool {tool.name!r} is already registered")
        self._tools[tool.name] = tool
        logger.info("Registered tool %r", tool.name)

    def register_builtin_tools(self) -> None:
        """Register all :class:`BuiltinTools` methods as Tool instances."""
        bt = BuiltinTools()
        builtin_defs = [
            {
                "name": "calculator",
                "description": "Evaluate a mathematical expression safely.",
                "func": bt.calculator,
                "params": [
                    {"name": "expression", "type": "string", "required": True,
                     "description": "Math expression to evaluate"},
                ],
            },
            {
                "name": "text_transform",
                "description": "Apply a text transformation (upper, lower, title, etc.).",
                "func": bt.text_transform,
                "params": [
                    {"name": "text", "type": "string", "required": True,
                     "description": "Input text"},
                    {"name": "operation", "type": "string", "required": True,
                     "description": "Operation: upper, lower, title, capitalize, reverse, strip, swapcase"},
                ],
            },
            {
                "name": "json_parser",
                "description": "Parse a JSON string and optionally extract a key.",
                "func": bt.json_parser,
                "params": [
                    {"name": "json_str", "type": "string", "required": True,
                     "description": "JSON string to parse"},
                    {"name": "key", "type": "string", "required": False,
                     "description": "Dot-separated key path to extract"},
                ],
            },
            {
                "name": "file_reader",
                "description": "Read a text file and return its content.",
                "func": bt.file_reader,
                "params": [
                    {"name": "path", "type": "string", "required": True,
                     "description": "File path"},
                    {"name": "encoding", "type": "string", "required": False,
                     "description": "Text encoding (default: utf-8)"},
                ],
            },
            {
                "name": "table_formatter",
                "description": "Format data as a text table (simple, markdown, csv).",
                "func": bt.table_formatter,
                "params": [
                    {"name": "data", "type": "array", "required": True,
                     "description": "List of row dictionaries"},
                    {"name": "format", "type": "string", "required": False,
                     "description": "Output format: simple, markdown, csv"},
                ],
            },
            {
                "name": "unit_converter",
                "description": "Convert a value between common units (length, weight, temperature).",
                "func": bt.unit_converter,
                "params": [
                    {"name": "value", "type": "number", "required": True,
                     "description": "Numeric value to convert"},
                    {"name": "from_unit", "type": "string", "required": True,
                     "description": "Source unit"},
                    {"name": "to_unit", "type": "string", "required": True,
                     "description": "Target unit"},
                ],
            },
        ]

        for defn in builtin_defs:
            builder = ToolBuilder().name(defn["name"]).description(defn["description"]).function(defn["func"])
            for p in defn["params"]:
                builder.param(
                    p["name"],
                    type=p["type"],
                    required=p["required"],
                    description=p.get("description", ""),
                )
            tool = builder.build()
            self.register(tool)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_tool(self, name: str) -> Tool:
        """Return the tool registered under *name*.

        Raises
        ------
        ToolNotFoundError
            If *name* is not registered.
        """
        if name not in self._tools:
            raise ToolNotFoundError(name)
        return self._tools[name]

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------

    def list_tools(self) -> List[str]:
        """Return a sorted list of registered tool names."""
        return sorted(self._tools.keys())

    def list_tools_details(self) -> List[Dict[str, Any]]:
        """Return a list of dicts with name and description for each tool."""
        return [tool.to_dict() for tool in self._tools.values()]

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(self, name: str, **kwargs: Any) -> Any:
        """Look up a tool by *name* and execute it.

        Raises
        ------
        ToolNotFoundError
            If *name* is not registered.
        """
        tool = self.get_tool(name)
        logger.info("Executing tool %r", name)
        return tool.execute(**kwargs)

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def unregister(self, name: str) -> None:
        """Remove a registered tool."""
        if name not in self._tools:
            raise ToolNotFoundError(name)
        del self._tools[name]

    def clear(self) -> None:
        """Remove all registered tools."""
        self._tools.clear()
