"""ToolBuilder — fluent API for constructing Tool instances."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from nexus_llm.tools.tool import Tool

logger = logging.getLogger(__name__)


class ToolBuilder:
    """Fluent builder for :class:`Tool` instances.

    Example
    -------
    >>> tool = (
    ...     ToolBuilder()
    ...     .name("greet")
    ...     .description("Say hello")
    ...     .function(lambda name: f"Hello, {name}!")
    ...     .param("name", type="string", required=True, description="Person's name")
    ...     .build()
    ... )
    >>> tool.execute(name="Alice")
    'Hello, Alice!'
    """

    def __init__(self) -> None:
        self._name: Optional[str] = None
        self._description: Optional[str] = None
        self._func: Optional[Callable] = None
        self._parameters: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Fluent setters
    # ------------------------------------------------------------------

    def name(self, name: str) -> "ToolBuilder":
        """Set the tool name."""
        self._name = name
        return self

    def description(self, desc: str) -> "ToolBuilder":
        """Set the tool description."""
        self._description = desc
        return self

    def function(self, func: Callable) -> "ToolBuilder":
        """Set the underlying callable."""
        if not callable(func):
            raise TypeError(f"func must be callable, got {type(func)!r}")
        self._func = func
        return self

    def param(
        self,
        name: str,
        *,
        type: str = "string",
        required: bool = False,
        default: Any = None,
        description: str = "",
    ) -> "ToolBuilder":
        """Add a parameter declaration.

        Parameters
        ----------
        name:
            Parameter name.
        type:
            JSON-schema type (``"string"``, ``"integer"``, ``"number"``,
            ``"boolean"``, ``"array"``, ``"object"``).
        required:
            Whether the parameter is mandatory.
        default:
            Default value (not yet used at execution time; documented in schema).
        description:
            Human-readable description of the parameter.
        """
        self._parameters.append({
            "name": name,
            "type": type,
            "required": required,
            "default": default,
            "description": description,
        })
        return self

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> Tool:
        """Construct and return a :class:`Tool` from the accumulated state.

        Raises
        ------
        ValueError
            If ``name``, ``description``, or ``function`` have not been set.
        """
        if not self._name:
            raise ValueError("Tool name is required")
        if not self._description:
            raise ValueError("Tool description is required")
        if self._func is None:
            raise ValueError("Tool function is required")

        schema = self._build_schema()
        tool = Tool(
            name=self._name,
            description=self._description,
            func=self._func,
            parameters_schema=schema,
        )
        logger.debug("Built tool %r", self._name)
        return tool

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_schema(self) -> Dict[str, Any]:
        """Convert accumulated parameters into a JSON-schema dict."""
        properties: Dict[str, Any] = {}
        required: List[str] = []

        for p in self._parameters:
            prop: Dict[str, Any] = {"type": p["type"]}
            if p["description"]:
                prop["description"] = p["description"]
            if p["default"] is not None:
                prop["default"] = p["default"]
            properties[p["name"]] = prop
            if p["required"]:
                required.append(p["name"])

        schema: Dict[str, Any] = {
            "type": "object",
            "properties": properties,
        }
        if required:
            schema["required"] = required
        return schema

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> "ToolBuilder":
        """Clear all builder state and return *self* for reuse."""
        self._name = None
        self._description = None
        self._func = None
        self._parameters.clear()
        return self
