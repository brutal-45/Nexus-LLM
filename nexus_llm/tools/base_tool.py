"""Nexus-LLM Base Tool Class.

Provides the BaseTool abstract class and supporting data structures
that all tools must implement.
"""

import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ParameterType(Enum):
    """Types for tool parameters."""

    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"


@dataclass
class ToolParameter:
    """Definition of a tool parameter.

    Attributes:
        name: Parameter name.
        type: Parameter type.
        description: Human-readable description.
        required: Whether the parameter is required.
        default: Default value if not provided.
        choices: List of allowed values.
    """

    name: str
    type: ParameterType = ParameterType.STRING
    description: str = ""
    required: bool = True
    default: Any = None
    choices: Optional[List[Any]] = None


@dataclass
class ToolResult:
    """Result from a tool execution.

    Attributes:
        id: Unique result identifier.
        tool_name: Name of the tool that produced this result.
        success: Whether execution was successful.
        output: The output data.
        error: Error message if execution failed.
        duration_ms: Execution duration in milliseconds.
        metadata: Additional result metadata.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tool_name: str = ""
    success: bool = True
    output: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "tool_name": self.tool_name,
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }


class BaseTool(ABC):
    """Abstract base class for all Nexus-LLM tools.

    Every tool must implement the `execute` method and declare
    its parameters via the `parameters` property.

    Attributes:
        name: Unique tool name.
        description: Human-readable tool description.
    """

    def __init__(self, name: Optional[str] = None, description: Optional[str] = None) -> None:
        self._name = name or self.__class__.__name__.lower().replace("tool", "")
        self._description = description or self.__doc__ or ""
        self._enabled = True
        logger.debug("Tool initialized: %s", self._name)

    @property
    def name(self) -> str:
        """Unique tool name."""
        return self._name

    @property
    def description(self) -> str:
        """Human-readable tool description."""
        return self._description.strip()

    @property
    def enabled(self) -> bool:
        """Whether the tool is currently enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    @property
    @abstractmethod
    def parameters(self) -> List[ToolParameter]:
        """Declare the parameters this tool accepts."""
        ...

    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with the given parameters.

        Args:
            **kwargs: Tool parameters.

        Returns:
            A ToolResult with execution output.
        """
        ...

    def validate_inputs(self, kwargs: Dict[str, Any]) -> List[str]:
        """Validate input parameters against declared parameters.

        Args:
            kwargs: Provided parameters.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors: List[str] = []
        param_map = {p.name: p for p in self.parameters}

        # Check required parameters
        for param in self.parameters:
            if param.required and param.name not in kwargs and param.default is None:
                errors.append(f"Missing required parameter: {param.name}")

        # Check types and choices
        for key, value in kwargs.items():
            if key not in param_map:
                errors.append(f"Unknown parameter: {key}")
                continue

            param = param_map[key]
            if param.choices is not None and value not in param.choices:
                errors.append(
                    f"Parameter '{key}' value {value!r} not in allowed choices: {param.choices}"
                )

        return errors

    def run(self, **kwargs: Any) -> ToolResult:
        """Run the tool with validation and timing.

        Args:
            **kwargs: Tool parameters.

        Returns:
            A ToolResult with execution output.
        """
        if not self._enabled:
            return ToolResult(
                tool_name=self._name,
                success=False,
                error=f"Tool '{self._name}' is disabled",
            )

        errors = self.validate_inputs(kwargs)
        if errors:
            return ToolResult(
                tool_name=self._name,
                success=False,
                error="Validation errors: " + "; ".join(errors),
            )

        # Apply defaults
        for param in self.parameters:
            if param.name not in kwargs and param.default is not None:
                kwargs[param.name] = param.default

        start = time.perf_counter()
        try:
            result = self.execute(**kwargs)
            result.tool_name = self._name
        except Exception as exc:
            result = ToolResult(
                tool_name=self._name,
                success=False,
                error=str(exc),
            )
        result.duration_ms = (time.perf_counter() - start) * 1000
        return result

    def schema(self) -> Dict[str, Any]:
        """Return a JSON schema describing this tool.

        Returns:
            Dictionary with tool name, description, and parameters.
        """
        return {
            "name": self._name,
            "description": self.description,
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type.value,
                    "description": p.description,
                    "required": p.required,
                    "default": p.default,
                    "choices": p.choices,
                }
                for p in self.parameters
            ],
        }
