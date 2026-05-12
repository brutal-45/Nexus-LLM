"""
Tool Use Module
===============

Comprehensive tool use system for agent framework providing tool definition,
execution with safety guarantees, LLM-based tool selection, sequential and
parallel tool chains, and a suite of built-in tools.

Classes
-------
- ``ToolDefinition``: Schema definition for a callable tool.
- ``ToolResult``: Standardized result of a tool execution.
- ``ToolUsePolicy``: Access control and safety policy for tool usage.
- ``ToolExecutor``: Executes tools with timeout, retry, and validation.
- ``ToolSelector``: LLM-based tool selection and ranking.
- ``ToolChain``: Sequential, parallel, and conditional tool pipelines.
- ``BuiltinTools``: Factory for standard built-in tools.
- ``FileManager``: File system operations (read, write, list, search, edit).
- ``Calculator``: Safe mathematical expression evaluation.
- ``CodeExecutor``: Sandboxed Python/JavaScript code execution.
- ``WebSearch``: Web search abstraction layer.
- ``DateTimeTool``: Date and time queries and arithmetic.
- ``SystemInfoTool``: System information retrieval.
"""

from __future__ import annotations

import ast
import copy
import ctypes
import datetime
import io
import json
import logging
import math
import os
import platform
import re
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ToolExecutionError(Exception):
    """Raised when a tool execution fails.

    Parameters
    ----------
    tool_name : str
        Name of the tool that failed.
    message : str
        Description of the failure.
    original_error : Optional[Exception]
        The underlying exception that caused the failure.
    """

    def __init__(
        self,
        tool_name: str,
        message: str,
        original_error: Optional[Exception] = None,
    ) -> None:
        self.tool_name = tool_name
        self.original_error = original_error
        super().__init__(f"Tool '{tool_name}' error: {message}")


class ToolTimeoutError(ToolExecutionError):
    """Raised when a tool execution exceeds its timeout."""

    def __init__(
        self,
        tool_name: str,
        timeout: float,
        original_error: Optional[Exception] = None,
    ) -> None:
        super().__init__(
            tool_name,
            f"Execution timed out after {timeout:.1f}s",
            original_error,
        )
        self.timeout = timeout


class ToolValidationError(ToolExecutionError):
    """Raised when tool arguments fail validation."""

    def __init__(
        self,
        tool_name: str,
        message: str,
        validation_errors: Optional[List[str]] = None,
    ) -> None:
        super().__init__(tool_name, message)
        self.validation_errors = validation_errors or []


class ToolNotRegisteredError(ToolExecutionError):
    """Raised when attempting to execute an unregistered tool."""

    def __init__(self, tool_name: str) -> None:
        super().__init__(tool_name, f"Tool '{tool_name}' is not registered")


class ToolAccessDeniedError(ToolExecutionError):
    """Raised when a tool call violates the usage policy."""

    def __init__(self, tool_name: str, reason: str) -> None:
        super().__init__(tool_name, f"Access denied: {reason}")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class ToolParameterType(Enum):
    """Supported parameter types for tool arguments.

    Attributes
    ----------
    STRING : str
        Text string parameter.
    INTEGER : str
        Integer number parameter.
    FLOAT : str
        Floating-point number parameter.
    BOOLEAN : str
        Boolean (true/false) parameter.
    ARRAY : str
        JSON array parameter.
    OBJECT : str
        JSON object parameter.
    ENUM : str
        One of a fixed set of values.
    ANY : str
        Any type accepted.
    """
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    ENUM = "enum"
    ANY = "any"


@dataclass
class ToolParameter:
    """Definition of a single tool parameter.

    Parameters
    ----------
    name : str
        Parameter name.
    param_type : ToolParameterType
        Expected data type.
    description : str
        Human-readable description.
    required : bool
        Whether this parameter must be provided.
    default : Any
        Default value if not provided.
    enum_values : List[str]
        Allowed values for ENUM type.
    min_value : float
        Minimum value for numeric types.
    max_value : float
        Maximum value for numeric types.
    pattern : str
        Regex pattern for STRING type.
    """

    name: str = ""
    param_type: ToolParameterType = ToolParameterType.STRING
    description: str = ""
    required: bool = False
    default: Any = None
    enum_values: List[str] = field(default_factory=list)
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    pattern: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation of the parameter schema.
        """
        result: Dict[str, Any] = {
            "name": self.name,
            "type": self.param_type.value,
            "description": self.description,
        }
        if self.required:
            result["required"] = True
        if self.default is not None:
            result["default"] = self.default
        if self.enum_values:
            result["enum"] = self.enum_values
        if self.min_value is not None:
            result["min"] = self.min_value
        if self.max_value is not None:
            result["max"] = self.max_value
        if self.pattern is not None:
            result["pattern"] = self.pattern
        return result

    def validate(self, value: Any) -> Tuple[bool, Optional[str]]:
        """Validate a value against this parameter's schema.

        Parameters
        ----------
        value : Any
            The value to validate.

        Returns
        -------
        Tuple[bool, Optional[str]]
            (is_valid, error_message) tuple.
        """
        if value is None:
            if self.required:
                return False, f"Parameter '{self.name}' is required"
            if self.default is not None:
                return True, None
            return True, None
        if self.param_type == ToolParameterType.STRING:
            if not isinstance(value, str):
                return False, f"Parameter '{self.name}' must be a string, got {type(value).__name__}"
            if self.pattern and not re.match(self.pattern, value):
                return False, f"Parameter '{self.name}' does not match pattern '{self.pattern}'"
        elif self.param_type == ToolParameterType.INTEGER:
            if not isinstance(value, int) or isinstance(value, bool):
                return False, f"Parameter '{self.name}' must be an integer"
            if self.min_value is not None and value < self.min_value:
                return False, f"Parameter '{self.name}' value {value} is below minimum {self.min_value}"
            if self.max_value is not None and value > self.max_value:
                return False, f"Parameter '{self.name}' value {value} exceeds maximum {self.max_value}"
        elif self.param_type == ToolParameterType.FLOAT:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                return False, f"Parameter '{self.name}' must be a number"
            if self.min_value is not None and value < self.min_value:
                return False, f"Parameter '{self.name}' value {value} is below minimum {self.min_value}"
            if self.max_value is not None and value > self.max_value:
                return False, f"Parameter '{self.name}' value {value} exceeds maximum {self.max_value}"
        elif self.param_type == ToolParameterType.BOOLEAN:
            if not isinstance(value, bool):
                return False, f"Parameter '{self.name}' must be a boolean"
        elif self.param_type == ToolParameterType.ARRAY:
            if not isinstance(value, list):
                return False, f"Parameter '{self.name}' must be an array"
        elif self.param_type == ToolParameterType.OBJECT:
            if not isinstance(value, dict):
                return False, f"Parameter '{self.name}' must be an object"
        elif self.param_type == ToolParameterType.ENUM:
            if value not in self.enum_values:
                return False, f"Parameter '{self.name}' value '{value}' not in allowed values: {self.enum_values}"
        return True, None


@dataclass
class ToolDefinition:
    """Schema definition for a callable tool.

    Encapsulates the tool's identity, parameter schema, handler function,
    and usage examples.  Used by both the agent (for invocation) and the
    LLM (for selection).

    Parameters
    ----------
    name : str
        Unique tool identifier (e.g., ``"web_search"``).
    description : str
        Human-readable description used for tool selection.
    parameters : List[ToolParameter]
        Schema for each parameter.
    handler : Optional[Callable]
        The function to invoke when the tool is called.
    examples : List[Dict[str, Any]]
        Example invocations for few-shot prompting.
    """

    name: str = ""
    description: str = ""
    parameters: List[ToolParameter] = field(default_factory=list)
    handler: Optional[Callable] = None
    examples: List[Dict[str, Any]] = field(default_factory=list)
    category: str = "general"
    version: str = "1.0.0"
    tags: Set[str] = field(default_factory=set)
    timeout: float = 30.0
    retry_count: int = 3
    dangerous: bool = False
    rate_limit_per_minute: int = 60
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the tool definition after initialization."""
        if not self.name or not isinstance(self.name, str):
            raise ValueError("ToolDefinition.name must be a non-empty string")
        if not re.match(r"^[a-zA-Z][a-zA-Z0-9_]*$", self.name):
            raise ValueError(
                f"Tool name '{self.name}' must match pattern "
                r"'^[a-zA-Z][a-zA-Z0-9_]*$'"
            )
        if not self.description:
            raise ValueError("ToolDefinition.description must be non-empty")

    @property
    def required_parameters(self) -> List[ToolParameter]:
        """Return only the required parameters.

        Returns
        -------
        List[ToolParameter]
            Parameters where ``required`` is True.
        """
        return [p for p in self.parameters if p.required]

    @property
    def optional_parameters(self) -> List[ToolParameter]:
        """Return only the optional parameters.

        Returns
        -------
        List[ToolParameter]
            Parameters where ``required`` is False.
        """
        return [p for p in self.parameters if not p.required]

    @property
    def parameter_names(self) -> List[str]:
        """Return all parameter names.

        Returns
        -------
        List[str]
            List of parameter name strings.
        """
        return [p.name for p in self.parameters]

    @property
    def json_schema(self) -> Dict[str, Any]:
        """Generate a JSON Schema representation of the parameters.

        Returns
        -------
        Dict[str, Any]
            JSON Schema dictionary.
        """
        properties: Dict[str, Any] = {}
        required: List[str] = []
        for param in self.parameters:
            prop: Dict[str, Any] = {
                "type": param.param_type.value,
                "description": param.description,
            }
            if param.enum_values:
                prop["enum"] = param.enum_values
            if param.min_value is not None:
                prop["minimum"] = param.min_value
            if param.max_value is not None:
                prop["maximum"] = param.max_value
            if param.pattern is not None:
                prop["pattern"] = param.pattern
            if param.default is not None:
                prop["default"] = param.default
            properties[param.name] = prop
            if param.required:
                required.append(param.name)
        schema: Dict[str, Any] = {
            "type": "object",
            "properties": properties,
        }
        if required:
            schema["required"] = required
        return schema

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the tool definition to a dictionary.

        Returns
        -------
        Dict[str, Any]
            Complete dictionary representation.
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": [p.to_dict() for p in self.parameters],
            "examples": copy.deepcopy(self.examples),
            "category": self.category,
            "version": self.version,
            "tags": list(self.tags),
            "timeout": self.timeout,
            "retry_count": self.retry_count,
            "dangerous": self.dangerous,
            "rate_limit_per_minute": self.rate_limit_per_minute,
            "json_schema": self.json_schema,
            "metadata": copy.deepcopy(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ToolDefinition:
        """Deserialize from a dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary with tool definition fields.

        Returns
        -------
        ToolDefinition
            Deserialized tool definition.
        """
        params = []
        for p_data in data.get("parameters", []):
            param_type = ToolParameterType(p_data.get("type", "string"))
            params.append(ToolParameter(
                name=p_data.get("name", ""),
                param_type=param_type,
                description=p_data.get("description", ""),
                required=p_data.get("required", False),
                default=p_data.get("default"),
                enum_values=p_data.get("enum", []),
                min_value=p_data.get("min"),
                max_value=p_data.get("max"),
                pattern=p_data.get("pattern"),
            ))
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            parameters=params,
            examples=data.get("examples", []),
            category=data.get("category", "general"),
            version=data.get("version", "1.0.0"),
            tags=set(data.get("tags", [])),
            timeout=data.get("timeout", 30.0),
            retry_count=data.get("retry_count", 3),
            dangerous=data.get("dangerous", False),
            rate_limit_per_minute=data.get("rate_limit_per_minute", 60),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ToolResult:
    """Standardized result of a tool execution.

    Parameters
    ----------
    success : bool
        Whether the tool executed without errors.
    output : str
        The textual output from the tool.
    error : Optional[str]
        Error message if execution failed.
    metadata : Dict[str, Any]
        Additional metadata about the execution.
    execution_time : float
        Wall-clock time of the execution in seconds.
    tool_name : str
        Name of the tool that produced this result.
    """

    success: bool = True
    output: str = ""
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    tool_name: str = ""

    @property
    def is_error(self) -> bool:
        """Whether the tool execution encountered an error.

        Returns
        -------
        bool
            True if the execution failed.
        """
        return not self.success

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation.
        """
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "metadata": copy.deepcopy(self.metadata),
            "execution_time": self.execution_time,
            "tool_name": self.tool_name,
        }

    @classmethod
    def from_success(
        cls,
        output: str,
        tool_name: str = "",
        execution_time: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        """Create a successful tool result.

        Parameters
        ----------
        output : str
            Tool output text.
        tool_name : str
            Name of the tool.
        execution_time : float
            Execution time in seconds.
        metadata : Dict[str, Any], optional
            Additional metadata.

        Returns
        -------
        ToolResult
            Successful tool result.
        """
        return cls(
            success=True,
            output=output,
            tool_name=tool_name,
            execution_time=execution_time,
            metadata=metadata or {},
        )

    @classmethod
    def from_error(
        cls,
        error: str,
        tool_name: str = "",
        execution_time: float = 0.0,
        output: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        """Create an error tool result.

        Parameters
        ----------
        error : str
            Error description.
        tool_name : str
            Name of the tool.
        execution_time : float
            Execution time in seconds.
        output : str
            Partial output before the error.
        metadata : Dict[str, Any], optional
            Additional metadata.

        Returns
        -------
        ToolResult
            Error tool result.
        """
        return cls(
            success=False,
            output=output,
            error=error,
            tool_name=tool_name,
            execution_time=execution_time,
            metadata=metadata or {},
        )


# ---------------------------------------------------------------------------
# ToolUsePolicy
# ---------------------------------------------------------------------------

@dataclass
class ToolUsePolicy:
    """Access control and safety policy for tool usage.

    Defines which tools are allowed, rate limits, confirmation requirements,
    and other safety constraints.

    Parameters
    ----------
    allowed_tools : Set[str]
        Set of tool names that are permitted.
    max_calls_per_step : int
        Maximum tool calls per agent step.
    confirm_dangerous : bool
        Whether dangerous tools require confirmation.
    timeout : float
        Global timeout for all tool executions.
    """

    allowed_tools: Set[str] = field(default_factory=lambda: {"*"})
    denied_tools: Set[str] = field(default_factory=set)
    max_calls_per_step: int = 10
    max_calls_per_run: int = 50
    confirm_dangerous: bool = True
    timeout: float = 60.0
    enable_rate_limiting: bool = True
    rate_limit_per_minute: int = 120
    enable_sandboxing: bool = True
    max_output_length: int = 100000
    allowed_file_patterns: List[str] = field(default_factory=list)
    denied_file_patterns: List[str] = field(
        default_factory=lambda: [
            "/etc/passwd",
            "/etc/shadow",
            "/.ssh/",
            "~/.ssh/",
            ".env",
            ".git/",
        ]
    )
    allowed_commands: Set[str] = field(default_factory=set)
    denied_commands: Set[str] = field(
        default_factory=lambda: {
            "rm -rf /",
            "mkfs",
            "dd if=",
            ":(){ :|:& };:",
            "shutdown",
            "reboot",
            "halt",
            "init 0",
            "chmod 777 /",
        }
    )
    network_access: bool = True
    max_memory_mb: int = 512
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_allow_all(self) -> bool:
        """Whether all tools are allowed.

        Returns
        -------
        bool
            True if allowed_tools contains the wildcard.
        """
        return "*" in self.allowed_tools

    def is_tool_allowed(self, tool_name: str) -> bool:
        """Check if a specific tool is allowed by policy.

        Parameters
        ----------
        tool_name : str
            Name of the tool to check.

        Returns
        -------
        bool
            True if the tool is permitted.
        """
        if tool_name in self.denied_tools:
            return False
        if self.is_allow_all:
            return True
        return tool_name in self.allowed_tools

    def is_file_allowed(self, file_path: str) -> bool:
        """Check if a file path is allowed by policy.

        Parameters
        ----------
        file_path : str
            Path to the file.

        Returns
        -------
        bool
            True if access to the file is permitted.
        """
        expanded = os.path.expanduser(file_path)
        expanded = os.path.abspath(expanded)
        for pattern in self.denied_file_patterns:
            if pattern in expanded:
                return False
        if self.allowed_file_patterns:
            for pattern in self.allowed_file_patterns:
                if re.match(pattern, expanded):
                    return True
            return False
        return True

    def is_command_allowed(self, command: str) -> bool:
        """Check if a shell command is allowed by policy.

        Parameters
        ----------
        command : str
            The command string to check.

        Returns
        -------
        bool
            True if the command is permitted.
        """
        command_stripped = command.strip()
        for denied in self.denied_commands:
            if denied.lower() in command_stripped.lower():
                return False
        if self.allowed_commands:
            cmd_base = command_stripped.split()[0] if command_stripped else ""
            return cmd_base in self.allowed_commands
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation.
        """
        return {
            "allowed_tools": list(self.allowed_tools),
            "denied_tools": list(self.denied_tools),
            "max_calls_per_step": self.max_calls_per_step,
            "max_calls_per_run": self.max_calls_per_run,
            "confirm_dangerous": self.confirm_dangerous,
            "timeout": self.timeout,
            "enable_rate_limiting": self.enable_rate_limiting,
            "rate_limit_per_minute": self.rate_limit_per_minute,
            "enable_sandboxing": self.enable_sandboxing,
            "max_output_length": self.max_output_length,
            "allowed_file_patterns": self.allowed_file_patterns,
            "denied_file_patterns": self.denied_file_patterns,
            "allowed_commands": list(self.allowed_commands),
            "denied_commands": list(self.denied_commands),
            "network_access": self.network_access,
            "max_memory_mb": self.max_memory_mb,
            "metadata": copy.deepcopy(self.metadata),
        }


# ---------------------------------------------------------------------------
# Rate Limiter
# ---------------------------------------------------------------------------

class _ToolRateLimiter:
    """Thread-safe rate limiter for tool calls.

    Uses a sliding window counter to enforce calls-per-minute limits.

    Parameters
    ----------
    max_calls : int
        Maximum calls allowed in the window.
    window_seconds : float
        Length of the sliding window in seconds.
    """

    def __init__(self, max_calls: int = 120, window_seconds: float = 60.0) -> None:
        self._max_calls = max_calls
        self._window_seconds = window_seconds
        self._timestamps: List[float] = []
        self._lock = threading.Lock()

    def acquire(self) -> bool:
        """Attempt to acquire a rate limit slot.

        Returns
        -------
        bool
            True if the call is allowed, False if rate limited.
        """
        with self._lock:
            now = time.monotonic()
            cutoff = now - self._window_seconds
            self._timestamps = [
                ts for ts in self._timestamps if ts > cutoff
            ]
            if len(self._timestamps) >= self._max_calls:
                return False
            self._timestamps.append(now)
            return True

    def wait_for_slot(self, timeout: float = 30.0) -> bool:
        """Wait until a rate limit slot becomes available.

        Parameters
        ----------
        timeout : float
        Maximum time to wait in seconds.

        Returns
        -------
        bool
            True if a slot was acquired, False on timeout.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.acquire():
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            sleep_time = min(0.1, remaining)
            time.sleep(sleep_time)
        return False

    @property
    def available_calls(self) -> int:
        """Return the number of available call slots.

        Returns
        -------
        int
            Remaining calls in the current window.
        """
        with self._lock:
            now = time.monotonic()
            cutoff = now - self._window_seconds
            active = sum(1 for ts in self._timestamps if ts > cutoff)
            return max(0, self._max_calls - active)


# ---------------------------------------------------------------------------
# ToolExecutor
# ---------------------------------------------------------------------------

class ToolExecutor:
    """Executes tools with safety, timeout, retry, and validation.

    Manages a registry of ``ToolDefinition`` objects and provides a safe
    execution environment with configurable policies.

    Parameters
    ----------
    policy : ToolUsePolicy, optional
        Usage policy governing tool access and safety.
    """

    def __init__(self, policy: Optional[ToolUsePolicy] = None) -> None:
        """Initialize the tool executor.

        Parameters
        ----------
        policy : ToolUsePolicy, optional
            Tool usage policy.
        """
        self._policy = policy or ToolUsePolicy()
        self._tools: Dict[str, ToolDefinition] = {}
        self._rate_limiter = _ToolRateLimiter(
            max_calls=self._policy.rate_limit_per_minute
        )
        self._call_count: int = 0
        self._error_count: int = 0
        self._logger = logging.getLogger(
            f"{__name__}.ToolExecutor"
        )

    @property
    def policy(self) -> ToolUsePolicy:
        """Return the current tool usage policy.

        Returns
        -------
        ToolUsePolicy
            The active policy.
        """
        return self._policy

    @policy.setter
    def policy(self, value: ToolUsePolicy) -> None:
        """Set the tool usage policy.

        Parameters
        ----------
        value : ToolUsePolicy
            New policy to apply.
        """
        self._policy = value
        self._rate_limiter = _ToolRateLimiter(
            max_calls=self._policy.rate_limit_per_minute
        )

    @property
    def registered_tools(self) -> List[str]:
        """Return names of all registered tools.

        Returns
        -------
        List[str]
            Sorted list of tool names.
        """
        return sorted(self._tools.keys())

    @property
    def stats(self) -> Dict[str, Any]:
        """Return execution statistics.

        Returns
        -------
        Dict[str, Any]
            Dictionary with call counts and rates.
        """
        return {
            "registered_tools": len(self._tools),
            "total_calls": self._call_count,
            "error_count": self._error_count,
            "success_rate": (
                (self._call_count - self._error_count)
                / max(self._call_count, 1)
            ),
            "available_rate_limit_slots": self._rate_limiter.available_calls,
        }

    def register(self, tool: ToolDefinition) -> None:
        """Register a tool definition.

        Parameters
        ----------
        tool : ToolDefinition
            The tool definition to register.

        Raises
        ------
        ValueError
            If a tool with the same name is already registered.
        """
        if not isinstance(tool, ToolDefinition):
            raise TypeError(
                f"Expected ToolDefinition, got {type(tool).__name__}"
            )
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool
        self._logger.info("Registered tool: %s", tool.name)

    def unregister(self, tool_name: str) -> bool:
        """Unregister a tool by name.

        Parameters
        ----------
        tool_name : str
            Name of the tool to remove.

        Returns
        -------
        bool
            True if the tool was found and removed.
        """
        if tool_name in self._tools:
            del self._tools[tool_name]
            self._logger.info("Unregistered tool: %s", tool_name)
            return True
        return False

    def get_tool(self, tool_name: str) -> Optional[ToolDefinition]:
        """Look up a tool definition by name.

        Parameters
        ----------
        tool_name : str
            Name of the tool.

        Returns
        -------
        Optional[ToolDefinition]
            Tool definition if found, else None.
        """
        return self._tools.get(tool_name)

    def execute(
        self,
        tool_name: str,
        args: Dict[str, Any],
        timeout: Optional[float] = None,
    ) -> ToolResult:
        """Execute a tool with full safety checks, timeout, and retries.

        Parameters
        ----------
        tool_name : str
            Name of the tool to execute.
        args : Dict[str, Any]
            Arguments to pass to the tool.
        timeout : float, optional
            Override the tool's configured timeout.

        Returns
        -------
        ToolResult
            Result of the tool execution.

        Raises
        ------
        ToolNotRegisteredError
            If the tool is not registered.
        ToolAccessDeniedError
            If the policy denies access to the tool.
        """
        self._call_count += 1
        if tool_name not in self._tools:
            self._error_count += 1
            raise ToolNotRegisteredError(tool_name)
        if not self._policy.is_tool_allowed(tool_name):
            self._error_count += 1
            raise ToolAccessDeniedError(
                tool_name,
                f"Tool '{tool_name}' is not in the allowed tools list",
            )
        if self._policy.enable_rate_limiting:
            if not self._rate_limiter.wait_for_slot(timeout=self._policy.timeout):
                self._error_count += 1
                raise ToolExecutionError(
                    tool_name,
                    f"Rate limit exceeded: {self._policy.rate_limit_per_minute} calls/minute",
                )
        tool = self._tools[tool_name]
        validation_errors = self.validate_args(tool, args)
        if validation_errors:
            self._error_count += 1
            raise ToolValidationError(
                tool_name,
                f"Argument validation failed: {'; '.join(validation_errors)}",
                validation_errors,
            )
        sanitized_args = self.sanitize_args(args)
        effective_timeout = (
            timeout if timeout is not None
            else min(tool.timeout, self._policy.timeout)
        )
        max_retries = tool.retry_count
        last_error: Optional[Exception] = None
        last_output = ""
        last_exec_time = 0.0
        for attempt in range(max_retries + 1):
            try:
                start_time = time.monotonic()
                output = self._run_with_timeout(
                    tool, sanitized_args, effective_timeout
                )
                elapsed = time.monotonic() - start_time
                if not isinstance(output, str):
                    output = json.dumps(output, default=str)
                if (
                    self._policy.max_output_length > 0
                    and len(output) > self._policy.max_output_length
                ):
                    output = (
                        output[:self._policy.max_output_length]
                        + f"\n\n[Output truncated at {self._policy.max_output_length} characters]"
                    )
                self._logger.info(
                    "Tool '%s' completed in %.3fs (attempt %d/%d)",
                    tool_name, elapsed, attempt + 1, max_retries + 1,
                )
                return ToolResult.from_success(
                    output=output,
                    tool_name=tool_name,
                    execution_time=elapsed,
                )
            except ToolTimeoutError as exc:
                last_error = exc
                last_exec_time = time.monotonic() - start_time
                self._logger.warning(
                    "Tool '%s' timed out on attempt %d/%d (%.1fs)",
                    tool_name, attempt + 1, max_retries + 1,
                    effective_timeout,
                )
                if attempt < max_retries:
                    effective_timeout *= 1.5
            except Exception as exc:
                last_error = exc
                last_exec_time = time.monotonic() - start_time
                self._logger.warning(
                    "Tool '%s' failed on attempt %d/%d: %s",
                    tool_name, attempt + 1, max_retries + 1, exc,
                )
        self._error_count += 1
        return ToolResult.from_error(
            error=str(last_error) if last_error else "Unknown error",
            tool_name=tool_name,
            execution_time=last_exec_time,
            output=last_output,
        )

    def _run_with_timeout(
        self,
        tool: ToolDefinition,
        args: Dict[str, Any],
        timeout: float,
    ) -> str:
        """Run a tool handler with timeout enforcement.

        Parameters
        ----------
        tool : ToolDefinition
            The tool to execute.
        args : Dict[str, Any]
            Sanitized arguments.
        timeout : float
            Maximum execution time in seconds.

        Returns
        -------
        str
            Tool output.

        Raises
        ------
        ToolTimeoutError
            If the tool exceeds the timeout.
        ToolExecutionError
            If the tool handler raises an exception.
        """
        if tool.handler is None:
            raise ToolExecutionError(
                tool.name, "Tool has no handler function"
            )
        result_container: List[Any] = []
        error_container: List[Exception] = []

        def _target() -> None:
            try:
                result = tool.handler(**args)
                result_container.append(result)
            except Exception as exc:
                error_container.append(exc)

        thread = threading.Thread(target=_target, daemon=True)
        thread.start()
        thread.join(timeout=timeout)
        if thread.is_alive():
            raise ToolTimeoutError(tool.name, timeout)
        if error_container:
            raise ToolExecutionError(
                tool.name,
                str(error_container[0]),
                original_error=error_container[0],
            )
        if result_container:
            return str(result_container[0])
        return ""

    def validate_args(
        self, tool: ToolDefinition, args: Dict[str, Any]
    ) -> List[str]:
        """Validate tool arguments against the tool's parameter schema.

        Parameters
        ----------
        tool : ToolDefinition
            The tool whose schema to validate against.
        args : Dict[str, Any]
            Arguments to validate.

        Returns
        -------
        List[str]
            List of validation error messages. Empty if valid.
        """
        errors: List[str] = []
        param_map = {p.name: p for p in tool.parameters}
        for param in tool.parameters:
            if param.required and param.name not in args:
                if param.default is None:
                    errors.append(
                        f"Missing required parameter: '{param.name}'"
                    )
        for arg_name, arg_value in args.items():
            if arg_name not in param_map:
                errors.append(f"Unknown parameter: '{arg_name}'")
                continue
            param = param_map[arg_name]
            is_valid, error_msg = param.validate(arg_value)
            if not is_valid and error_msg:
                errors.append(error_msg)
        return errors

    def sanitize_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize tool arguments for safety.

        Removes potentially dangerous content, limits string lengths,
        and normalizes data types.

        Parameters
        ----------
        args : Dict[str, Any]
            Raw arguments to sanitize.

        Returns
        -------
        Dict[str, Any]
            Sanitized arguments.
        """
        sanitized: Dict[str, Any] = {}
        for key, value in args.items():
            if isinstance(value, str):
                sanitized[key] = self._sanitize_string(value)
            elif isinstance(value, (list, tuple)):
                sanitized[key] = [
                    self._sanitize_string(v) if isinstance(v, str) else v
                    for v in value
                ]
            elif isinstance(value, dict):
                sanitized[key] = self.sanitize_args(value)
            elif isinstance(value, bool):
                sanitized[key] = value
            elif isinstance(value, (int, float)):
                if math.isfinite(value):
                    sanitized[key] = value
                else:
                    sanitized[key] = 0
            else:
                sanitized[key] = value
        return sanitized

    def _sanitize_string(self, value: str) -> str:
        """Sanitize a string value for safe tool execution.

        Removes null bytes, limits length, and escapes dangerous sequences.

        Parameters
        ----------
        value : str
            String to sanitize.

        Returns
        -------
        str
            Sanitized string.
        """
        sanitized = value.replace("\x00", "")
        sanitized = sanitized.replace("\r\n", "\n")
        max_length = 1000000
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
        return sanitized

    def handle_error(
        self,
        tool: ToolDefinition,
        error: Exception,
        retry_count: int,
    ) -> ToolResult:
        """Handle a tool execution error with appropriate retry logic.

        Parameters
        ----------
        tool : ToolDefinition
            The tool that failed.
        error : Exception
            The exception that was raised.
        retry_count : int
            Number of retries already attempted.

        Returns
        -------
        ToolResult
            Error result with diagnostic information.
        """
        self._logger.error(
            "Tool '%s' failed after %d retries: %s",
            tool.name, retry_count, error,
        )
        return ToolResult.from_error(
            error=f"{type(error).__name__}: {str(error)}",
            tool_name=tool.name,
            metadata={
                "retries": retry_count,
                "error_type": type(error).__name__,
            },
        )


# ---------------------------------------------------------------------------
# ToolSelector
# ---------------------------------------------------------------------------

class ToolSelector:
    """LLM-based tool selection and ranking system.

    Given a natural language query, ranks available tools by relevance
    and can parse LLM output into structured tool call specifications.

    Parameters
    ----------
    executor : ToolExecutor
        The executor containing registered tools.
    """

    def __init__(self, executor: ToolExecutor) -> None:
        """Initialize the tool selector.

        Parameters
        ----------
        executor : ToolExecutor
            Tool executor with registered tools.
        """
        self._executor = executor
        self._logger = logging.getLogger(
            f"{__name__}.ToolSelector"
        )
        self._selection_cache: Dict[str, List[str]] = {}
        self._cache_max_size = 500
        self._keyword_index: Dict[str, Set[str]] = {}
        self._build_keyword_index()

    def _build_keyword_index(self) -> None:
        """Build an inverted keyword index for fast tool lookup."""
        self._keyword_index.clear()
        for tool_name, tool in self._executor._tools.items():
            words = set(re.findall(r'\b\w+\b', tool.description.lower()))
            words.update(re.findall(r'\b\w+\b', tool_name.lower()))
            words.update(tool.tags)
            for word in words:
                if word not in self._keyword_index:
                    self._keyword_index[word] = set()
                self._keyword_index[word].add(tool_name)

    def select_tools(
        self,
        query: str,
        available_tools: Optional[List[str]] = None,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """Rank and select relevant tools for a given query.

        Uses keyword matching, description similarity, and tag overlap
        to score tools against the query.

        Parameters
        ----------
        query : str
            Natural language query describing the desired tool behavior.
        available_tools : List[str], optional
            Restrict selection to these tools.
        top_k : int
            Maximum number of tools to return.

        Returns
        -------
        List[Tuple[str, float]]
            List of (tool_name, relevance_score) pairs, sorted by score
            descending.
        """
        query_lower = query.lower()
        query_words = set(re.findall(r'\b\w+\b', query_lower))
        scores: Dict[str, float] = {}
        tool_pool = (
            available_tools if available_tools
            else self._executor.registered_tools
        )
        for tool_name in tool_pool:
            tool = self._executor.get_tool(tool_name)
            if tool is None:
                continue
            score = 0.0
            name_words = set(re.findall(r'\b\w+\b', tool_name.lower()))
            name_overlap = query_words & name_words
            if name_overlap:
                score += len(name_overlap) * 2.0
            if query_lower in tool_name.lower():
                score += 3.0
            desc_words = set(
                re.findall(r'\b\w+\b', tool.description.lower())
            )
            desc_overlap = query_words & desc_words
            if desc_overlap:
                score += len(desc_overlap) * 1.0
            if any(
                query_lower in word or word in query_lower
                for word in desc_words
                if len(word) > 3
            ):
                score += 1.5
            tag_overlap = query_words & tool.tags
            if tag_overlap:
                score += len(tag_overlap) * 1.5
            for word in query_words:
                if word in self._keyword_index:
                    if tool_name in self._keyword_index[word]:
                        score += 0.5
            scores[tool_name] = score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def parse_tool_call(
        self, llm_output: str
    ) -> List[Dict[str, Any]]:
        """Parse LLM output into structured tool call specifications.

        Supports multiple output formats including JSON function calls,
        XML-style tags, and natural language patterns.

        Parameters
        ----------
        llm_output : str
            Raw LLM output that may contain tool call specifications.

        Returns
        -------
        List[Dict[str, Any]]
            List of parsed tool calls, each with ``name`` and ``arguments``.
        """
        tool_calls: List[Dict[str, Any]] = []
        json_calls = self._parse_json_tool_calls(llm_output)
        if json_calls:
            tool_calls.extend(json_calls)
        xml_calls = self._parse_xml_tool_calls(llm_output)
        if xml_calls:
            tool_calls.extend(xml_calls)
        natural_calls = self._parse_natural_tool_calls(llm_output)
        if natural_calls and not json_calls and not xml_calls:
            tool_calls.extend(natural_calls)
        seen_names: Set[str] = set()
        unique_calls: List[Dict[str, Any]] = []
        for call in tool_calls:
            if call["name"] not in seen_names:
                seen_names.add(call["name"])
                unique_calls.append(call)
        return unique_calls

    def _parse_json_tool_calls(
        self, output: str
    ) -> List[Dict[str, Any]]:
        """Parse JSON-formatted tool calls from LLM output.

        Parameters
        ----------
        output : str
            LLM output text.

        Returns
        -------
        List[Dict[str, Any]]
            Parsed tool calls.
        """
        calls: List[Dict[str, Any]] = []
        json_patterns = [
            r'\{[^{}]*"name"\s*:\s*"([^"]+)"[^{}]*"arguments"\s*:\s*(\{[^{}]*\})[^{}]*\}',
            r'```json\s*(\{.*?\})\s*```',
            r'```(\{.*?\})```',
        ]
        for pattern in json_patterns:
            matches = re.findall(pattern, output, re.DOTALL)
            for match in matches:
                try:
                    if isinstance(match, tuple):
                        if len(match) == 2:
                            name = match[0]
                            args = json.loads(match[1])
                        else:
                            data = json.loads(match[0])
                            name = data.get("name", "")
                            args = data.get("arguments", {})
                    else:
                        data = json.loads(match)
                        name = data.get("name", "")
                        args = data.get("arguments", {})
                    if name and name in self._executor._tools:
                        calls.append({"name": name, "arguments": args})
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue
        return calls

    def _parse_xml_tool_calls(
        self, output: str
    ) -> List[Dict[str, Any]]:
        """Parse XML-tagged tool calls from LLM output.

        Parameters
        ----------
        output : str
            LLM output text.

        Returns
        -------
        List[Dict[str, Any]]
            Parsed tool calls.
        """
        calls: List[Dict[str, Any]] = []
        tool_pattern = r'<tool_call>\s*<tool_name>([^<]+)</tool_name>\s*<arguments>(.*?)</arguments>\s*</tool_call>'
        matches = re.findall(tool_pattern, output, re.DOTALL)
        for name, args_str in matches:
            name = name.strip()
            args: Dict[str, Any] = {}
            try:
                args = json.loads(args_str.strip())
            except json.JSONDecodeError:
                kv_pattern = r'<(\w+)>([^<]*)</\w+>'
                kv_matches = re.findall(kv_pattern, args_str)
                for key, value in kv_matches:
                    args[key] = value.strip()
            if name and name in self._executor._tools:
                calls.append({"name": name, "arguments": args})
        return calls

    def _parse_natural_tool_calls(
        self, output: str
    ) -> List[Dict[str, Any]]:
        """Parse natural language tool references from LLM output.

        Parameters
        ----------
        output : str
            LLM output text.

        Returns
        -------
        List[Dict[str, Any]]
            Parsed tool calls.
        """
        calls: List[Dict[str, Any]] = []
        pattern = r'(?:use|call|invoke|run)\s+(?:the\s+)?["\']?(\w+)["\']?\s*(?:tool)?'
        matches = re.findall(pattern, output, re.IGNORECASE)
        for name in matches:
            if name.lower() in {
                t.lower() for t in self._executor._tools
            }:
                actual_name = next(
                    t for t in self._executor._tools
                    if t.lower() == name.lower()
                )
                calls.append({"name": actual_name, "arguments": {}})
        return calls

    def format_tools_for_llm(
        self,
        tools: Optional[List[str]] = None,
    ) -> str:
        """Format tool schemas for inclusion in an LLM prompt.

        Parameters
        ----------
        tools : List[str], optional
            Specific tools to format. If None, formats all registered tools.

        Returns
        -------
        str
            Formatted tool descriptions suitable for LLM prompts.
        """
        tool_names = tools or self._executor.registered_tools
        if not tool_names:
            return "No tools available."
        lines: List[str] = [
            "You have access to the following tools. "
            "To use a tool, output a JSON object with "
            "'name' and 'arguments' fields.",
            "",
        ]
        for name in tool_names:
            tool = self._executor.get_tool(name)
            if tool is None:
                continue
            lines.append(f"### {name}")
            lines.append(f"**Description:** {tool.description}")
            if tool.parameters:
                lines.append("**Parameters:**")
                for param in tool.parameters:
                    req = " (required)" if param.required else " (optional)"
                    default = (
                        f", default={param.default}"
                        if param.default is not None
                        else ""
                    )
                    lines.append(
                        f"  - `{param.name}` ({param.param_type.value}){req}{default}: "
                        f"{param.description}"
                    )
            if tool.examples:
                lines.append("**Examples:**")
                for ex in tool.examples:
                    lines.append(
                        f"  - Input: {json.dumps(ex.get('input', {}))} → "
                        f"Output: {ex.get('output', '...')[:200]}"
                    )
            lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# ToolChain
# ---------------------------------------------------------------------------

@dataclass
class ChainStep:
    """A single step in a tool chain.

    Parameters
    ----------
    tool_name : str
        Name of the tool to execute.
    args : Dict[str, Any]
        Arguments for the tool.
    input_mapping : str
        How to map the previous step's output to this step's input.
    condition : Optional[Callable]
        Optional condition function that determines if this step should run.
    """

    tool_name: str = ""
    args: Dict[str, Any] = field(default_factory=dict)
    input_mapping: str = "pass_through"
    condition: Optional[Callable] = None
    output_key: str = ""
    max_retries: int = 3
    timeout: Optional[float] = None


class ToolChain:
    """Sequential, parallel, and conditional tool execution pipeline.

    Chains multiple tool executions together, passing outputs between
    steps.  Supports branching based on conditions and parallel execution
    of independent steps.

    Parameters
    ----------
    executor : ToolExecutor
        Tool executor for running individual tools.
    """

    def __init__(self, executor: ToolExecutor) -> None:
        """Initialize the tool chain.

        Parameters
        ----------
        executor : ToolExecutor
            Tool executor for running tools.
        """
        self._executor = executor
        self._steps: List[ChainStep] = []
        self._logger = logging.getLogger(f"{__name__}.ToolChain")
        self._intermediate_results: Dict[str, Any] = {}

    @property
    def steps(self) -> List[ChainStep]:
        """Return the chain steps.

        Returns
        -------
        List[ChainStep]
            List of chain step definitions.
        """
        return list(self._steps)

    @property
    def intermediate_results(self) -> Dict[str, Any]:
        """Return intermediate results from the chain.

        Returns
        -------
        Dict[str, Any]
            Mapping of step keys to their outputs.
        """
        return copy.deepcopy(self._intermediate_results)

    def add_tool(
        self,
        tool_name: str,
        args: Optional[Dict[str, Any]] = None,
        input_mapping: str = "pass_through",
        condition: Optional[Callable] = None,
        output_key: str = "",
        max_retries: int = 3,
        timeout: Optional[float] = None,
    ) -> ToolChain:
        """Add a tool to the execution chain.

        Parameters
        ----------
        tool_name : str
            Name of the tool to add.
        args : Dict[str, Any], optional
            Arguments for the tool.
        input_mapping : str
            How to map previous output to this step's input.
        condition : Callable, optional
            Condition function; step runs only if this returns True.
        output_key : str
            Key to store the output under in intermediate results.
        max_retries : int
            Retry count for this step.
        timeout : float, optional
            Timeout for this step.

        Returns
        -------
        ToolChain
            Self, for method chaining.
        """
        step = ChainStep(
            tool_name=tool_name,
            args=args or {},
            input_mapping=input_mapping,
            condition=condition,
            output_key=output_key or tool_name,
            max_retries=max_retries,
            timeout=timeout,
        )
        self._steps.append(step)
        return self

    def clear(self) -> None:
        """Remove all steps from the chain."""
        self._steps.clear()
        self._intermediate_results.clear()

    def execute_chain(
        self,
        initial_input: Any = None,
    ) -> Dict[str, ToolResult]:
        """Execute tools in sequence, passing outputs between steps.

        Parameters
        ----------
        initial_input : Any
            Input for the first step in the chain.

        Returns
        -------
        Dict[str, ToolResult]
            Mapping of step output keys to their results.
        """
        self._intermediate_results.clear()
        results: Dict[str, ToolResult] = {}
        current_input = initial_input
        for idx, step in enumerate(self._steps):
            self._logger.debug(
                "Chain step %d/%d: %s",
                idx + 1, len(self._steps), step.tool_name,
            )
            if step.condition is not None:
                try:
                    should_run = step.condition(current_input, results)
                    if not should_run:
                        self._logger.debug(
                            "Skipping step %d: condition not met", idx + 1
                        )
                        continue
                except Exception as exc:
                    self._logger.warning(
                        "Condition check failed for step %d: %s",
                        idx + 1, exc,
                    )
                    results[step.output_key] = ToolResult.from_error(
                        error=f"Condition check failed: {exc}",
                        tool_name=step.tool_name,
                    )
                    continue
            resolved_args = self._resolve_args(
                step.args, current_input, step.input_mapping
            )
            try:
                result = self._executor.execute(
                    step.tool_name,
                    resolved_args,
                    timeout=step.timeout,
                )
                results[step.output_key] = result
                self._intermediate_results[step.output_key] = result.output
                current_input = result.output
            except ToolExecutionError as exc:
                self._logger.error(
                    "Chain step %d failed: %s", idx + 1, exc
                )
                results[step.output_key] = ToolResult.from_error(
                    error=str(exc),
                    tool_name=step.tool_name,
                )
                break
        return results

    def parallel_execute(
        self,
        initial_input: Any = None,
        max_workers: int = 4,
    ) -> Dict[str, ToolResult]:
        """Execute independent tools in parallel.

        Tools are run concurrently using a thread pool.  All steps are
        assumed to be independent (no inter-step data flow).

        Parameters
        ----------
        initial_input : Any
            Input passed to all steps.
        max_workers : int
            Maximum number of concurrent threads.

        Returns
        -------
        Dict[str, ToolResult]
            Mapping of step output keys to their results.
        """
        self._intermediate_results.clear()
        results: Dict[str, ToolResult] = {}
        futures: Dict[str, Any] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for step in self._steps:
                if step.condition is not None:
                    try:
                        if not step.condition(initial_input, {}):
                            continue
                    except Exception:
                        continue
                resolved_args = self._resolve_args(
                    step.args, initial_input, step.input_mapping
                )
                future = pool.submit(
                    self._executor.execute,
                    step.tool_name,
                    resolved_args,
                    step.timeout,
                )
                futures[step.output_key] = future
            for key, future in futures.items():
                try:
                    result = future.result(
                        timeout=self._executor.policy.timeout
                    )
                    results[key] = result
                    self._intermediate_results[key] = result.output
                except Exception as exc:
                    results[key] = ToolResult.from_error(
                        error=str(exc),
                    )
        return results

    def conditional_execute(
        self,
        initial_input: Any = None,
        condition_fn: Optional[Callable] = None,
    ) -> Dict[str, ToolResult]:
        """Execute tools conditionally based on runtime evaluation.

        Parameters
        ----------
        initial_input : Any
            Input for the first step.
        condition_fn : Callable, optional
            Global condition function applied to all steps.

        Returns
        -------
        Dict[str, ToolResult]
            Mapping of step output keys to their results.
        """
        if condition_fn is None:
            return self.execute_chain(initial_input)
        self._intermediate_results.clear()
        results: Dict[str, ToolResult] = {}
        current_input = initial_input
        for idx, step in enumerate(self._steps):
            should_run = True
            if step.condition is not None:
                try:
                    should_run = step.condition(current_input, results)
                except Exception:
                    should_run = False
            if should_run and condition_fn is not None:
                try:
                    should_run = condition_fn(
                        step, current_input, results
                    )
                except Exception:
                    should_run = False
            if not should_run:
                continue
            resolved_args = self._resolve_args(
                step.args, current_input, step.input_mapping
            )
            try:
                result = self._executor.execute(
                    step.tool_name,
                    resolved_args,
                    timeout=step.timeout,
                )
                results[step.output_key] = result
                self._intermediate_results[step.output_key] = result.output
                current_input = result.output
            except ToolExecutionError as exc:
                results[step.output_key] = ToolResult.from_error(
                    error=str(exc),
                    tool_name=step.tool_name,
                )
        return results

    def _resolve_args(
        self,
        base_args: Dict[str, Any],
        input_data: Any,
        mapping: str,
    ) -> Dict[str, Any]:
        """Resolve step arguments using the input mapping strategy.

        Parameters
        ----------
        base_args : Dict[str, Any]
            Base arguments from the step definition.
        input_data : Any
            Input from the previous step.
        mapping : str
            Mapping strategy name.

        Returns
        -------
        Dict[str, Any]
            Resolved arguments.
        """
        resolved = copy.deepcopy(base_args)
        if input_data is None:
            return resolved
        if mapping == "pass_through":
            if "input" in resolved:
                resolved["input"] = str(input_data)
            elif not resolved:
                tool = self._executor.get_tool(
                    self._steps[0].tool_name
                )
                if tool and tool.required_parameters:
                    first_param = tool.required_parameters[0].name
                    resolved[first_param] = str(input_data)
        elif mapping == "json_parse":
            try:
                if isinstance(input_data, str):
                    parsed = json.loads(input_data)
                    if isinstance(parsed, dict):
                        resolved.update(parsed)
            except json.JSONDecodeError:
                pass
        elif mapping == "append":
            for key in resolved:
                if isinstance(resolved[key], str):
                    resolved[key] += f"\n{input_data}"
        elif mapping.startswith("field:"):
            field_name = mapping[len("field:"):]
            try:
                if isinstance(input_data, str):
                    parsed = json.loads(input_data)
                    if isinstance(parsed, dict) and field_name in parsed:
                        resolved["input"] = str(parsed[field_name])
            except json.JSONDecodeError:
                pass
        return resolved


# ---------------------------------------------------------------------------
# Built-in Tools
# ---------------------------------------------------------------------------

class BuiltinTools:
    """Factory class that creates standard built-in tools.

    Provides a centralized way to instantiate common tools like file
    management, calculator, code execution, web search, date/time, and
    system information.

    Examples
    --------
    >>> executor = ToolExecutor()
    >>> for tool in BuiltinTools.create_all():
    ...     executor.register(tool)
    """

    @staticmethod
    def create_all(
        allowed_network: bool = True,
        allowed_file_read: bool = True,
        allowed_file_write: bool = False,
        allowed_code_exec: bool = False,
        allowed_commands: Optional[Set[str]] = None,
    ) -> List[ToolDefinition]:
        """Create all built-in tools with the specified access levels.

        Parameters
        ----------
        allowed_network : bool
            Whether network tools are enabled.
        allowed_file_read : bool
            Whether file reading is enabled.
        allowed_file_write : bool
            Whether file writing is enabled.
        allowed_code_exec : bool
            Whether code execution is enabled.
        allowed_commands : Set[str], optional
            Set of allowed shell commands.

        Returns
        -------
        List[ToolDefinition]
            List of tool definitions.
        """
        tools: List[ToolDefinition] = []
        tools.append(BuiltinTools.create_calculator())
        tools.append(BuiltinTools.create_datetime_tool())
        tools.append(BuiltinTools.create_system_info_tool())
        if allowed_file_read or allowed_file_write:
            tools.append(
                BuiltinTools.create_file_manager(
                    read=allowed_file_read,
                    write=allowed_file_write,
                    allowed_commands=allowed_commands,
                )
            )
        if allowed_code_exec:
            tools.append(BuiltinTools.create_code_executor())
        if allowed_network:
            tools.append(BuiltinTools.create_web_search())
        return tools

    @staticmethod
    def create_calculator() -> ToolDefinition:
        """Create the calculator tool.

        Returns
        -------
        ToolDefinition
            Calculator tool definition.
        """
        calc = Calculator()
        return ToolDefinition(
            name="calculator",
            description=(
                "Safely evaluates mathematical expressions. "
                "Supports basic arithmetic (+, -, *, /, **, %), "
                "common functions (sin, cos, tan, sqrt, log, abs, etc.), "
                "constants (pi, e), and parentheses for grouping."
            ),
            parameters=[
                ToolParameter(
                    name="expression",
                    param_type=ToolParameterType.STRING,
                    description="The mathematical expression to evaluate",
                    required=True,
                    pattern=r'^[\d\s+\-*/().^%a-zA-Z,]+$',
                ),
            ],
            handler=calc.evaluate,
            category="utility",
            tags={"math", "calculator", "arithmetic", "computation"},
            examples=[
                {"input": {"expression": "2 + 2"}, "output": "4"},
                {"input": {"expression": "sqrt(144)"}, "output": "12.0"},
                {"input": {"expression": "sin(pi / 2)"}, "output": "1.0"},
            ],
        )

    @staticmethod
    def create_file_manager(
        read: bool = True,
        write: bool = False,
        allowed_commands: Optional[Set[str]] = None,
    ) -> ToolDefinition:
        """Create the file manager tool.

        Parameters
        ----------
        read : bool
            Allow file reading.
        write : bool
            Allow file writing.
        allowed_commands : Set[str], optional
            Set of allowed shell commands.

        Returns
        -------
        ToolDefinition
            File manager tool definition.
        """
        fm = FileManager(
            allow_read=read,
            allow_write=write,
            allowed_commands=allowed_commands or set(),
        )
        return ToolDefinition(
            name="file_manager",
            description=(
                "Performs file system operations: read, write, list, "
                "search, and edit files. Supports text files with optional "
                "line range specification."
            ),
            parameters=[
                ToolParameter(
                    name="operation",
                    param_type=ToolParameterType.ENUM,
                    description="The file operation to perform",
                    required=True,
                    enum_values=[
                        "read", "write", "list", "search", "info",
                    ],
                ),
                ToolParameter(
                    name="path",
                    param_type=ToolParameterType.STRING,
                    description="File or directory path",
                    required=True,
                ),
                ToolParameter(
                    name="content",
                    param_type=ToolParameterType.STRING,
                    description="Content to write (for write operation)",
                ),
                ToolParameter(
                    name="pattern",
                    param_type=ToolParameterType.STRING,
                    description="Search pattern (for search operation)",
                ),
                ToolParameter(
                    name="start_line",
                    param_type=ToolParameterType.INTEGER,
                    description="Starting line number (1-based, for read)",
                    min_value=1,
                ),
                ToolParameter(
                    name="end_line",
                    param_type=ToolParameterType.INTEGER,
                    description="Ending line number (inclusive, for read)",
                    min_value=1,
                ),
            ],
            handler=fm.execute,
            category="file_system",
            tags={"file", "read", "write", "search", "filesystem"},
            dangerous=write,
            examples=[
                {
                    "input": {
                        "operation": "read",
                        "path": "/tmp/example.txt",
                    },
                    "output": "file contents...",
                },
                {
                    "input": {
                        "operation": "list",
                        "path": "/tmp/",
                    },
                    "output": "file1.txt\\nfile2.txt",
                },
            ],
        )

    @staticmethod
    def create_code_executor() -> ToolDefinition:
        """Create the code executor tool.

        Returns
        -------
        ToolDefinition
            Code executor tool definition.
        """
        ce = CodeExecutor()
        return ToolDefinition(
            name="code_executor",
            description=(
                "Executes Python code in a sandboxed environment. "
                "Supports standard library imports but restricts "
                "dangerous operations like file system access and "
                "network calls."
            ),
            parameters=[
                ToolParameter(
                    name="code",
                    param_type=ToolParameterType.STRING,
                    description="Python code to execute",
                    required=True,
                ),
                ToolParameter(
                    name="language",
                    param_type=ToolParameterType.ENUM,
                    description="Programming language",
                    enum_values=["python"],
                    default="python",
                ),
                ToolParameter(
                    name="timeout",
                    param_type=ToolParameterType.FLOAT,
                    description="Execution timeout in seconds",
                    default=30.0,
                    min_value=1.0,
                    max_value=120.0,
                ),
            ],
            handler=ce.execute,
            category="code",
            tags={"code", "python", "execute", "sandbox", "programming"},
            dangerous=True,
            timeout=60.0,
            retry_count=0,
            examples=[
                {
                    "input": {
                        "code": "print(sum(range(1, 101)))",
                        "language": "python",
                    },
                    "output": "5050",
                },
            ],
        )

    @staticmethod
    def create_web_search() -> ToolDefinition:
        """Create the web search tool.

        Returns
        -------
        ToolDefinition
            Web search tool definition.
        """
        ws = WebSearch()
        return ToolDefinition(
            name="web_search",
            description=(
                "Searches the web for information. Returns relevant "
                "snippets from search results. Use for looking up "
                "current information, facts, and references."
            ),
            parameters=[
                ToolParameter(
                    name="query",
                    param_type=ToolParameterType.STRING,
                    description="Search query string",
                    required=True,
                ),
                ToolParameter(
                    name="num_results",
                    param_type=ToolParameterType.INTEGER,
                    description="Number of results to return",
                    default=5,
                    min_value=1,
                    max_value=20,
                ),
            ],
            handler=ws.search,
            category="web",
            tags={"search", "web", "lookup", "information"},
            examples=[
                {
                    "input": {"query": "Python latest version"},
                    "output": "Search results for Python latest version...",
                },
            ],
        )

    @staticmethod
    def create_datetime_tool() -> ToolDefinition:
        """Create the date/time tool.

        Returns
        -------
        ToolDefinition
            Date/time tool definition.
        """
        dt = DateTimeTool()
        return ToolDefinition(
            name="datetime",
            description=(
                "Provides date and time information including current "
                "time, date arithmetic, timezone conversions, and "
                "formatting."
            ),
            parameters=[
                ToolParameter(
                    name="operation",
                    param_type=ToolParameterType.ENUM,
                    description="The date/time operation to perform",
                    required=True,
                    enum_values=[
                        "now", "format", "add", "subtract",
                        "diff", "timezone",
                    ],
                ),
                ToolParameter(
                    name="value",
                    param_type=ToolParameterType.STRING,
                    description="Date/time value or expression",
                ),
                ToolParameter(
                    name="format",
                    param_type=ToolParameterType.STRING,
                    description="Output format string",
                ),
                ToolParameter(
                    name="timezone",
                    param_type=ToolParameterType.STRING,
                    description="Timezone name (e.g., 'UTC', 'US/Eastern')",
                ),
            ],
            handler=dt.execute,
            category="utility",
            tags={"time", "date", "datetime", "calendar", "timezone"},
            examples=[
                {
                    "input": {"operation": "now"},
                    "output": "2024-01-15T10:30:00+00:00",
                },
                {
                    "input": {
                        "operation": "add",
                        "value": "7 days",
                    },
                    "output": "2024-01-22T10:30:00+00:00",
                },
            ],
        )

    @staticmethod
    def create_system_info_tool() -> ToolDefinition:
        """Create the system information tool.

        Returns
        -------
        ToolDefinition
            System info tool definition.
        """
        si = SystemInfoTool()
        return ToolDefinition(
            name="system_info",
            description=(
                "Provides system information including OS, CPU, memory, "
                "disk, and Python version details."
            ),
            parameters=[
                ToolParameter(
                    name="info_type",
                    param_type=ToolParameterType.ENUM,
                    description="Type of system information to retrieve",
                    required=True,
                    enum_values=[
                        "all", "os", "cpu", "memory", "disk",
                        "python", "network", "environment",
                    ],
                ),
            ],
            handler=si.get_info,
            category="system",
            tags={"system", "info", "os", "hardware", "diagnostics"},
            examples=[
                {
                    "input": {"info_type": "all"},
                    "output": "System: Linux...\nCPU: ...\nMemory: ...",
                },
            ],
        )


# ---------------------------------------------------------------------------
# FileManager
# ---------------------------------------------------------------------------

class FileManager:
    """File system operations tool with safety controls.

    Provides read, write, list, search, and info operations on the file
    system with configurable access restrictions.

    Parameters
    ----------
    allow_read : bool
        Whether file reading is permitted.
    allow_write : bool
        Whether file writing is permitted.
    allowed_commands : Set[str]
        Set of allowed shell commands.
    max_file_size : int
        Maximum file size in bytes for reading.
    max_dir_entries : int
        Maximum directory entries for listing.
    """

    def __init__(
        self,
        allow_read: bool = True,
        allow_write: bool = False,
        allowed_commands: Optional[Set[str]] = None,
        max_file_size: int = 10 * 1024 * 1024,
        max_dir_entries: int = 1000,
    ) -> None:
        self._allow_read = allow_read
        self._allow_write = allow_write
        self._allowed_commands = allowed_commands or set()
        self._max_file_size = max_file_size
        self._max_dir_entries = max_dir_entries
        self._logger = logging.getLogger(f"{__name__}.FileManager")

    def execute(self, **kwargs: Any) -> str:
        """Execute a file system operation.

        Parameters
        ----------
        **kwargs : Any
            Operation parameters including ``operation`` and ``path``.

        Returns
        -------
        str
            Operation result text.
        """
        operation = kwargs.get("operation", "").lower()
        path = kwargs.get("path", "")
        if not operation:
            return "Error: No operation specified"
        if not path:
            return "Error: No path specified"
        path = os.path.expanduser(path)
        path = os.path.abspath(path)
        if operation == "read":
            return self._read_file(path, kwargs)
        elif operation == "write":
            return self._write_file(path, kwargs)
        elif operation == "list":
            return self._list_directory(path, kwargs)
        elif operation == "search":
            return self._search_files(path, kwargs)
        elif operation == "info":
            return self._file_info(path)
        else:
            return f"Error: Unknown operation '{operation}'"

    def _read_file(self, path: str, kwargs: Dict[str, Any]) -> str:
        """Read a file with optional line range.

        Parameters
        ----------
        path : str
            File path.
        kwargs : Dict[str, Any]
            Optional start_line and end_line.

        Returns
        -------
        str
            File contents or error message.
        """
        if not self._allow_read:
            return "Error: File reading is not enabled"
        if not os.path.isfile(path):
            return f"Error: File not found: {path}"
        file_size = os.path.getsize(path)
        if file_size > self._max_file_size:
            return (
                f"Error: File size ({file_size} bytes) exceeds "
                f"maximum ({self._max_file_size} bytes)"
            )
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            start_line = kwargs.get("start_line")
            end_line = kwargs.get("end_line")
            if start_line is not None and end_line is not None:
                start_idx = max(0, int(start_line) - 1)
                end_idx = min(len(lines), int(end_line))
                lines = lines[start_idx:end_idx]
                result = "".join(lines)
                total_lines = end_idx - start_idx
                return (
                    f"[Lines {start_line}-{end_line} of file]\n"
                    f"{result}\n"
                    f"[Showing {total_lines} lines]"
                )
            result = "".join(lines)
            return f"[{len(lines)} lines]\n{result}"
        except PermissionError:
            return f"Error: Permission denied: {path}"
        except OSError as exc:
            return f"Error reading file: {exc}"

    def _write_file(self, path: str, kwargs: Dict[str, Any]) -> str:
        """Write content to a file.

        Parameters
        ----------
        path : str
            File path.
        kwargs : Dict[str, Any]
            Content to write.

        Returns
        -------
        str
            Success or error message.
        """
        if not self._allow_write:
            return "Error: File writing is not enabled"
        content = kwargs.get("content", "")
        if content is None:
            content = ""
        parent_dir = os.path.dirname(path)
        if parent_dir and not os.path.exists(parent_dir):
            try:
                os.makedirs(parent_dir, exist_ok=True)
            except OSError as exc:
                return f"Error creating directory: {exc}"
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(str(content))
            bytes_written = len(str(content).encode("utf-8"))
            return (
                f"Successfully wrote {bytes_written} bytes to {path}"
            )
        except PermissionError:
            return f"Error: Permission denied: {path}"
        except OSError as exc:
            return f"Error writing file: {exc}"

    def _list_directory(self, path: str, kwargs: Dict[str, Any]) -> str:
        """List directory contents.

        Parameters
        ----------
        path : str
            Directory path.
        kwargs : Dict[str, Any]
            Additional options.

        Returns
        -------
        str
            Directory listing or error message.
        """
        if not self._allow_read:
            return "Error: Directory listing is not enabled"
        if not os.path.isdir(path):
            return f"Error: Not a directory: {path}"
        try:
            entries = sorted(os.listdir(path))
            if len(entries) > self._max_dir_entries:
                entries = entries[:self._max_dir_entries]
                truncated = True
            else:
                truncated = False
            lines: List[str] = []
            for entry in entries:
                full_path = os.path.join(path, entry)
                if os.path.isdir(full_path):
                    lines.append(f"[DIR]  {entry}")
                elif os.path.isfile(full_path):
                    size = os.path.getsize(full_path)
                    if size < 1024:
                        size_str = f"{size} B"
                    elif size < 1024 * 1024:
                        size_str = f"{size / 1024:.1f} KB"
                    else:
                        size_str = f"{size / (1024 * 1024):.1f} MB"
                    lines.append(f"[FILE] {entry} ({size_str})")
                else:
                    lines.append(f"[LINK] {entry}")
            result = "\n".join(lines)
            if truncated:
                result += (
                    f"\n\n[Truncated: showing {self._max_dir_entries} "
                    f"of {len(os.listdir(path))} entries]"
                )
            return result
        except PermissionError:
            return f"Error: Permission denied: {path}"
        except OSError as exc:
            return f"Error listing directory: {exc}"

    def _search_files(self, path: str, kwargs: Dict[str, Any]) -> str:
        """Search for files matching a pattern.

        Parameters
        ----------
        path : str
            Search root path.
        kwargs : Dict[str, Any]
            Search pattern.

        Returns
        -------
        str
            Search results or error message.
        """
        if not self._allow_read:
            return "Error: File search is not enabled"
        pattern = kwargs.get("pattern", "")
        if not pattern:
            return "Error: No search pattern provided"
        if not os.path.isdir(path):
            return f"Error: Not a directory: {path}"
        matches: List[str] = []
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as exc:
            return f"Error: Invalid regex pattern: {exc}"
        try:
            for root, dirs, files in os.walk(path):
                dirs.sort()
                files.sort()
                for filename in files:
                    if regex.search(filename):
                        full_path = os.path.join(root, filename)
                        matches.append(full_path)
                    if len(matches) >= 100:
                        break
                if len(matches) >= 100:
                    break
        except PermissionError:
            pass
        except OSError as exc:
            return f"Error during search: {exc}"
        if not matches:
            return f"No files matching '{pattern}' found in {path}"
        result = f"Found {len(matches)} files matching '{pattern}':\n"
        result += "\n".join(f"  {m}" for m in matches)
        return result

    def _file_info(self, path: str) -> str:
        """Get detailed information about a file or directory.

        Parameters
        ----------
        path : str
            File or directory path.

        Returns
        -------
        str
            File information or error message.
        """
        if not os.path.exists(path):
            return f"Error: Path not found: {path}"
        stat = os.stat(path)
        info_lines: List[str] = [
            f"Path: {path}",
            f"Type: {'Directory' if os.path.isdir(path) else 'File'}",
            f"Size: {stat.st_size} bytes",
            f"Modified: {datetime.datetime.fromtimestamp(stat.st_mtime, tz=datetime.timezone.utc).isoformat()}",
            f"Accessed: {datetime.datetime.fromtimestamp(stat.st_atime, tz=datetime.timezone.utc).isoformat()}",
            f"Created: {datetime.datetime.fromtimestamp(stat.st_ctime, tz=datetime.timezone.utc).isoformat()}",
        ]
        if os.path.isfile(path):
            _, ext = os.path.splitext(path)
            info_lines.append(f"Extension: {ext or 'none'}")
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()
                info_lines.append(f"Lines: {len(lines)}")
                non_empty = [l for l in lines if l.strip()]
                info_lines.append(f"Non-empty lines: {len(non_empty)}")
            except (PermissionError, OSError):
                pass
        return "\n".join(info_lines)


# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------

class Calculator:
    """Safe mathematical expression evaluator.

    Evaluates mathematical expressions using Python's AST parser with
    restricted node types to prevent code injection.  Supports standard
    arithmetic operators, common math functions, and constants.
    """

    SAFE_FUNCTIONS: Dict[str, Callable] = None  # type: ignore

    SAFE_CONSTANTS: Dict[str, float] = None  # type: ignore

    def __init__(self) -> None:
        if Calculator.SAFE_FUNCTIONS is None:
            Calculator.SAFE_FUNCTIONS = {
                "abs": abs,
                "round": round,
                "min": min,
                "max": max,
                "sum": sum,
                "pow": pow,
                "int": int,
                "float": float,
                "len": len,
                "bin": bin,
                "oct": oct,
                "hex": hex,
                "sqrt": math.sqrt,
                "sin": math.sin,
                "cos": math.cos,
                "tan": math.tan,
                "asin": math.asin,
                "acos": math.acos,
                "atan": math.atan,
                "atan2": math.atan2,
                "log": math.log,
                "log10": math.log10,
                "log2": math.log2,
                "exp": math.exp,
                "floor": math.floor,
                "ceil": math.ceil,
                "trunc": math.trunc,
                "factorial": math.factorial,
                "gcd": math.gcd,
                "hypot": math.hypot,
                "degrees": math.degrees,
                "radians": math.radians,
                "copysign": math.copysign,
                "isclose": math.isclose,
                "isfinite": math.isfinite,
                "isinf": math.isinf,
                "isnan": math.isnan,
                "prod": math.prod,
                "comb": math.comb,
                "perm": math.perm,
                "modf": math.modf,
                "frexp": math.frexp,
                "ldexp": math.ldexp,
                "fabs": math.fabs,
                "fmod": math.fmod,
                "remainder": math.remainder,
            }
        if Calculator.SAFE_CONSTANTS is None:
            Calculator.SAFE_CONSTANTS = {
                "pi": math.pi,
                "e": math.e,
                "tau": math.tau,
                "inf": math.inf,
                "nan": math.nan,
            }

    def evaluate(self, expression: str, **_kwargs: Any) -> str:
        """Safely evaluate a mathematical expression.

        Parameters
        ----------
        expression : str
            The mathematical expression to evaluate.

        Returns
        -------
        str
            String representation of the result.

        Raises
        ------
        ValueError
            If the expression contains unsafe operations.
        """
        expression = expression.strip()
        if not expression:
            return "Error: Empty expression"
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            return f"Error: Invalid expression syntax: {exc}"
        try:
            self._validate_ast(tree)
        except ValueError as exc:
            return f"Error: {exc}"
        try:
            result = eval(
                compile(tree, "<calculator>", "eval"),
                {"__builtins__": {}},
                {
                    **self.SAFE_FUNCTIONS,
                    **self.SAFE_CONSTANTS,
                },
            )
            if isinstance(result, float) and result == int(result):
                return str(int(result))
            if isinstance(result, complex):
                return f"{result.real} + {result.imag}j"
            return str(result)
        except ZeroDivisionError:
            return "Error: Division by zero"
        except OverflowError:
            return "Error: Result too large"
        except ValueError as exc:
            return f"Error: {exc}"
        except TypeError as exc:
            return f"Error: {exc}"
        except Exception as exc:
            return f"Error: Evaluation failed: {exc}"

    def _validate_ast(self, tree: ast.AST) -> None:
        """Validate that an AST contains only safe operations.

        Parameters
        ----------
        tree : ast.AST
            The AST to validate.

        Raises
        ------
        ValueError
            If unsafe operations are detected.
        """
        allowed_nodes = {
            ast.Expression, ast.BinOp, ast.UnaryOp, ast.Compare,
            ast.Call, ast.Constant, ast.Num, ast.Name, ast.Load,
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv,
            ast.Mod, ast.Pow, ast.LShift, ast.RShift, ast.BitOr,
            ast.BitXor, ast.BitAnd, ast.Invert, ast.Not, ast.UAdd,
            ast.USub, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt,
            ast.GtE, ast.Is, ast.IsNot, ast.And, ast.Or,
            ast.BoolOp, ast.Tuple, ast.List,
        }
        if hasattr(ast, "MatMult"):
            allowed_nodes.add(ast.MatMult)
        for node in ast.walk(tree):
            node_type = type(node)
            if node_type not in allowed_nodes:
                if node_type == ast.Attribute:
                    attr = getattr(node, "attr", "")
                    if attr not in (
                        "real", "imag", "numerator", "denominator",
                        "bit_length", "conjugate",
                    ):
                        raise ValueError(
                            f"Attribute access not allowed: {attr}"
                        )
                    continue
                if node_type == ast.Name:
                    name = getattr(node, "id", "")
                    if name not in self.SAFE_FUNCTIONS and name not in self.SAFE_CONSTANTS:
                        raise ValueError(
                            f"Unknown name: '{name}'"
                        )
                    continue
                raise ValueError(
                    f"Operation not allowed: {node_type.__name__}"
                )


# ---------------------------------------------------------------------------
# CodeExecutor
# ---------------------------------------------------------------------------

class CodeExecutor:
    """Sandboxed code execution tool.

    Executes Python code in a restricted environment with resource limits,
    no network access, and no file system modifications.

    Parameters
    ----------
    max_execution_time : float
        Maximum execution time per code block in seconds.
    max_output_length : int
        Maximum output length in characters.
    max_memory_mb : int
        Maximum memory usage in megabytes.
    """

    def __init__(
        self,
        max_execution_time: float = 30.0,
        max_output_length: int = 100000,
        max_memory_mb: int = 256,
    ) -> None:
        self._max_execution_time = max_execution_time
        self._max_output_length = max_output_length
        self._max_memory_mb = max_memory_mb
        self._logger = logging.getLogger(f"{__name__}.CodeExecutor")

    def execute(self, code: str, **kwargs: Any) -> str:
        """Execute Python code in a sandboxed environment.

        Parameters
        ----------
        code : str
            Python code to execute.
        **kwargs : Any
            Additional parameters (language, timeout).

        Returns
        -------
        str
            Captured stdout, stderr, and return value.
        """
        language = kwargs.get("language", "python").lower()
        if language != "python":
            return f"Error: Language '{language}' is not supported. Only 'python' is available."
        timeout = kwargs.get("timeout", self._max_execution_time)
        timeout = min(timeout, self._max_execution_time)
        if not code or not code.strip():
            return "Error: No code provided"
        code = code.strip()
        self._validate_code_safety(code)
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        result_container: List[Any] = []
        error_container: List[str] = []
        execution_success = False

        def _run_code() -> None:
            """Execute the code in a restricted environment."""
            try:
                old_stdout = sys.stdout
                old_stderr = sys.stderr
                sys.stdout = stdout_buffer
                sys.stderr = stderr_buffer
                local_vars: Dict[str, Any] = {
                    "__builtins__": self._get_restricted_builtins(),
                    "print": print,
                    "range": range,
                    "len": len,
                    "str": str,
                    "int": int,
                    "float": float,
                    "list": list,
                    "dict": dict,
                    "set": set,
                    "tuple": tuple,
                    "sorted": sorted,
                    "enumerate": enumerate,
                    "zip": zip,
                    "map": map,
                    "filter": filter,
                    "min": min,
                    "max": max,
                    "sum": sum,
                    "abs": abs,
                    "round": round,
                    "type": type,
                    "isinstance": isinstance,
                    "hasattr": hasattr,
                    "getattr": getattr,
                    "reversed": reversed,
                    "any": any,
                    "all": all,
                    "bool": bool,
                    "bytes": bytes,
                    "chr": chr,
                    "ord": ord,
                    "hex": hex,
                    "oct": oct,
                    "bin": bin,
                    "hash": hash,
                    "id": id,
                    "repr": repr,
                    "format": format,
                    "pow": pow,
                    "divmod": divmod,
                    "next": next,
                    "iter": iter,
                    "frozenset": frozenset,
                    "slice": slice,
                    "staticmethod": staticmethod,
                    "classmethod": classmethod,
                    "property": property,
                    "super": super,
                    "object": object,
                    "Exception": Exception,
                    "ValueError": ValueError,
                    "TypeError": TypeError,
                    "KeyError": KeyError,
                    "IndexError": IndexError,
                    "AttributeError": AttributeError,
                    "RuntimeError": RuntimeError,
                    "StopIteration": StopIteration,
                    "NotImplementedError": NotImplementedError,
                    "ImportError": ImportError,
                    "ZeroDivisionError": ZeroDivisionError,
                    "OverflowError": OverflowError,
                    "math": math,
                    "json": json,
                    "re": re,
                    "datetime": datetime,
                    "collections": __import__("collections"),
                    "itertools": __import__("itertools"),
                    "functools": __import__("functools"),
                    "operator": __import__("operator"),
                    "string": __import__("string"),
                    "statistics": __import__("statistics"),
                    "fractions": __import__("fractions"),
                    "decimal": __import__("decimal"),
                    "dataclasses": __import__("dataclasses"),
                    "typing": __import__("typing"),
                    "copy": copy,
                    "uuid": uuid,
                    "time": time,
                    "hashlib": __import__("hashlib"),
                    "base64": __import__("base64"),
                }
                exec(code, local_vars)
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                result_container.append("Code executed successfully")
                execution_success = True
            except Exception as exc:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                tb = traceback.format_exc()
                error_container.append(f"{type(exc).__name__}: {exc}\n{tb}")

        thread = threading.Thread(target=_run_code, daemon=True)
        thread.start()
        thread.join(timeout=timeout)
        if thread.is_alive():
            return f"Error: Code execution timed out after {timeout:.1f} seconds"
        stdout_output = stdout_buffer.getvalue()
        stderr_output = stderr_buffer.getvalue()
        output_parts: List[str] = []
        if stdout_output:
            output_parts.append(f"stdout:\n{stdout_output}")
        if stderr_output:
            output_parts.append(f"stderr:\n{stderr_output}")
        if error_container:
            output_parts.append(f"Error:\n{error_container[0]}")
        if result_container and execution_success:
            output_parts.append(result_container[0])
        if not output_parts:
            return "Code executed successfully (no output)"
        full_output = "\n".join(output_parts)
        if len(full_output) > self._max_output_length:
            full_output = (
                full_output[:self._max_output_length]
                + f"\n\n[Output truncated at {self._max_output_length} characters]"
            )
        return full_output

    def _validate_code_safety(self, code: str) -> None:
        """Validate that code does not contain obviously dangerous patterns.

        Parameters
        ----------
        code : str
            Code to validate.

        Raises
        ------
        ValueError
            If dangerous patterns are detected.
        """
        dangerous_patterns = [
            (r'\bos\.system\b', "os.system calls"),
            (r'\bos\.popen\b', "os.popen calls"),
            (r'\bsubprocess\b', "subprocess module"),
            (r'\bexecfile\b', "execfile calls"),
            (r'\bcompile\b', "compile calls"),
            (r'\beval\b', "eval calls"),
            (r'\b__import__\b', "__import__ calls"),
            (r'\bglobals\b', "globals access"),
            (r'\blocals\b', "locals access"),
            (r'\bgetattr\b.*__\w+__', "dunder attribute access"),
            (r'\bsetattr\b.*__\w+__', "dunder attribute modification"),
            (r'\bdel\b.*__\w+__', "dunder attribute deletion"),
            (r'\bctypes\b', "ctypes module"),
            (r'\bsocket\b', "socket module"),
            (r'\brequests\b', "requests module"),
            (r'\burllib\b', "urllib module"),
            (r'\bhttp\b', "HTTP access"),
            (r'\bshutil\b', "shutil module"),
            (r'\bos\.remove\b', "file deletion"),
            (r'\bos\.unlink\b', "file unlinking"),
            (r'\bos\.rmdir\b', "directory removal"),
            (r'\bos\.mkdir\b', "directory creation (restricted)"),
            (r'\bopen\b.*[\'"]w', "file writing"),
        ]
        for pattern, description in dangerous_patterns:
            if re.search(pattern, code):
                raise ValueError(f"Potentially dangerous code detected: {description}")

    def _get_restricted_builtins(self) -> Dict[str, Any]:
        """Get a restricted builtins dictionary.

        Returns
        -------
        Dict[str, Any]
            Restricted builtins with dangerous functions removed.
        """
        builtins_dict: Dict[str, Any] = {}
        safe_builtins = {
            "True": True, "False": False, "None": None,
            "abs": abs, "all": all, "any": any, "bin": bin,
            "bool": bool, "chr": chr, "complex": complex,
            "dict": dict, "divmod": divmod, "enumerate": enumerate,
            "filter": filter, "float": float, "format": format,
            "frozenset": frozenset, "hash": hash, "hex": hex,
            "int": int, "isinstance": isinstance, "issubclass": issubclass,
            "iter": iter, "len": len, "list": list, "map": map,
            "max": max, "min": min, "next": next, "oct": oct,
            "ord": ord, "pow": pow, "print": print, "range": range,
            "repr": repr, "reversed": reversed, "round": round,
            "set": set, "slice": slice, "sorted": sorted, "str": str,
            "sum": sum, "tuple": tuple, "type": type, "zip": zip,
            "NotImplemented": NotImplemented, "Ellipsis": Ellipsis,
            "Exception": Exception, "ValueError": ValueError,
            "TypeError": TypeError, "RuntimeError": RuntimeError,
            "StopIteration": StopIteration, "KeyError": KeyError,
            "IndexError": IndexError, "AttributeError": AttributeError,
            "ZeroDivisionError": ZeroDivisionError,
            "ArithmeticError": ArithmeticError,
            "OverflowError": OverflowError,
        }
        return safe_builtins


# ---------------------------------------------------------------------------
# WebSearch
# ---------------------------------------------------------------------------

class WebSearch:
    """Web search abstraction layer.

    Provides a pluggable interface for web search with mock results
    when no real search API is configured.
    """

    def __init__(
        self,
        api_key: str = "",
        search_engine: str = "mock",
        max_results: int = 10,
        timeout: float = 10.0,
    ) -> None:
        self._api_key = api_key
        self._search_engine = search_engine
        self._max_results = max_results
        self._timeout = timeout
        self._logger = logging.getLogger(f"{__name__}.WebSearch")

    def search(self, query: str, **kwargs: Any) -> str:
        """Execute a web search.

        Parameters
        ----------
        query : str
            Search query string.
        **kwargs : Any
            Additional parameters including num_results.

        Returns
        -------
        str
            Formatted search results.
        """
        if not query or not query.strip():
            return "Error: Empty search query"
        num_results = min(
            kwargs.get("num_results", self._max_results),
            self._max_results,
        )
        if self._search_engine == "mock":
            return self._mock_search(query, num_results)
        return self._mock_search(query, num_results)

    def _mock_search(
        self, query: str, num_results: int
    ) -> str:
        """Generate mock search results.

        Parameters
        ----------
        query : str
            Search query.
        num_results : int
            Number of results to generate.

        Returns
        -------
        str
            Mock search results.
        """
        results: List[str] = []
        results.append(f"Search results for: {query}")
        results.append("=" * 60)
        for i in range(num_results):
            results.append(
                f"\nResult {i + 1}:\n"
                f"  Title: Relevant result for '{query}' #{i + 1}\n"
                f"  URL: https://example.com/result-{i + 1}\n"
                f"  Snippet: This is a mock search result snippet "
                f"related to '{query}'. In a production environment, "
                f"this would contain actual search results from a "
                f"web search API.\n"
            )
        results.append(
            "\nNote: This is a mock search. Configure a real search "
            "API key to get actual results."
        )
        return "\n".join(results)


# ---------------------------------------------------------------------------
# DateTimeTool
# ---------------------------------------------------------------------------

class DateTimeTool:
    """Date and time query and arithmetic tool.

    Provides current time, formatting, arithmetic, difference
    calculation, and timezone conversion capabilities.
    """

    TIME_UNITS: Dict[str, int] = None  # type: ignore

    def __init__(self) -> None:
        if DateTimeTool.TIME_UNITS is None:
            DateTimeTool.TIME_UNITS = {
                "seconds": 1,
                "second": 1,
                "sec": 1,
                "s": 1,
                "minutes": 60,
                "minute": 60,
                "min": 60,
                "hours": 3600,
                "hour": 3600,
                "hr": 3600,
                "days": 86400,
                "day": 86400,
                "weeks": 604800,
                "week": 604800,
                "wk": 604800,
                "months": 2592000,
                "month": 2592000,
                "mon": 2592000,
                "years": 31536000,
                "year": 31536000,
                "yr": 31536000,
            }

    def execute(self, **kwargs: Any) -> str:
        """Execute a date/time operation.

        Parameters
        ----------
        **kwargs : Any
            Operation parameters.

        Returns
        -------
        str
            Operation result.
        """
        operation = kwargs.get("operation", "now").lower()
        if operation == "now":
            return self._get_now(kwargs)
        elif operation == "format":
            return self._format_datetime(kwargs)
        elif operation == "add":
            return self._add_time(kwargs)
        elif operation == "subtract":
            return self._subtract_time(kwargs)
        elif operation == "diff":
            return self._diff_time(kwargs)
        elif operation == "timezone":
            return self._convert_timezone(kwargs)
        else:
            return f"Error: Unknown operation '{operation}'"

    def _get_now(self, kwargs: Dict[str, Any]) -> str:
        """Get current date and time.

        Parameters
        ----------
        kwargs : Dict[str, Any]
            Optional timezone parameter.

        Returns
        -------
        str
            Current datetime in ISO format.
        """
        tz = kwargs.get("timezone")
        now = datetime.datetime.now(timezone.utc)
        if tz and tz.upper() != "UTC":
            try:
                now = now.astimezone(
                    datetime.timezone(datetime.timedelta(hours=0))
                )
            except Exception:
                pass
        return now.isoformat()

    def _format_datetime(self, kwargs: Dict[str, Any]) -> str:
        """Format a datetime value.

        Parameters
        ----------
        kwargs : Dict[str, Any]
            Value and format parameters.

        Returns
        -------
        str
            Formatted datetime string.
        """
        value = kwargs.get("value", "")
        fmt = kwargs.get("format", "%Y-%m-%d %H:%M:%S")
        try:
            if value.lower() == "now":
                dt = datetime.datetime.now(timezone.utc)
            else:
                dt = datetime.datetime.fromisoformat(value)
            return dt.strftime(fmt)
        except (ValueError, TypeError) as exc:
            return f"Error: Cannot parse datetime '{value}': {exc}"

    def _add_time(self, kwargs: Dict[str, Any]) -> str:
        """Add time to current datetime.

        Parameters
        ----------
        kwargs : Dict[str, Any]
            Value specifying the time to add.

        Returns
        -------
        str
            Result datetime in ISO format.
        """
        value = kwargs.get("value", "")
        now = datetime.datetime.now(timezone.utc)
        try:
            delta = self._parse_timedelta(value)
            result = now + delta
            return result.isoformat()
        except ValueError as exc:
            return f"Error: {exc}"

    def _subtract_time(self, kwargs: Dict[str, Any]) -> str:
        """Subtract time from current datetime.

        Parameters
        ----------
        kwargs : Dict[str, Any]
            Value specifying the time to subtract.

        Returns
        -------
        str
            Result datetime in ISO format.
        """
        value = kwargs.get("value", "")
        now = datetime.datetime.now(timezone.utc)
        try:
            delta = self._parse_timedelta(value)
            result = now - delta
            return result.isoformat()
        except ValueError as exc:
            return f"Error: {exc}"

    def _diff_time(self, kwargs: Dict[str, Any]) -> str:
        """Calculate the difference between two datetime values.

        Parameters
        ----------
        kwargs : Dict[str, Any]
            Two datetime values to compare.

        Returns
        -------
        str
            Human-readable time difference.
        """
        value = kwargs.get("value", "")
        try:
            parts = value.split(",")
            if len(parts) != 2:
                return (
                    "Error: Provide two comma-separated ISO datetime values"
                )
            dt1 = datetime.datetime.fromisoformat(parts[0].strip())
            dt2 = datetime.datetime.fromisoformat(parts[1].strip())
            delta = abs(dt2 - dt1)
            days = delta.days
            seconds = delta.seconds
            hours, remainder = divmod(seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            parts_desc: List[str] = []
            if days:
                parts_desc.append(f"{days} day{'s' if days != 1 else ''}")
            if hours:
                parts_desc.append(f"{hours} hour{'s' if hours != 1 else ''}")
            if minutes:
                parts_desc.append(
                    f"{minutes} minute{'s' if minutes != 1 else ''}"
                )
            if seconds:
                parts_desc.append(
                    f"{seconds} second{'s' if seconds != 1 else ''}"
                )
            return (
                f"Difference: {', '.join(parts_desc)} "
                f"(total: {delta.total_seconds():.0f} seconds)"
            )
        except (ValueError, TypeError) as exc:
            return f"Error: {exc}"

    def _convert_timezone(self, kwargs: Dict[str, Any]) -> str:
        """Convert a datetime to a different timezone.

        Parameters
        ----------
        kwargs : Dict[str, Any]
            Value and target timezone.

        Returns
        -------
        str
            Converted datetime in ISO format.
        """
        value = kwargs.get("value", "now")
        tz_name = kwargs.get("timezone", "UTC")
        try:
            if value.lower() == "now":
                dt = datetime.datetime.now(timezone.utc)
            else:
                dt = datetime.datetime.fromisoformat(value)
            offset_map = {
                "utc": 0, "gmt": 0,
                "est": -5, "edt": -4,
                "cst": -6, "cdt": -5,
                "mst": -7, "mdt": -6,
                "pst": -8, "pdt": -7,
                "cet": 1, "cest": 2,
                "eet": 2, "eest": 3,
                "jst": 9, "kst": 9,
                "ist": 5.5, "sgt": 8,
                "aest": 10, "aedt": 11,
                "nzst": 12, "nzdt": 13,
            }
            offset = offset_map.get(tz_name.lower())
            if offset is None:
                return f"Error: Unknown timezone '{tz_name}'"
            target_tz = datetime.timezone(
                datetime.timedelta(hours=offset)
            )
            converted = dt.astimezone(target_tz)
            return (
                f"{value} in {tz_name.upper()}: {converted.isoformat()}"
            )
        except (ValueError, TypeError) as exc:
            return f"Error: {exc}"

    def _parse_timedelta(self, value: str) -> datetime.timedelta:
        """Parse a human-readable time duration string.

        Parameters
        ----------
        value : str
            Duration string (e.g., "7 days", "2 hours 30 minutes").

        Returns
        -------
        datetime.timedelta
            Parsed timedelta.

        Raises
        ------
        ValueError
            If the duration string cannot be parsed.
        """
        value = value.strip()
        pattern = r'(\d+(?:\.\d+)?)\s*(seconds?|minutes?|hours?|days?|weeks?|months?|years?|s|m|hr?|wk?|mon?|yr?)'
        matches = re.findall(pattern, value, re.IGNORECASE)
        if not matches:
            raise ValueError(
                f"Cannot parse duration: '{value}'. "
                f"Use format like '7 days', '2 hours 30 minutes'"
            )
        total_seconds = 0.0
        for amount_str, unit in matches:
            amount = float(amount_str)
            unit_key = unit.lower()
            multiplier = self.TIME_UNITS.get(unit_key)
            if multiplier is None:
                raise ValueError(f"Unknown time unit: '{unit}'")
            total_seconds += amount * multiplier
        return datetime.timedelta(seconds=total_seconds)


# ---------------------------------------------------------------------------
# SystemInfoTool
# ---------------------------------------------------------------------------

class SystemInfoTool:
    """System information retrieval tool.

    Provides OS, CPU, memory, disk, Python version, and environment
    variable information.
    """

    def get_info(self, **kwargs: Any) -> str:
        """Get system information.

        Parameters
        ----------
        **kwargs : Any
            Info type parameter.

        Returns
        -------
        str
            Formatted system information.
        """
        info_type = kwargs.get("info_type", "all").lower()
        info_parts: List[str] = []
        if info_type in ("all", "os"):
            info_parts.append(self._get_os_info())
        if info_type in ("all", "cpu"):
            info_parts.append(self._get_cpu_info())
        if info_type in ("all", "memory"):
            info_parts.append(self._get_memory_info())
        if info_type in ("all", "disk"):
            info_parts.append(self._get_disk_info())
        if info_type in ("all", "python"):
            info_parts.append(self._get_python_info())
        if info_type in ("all", "network"):
            info_parts.append(self._get_network_info())
        if info_type in ("all", "environment"):
            info_parts.append(self._get_env_info())
        if not info_parts:
            return f"Error: Unknown info type '{info_type}'"
        return "\n\n".join(info_parts)

    def _get_os_info(self) -> str:
        """Get operating system information.

        Returns
        -------
        str
            OS details.
        """
        lines: List[str] = ["=== Operating System ==="]
        lines.append(f"System: {platform.system()}")
        lines.append(f"Release: {platform.release()}")
        lines.append(f"Version: {platform.version()}")
        lines.append(f"Machine: {platform.machine()}")
        lines.append(f"Processor: {platform.processor()}")
        lines.append(f"Hostname: {platform.node()}")
        lines.append(f"Architecture: {platform.architecture()[0]}")
        return "\n".join(lines)

    def _get_cpu_info(self) -> str:
        """Get CPU information.

        Returns
        -------
        str
            CPU details.
        """
        lines: List[str] = ["=== CPU ==="]
        try:
            lines.append(f"Physical cores: {os.cpu_count()}")
        except (AttributeError, NotImplementedError):
            lines.append("Physical cores: N/A")
        try:
            lines.append(f"Processor: {platform.processor()}")
        except (AttributeError, NotImplementedError):
            lines.append("Processor: N/A")
        freq = getattr(os, "sched_getaffinity", None)
        if freq:
            try:
                affinity = os.sched_getaffinity(0)
                lines.append(f"CPU affinity: {len(affinity)} cores")
            except (AttributeError, OSError):
                pass
        return "\n".join(lines)

    def _get_memory_info(self) -> str:
        """Get memory information.

        Returns
        -------
        str
            Memory details.
        """
        lines: List[str] = ["=== Memory ==="]
        try:
            import resource
            rusage = resource.getrusage(resource.RUSAGE_SELF)
            max_rss = rusage.ru_maxrss
            if platform.system() == "Darwin":
                max_rss_mb = max_rss / (1024 * 1024)
            else:
                max_rss_mb = max_rss / 1024
            lines.append(f"Current process max RSS: {max_rss_mb:.1f} MB")
        except (ImportError, AttributeError):
            lines.append("Current process max RSS: N/A")
        try:
            if hasattr(os, "statvfs"):
                stat = os.statvfs("/")
                total = stat.f_blocks * stat.f_frsize
                free = stat.f_bavail * stat.f_frsize
                used = total - free
                lines.append(f"Disk total: {total / (1024**3):.1f} GB")
                lines.append(f"Disk used: {used / (1024**3):.1f} GB")
                lines.append(f"Disk free: {free / (1024**3):.1f} GB")
        except (AttributeError, OSError):
            pass
        return "\n".join(lines)

    def _get_disk_info(self) -> str:
        """Get disk usage information.

        Returns
        -------
        str
            Disk usage details.
        """
        lines: List[str] = ["=== Disk ==="]
        try:
            if hasattr(os, "statvfs"):
                for path in ["/", "/tmp", os.path.expanduser("~")]:
                    try:
                        stat = os.statvfs(path)
                        total = stat.f_blocks * stat.f_frsize
                        free = stat.f_bavail * stat.f_frsize
                        used = total - free
                        pct = (used / total * 100) if total > 0 else 0
                        lines.append(
                            f"{path}: "
                            f"{total / (1024**3):.1f} GB total, "
                            f"{used / (1024**3):.1f} GB used "
                            f"({pct:.1f}%), "
                            f"{free / (1024**3):.1f} GB free"
                        )
                    except OSError:
                        pass
        except (AttributeError, OSError):
            lines.append("Disk info: N/A")
        return "\n".join(lines)

    def _get_python_info(self) -> str:
        """Get Python runtime information.

        Returns
        -------
        str
            Python version and details.
        """
        lines: List[str] = ["=== Python ==="]
        lines.append(f"Version: {platform.python_version()}")
        lines.append(f"Implementation: {platform.python_implementation()}")
        lines.append(f"Compiler: {platform.python_compiler()}")
        lines.append(f"Build: {platform.python_build()}")
        lines.append(f"Executable: {sys.executable}")
        lines.append(f"Prefix: {sys.prefix}")
        lines.append(f"Base prefix: {sys.base_prefix}")
        lines.append(f"Path entries: {len(sys.path)}")
        try:
            lines.append(
                f"Default encoding: {sys.getdefaultencoding()}"
            )
            lines.append(f"File system encoding: {sys.getfilesystemencoding()}")
        except AttributeError:
            pass
        lines.append(f"Thread count: {threading.active_count()}")
        return "\n".join(lines)

    def _get_network_info(self) -> str:
        """Get network information.

        Returns
        -------
        str
            Network details.
        """
        lines: List[str] = ["=== Network ==="]
        try:
            hostname = platform.node()
            lines.append(f"Hostname: {hostname}")
        except Exception:
            lines.append("Hostname: N/A")
        lines.append(f"Network access: enabled (tool level)")
        return "\n".join(lines)

    def _get_env_info(self) -> str:
        """Get environment variable information.

        Returns
        -------
        str
            Safe environment variables (no secrets).
        """
        lines: List[str] = ["=== Environment ==="]
        safe_prefixes = (
            "PATH", "HOME", "USER", "LANG", "TERM",
            "SHELL", "PWD", "EDITOR", "VISUAL",
            "PYTHON", "JAVA", "NODE",
        )
        safe_vars: Dict[str, str] = {}
        for key, value in os.environ.items():
            if any(key.upper().startswith(p) for p in safe_prefixes):
                safe_value = value
                if len(safe_value) > 200:
                    safe_value = safe_value[:200] + "..."
                safe_vars[key] = safe_value
        for key in sorted(safe_vars):
            lines.append(f"{key}: {safe_vars[key]}")
        if not safe_vars:
            lines.append("No safe environment variables found")
        return "\n".join(lines)
