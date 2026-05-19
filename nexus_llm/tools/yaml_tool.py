"""Nexus-LLM YAML Processing Tool.

Provides YAML parsing, serialization, and validation capabilities.
"""

import logging
from typing import Any, Dict, List, Optional

from nexus_llm.tools.base_tool import BaseTool, ToolParameter, ToolResult, ParameterType

logger = logging.getLogger(__name__)

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


class YamlTool(BaseTool):
    """Tool for YAML data processing.

    Supports parsing, serialization, merging, and validation.

    Example::

        tool = YamlTool()
        result = tool.run(operation="parse", data="key: value\nlist:\n  - item1")
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name="yaml", description="Process YAML data", **kwargs)

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="operation", type=ParameterType.STRING, description="Operation to perform",
                         required=True, choices=["parse", "stringify", "validate", "merge", "get_key"]),
            ToolParameter(name="data", type=ParameterType.STRING, description="YAML data string", required=True),
            ToolParameter(name="key", type=ParameterType.STRING, description="Key to retrieve", required=False),
            ToolParameter(name="merge_data", type=ParameterType.STRING, description="Second YAML for merging", required=False),
        ]

    def execute(self, **kwargs: Any) -> ToolResult:
        if not _HAS_YAML:
            return ToolResult(success=False, error="PyYAML is not installed")

        operation = kwargs.get("operation", "")
        data = kwargs.get("data", "")

        if not operation:
            return ToolResult(success=False, error="No operation specified")

        try:
            if operation == "parse":
                return self._parse(data)
            elif operation == "stringify":
                return self._stringify(data)
            elif operation == "validate":
                return self._validate(data)
            elif operation == "merge":
                merge_data = kwargs.get("merge_data", "")
                return self._merge(data, merge_data)
            elif operation == "get_key":
                key = kwargs.get("key", "")
                return self._get_key(data, key)
            else:
                return ToolResult(success=False, error=f"Unknown operation: {operation}")
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))

    def _parse(self, data: str) -> ToolResult:
        """Parse YAML string to Python object."""
        result = yaml.safe_load(data)
        return ToolResult(success=True, output=result)

    def _stringify(self, data: str) -> ToolResult:
        """Serialize a YAML-parsed object back to string."""
        parsed = yaml.safe_load(data)
        output = yaml.dump(parsed, default_flow_style=False)
        return ToolResult(success=True, output=output)

    def _validate(self, data: str) -> ToolResult:
        """Validate YAML syntax."""
        try:
            yaml.safe_load(data)
            return ToolResult(success=True, output={"valid": True})
        except yaml.YAMLError as e:
            return ToolResult(success=True, output={"valid": False, "error": str(e)})

    def _merge(self, data1: str, data2: str) -> ToolResult:
        """Merge two YAML documents."""
        obj1 = yaml.safe_load(data1) or {}
        obj2 = yaml.safe_load(data2) or {}
        merged = {**obj1, **obj2}
        return ToolResult(success=True, output=merged)

    def _get_key(self, data: str, key: str) -> ToolResult:
        """Get a specific key from YAML data."""
        if not key:
            return ToolResult(success=False, error="No key specified")
        parsed = yaml.safe_load(data)
        if not isinstance(parsed, dict):
            return ToolResult(success=False, error="YAML data is not a mapping")
        value = parsed.get(key)
        if value is None:
            return ToolResult(success=False, error=f"Key '{key}' not found")
        return ToolResult(success=True, output=value)
