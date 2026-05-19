"""Nexus-LLM JSON Processor Tool.

Provides the JSONTool for parsing, querying, validating, and
transforming JSON data.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from nexus_llm.tools.base_tool import BaseTool, ToolParameter, ToolResult, ParameterType

logger = logging.getLogger(__name__)


class JSONTool(BaseTool):
    """JSON processor tool for parsing, querying, and transforming JSON.

    Supports operations like parse, format, get (path access), set,
    delete, keys, validate, merge, and diff.

    Example::

        jtool = JSONTool()
        result = jtool.execute(operation="parse", data='{"name": "Alice", "age": 30}')
        result = jtool.execute(operation="get", data='{"a": {"b": 1}}', path="a.b")
    """

    def __init__(self) -> None:
        super().__init__(name="json", description="Parse, query, and transform JSON data")

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="operation", type=ParameterType.STRING, description="JSON operation", required=True,
                          choices=["parse", "format", "get", "set", "delete", "keys", "validate", "merge", "diff"]),
            ToolParameter(name="data", type=ParameterType.STRING, description="JSON string input", required=True),
            ToolParameter(name="path", type=ParameterType.STRING, description="Dot-notation path (for get/set/delete)", required=False),
            ToolParameter(name="value", type=ParameterType.STRING, description="Value to set (JSON encoded)", required=False),
            ToolParameter(name="data2", type=ParameterType.STRING, description="Second JSON string (for merge/diff)", required=False),
        ]

    def execute(
        self,
        operation: str = "",
        data: str = "",
        path: str = "",
        value: str = "",
        data2: str = "",
        **kwargs: Any,
    ) -> ToolResult:
        """Execute a JSON operation.

        Args:
            operation: The operation to perform.
            data: Primary JSON string.
            path: Dot-notation path.
            value: Value for set operations.
            data2: Secondary JSON string for merge/diff.

        Returns:
            ToolResult with operation output.
        """
        ops = {
            "parse": self._parse,
            "format": self._format,
            "get": self._get,
            "set": self._set,
            "delete": self._delete,
            "keys": self._keys,
            "validate": self._validate,
            "merge": self._merge,
            "diff": self._diff,
        }

        handler = ops.get(operation)
        if handler is None:
            return ToolResult(tool_name=self.name, success=False, error=f"Unknown operation: {operation}")

        try:
            return handler(data, path=path, value=value, data2=data2)
        except Exception as exc:
            return ToolResult(tool_name=self.name, success=False, error=str(exc))

    def _parse(self, data: str, **kw: Any) -> ToolResult:
        parsed = json.loads(data)
        return ToolResult(tool_name=self.name, success=True, output=parsed, metadata={"type": type(parsed).__name__})

    def _format(self, data: str, **kw: Any) -> ToolResult:
        parsed = json.loads(data)
        formatted = json.dumps(parsed, indent=2, ensure_ascii=False)
        return ToolResult(tool_name=self.name, success=True, output=formatted)

    def _get(self, data: str, path: str = "", **kw: Any) -> ToolResult:
        if not path:
            return ToolResult(tool_name=self.name, success=False, error="Path is required for 'get'")
        obj = json.loads(data)
        result = self._resolve_path(obj, path)
        return ToolResult(tool_name=self.name, success=True, output=result, metadata={"path": path})

    def _set(self, data: str, path: str = "", value: str = "", **kw: Any) -> ToolResult:
        if not path:
            return ToolResult(tool_name=self.name, success=False, error="Path is required for 'set'")
        obj = json.loads(data)
        val = json.loads(value)
        self._set_path(obj, path, val)
        return ToolResult(tool_name=self.name, success=True, output=obj, metadata={"path": path})

    def _delete(self, data: str, path: str = "", **kw: Any) -> ToolResult:
        if not path:
            return ToolResult(tool_name=self.name, success=False, error="Path is required for 'delete'")
        obj = json.loads(data)
        self._delete_path(obj, path)
        return ToolResult(tool_name=self.name, success=True, output=obj, metadata={"deleted_path": path})

    def _keys(self, data: str, **kw: Any) -> ToolResult:
        obj = json.loads(data)
        if isinstance(obj, dict):
            return ToolResult(tool_name=self.name, success=True, output=list(obj.keys()))
        return ToolResult(tool_name=self.name, success=False, error="Not a JSON object")

    def _validate(self, data: str, **kw: Any) -> ToolResult:
        try:
            json.loads(data)
            return ToolResult(tool_name=self.name, success=True, output=True)
        except json.JSONDecodeError as exc:
            return ToolResult(tool_name=self.name, success=True, output=False, error=str(exc))

    def _merge(self, data: str, data2: str = "", **kw: Any) -> ToolResult:
        if not data2:
            return ToolResult(tool_name=self.name, success=False, error="data2 is required for 'merge'")
        obj1 = json.loads(data)
        obj2 = json.loads(data2)
        merged = self._deep_merge(obj1, obj2)
        return ToolResult(tool_name=self.name, success=True, output=merged)

    def _diff(self, data: str, data2: str = "", **kw: Any) -> ToolResult:
        if not data2:
            return ToolResult(tool_name=self.name, success=False, error="data2 is required for 'diff'")
        obj1 = json.loads(data)
        obj2 = json.loads(data2)
        diffs = self._compute_diff(obj1, obj2)
        return ToolResult(tool_name=self.name, success=True, output=diffs)

    @staticmethod
    def _resolve_path(obj: Any, path: str) -> Any:
        """Navigate a dot-notation path into a nested structure."""
        current = obj
        for part in path.split("."):
            if isinstance(current, dict):
                current = current[part]
            elif isinstance(current, list):
                current = current[int(part)]
            else:
                raise KeyError(f"Cannot navigate into {type(current).__name__} with key '{part}'")
        return current

    @staticmethod
    def _set_path(obj: Any, path: str, value: Any) -> None:
        """Set a value at a dot-notation path."""
        parts = path.split(".")
        current = obj
        for part in parts[:-1]:
            if isinstance(current, dict):
                current = current.setdefault(part, {})
            elif isinstance(current, list):
                current = current[int(part)]
        last = parts[-1]
        if isinstance(current, dict):
            current[last] = value
        elif isinstance(current, list):
            current[int(last)] = value

    @staticmethod
    def _delete_path(obj: Any, path: str) -> None:
        """Delete a key at a dot-notation path."""
        parts = path.split(".")
        current = obj
        for part in parts[:-1]:
            if isinstance(current, dict):
                current = current[part]
            elif isinstance(current, list):
                current = current[int(part)]
        last = parts[-1]
        if isinstance(current, dict):
            current.pop(last, None)
        elif isinstance(current, list):
            del current[int(last)]

    @staticmethod
    def _deep_merge(a: Any, b: Any) -> Any:
        """Deep merge two dicts; b overrides a."""
        if isinstance(a, dict) and isinstance(b, dict):
            result = dict(a)
            for key, val in b.items():
                result[key] = JSONTool._deep_merge(a.get(key), val)
            return result
        return b

    @staticmethod
    def _compute_diff(a: Any, b: Any, prefix: str = "") -> List[Dict[str, Any]]:
        """Compute differences between two JSON structures."""
        diffs: List[Dict[str, Any]] = []
        if isinstance(a, dict) and isinstance(b, dict):
            all_keys = set(a.keys()) | set(b.keys())
            for key in all_keys:
                path = f"{prefix}.{key}" if prefix else key
                if key not in a:
                    diffs.append({"path": path, "change": "added", "value": b[key]})
                elif key not in b:
                    diffs.append({"path": path, "change": "removed", "value": a[key]})
                elif a[key] != b[key]:
                    if isinstance(a[key], dict) and isinstance(b[key], dict):
                        diffs.extend(JSONTool._compute_diff(a[key], b[key], path))
                    else:
                        diffs.append({"path": path, "change": "modified", "old": a[key], "new": b[key]})
        return diffs
