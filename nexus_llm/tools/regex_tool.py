"""Nexus-LLM Regex Tool.

Provides regular expression operations including matching, searching,
replacing, and splitting text using Python's re module.
"""

import logging
import re
from typing import Any, Dict, List, Optional

from nexus_llm.tools.base_tool import BaseTool, ToolParameter, ToolResult, ParameterType

logger = logging.getLogger(__name__)


class RegexTool(BaseTool):
    """Tool for regular expression operations.

    Supports matching, searching, finding all matches, replacing,
    and splitting text.

    Example::

        tool = RegexTool()
        result = tool.run(operation="findall", text="Hello 123 World 456", pattern=r"\d+")
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name="regex", description="Regular expression operations", **kwargs)

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="operation", type=ParameterType.STRING, description="Operation to perform",
                         required=True, choices=["match", "search", "findall", "sub", "split"]),
            ToolParameter(name="text", type=ParameterType.STRING, description="Input text", required=True),
            ToolParameter(name="pattern", type=ParameterType.STRING, description="Regex pattern", required=True),
            ToolParameter(name="replacement", type=ParameterType.STRING, description="Replacement string for sub", required=False),
            ToolParameter(name="flags", type=ParameterType.STRING, description="Regex flags (e.g., IGNORECASE)", required=False),
        ]

    def execute(self, **kwargs: Any) -> ToolResult:
        operation = kwargs.get("operation", "")
        text = kwargs.get("text", "")
        pattern = kwargs.get("pattern", "")

        if not operation:
            return ToolResult(success=False, error="No operation specified")
        if pattern is None:
            return ToolResult(success=False, error="No pattern specified")

        try:
            flags = self._parse_flags(kwargs.get("flags", ""))
            compiled = re.compile(pattern, flags)

            if operation == "match":
                return self._match(compiled, text)
            elif operation == "search":
                return self._search(compiled, text)
            elif operation == "findall":
                return self._findall(compiled, text)
            elif operation == "sub":
                replacement = kwargs.get("replacement", "")
                return self._sub(compiled, text, replacement)
            elif operation == "split":
                return self._split(compiled, text)
            else:
                return ToolResult(success=False, error=f"Unknown operation: {operation}")
        except re.error as exc:
            return ToolResult(success=False, error=f"Invalid regex: {exc}")
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))

    def _parse_flags(self, flags_str: str) -> int:
        """Parse flag string into re flags."""
        flag_map = {
            "IGNORECASE": re.IGNORECASE,
            "MULTILINE": re.MULTILINE,
            "DOTALL": re.DOTALL,
            "VERBOSE": re.VERBOSE,
        }
        result = 0
        for name in flags_str.split(","):
            name = name.strip().upper()
            if name in flag_map:
                result |= flag_map[name]
        return result

    def _match(self, pattern: re.Pattern, text: str) -> ToolResult:
        match = pattern.match(text)
        if match:
            return ToolResult(success=True, output={
                "matched": True,
                "match": match.group(),
                "groups": match.groups(),
                "span": match.span(),
            })
        return ToolResult(success=True, output={"matched": False})

    def _search(self, pattern: re.Pattern, text: str) -> ToolResult:
        match = pattern.search(text)
        if match:
            return ToolResult(success=True, output={
                "found": True,
                "match": match.group(),
                "groups": match.groups(),
                "span": match.span(),
            })
        return ToolResult(success=True, output={"found": False})

    def _findall(self, pattern: re.Pattern, text: str) -> ToolResult:
        matches = pattern.findall(text)
        return ToolResult(success=True, output=matches, metadata={"match_count": len(matches)})

    def _sub(self, pattern: re.Pattern, text: str, replacement: str) -> ToolResult:
        result = pattern.sub(replacement, text)
        count = len(pattern.findall(text))
        return ToolResult(success=True, output=result, metadata={"replacements": count})

    def _split(self, pattern: re.Pattern, text: str) -> ToolResult:
        parts = pattern.split(text)
        return ToolResult(success=True, output=parts, metadata={"part_count": len(parts)})
