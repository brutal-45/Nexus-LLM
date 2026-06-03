"""Nexus-LLM Text Manipulation Tool.

Provides the TextTool for common text operations such as search/replace,
splitting, joining, counting, and formatting.
"""

import logging
import re
from typing import Any, Dict, List

from nexus_llm.tools.base_tool import BaseTool, ToolParameter, ToolResult, ParameterType

logger = logging.getLogger(__name__)


class TextTool(BaseTool):
    """Text manipulation tool for common string operations.

    Supports operations: search, replace, split, join, count, trim,
    reverse, sort_lines, unique_lines, pad, wrap, truncate.

    Example::

        ttool = TextTool()
        result = ttool.execute(operation="replace", text="hello world", pattern="world", replacement="Nexus")
    """

    def __init__(self) -> None:
        super().__init__(name="text", description="Common text manipulation operations")

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="operation", type=ParameterType.STRING, description="Text operation", required=True,
                          choices=["search", "replace", "split", "join", "count", "trim", "reverse", "sort_lines",
                                   "unique_lines", "pad", "wrap", "truncate"]),
            ToolParameter(name="text", type=ParameterType.STRING, description="Input text", required=True),
            ToolParameter(name="pattern", type=ParameterType.STRING, description="Search pattern or delimiter", required=False),
            ToolParameter(name="replacement", type=ParameterType.STRING, description="Replacement string", required=False),
            ToolParameter(name="max_length", type=ParameterType.INTEGER, description="Max length for truncate/wrap", required=False, default=80),
        ]

    def execute(
        self,
        operation: str = "",
        text: str = "",
        pattern: str = "",
        replacement: str = "",
        max_length: int = 80,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute a text operation.

        Args:
            operation: The operation to perform.
            text: Input text.
            pattern: Pattern or delimiter.
            replacement: Replacement string.
            max_length: Max length for truncate/wrap.

        Returns:
            ToolResult with the operation output.
        """
        ops = {
            "search": self._search,
            "replace": self._replace,
            "split": self._split,
            "join": self._join,
            "count": self._count,
            "trim": self._trim,
            "reverse": self._reverse,
            "sort_lines": self._sort_lines,
            "unique_lines": self._unique_lines,
            "pad": self._pad,
            "wrap": self._wrap,
            "truncate": self._truncate,
        }

        handler = ops.get(operation)
        if handler is None:
            return ToolResult(tool_name=self.name, success=False, error=f"Unknown operation: {operation}")

        try:
            return handler(text, pattern=pattern, replacement=replacement, max_length=max_length)
        except Exception as exc:
            return ToolResult(tool_name=self.name, success=False, error=str(exc))

    def _search(self, text: str, pattern: str = "", **kw: Any) -> ToolResult:
        if not pattern:
            return ToolResult(tool_name=self.name, success=False, error="Pattern required for search")
        matches = [(m.start(), m.end(), m.group()) for m in re.finditer(pattern, text)]
        return ToolResult(tool_name=self.name, success=True, output=matches, metadata={"match_count": len(matches)})

    def _replace(self, text: str, pattern: str = "", replacement: str = "", **kw: Any) -> ToolResult:
        if not pattern:
            return ToolResult(tool_name=self.name, success=False, error="Pattern required for replace")
        result = re.sub(pattern, replacement, text)
        changes = len(re.findall(pattern, text))
        return ToolResult(tool_name=self.name, success=True, output=result, metadata={"replacements": changes})

    def _split(self, text: str, pattern: str = "", **kw: Any) -> ToolResult:
        delimiter = pattern or r'\s+'
        parts = re.split(delimiter, text)
        return ToolResult(tool_name=self.name, success=True, output=parts, metadata={"part_count": len(parts)})

    def _join(self, text: str, pattern: str = "", **kw: Any) -> ToolResult:
        # text is expected to be JSON-like list or newline-separated
        import json
        try:
            items = json.loads(text)
            if not isinstance(items, list):
                items = text.split('\n')
        except (json.JSONDecodeError, TypeError):
            items = text.split('\n')
        delimiter = pattern or " "
        result = delimiter.join(str(i) for i in items)
        return ToolResult(tool_name=self.name, success=True, output=result)

    def _count(self, text: str, pattern: str = "", **kw: Any) -> ToolResult:
        if pattern:
            count = len(re.findall(pattern, text))
        else:
            count = len(text)
        return ToolResult(tool_name=self.name, success=True, output=count, metadata={"pattern": pattern or "chars"})

    def _trim(self, text: str, **kw: Any) -> ToolResult:
        return ToolResult(tool_name=self.name, success=True, output=text.strip())

    def _reverse(self, text: str, **kw: Any) -> ToolResult:
        return ToolResult(tool_name=self.name, success=True, output=text[::-1])

    def _sort_lines(self, text: str, **kw: Any) -> ToolResult:
        lines = text.split('\n')
        return ToolResult(tool_name=self.name, success=True, output='\n'.join(sorted(lines)))

    def _unique_lines(self, text: str, **kw: Any) -> ToolResult:
        lines = text.split('\n')
        seen: List[str] = []
        for line in lines:
            if line not in seen:
                seen.append(line)
        return ToolResult(tool_name=self.name, success=True, output='\n'.join(seen))

    def _pad(self, text: str, max_length: int = 80, **kw: Any) -> ToolResult:
        padded = text.ljust(max_length)
        return ToolResult(tool_name=self.name, success=True, output=padded, metadata={"width": max_length})

    def _wrap(self, text: str, max_length: int = 80, **kw: Any) -> ToolResult:
        import textwrap
        wrapped = textwrap.fill(text, width=max_length)
        return ToolResult(tool_name=self.name, success=True, output=wrapped, metadata={"width": max_length})

    def _truncate(self, text: str, max_length: int = 80, **kw: Any) -> ToolResult:
        if len(text) <= max_length:
            return ToolResult(tool_name=self.name, success=True, output=text)
        truncated = text[:max_length - 3] + "..."
        return ToolResult(tool_name=self.name, success=True, output=truncated, metadata={"original_length": len(text)})
