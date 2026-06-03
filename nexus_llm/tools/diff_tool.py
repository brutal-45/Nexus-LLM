"""Nexus-LLM Diff Comparison Tool.

Provides text and structured data comparison capabilities, producing
unified diffs and identifying additions, deletions, and modifications.
"""

import difflib
import logging
from typing import Any, Dict, List, Optional

from nexus_llm.tools.base_tool import BaseTool, ToolParameter, ToolResult, ParameterType

logger = logging.getLogger(__name__)


class DiffTool(BaseTool):
    """Tool for comparing text and data.

    Supports unified diff, side-by-side comparison, and structured
    change detection.

    Example::

        tool = DiffTool()
        result = tool.run(text1="hello world", text2="hello earth")
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name="diff", description="Compare text and data", **kwargs)

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="text1", type=ParameterType.STRING, description="First text to compare", required=True),
            ToolParameter(name="text2", type=ParameterType.STRING, description="Second text to compare", required=True),
            ToolParameter(name="mode", type=ParameterType.STRING, description="Comparison mode",
                         required=False, default="unified", choices=["unified", "context", "changes"]),
            ToolParameter(name="context_lines", type=ParameterType.INTEGER, description="Number of context lines",
                         required=False, default=3),
        ]

    def execute(self, **kwargs: Any) -> ToolResult:
        text1 = kwargs.get("text1", "")
        text2 = kwargs.get("text2", "")
        mode = kwargs.get("mode", "unified")
        context_lines = kwargs.get("context_lines", 3)

        try:
            if mode == "unified":
                diff = self._unified_diff(text1, text2, context_lines)
            elif mode == "context":
                diff = self._context_diff(text1, text2, context_lines)
            elif mode == "changes":
                diff = self._changes(text1, text2)
            else:
                return ToolResult(success=False, error=f"Unknown mode: {mode}")

            return ToolResult(success=True, output=diff)
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))

    def _unified_diff(self, text1: str, text2: str, context_lines: int) -> Dict[str, Any]:
        """Generate a unified diff."""
        lines1 = text1.splitlines(keepends=True)
        lines2 = text2.splitlines(keepends=True)
        diff_lines = list(difflib.unified_diff(lines1, lines2, n=context_lines))
        return {
            "format": "unified",
            "diff": "".join(diff_lines),
            "has_changes": len(diff_lines) > 0,
        }

    def _context_diff(self, text1: str, text2: str, context_lines: int) -> Dict[str, Any]:
        """Generate a context diff."""
        lines1 = text1.splitlines(keepends=True)
        lines2 = text2.splitlines(keepends=True)
        diff_lines = list(difflib.context_diff(lines1, lines2, n=context_lines))
        return {
            "format": "context",
            "diff": "".join(diff_lines),
            "has_changes": len(diff_lines) > 0,
        }

    def _changes(self, text1: str, text2: str) -> Dict[str, Any]:
        """Identify structural changes between two texts."""
        lines1 = set(text1.splitlines())
        lines2 = set(text2.splitlines())
        added = lines2 - lines1
        removed = lines1 - lines2
        common = lines1 & lines2
        return {
            "format": "changes",
            "added": sorted(added),
            "removed": sorted(removed),
            "common_count": len(common),
            "added_count": len(added),
            "removed_count": len(removed),
            "similarity": len(common) / max(len(lines1 | lines2), 1),
        }
