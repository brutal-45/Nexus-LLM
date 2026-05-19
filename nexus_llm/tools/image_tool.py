"""Nexus-LLM Image Processing Tool.

Provides basic image processing capabilities including resizing,
format conversion, and metadata extraction.
"""

import logging
from typing import Any, Dict, List, Optional

from nexus_llm.tools.base_tool import BaseTool, ToolParameter, ToolResult, ParameterType

logger = logging.getLogger(__name__)


class ImageTool(BaseTool):
    """Tool for image processing operations.

    Supports metadata extraction, format information, and basic
    image manipulation descriptors.

    Example::

        tool = ImageTool()
        result = tool.run(operation="info", path="/path/to/image.png")
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name="image", description="Process and analyze images", **kwargs)

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="operation", type=ParameterType.STRING, description="Operation to perform", required=True,
                         choices=["info", "resize", "convert", "metadata"]),
            ToolParameter(name="path", type=ParameterType.STRING, description="Path to image file", required=True),
            ToolParameter(name="width", type=ParameterType.INTEGER, description="Target width for resize", required=False),
            ToolParameter(name="height", type=ParameterType.INTEGER, description="Target height for resize", required=False),
            ToolParameter(name="format", type=ParameterType.STRING, description="Target format for conversion", required=False),
        ]

    def execute(self, **kwargs: Any) -> ToolResult:
        operation = kwargs.get("operation", "")
        path = kwargs.get("path", "")

        if not operation:
            return ToolResult(success=False, error="No operation specified")
        if not path:
            return ToolResult(success=False, error="No image path specified")

        try:
            if operation == "info":
                return self._get_info(path)
            elif operation == "resize":
                width = kwargs.get("width")
                height = kwargs.get("height")
                return self._resize(path, width, height)
            elif operation == "convert":
                fmt = kwargs.get("format", "png")
                return self._convert(path, fmt)
            elif operation == "metadata":
                return self._get_metadata(path)
            else:
                return ToolResult(success=False, error=f"Unknown operation: {operation}")
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))

    def _get_info(self, path: str) -> ToolResult:
        """Get basic image information."""
        import os
        if not os.path.exists(path):
            return ToolResult(success=False, error=f"File not found: {path}")
        stat = os.stat(path)
        ext = os.path.splitext(path)[1].lower()
        return ToolResult(success=True, output={
            "path": path,
            "extension": ext,
            "size_bytes": stat.st_size,
            "modified": stat.st_mtime,
        })

    def _resize(self, path: str, width: Optional[int], height: Optional[int]) -> ToolResult:
        """Describe a resize operation."""
        if not width and not height:
            return ToolResult(success=False, error="Must specify at least width or height")
        return ToolResult(success=True, output={
            "path": path,
            "operation": "resize",
            "target_width": width,
            "target_height": height,
            "status": "described",
        })

    def _convert(self, path: str, fmt: str) -> ToolResult:
        """Describe a format conversion."""
        return ToolResult(success=True, output={
            "path": path,
            "operation": "convert",
            "target_format": fmt,
            "status": "described",
        })

    def _get_metadata(self, path: str) -> ToolResult:
        """Extract image metadata."""
        import os
        if not os.path.exists(path):
            return ToolResult(success=False, error=f"File not found: {path}")
        return ToolResult(success=True, output={
            "path": path,
            "metadata": {
                "size_bytes": os.path.getsize(path),
            },
        })
