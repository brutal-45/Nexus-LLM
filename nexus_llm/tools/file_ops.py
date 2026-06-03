"""Nexus-LLM File Operations Tool.

Provides the FileOpsTool for safe file system operations including
reading, writing, listing, and basic file management.
"""

import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List

from nexus_llm.tools.base_tool import BaseTool, ToolParameter, ToolResult, ParameterType

logger = logging.getLogger(__name__)

# Default sandbox root — all operations are constrained to this directory
_DEFAULT_SANDBOX = os.getcwd()


class FileOpsTool(BaseTool):
    """File operations tool for safe file system access.

    All file paths are resolved relative to a configurable sandbox root
    to prevent directory traversal attacks.

    Supported operations: read, write, append, delete, list, exists, mkdir, copy, move.

    Example::

        fops = FileOpsTool(sandbox_root="/tmp/workspace")
        result = fops.execute(operation="write", path="hello.txt", content="Hello!")
        result = fops.execute(operation="read", path="hello.txt")
    """

    def __init__(self, sandbox_root: str = _DEFAULT_SANDBOX) -> None:
        super().__init__(name="file_ops", description="Safe file system operations within a sandbox")
        self._sandbox = os.path.abspath(sandbox_root)

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="operation",
                type=ParameterType.STRING,
                description="File operation to perform",
                required=True,
                choices=["read", "write", "append", "delete", "list", "exists", "mkdir", "copy", "move"],
            ),
            ToolParameter(name="path", type=ParameterType.STRING, description="File or directory path", required=True),
            ToolParameter(name="content", type=ParameterType.STRING, description="Content for write/append", required=False),
            ToolParameter(name="destination", type=ParameterType.STRING, description="Destination path for copy/move", required=False),
        ]

    def execute(
        self,
        operation: str = "",
        path: str = "",
        content: str = "",
        destination: str = "",
        **kwargs: Any,
    ) -> ToolResult:
        """Execute a file operation.

        Args:
            operation: The operation to perform.
            path: Target file or directory path.
            content: Content for write/append operations.
            destination: Destination for copy/move operations.

        Returns:
            ToolResult with operation output.
        """
        try:
            safe_path = self._resolve(path)
        except ValueError as exc:
            return ToolResult(tool_name=self.name, success=False, error=str(exc))

        ops = {
            "read": self._read,
            "write": self._write,
            "append": self._append,
            "delete": self._delete,
            "list": self._list,
            "exists": self._exists,
            "mkdir": self._mkdir,
            "copy": self._copy,
            "move": self._move,
        }

        handler = ops.get(operation)
        if handler is None:
            return ToolResult(tool_name=self.name, success=False, error=f"Unknown operation: {operation}")

        try:
            if operation in ("write", "append"):
                return handler(safe_path, content)
            elif operation in ("copy", "move"):
                try:
                    safe_dest = self._resolve(destination)
                except ValueError as exc:
                    return ToolResult(tool_name=self.name, success=False, error=str(exc))
                return handler(safe_path, safe_dest)
            else:
                return handler(safe_path)
        except Exception as exc:
            return ToolResult(tool_name=self.name, success=False, error=str(exc))

    def _resolve(self, path: str) -> str:
        """Resolve a path within the sandbox, preventing traversal.

        Args:
            path: Relative or absolute path.

        Returns:
            Absolute path within the sandbox.

        Raises:
            ValueError: If the path escapes the sandbox.
        """
        resolved = os.path.abspath(os.path.join(self._sandbox, path))
        if not resolved.startswith(self._sandbox):
            raise ValueError(f"Path escapes sandbox: {path}")
        return resolved

    def _read(self, path: str) -> ToolResult:
        if not os.path.isfile(path):
            return ToolResult(tool_name=self.name, success=False, error=f"File not found: {path}")
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        return ToolResult(tool_name=self.name, success=True, output=content, metadata={"path": path, "size": len(content)})

    def _write(self, path: str, content: str) -> ToolResult:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return ToolResult(tool_name=self.name, success=True, output=f"Wrote {len(content)} chars", metadata={"path": path})

    def _append(self, path: str, content: str) -> ToolResult:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(content)
        return ToolResult(tool_name=self.name, success=True, output=f"Appended {len(content)} chars", metadata={"path": path})

    def _delete(self, path: str) -> ToolResult:
        if os.path.isfile(path):
            os.remove(path)
            return ToolResult(tool_name=self.name, success=True, output=f"Deleted file: {path}")
        elif os.path.isdir(path):
            shutil.rmtree(path)
            return ToolResult(tool_name=self.name, success=True, output=f"Deleted directory: {path}")
        return ToolResult(tool_name=self.name, success=False, error=f"Not found: {path}")

    def _list(self, path: str) -> ToolResult:
        if not os.path.isdir(path):
            return ToolResult(tool_name=self.name, success=False, error=f"Not a directory: {path}")
        entries = os.listdir(path)
        return ToolResult(tool_name=self.name, success=True, output=entries, metadata={"path": path, "count": len(entries)})

    def _exists(self, path: str) -> ToolResult:
        exists = os.path.exists(path)
        return ToolResult(tool_name=self.name, success=True, output=exists, metadata={"path": path})

    def _mkdir(self, path: str) -> ToolResult:
        os.makedirs(path, exist_ok=True)
        return ToolResult(tool_name=self.name, success=True, output=f"Created directory: {path}")

    def _copy(self, src: str, dst: str) -> ToolResult:
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
        return ToolResult(tool_name=self.name, success=True, output=f"Copied to {dst}")

    def _move(self, src: str, dst: str) -> ToolResult:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)
        return ToolResult(tool_name=self.name, success=True, output=f"Moved to {dst}")
