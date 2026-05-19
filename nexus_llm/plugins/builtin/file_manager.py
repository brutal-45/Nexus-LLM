"""File manager plugin for reading, writing, and listing files.

A builtin plugin providing safe file system operations with
configurable access controls and path restrictions.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from nexus_llm.plugins.hook import HookManager, HookPriority

logger = logging.getLogger(__name__)


class FileManagerPlugin:
    """Plugin providing file system operations.

    Supports reading, writing, listing, and managing files with
    configurable allowed directories for security.
    """

    name = "file_manager"
    version = "1.0.0"
    description = "Read, write, list, and manage files on the filesystem"
    dependencies: List[str] = []
    tags = ["file", "filesystem", "io", "builtin"]

    def __init__(
        self,
        hook_manager: Optional[HookManager] = None,
        allowed_dirs: Optional[List[str]] = None,
        max_file_size: int = 10 * 1024 * 1024,  # 10 MB
        **kwargs,
    ):
        """Initialize the file manager plugin.

        Args:
            hook_manager: Optional hook manager.
            allowed_dirs: List of allowed directory paths. None = all allowed.
            max_file_size: Maximum file size in bytes for read operations.
        """
        self.hook_manager = hook_manager
        self.allowed_dirs = [os.path.abspath(d) for d in (allowed_dirs or [])]
        self.max_file_size = max_file_size
        self._active = False
        self._operation_log: List[Dict[str, Any]] = []

    def activate(self) -> None:
        """Activate the file manager plugin."""
        if self.hook_manager:
            self.hook_manager.register(
                "tool_request",
                self._handle_tool_request,
                name="file_manager_tool_handler",
                priority=HookPriority.NORMAL,
                owner=self.name,
            )
        self._active = True
        logger.info("File manager plugin activated.")

    def deactivate(self) -> None:
        """Deactivate the file manager plugin."""
        if self.hook_manager:
            self.hook_manager.unregister_by_owner(self.name)
        self._active = False
        logger.info("File manager plugin deactivated.")

    def _is_path_allowed(self, path: str) -> bool:
        """Check if a path is within allowed directories."""
        if not self.allowed_dirs:
            return True
        abs_path = os.path.abspath(path)
        return any(abs_path.startswith(allowed) for allowed in self.allowed_dirs)

    def _log_operation(self, operation: str, path: str, success: bool, detail: str = "") -> None:
        """Log a file operation."""
        self._operation_log.append({
            "operation": operation,
            "path": path,
            "success": success,
            "detail": detail,
        })

    def read_file(self, path: str, encoding: str = "utf-8", max_lines: Optional[int] = None) -> Dict[str, Any]:
        """Read the contents of a file.

        Args:
            path: File path to read.
            encoding: File encoding.
            max_lines: Maximum number of lines to return.

        Returns:
            Dict with file contents and metadata.
        """
        if not self._is_path_allowed(path):
            self._log_operation("read", path, False, "Access denied")
            return {"success": False, "error": f"Access denied: '{path}' is not in allowed directories."}

        try:
            file_path = Path(path)
            if not file_path.exists():
                self._log_operation("read", path, False, "File not found")
                return {"success": False, "error": f"File not found: {path}"}
            if not file_path.is_file():
                self._log_operation("read", path, False, "Not a file")
                return {"success": False, "error": f"Not a file: {path}"}

            size = file_path.stat().st_size
            if size > self.max_file_size:
                self._log_operation("read", path, False, "File too large")
                return {"success": False, "error": f"File too large ({size} bytes). Max: {self.max_file_size}"}

            content = file_path.read_text(encoding=encoding)
            total_lines = content.count("\n") + 1

            if max_lines:
                lines = content.split("\n")[:max_lines]
                content = "\n".join(lines)
                if total_lines > max_lines:
                    content += f"\n... (truncated at {max_lines}/{total_lines} lines)"

            self._log_operation("read", path, True, f"Read {size} bytes")
            return {
                "success": True,
                "content": content,
                "path": str(file_path),
                "size": size,
                "total_lines": total_lines,
            }
        except UnicodeDecodeError:
            self._log_operation("read", path, False, "Encoding error")
            return {"success": False, "error": f"Cannot read file as {encoding}: {path}"}
        except Exception as e:
            self._log_operation("read", path, False, str(e))
            return {"success": False, "error": f"Error reading file: {e}"}

    def write_file(self, path: str, content: str, mode: str = "write", encoding: str = "utf-8") -> Dict[str, Any]:
        """Write content to a file.

        Args:
            path: File path to write.
            content: Content to write.
            mode: 'write' to overwrite, 'append' to append.
            encoding: File encoding.

        Returns:
            Dict with operation result.
        """
        if not self._is_path_allowed(path):
            self._log_operation("write", path, False, "Access denied")
            return {"success": False, "error": f"Access denied: '{path}' is not in allowed directories."}

        try:
            file_path = Path(path)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            write_mode = "a" if mode == "append" else "w"
            with open(file_path, write_mode, encoding=encoding) as f:
                f.write(content)

            chars = len(content)
            self._log_operation("write", path, True, f"Wrote {chars} chars in {mode} mode")
            return {
                "success": True,
                "path": str(file_path),
                "chars_written": chars,
                "mode": mode,
            }
        except Exception as e:
            self._log_operation("write", path, False, str(e))
            return {"success": False, "error": f"Error writing file: {e}"}

    def list_directory(self, path: str = ".", pattern: str = "*", recursive: bool = False) -> Dict[str, Any]:
        """List files and directories.

        Args:
            path: Directory path to list.
            pattern: Glob pattern for filtering.
            recursive: Whether to list recursively.

        Returns:
            Dict with listing results.
        """
        if not self._is_path_allowed(path):
            return {"success": False, "error": f"Access denied: '{path}' is not in allowed directories."}

        try:
            dir_path = Path(path)
            if not dir_path.exists():
                return {"success": False, "error": f"Directory not found: {path}"}
            if not dir_path.is_dir():
                return {"success": False, "error": f"Not a directory: {path}"}

            entries = []
            glob_pattern = f"**/{pattern}" if recursive else pattern

            for entry in sorted(dir_path.glob(glob_pattern)):
                if entry.name.startswith(".") or entry.name == "__pycache__":
                    continue
                entries.append({
                    "name": entry.name,
                    "path": str(entry),
                    "is_dir": entry.is_dir(),
                    "size": entry.stat().st_size if entry.is_file() else 0,
                })

            self._log_operation("list", path, True, f"Listed {len(entries)} entries")
            return {
                "success": True,
                "path": str(dir_path),
                "entries": entries,
                "count": len(entries),
            }
        except Exception as e:
            self._log_operation("list", path, False, str(e))
            return {"success": False, "error": f"Error listing directory: {e}"}

    def delete_file(self, path: str) -> Dict[str, Any]:
        """Delete a file.

        Args:
            path: File path to delete.

        Returns:
            Dict with operation result.
        """
        if not self._is_path_allowed(path):
            self._log_operation("delete", path, False, "Access denied")
            return {"success": False, "error": f"Access denied: '{path}' is not in allowed directories."}

        try:
            file_path = Path(path)
            if not file_path.exists():
                return {"success": False, "error": f"File not found: {path}"}

            if file_path.is_file():
                file_path.unlink()
                self._log_operation("delete", path, True, "Deleted file")
                return {"success": True, "path": str(file_path), "action": "deleted"}
            elif file_path.is_dir():
                shutil.rmtree(file_path)
                self._log_operation("delete", path, True, "Deleted directory")
                return {"success": True, "path": str(file_path), "action": "deleted_directory"}

            return {"success": False, "error": f"Unknown file type: {path}"}
        except Exception as e:
            self._log_operation("delete", path, False, str(e))
            return {"success": False, "error": f"Error deleting: {e}"}

    def file_info(self, path: str) -> Dict[str, Any]:
        """Get detailed information about a file.

        Args:
            path: File path.

        Returns:
            Dict with file metadata.
        """
        if not self._is_path_allowed(path):
            return {"success": False, "error": f"Access denied: '{path}' is not in allowed directories."}

        try:
            file_path = Path(path)
            if not file_path.exists():
                return {"success": False, "error": f"Path not found: {path}"}

            stat = file_path.stat()
            return {
                "success": True,
                "path": str(file_path),
                "name": file_path.name,
                "suffix": file_path.suffix,
                "is_file": file_path.is_file(),
                "is_dir": file_path.is_dir(),
                "size": stat.st_size,
                "modified": stat.st_mtime,
                "created": stat.st_ctime,
                "parent": str(file_path.parent),
            }
        except Exception as e:
            return {"success": False, "error": f"Error getting file info: {e}"}

    def get_operation_log(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent file operation log."""
        return self._operation_log[-limit:]

    def _handle_tool_request(self, result, *args, **kwargs):
        """Handle tool requests for file operations."""
        tool_name = kwargs.get("tool_name", "")
        if tool_name == "file_read":
            path = kwargs.get("path", "")
            if path:
                read_result = self.read_file(path)
                return read_result.get("content", read_result.get("error", ""))
        elif tool_name == "file_write":
            path = kwargs.get("path", "")
            content = kwargs.get("content", "")
            if path and content:
                write_result = self.write_file(path, content)
                return write_result.get("success", False)
        return result
