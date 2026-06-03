"""Nexus-LLM Shell Command Tool.

Provides the ShellTool for executing shell commands in a controlled
environment with timeout and output capture.
"""

import logging
import os
import subprocess
import tempfile
from typing import Any, Dict, List

from nexus_llm.tools.base_tool import BaseTool, ToolParameter, ToolResult, ParameterType

logger = logging.getLogger(__name__)


class ShellTool(BaseTool):
    """Shell command execution tool.

    Runs shell commands with configurable timeout, working directory,
    and environment variables. Output is captured and returned.

    .. warning::

        This tool executes arbitrary commands. Use with caution and
        only in trusted environments.

    Example::

        shell = ShellTool(timeout=30)
        result = shell.execute(command="echo hello")
    """

    def __init__(self, timeout: int = 60, allowed_commands: Optional[List[str]] = None) -> None:
        """Initialize the shell tool.

        Args:
            timeout: Default command timeout in seconds.
            allowed_commands: If set, only these command names are allowed.
        """
        super().__init__(name="shell", description="Execute shell commands with output capture")
        self._timeout = timeout
        self._allowed_commands = allowed_commands
        logger.debug("ShellTool initialized with timeout=%ds", timeout)

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="command", type=ParameterType.STRING, description="Shell command to execute", required=True),
            ToolParameter(name="timeout", type=ParameterType.INTEGER, description="Timeout in seconds", required=False, default=self._timeout),
            ToolParameter(name="working_dir", type=ParameterType.STRING, description="Working directory", required=False),
        ]

    def execute(self, command: str = "", timeout: int = 0, working_dir: str = "", **kwargs: Any) -> ToolResult:
        """Execute a shell command.

        Args:
            command: The shell command to run.
            timeout: Timeout in seconds (0 uses default).
            working_dir: Working directory for the command.

        Returns:
            ToolResult with stdout, stderr, and return code.
        """
        if not command.strip():
            return ToolResult(tool_name=self.name, success=False, error="Empty command")

        # Check allowed commands
        if self._allowed_commands is not None:
            cmd_name = command.split()[0] if command.split() else ""
            if cmd_name not in self._allowed_commands:
                return ToolResult(
                    tool_name=self.name,
                    success=False,
                    error=f"Command not allowed: {cmd_name}",
                )

        actual_timeout = timeout or self._timeout
        cwd = working_dir or None

        try:
            proc = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=actual_timeout,
                cwd=cwd,
            )
            success = proc.returncode == 0
            return ToolResult(
                tool_name=self.name,
                success=success,
                output=proc.stdout.strip() if proc.stdout else "",
                error=proc.stderr.strip() if proc.stderr else None,
                metadata={
                    "return_code": proc.returncode,
                    "command": command,
                    "timeout": actual_timeout,
                },
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"Command timed out after {actual_timeout}s",
                metadata={"command": command, "timeout": actual_timeout},
            )
        except Exception as exc:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"Execution error: {exc}",
            )
