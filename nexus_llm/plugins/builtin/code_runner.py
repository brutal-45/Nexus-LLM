"""Code runner plugin for executing Python code in a sandbox.

A builtin plugin providing safe Python code execution with
resource limits, output capture, and error reporting.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional

from nexus_llm.plugins.hook import HookManager, HookPriority

logger = logging.getLogger(__name__)


class CodeRunnerPlugin:
    """Plugin providing Python code execution in a subprocess sandbox.

    Executes Python code in an isolated subprocess with configurable
    timeout, output size limits, and detailed result reporting.
    """

    name = "code_runner"
    version = "1.0.0"
    description = "Execute Python code in a sandboxed environment"
    dependencies: List[str] = []
    tags = ["code", "execution", "python", "builtin"]

    def __init__(
        self,
        hook_manager: Optional[HookManager] = None,
        default_timeout: int = 30,
        max_output_size: int = 50000,
        allowed_imports: Optional[List[str]] = None,
        blocked_imports: Optional[List[str]] = None,
        **kwargs,
    ):
        """Initialize the code runner plugin.

        Args:
            hook_manager: Optional hook manager.
            default_timeout: Default execution timeout in seconds.
            max_output_size: Maximum output size in characters.
            allowed_imports: If set, only these modules can be imported.
            blocked_imports: Modules that cannot be imported.
        """
        self.hook_manager = hook_manager
        self.default_timeout = default_timeout
        self.max_output_size = max_output_size
        self.allowed_imports = allowed_imports
        self.blocked_imports = blocked_imports or [
            "socket", "subprocess", "os.system", "shutil",
            "ctypes", "multiprocessing", "signal",
        ]
        self._active = False
        self._execution_history: List[Dict[str, Any]] = []

    def activate(self) -> None:
        """Activate the code runner plugin."""
        if self.hook_manager:
            self.hook_manager.register(
                "tool_request",
                self._handle_tool_request,
                name="code_runner_tool_handler",
                priority=HookPriority.NORMAL,
                owner=self.name,
            )
        self._active = True
        logger.info("Code runner plugin activated.")

    def deactivate(self) -> None:
        """Deactivate the code runner plugin."""
        if self.hook_manager:
            self.hook_manager.unregister_by_owner(self.name)
        self._active = False
        logger.info("Code runner plugin deactivated.")

    def run_code(self, code: str, timeout: Optional[int] = None, stdin_input: str = "") -> Dict[str, Any]:
        """Execute Python code in a subprocess.

        Args:
            code: Python code to execute.
            timeout: Execution timeout in seconds.
            stdin_input: Optional input to provide via stdin.

        Returns:
            Dictionary with execution results.
        """
        timeout = timeout or self.default_timeout

        if not code.strip():
            return {"success": False, "error": "Empty code provided."}

        # Security check: scan for blocked imports
        security_check = self._check_security(code)
        if not security_check["safe"]:
            self._log_execution(code, False, error=security_check["reason"])
            return {"success": False, "error": security_check["reason"]}

        # Optionally inject import restrictions into the code
        wrapped_code = self._wrap_code(code)

        start_time = time.time()

        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(wrapped_code)
                temp_path = f.name

            result = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                input=stdin_input,
                cwd=tempfile.gettempdir(),
                env=self._get_safe_env(),
            )

            execution_time = time.time() - start_time

            # Clean up temp file
            try:
                os.unlink(temp_path)
            except OSError:
                pass

            stdout = result.stdout[:self.max_output_size]
            stderr = result.stderr[:self.max_output_size]

            if len(result.stdout) > self.max_output_size:
                stdout += f"\n... (output truncated at {self.max_output_size} characters)"
            if len(result.stderr) > self.max_output_size:
                stderr += f"\n... (error output truncated at {self.max_output_size} characters)"

            success = result.returncode == 0

            execution_result = {
                "success": success,
                "stdout": stdout.strip(),
                "stderr": stderr.strip(),
                "return_code": result.returncode,
                "execution_time": round(execution_time, 3),
                "timeout": timeout,
            }

            self._log_execution(
                code, success,
                output=stdout.strip() if success else "",
                error=stderr.strip() if not success else "",
                execution_time=execution_time,
            )

            return execution_result

        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            try:
                os.unlink(temp_path)
            except OSError:
                pass

            self._log_execution(code, False, error="Timeout", execution_time=execution_time)
            return {
                "success": False,
                "error": f"Code execution timed out after {timeout} seconds.",
                "execution_time": round(execution_time, 3),
            }

        except Exception as e:
            execution_time = time.time() - start_time
            self._log_execution(code, False, error=str(e), execution_time=execution_time)
            return {"success": False, "error": f"Execution error: {e}"}

    def _wrap_code(self, code: str) -> str:
        """Wrap user code with safety measures."""
        prefix = ""
        if self.blocked_imports:
            # Add import hook to block dangerous imports
            blocked = ", ".join(f"'{m}'" for m in self.blocked_imports)
            prefix += (
                "import importlib\n"
                "import sys\n"
                "_original_import = __builtins__.__import__\n"
                "_blocked = {" + blocked + "}\n"
                "def _safe_import(name, *args, **kwargs):\n"
                "    if name in _blocked:\n"
                "        raise ImportError(f\"Import of '{name}' is not allowed for security reasons.\")\n"
                "    return _original_import(name, *args, **kwargs)\n"
                "__builtins__.__import__ = _safe_import\n\n"
            )

        return prefix + code

    def _check_security(self, code: str) -> Dict[str, Any]:
        """Basic security check on code before execution."""
        # Check for obvious dangerous patterns
        dangerous_patterns = [
            (r"os\.system\s*\(", "os.system() is not allowed"),
            (r"eval\s*\(\s*input", "eval(input()) is not allowed"),
            (r"exec\s*\(\s*input", "exec(input()) is not allowed"),
            (r"__import__\s*\(\s*['\"]os['\"]", "Direct import of 'os' module is not allowed"),
        ]

        import re

        for pattern, reason in dangerous_patterns:
            if re.search(pattern, code):
                return {"safe": False, "reason": f"Security violation: {reason}"}

        return {"safe": True, "reason": ""}

    def _get_safe_env(self) -> Dict[str, str]:
        """Get a safe environment for subprocess execution."""
        env = os.environ.copy()
        # Set a restrictive PYTHONPATH
        env["PYTHONPATH"] = tempfile.gettempdir()
        # Disable writing .pyc files
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        return env

    def _log_execution(
        self,
        code: str,
        success: bool,
        output: str = "",
        error: str = "",
        execution_time: float = 0.0,
    ) -> None:
        """Log a code execution."""
        self._execution_history.append({
            "code_preview": code[:200],
            "success": success,
            "output_preview": output[:100] if output else "",
            "error_preview": error[:100] if error else "",
            "execution_time": round(execution_time, 3),
            "timestamp": time.time(),
        })

        # Keep only last 50 entries
        if len(self._execution_history) > 50:
            self._execution_history = self._execution_history[-50:]

    def get_execution_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent execution history."""
        return self._execution_history[-limit:]

    def clear_history(self) -> None:
        """Clear execution history."""
        self._execution_history.clear()

    def _handle_tool_request(self, result, *args, **kwargs):
        """Handle tool requests for code execution."""
        tool_name = kwargs.get("tool_name", "")
        if tool_name == "code_run":
            code = kwargs.get("code", "")
            if code:
                run_result = self.run_code(code)
                if run_result["success"]:
                    return run_result.get("stdout", "Code executed successfully.")
                return f"Error: {run_result.get('error', 'Unknown error')}"
        return result
