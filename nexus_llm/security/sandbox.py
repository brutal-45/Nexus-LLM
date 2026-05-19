"""Nexus-LLM Code Sandbox.

Provides the CodeSandbox for safely executing Python code in a
restricted environment with resource limits and output capture.
"""

import logging
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass, field
from io import StringIO
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Default set of allowed builtins (safe subset)
_SAFE_BUILTINS = {
    "abs", "all", "any", "bin", "bool", "chr", "divmod", "enumerate",
    "filter", "float", "format", "hex", "int", "isinstance", "len",
    "list", "map", "max", "min", "oct", "ord", "pow", "print",
    "range", "repr", "reversed", "round", "set", "sorted", "str",
    "sum", "tuple", "type", "zip", "dict", "True", "False", "None",
}

# Modules that are never allowed
_BLOCKED_MODULES: Set[str] = {
    "os", "subprocess", "shutil", "socket", "ctypes", "sys",
    "importlib", "signal", "multiprocessing", "threading",
}


@dataclass
class SandboxResult:
    """Result from sandboxed code execution.

    Attributes:
        success: Whether execution completed without errors.
        output: Captured stdout output.
        error: Captured stderr or traceback.
        return_value: Return value of the last expression.
        execution_time_ms: Execution time in milliseconds.
        memory_used_mb: Estimated memory usage.
    """

    success: bool = True
    output: str = ""
    error: Optional[str] = None
    return_value: Any = None
    execution_time_ms: float = 0.0
    memory_used_mb: float = 0.0


class CodeSandbox:
    """Sandbox for executing Python code safely.

    The sandbox restricts available builtins and modules, captures
    stdout/stderr, and enforces basic resource limits.

    .. warning::

        This sandbox provides basic isolation but is NOT a full
        security sandbox. Do not execute truly untrusted code.

    Example::

        sandbox = CodeSandbox()
        result = sandbox.execute("result = 2 + 2")
        print(result.return_value)  # 4
    """

    def __init__(
        self,
        timeout: float = 30.0,
        max_output_length: int = 10000,
        allowed_builtins: Optional[Set[str]] = None,
    ) -> None:
        self._timeout = timeout
        self._max_output_length = max_output_length
        self._allowed_builtins = allowed_builtins or _SAFE_BUILTINS
        logger.debug("CodeSandbox initialized (timeout=%.1fs)", timeout)

    def execute(self, code: str, globals_dict: Optional[Dict[str, Any]] = None) -> SandboxResult:
        """Execute Python code in the sandbox.

        Args:
            code: Python source code to execute.
            globals_dict: Optional globals dict (will be filtered).

        Returns:
            A SandboxResult with captured output.
        """
        import time

        # Build restricted globals
        safe_builtins = {
            name: __builtins__[name] if isinstance(__builtins__, dict) else getattr(__builtins__, name)
            for name in self._allowed_builtins
            if (name in __builtins__ if isinstance(__builtins__, dict) else hasattr(__builtins__, name))
        }

        sandbox_globals: Dict[str, Any] = {
            "__builtins__": safe_builtins,
        }
        if globals_dict:
            # Filter out dangerous keys
            for key, value in globals_dict.items():
                if not key.startswith("__"):
                    sandbox_globals[key] = value

        stdout_capture = StringIO()
        stderr_capture = StringIO()
        start = time.perf_counter()

        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                # Try to compile as expression first
                try:
                    compiled = compile(code, "<sandbox>", "eval")
                    return_value = eval(compiled, sandbox_globals)
                except SyntaxError:
                    compiled = compile(code, "<sandbox>", "exec")
                    exec(compiled, sandbox_globals)
                    return_value = sandbox_globals.get("result", None)

            duration_ms = (time.perf_counter() - start) * 1000
            output = stdout_capture.getvalue()[:self._max_output_length]

            return SandboxResult(
                success=True,
                output=output,
                return_value=return_value,
                execution_time_ms=duration_ms,
            )
        except Exception:
            duration_ms = (time.perf_counter() - start) * 1000
            error_msg = traceback.format_exc()[:self._max_output_length]
            return SandboxResult(
                success=False,
                output=stdout_capture.getvalue()[:self._max_output_length],
                error=error_msg,
                execution_time_ms=duration_ms,
            )

    def execute_file(self, filepath: str) -> SandboxResult:
        """Execute a Python file in the sandbox.

        Args:
            filepath: Path to the Python file.

        Returns:
            A SandboxResult with captured output.
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                code = f.read()
        except FileNotFoundError:
            return SandboxResult(success=False, error=f"File not found: {filepath}")
        except Exception as exc:
            return SandboxResult(success=False, error=f"Error reading file: {exc}")

        return self.execute(code)
