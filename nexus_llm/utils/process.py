"""Process management: subprocess runner, process pool, signal handling."""

import os
import sys
import signal
import subprocess
import logging
import time
from typing import Optional, Dict, Any, List, Tuple, Callable

logger = logging.getLogger(__name__)


class SubprocessResult:
    """Result of a subprocess execution."""

    def __init__(
        self,
        returncode: int,
        stdout: str,
        stderr: str,
        duration: float,
        command: str,
    ):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.duration = duration
        self.command = command

    @property
    def success(self) -> bool:
        return self.returncode == 0

    def __repr__(self) -> str:
        status = "success" if self.success else f"failed(rc={self.returncode})"
        return f"SubprocessResult({status}, duration={self.duration:.2f}s)"


def run_subprocess(
    command: List[str],
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    timeout: Optional[float] = None,
    capture_output: bool = True,
    shell: bool = False,
    check: bool = True,
    input_text: Optional[str] = None,
) -> SubprocessResult:
    """Run a subprocess command with proper error handling and timeout support.

    Args:
        command: Command and arguments as a list.
        cwd: Working directory.
        env: Environment variables.
        timeout: Timeout in seconds.
        capture_output: Whether to capture stdout/stderr.
        shell: Whether to use shell execution.
        check: Whether to raise on non-zero return code.
        input_text: Optional input to send to stdin.

    Returns:
        SubprocessResult with execution details.

    Raises:
        subprocess.TimeoutExpired: If the command times out.
        subprocess.CalledProcessError: If check=True and the command fails.
    """
    merged_env = None
    if env:
        merged_env = os.environ.copy()
        merged_env.update(env)

    command_str = " ".join(command) if isinstance(command, list) else str(command)
    logger.debug(f"Running subprocess: {command_str}")

    start_time = time.time()

    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            env=merged_env,
            timeout=timeout,
            capture_output=capture_output,
            text=True,
            shell=shell,
            input=input_text,
        )
        duration = time.time() - start_time

        if check and result.returncode != 0:
            logger.error(
                f"Subprocess failed (rc={result.returncode}): {command_str}\n"
                f"stderr: {result.stderr[:500] if result.stderr else ''}"
            )
            raise subprocess.CalledProcessError(
                result.returncode, command, result.stdout, result.stderr
            )

        return SubprocessResult(
            returncode=result.returncode,
            stdout=result.stdout or "",
            stderr=result.stderr or "",
            duration=duration,
            command=command_str,
        )

    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        logger.error(f"Subprocess timed out after {duration:.1f}s: {command_str}")
        raise
    except FileNotFoundError:
        logger.error(f"Command not found: {command[0]}")
        raise


class ProcessPool:
    """Simple process pool using subprocess for parallel task execution."""

    def __init__(
        self,
        max_workers: Optional[int] = None,
    ):
        self.max_workers = max_workers or os.cpu_count() or 4
        self._processes: List[subprocess.Popen] = []
        self._results: List[Optional[SubprocessResult]] = []

    def submit(
        self,
        command: List[str],
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> int:
        """Submit a command to the process pool.

        Args:
            command: Command and arguments.
            cwd: Working directory.
            env: Environment variables.

        Returns:
            Process index for tracking.
        """
        merged_env = None
        if env:
            merged_env = os.environ.copy()
            merged_env.update(env)

        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=merged_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        idx = len(self._processes)
        self._processes.append(process)
        self._results.append(None)
        return idx

    def wait_all(self, timeout: Optional[float] = None) -> List[SubprocessResult]:
        """Wait for all submitted processes to complete.

        Args:
            timeout: Maximum time to wait per process.

        Returns:
            List of SubprocessResult objects.
        """
        results = []
        for idx, process in enumerate(self._processes):
            try:
                stdout, stderr = process.communicate(timeout=timeout)
                result = SubprocessResult(
                    returncode=process.returncode,
                    stdout=stdout or "",
                    stderr=stderr or "",
                    duration=0.0,
                    command="",
                )
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                result = SubprocessResult(
                    returncode=-1,
                    stdout=stdout or "",
                    stderr=stderr or "",
                    duration=0.0,
                    command="",
                )

            self._results[idx] = result
            results.append(result)

        return results

    def wait_one(self, idx: int, timeout: Optional[float] = None) -> SubprocessResult:
        """Wait for a specific process to complete.

        Args:
            idx: Process index returned by submit().
            timeout: Maximum time to wait.

        Returns:
            SubprocessResult for the process.
        """
        if idx < 0 or idx >= len(self._processes):
            raise IndexError(f"Invalid process index: {idx}")

        process = self._processes[idx]
        start = time.time()

        try:
            stdout, stderr = process.communicate(timeout=timeout)
            duration = time.time() - start
            result = SubprocessResult(
                returncode=process.returncode,
                stdout=stdout or "",
                stderr=stderr or "",
                duration=duration,
                command="",
            )
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            duration = time.time() - start
            result = SubprocessResult(
                returncode=-1,
                stdout=stdout or "",
                stderr=stderr or "",
                duration=duration,
                command="",
            )

        self._results[idx] = result
        return result

    def kill_all(self):
        """Kill all running processes."""
        for process in self._processes:
            if process.poll() is None:
                process.kill()
        logger.info("All processes killed.")

    @property
    def active_count(self) -> int:
        """Number of still-running processes."""
        return sum(1 for p in self._processes if p.poll() is None)


class SignalHandler:
    """Graceful signal handler for process management."""

    def __init__(
        self,
        signals: Optional[List[int]] = None,
        callback: Optional[Callable] = None,
    ):
        self.signals = signals or [signal.SIGINT, signal.SIGTERM]
        self.callback = callback
        self._original_handlers: Dict[int, Any] = {}
        self._received_signals: List[int] = []
        self._shutdown_requested = False

    def install(self):
        """Install signal handlers."""
        for sig in self.signals:
            self._original_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, self._handle_signal)
        logger.debug(f"Signal handlers installed for {self.signals}")

    def _handle_signal(self, signum, frame):
        """Handle a received signal."""
        self._received_signals.append(signum)
        self._shutdown_requested = True
        logger.info(f"Received signal {signum}. Requesting graceful shutdown.")

        if self.callback:
            self.callback(signum)

        if len(self._received_signals) >= 3:
            logger.warning("Multiple signals received. Forcing exit.")
            sys.exit(1)

    @property
    def should_shutdown(self) -> bool:
        """Check if shutdown has been requested."""
        return self._shutdown_requested

    def restore(self):
        """Restore original signal handlers."""
        for sig, handler in self._original_handlers.items():
            signal.signal(sig, handler)
        self._original_handlers.clear()
