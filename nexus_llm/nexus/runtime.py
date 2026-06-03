"""Nexus-LLM Runtime Environment.

Provides the Runtime class that manages the execution environment for
Nexus-LLM, including resource tracking, environment variable management,
plugin context, and lifecycle hooks.
"""

import logging
import os
import platform
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RuntimeConfig:
    """Configuration for the runtime environment.

    Attributes:
        debug: Whether debug mode is enabled.
        log_level: Logging level string.
        max_memory_mb: Maximum memory usage in MB (0 = unlimited).
        timeout_seconds: Default operation timeout.
        working_dir: Working directory for file operations.
        env_vars: Additional environment variables.
    """

    debug: bool = False
    log_level: str = "INFO"
    max_memory_mb: int = 0
    timeout_seconds: float = 300.0
    working_dir: str = "."
    env_vars: Dict[str, str] = field(default_factory=dict)


@dataclass
class ResourceUsage:
    """Snapshot of current resource usage.

    Attributes:
        memory_rss_mb: Resident set size in MB.
        cpu_percent: CPU usage percentage.
        active_threads: Number of active threads.
        open_files: Number of open file descriptors (if available).
    """

    memory_rss_mb: float = 0.0
    cpu_percent: float = 0.0
    active_threads: int = 0
    open_files: int = 0


class Runtime:
    """Runtime environment manager for Nexus-LLM.

    The Runtime tracks the execution context, manages environment
    variables, monitors resource usage, and provides hooks for
    lifecycle events.

    Attributes:
        config: Active runtime configuration.
    """

    def __init__(self, config: Optional[RuntimeConfig] = None) -> None:
        self._config = config or RuntimeConfig()
        self._context: Dict[str, Any] = {}
        self._shutdown_hooks: List[Callable] = []
        self._startup_hooks: List[Callable] = []
        self._started = False
        self._apply_env_vars()
        logger.info("Runtime initialized (debug=%s, log_level=%s)", self._config.debug, self._config.log_level)

    @property
    def config(self) -> RuntimeConfig:
        """Active runtime configuration."""
        return self._config

    @property
    def is_started(self) -> bool:
        """Whether the runtime has been started."""
        return self._started

    def start(self) -> None:
        """Start the runtime, invoking all startup hooks."""
        if self._started:
            logger.warning("Runtime is already started")
            return
        logger.info("Starting runtime...")
        for hook in self._startup_hooks:
            try:
                hook(self)
            except Exception as exc:
                logger.error("Startup hook failed: %s", exc)
        self._started = True
        logger.info("Runtime started successfully")

    def stop(self) -> None:
        """Stop the runtime, invoking all shutdown hooks in reverse order."""
        if not self._started:
            return
        logger.info("Stopping runtime...")
        for hook in reversed(self._shutdown_hooks):
            try:
                hook(self)
            except Exception as exc:
                logger.error("Shutdown hook failed: %s", exc)
        self._started = False
        logger.info("Runtime stopped")

    def add_startup_hook(self, hook: Callable[["Runtime"], None]) -> None:
        """Register a callable to be invoked on startup.

        Args:
            hook: Callable that accepts the Runtime instance.
        """
        self._startup_hooks.append(hook)

    def add_shutdown_hook(self, hook: Callable[["Runtime"], None]) -> None:
        """Register a callable to be invoked on shutdown.

        Args:
            hook: Callable that accepts the Runtime instance.
        """
        self._shutdown_hooks.append(hook)

    def set_context(self, key: str, value: Any) -> None:
        """Set a context variable.

        Args:
            key: Context key.
            value: Context value.
        """
        self._context[key] = value

    def get_context(self, key: str, default: Any = None) -> Any:
        """Retrieve a context variable.

        Args:
            key: Context key.
            default: Default value if key is not found.

        Returns:
            The context value or default.
        """
        return self._context.get(key, default)

    def resource_usage(self) -> ResourceUsage:
        """Return a snapshot of current resource usage.

        Returns:
            A ResourceUsage dataclass with current metrics.
        """
        usage = ResourceUsage(active_threads=0)
        try:
            import threading
            usage.active_threads = threading.active_count()
        except Exception:
            pass
        try:
            import resource
            usage.memory_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        except Exception:
            pass
        return usage

    def platform_info(self) -> Dict[str, str]:
        """Return platform and Python information.

        Returns:
            Dictionary with system, Python version, and platform details.
        """
        return {
            "system": platform.system(),
            "machine": platform.machine(),
            "python_version": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
        }

    def health_check(self) -> Dict[str, Any]:
        """Return health status of the runtime.

        Returns:
            Dictionary with runtime status and resource usage.
        """
        usage = self.resource_usage()
        return {
            "status": "healthy" if self._started else "not_started",
            "started": self._started,
            "debug": self._config.debug,
            "memory_rss_mb": usage.memory_rss_mb,
            "active_threads": usage.active_threads,
        }

    def _apply_env_vars(self) -> None:
        """Apply configured environment variables."""
        for key, value in self._config.env_vars.items():
            os.environ[key] = value
            logger.debug("Set env var: %s", key)
