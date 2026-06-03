"""Nexus-LLM Signal Handling Module.

Provides graceful shutdown handling for the Nexus-LLM application.
Manages OS signals (SIGINT, SIGTERM, SIGUSR1) and ensures proper
cleanup of resources when the application is interrupted.
"""

import logging
import os
import signal
import sys
import threading
from typing import Any, Callable, Dict, List, Optional, Set

from nexus_llm.events import Event, EventBus, get_event_bus

logger = logging.getLogger(__name__)


class SignalHandler:
    """Handles OS signals for graceful application shutdown.

    Registers handlers for SIGINT, SIGTERM, and optionally SIGUSR1/SIGUSR2.
    Supports multiple shutdown callbacks and ensures orderly cleanup.

    Example:
        >>> handler = SignalHandler()
        >>> handler.register_shutdown_callback(cleanup_resources)
        >>> handler.install()
        >>> # Application runs...
        >>> # When SIGINT/SIGTERM is received, callbacks are called in order
    """

    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        graceful_timeout: float = 30.0,
        force_exit_code: int = 137,
    ) -> None:
        """Initialize the signal handler.

        Args:
            event_bus: Event bus for publishing shutdown events.
            graceful_timeout: Seconds to wait for graceful shutdown before forcing.
            force_exit_code: Exit code when forced shutdown occurs.
        """
        self._event_bus = event_bus or get_event_bus()
        self._graceful_timeout = graceful_timeout
        self._force_exit_code = force_exit_code
        self._shutdown_callbacks: List[Callable[[], Any]] = []
        self._pre_shutdown_callbacks: List[Callable[[], Any]] = []
        self._installed = False
        self._shutting_down = False
        self._shutdown_event = threading.Event()
        self._original_handlers: Dict[int, Any] = {}
        self._lock = threading.Lock()

    def register_shutdown_callback(self, callback: Callable[[], Any], priority: int = 0) -> None:
        """Register a callback to be called during shutdown.

        Callbacks are called in reverse order of registration (LIFO).

        Args:
            callback: Function to call during shutdown.
            priority: Priority level (higher = called first).
        """
        with self._lock:
            self._shutdown_callbacks.append((priority, callback))
            self._shutdown_callbacks.sort(key=lambda x: x[0], reverse=True)

    def register_pre_shutdown_callback(self, callback: Callable[[], Any]) -> None:
        """Register a callback called before the main shutdown sequence.

        Pre-shutdown callbacks are called before the main shutdown
        callbacks, allowing for early cleanup operations.

        Args:
            callback: Function to call before shutdown.
        """
        self._pre_shutdown_callbacks.append(callback)

    def install(self) -> None:
        """Install signal handlers.

        Replaces the default handlers for SIGINT and SIGTERM with
        graceful shutdown handlers.
        """
        if self._installed:
            logger.warning("Signal handlers already installed.")
            return

        signals_to_handle = [signal.SIGINT, signal.SIGTERM]

        # Add SIGUSR1/SIGUSR2 on Unix
        if hasattr(signal, "SIGUSR1"):
            signals_to_handle.append(signal.SIGUSR1)
        if hasattr(signal, "SIGUSR2"):
            signals_to_handle.append(signal.SIGUSR2)

        for sig in signals_to_handle:
            self._original_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, self._handle_signal)

        self._installed = True
        logger.info("Signal handlers installed for: %s", [s.name for s in signals_to_handle])

    def uninstall(self) -> None:
        """Restore original signal handlers."""
        if not self._installed:
            return

        for sig, handler in self._original_handlers.items():
            signal.signal(sig, handler)

        self._original_handlers.clear()
        self._installed = False
        logger.info("Signal handlers uninstalled.")

    def _handle_signal(self, signum: int, frame: Any) -> None:
        """Handle a received signal.

        Args:
            signum: The signal number received.
            frame: The current stack frame.
        """
        sig_name = signal.Signals(signum).name if signum in signal.Signals.__members__.values() else str(signum)

        if self._shutting_down:
            # Second signal - force exit
            logger.warning("Received %s during shutdown - forcing exit.", sig_name)
            sys.exit(self._force_exit_code)

        with self._lock:
            if self._shutting_down:
                return
            self._shutting_down = True

        logger.info("Received %s - initiating graceful shutdown...", sig_name)

        # Publish shutdown event
        self._event_bus.publish(Event(
            event_type="system.shutdown",
            data={"signal": sig_name, "signal_number": signum},
            source="SignalHandler",
        ))

        # Run pre-shutdown callbacks
        self._run_pre_shutdown_callbacks()

        # Run shutdown callbacks
        self._run_shutdown_callbacks()

        # Signal completion
        self._shutdown_event.set()

        # Restore original handler and re-raise
        if signum in self._original_handlers:
            signal.signal(signum, self._original_handlers[signum])

        # Exit cleanly
        sys.exit(0)

    def _run_pre_shutdown_callbacks(self) -> None:
        """Run pre-shutdown callbacks."""
        for callback in self._pre_shutdown_callbacks:
            try:
                callback()
            except Exception as exc:
                logger.error("Error in pre-shutdown callback %s: %s", callback.__name__, exc)

    def _run_shutdown_callbacks(self) -> None:
        """Run shutdown callbacks in priority order."""
        for priority, callback in self._shutdown_callbacks:
            try:
                logger.info("Running shutdown callback: %s (priority=%d)", callback.__name__, priority)
                callback()
            except Exception as exc:
                logger.error("Error in shutdown callback %s: %s", callback.__name__, exc)

    @property
    def is_shutting_down(self) -> bool:
        """Check if the application is shutting down."""
        return self._shutting_down

    def wait_for_shutdown(self, timeout: Optional[float] = None) -> bool:
        """Wait for the shutdown signal to be received.

        Args:
            timeout: Maximum time to wait in seconds. None waits indefinitely.

        Returns:
            True if shutdown was signaled, False if timeout occurred.
        """
        return self._shutdown_event.wait(timeout=timeout or self._graceful_timeout)

    def trigger_shutdown(self, reason: str = "manual") -> None:
        """Manually trigger a graceful shutdown.

        Args:
            reason: Reason for the shutdown.
        """
        if self._shutting_down:
            return

        with self._lock:
            if self._shutting_down:
                return
            self._shutting_down = True

        logger.info("Manual shutdown triggered: %s", reason)

        self._event_bus.publish(Event(
            event_type="system.shutdown",
            data={"reason": reason},
            source="SignalHandler",
        ))

        self._run_pre_shutdown_callbacks()
        self._run_shutdown_callbacks()
        self._shutdown_event.set()

    def __repr__(self) -> str:
        return (
            f"SignalHandler(installed={self._installed}, "
            f"shutting_down={self._shutting_down}, "
            f"callbacks={len(self._shutdown_callbacks)})"
        )


class GracefulContextManager:
    """Context manager that ensures graceful shutdown on signal.

    Use as a context manager to wrap the main application logic,
    ensuring cleanup happens even on interrupt.

    Example:
        >>> with GracefulContextManager() as ctx:
        >>>     run_application()
    """

    def __init__(self, timeout: float = 30.0) -> None:
        """Initialize the context manager.

        Args:
            timeout: Graceful shutdown timeout in seconds.
        """
        self._handler = SignalHandler(graceful_timeout=timeout)

    def __enter__(self) -> SignalHandler:
        self._handler.install()
        return self._handler

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._handler.uninstall()

    @property
    def handler(self) -> SignalHandler:
        """Access the underlying SignalHandler."""
        return self._handler
