"""Worker process with heartbeat monitoring."""

from __future__ import annotations

import enum
import threading
import time
import uuid
from typing import Any, Callable, Optional


class WorkerStatus(enum.Enum):
    """Possible states of a worker."""

    IDLE = "idle"
    BUSY = "busy"
    STOPPED = "stopped"

    def __str__(self) -> str:  # noqa: D105
        return self.value


class Worker:
    """A single worker that processes tasks with heartbeat monitoring.

    Each worker runs in its own thread and can execute arbitrary
    callables.  A heartbeat thread periodically records a timestamp so
    that external monitors can detect stalled workers.

    Attributes:
        worker_id: Unique identifier for this worker.
        status: Current worker status.
    """

    def __init__(
        self,
        worker_id: Optional[str] = None,
        heartbeat_interval: float = 5.0,
    ) -> None:
        self.worker_id: str = worker_id or uuid.uuid4().hex[:8]
        self._status = WorkerStatus.IDLE
        self._heartbeat_interval = heartbeat_interval
        self._last_heartbeat: float = time.monotonic()
        self._task_count: int = 0
        self._lock = threading.RLock()

        # Heartbeat thread
        self._hb_thread: Optional[threading.Thread] = None
        self._hb_stop = threading.Event()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def status(self) -> WorkerStatus:
        """Return the current worker status."""
        return self._status

    @property
    def last_heartbeat(self) -> float:
        """Return the monotonic timestamp of the last heartbeat."""
        return self._last_heartbeat

    @property
    def task_count(self) -> int:
        """Return the total number of tasks this worker has completed."""
        return self._task_count

    @property
    def is_alive(self) -> bool:
        """Return ``True`` if the worker is not stopped."""
        return self._status != WorkerStatus.STOPPED

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the heartbeat monitoring thread.

        Does **not** start a processing loop; task execution is driven
        externally (e.g. by :class:`WorkerPool`).
        """
        with self._lock:
            if self._status == WorkerStatus.STOPPED:
                self._status = WorkerStatus.IDLE

        self._hb_stop.clear()
        self._hb_thread = threading.Thread(
            target=self._heartbeat_loop,
            name=f"hb-{self.worker_id}",
            daemon=True,
        )
        self._hb_thread.start()

    def stop(self) -> None:
        """Stop the worker and the heartbeat thread."""
        self._hb_stop.set()
        with self._lock:
            self._status = WorkerStatus.STOPPED

        if self._hb_thread is not None and self._hb_thread.is_alive():
            self._hb_thread.join(timeout=self._heartbeat_interval * 2)
            self._hb_thread = None

    # ------------------------------------------------------------------
    # Task processing
    # ------------------------------------------------------------------

    def process(self, task: Callable, *args: Any, **kwargs: Any) -> Any:
        """Execute a task and return its result.

        Sets the worker status to ``BUSY`` for the duration of the
        task, then reverts to ``IDLE``.  The heartbeat is refreshed
        before and after execution.

        Args:
            task: Callable to execute.
            *args, **kwargs: Arguments forwarded to *task*.

        Returns:
            The return value of *task*.

        Raises:
            RuntimeError: If the worker is stopped.
        """
        with self._lock:
            if self._status == WorkerStatus.STOPPED:
                raise RuntimeError(
                    f"Worker '{self.worker_id}' is stopped and cannot process tasks."
                )
            self._status = WorkerStatus.BUSY

        self._touch_heartbeat()

        try:
            result = task(*args, **kwargs)
        except Exception:
            raise
        finally:
            with self._lock:
                self._status = WorkerStatus.IDLE
                self._task_count += 1
            self._touch_heartbeat()

        return result

    # ------------------------------------------------------------------
    # Heartbeat
    # ------------------------------------------------------------------

    def _touch_heartbeat(self) -> None:
        """Record a heartbeat timestamp."""
        self._last_heartbeat = time.monotonic()

    def _heartbeat_loop(self) -> None:
        """Background thread that periodically touches the heartbeat."""
        while not self._hb_stop.wait(self._heartbeat_interval):
            self._touch_heartbeat()

    def heartbeat_age(self) -> float:
        """Return seconds since the last heartbeat."""
        return time.monotonic() - self._last_heartbeat

    def is_healthy(self, max_age: Optional[float] = None) -> bool:
        """Check whether the worker's heartbeat is recent.

        Args:
            max_age: Maximum acceptable age in seconds; defaults to
                ``2 * heartbeat_interval``.

        Returns:
            ``True`` if the heartbeat is recent enough.
        """
        if self._status == WorkerStatus.STOPPED:
            return False
        threshold = max_age or (self._heartbeat_interval * 2)
        return self.heartbeat_age() < threshold

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"<Worker id={self.worker_id!r} "
            f"status={self._status.value} "
            f"tasks={self._task_count}>"
        )
