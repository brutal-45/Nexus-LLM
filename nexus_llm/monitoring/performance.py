"""Performance monitoring for Nexus-LLM."""

import functools
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class TimingRecord:
    """A single timing measurement."""
    operation: str
    start_time: float
    end_time: float
    elapsed: float


@dataclass
class OperationStats:
    """Aggregated statistics for an operation."""
    operation: str
    count: int
    total_time: float
    min_time: float
    max_time: float
    avg_time: float
    last_time: float


class PerformanceMonitor:
    """Monitors operation performance with timers and profiling.

    Provides manual start/stop timing, decorator-based profiling, and
    aggregated per-operation statistics.
    """

    def __init__(self, max_records_per_op: int = 5000) -> None:
        self._timers: Dict[int, Tuple[str, float]] = {}
        self._next_id: int = 0
        self._records: Dict[str, List[TimingRecord]] = defaultdict(list)
        self._max_records = max_records_per_op
        self._lock = threading.Lock()

    def start_timer(self, operation: str) -> int:
        """Start a timer for an operation.

        Args:
            operation: Name of the operation being timed.

        Returns:
            Timer ID that must be passed to ``stop_timer``.
        """
        with self._lock:
            timer_id = self._next_id
            self._next_id += 1
            self._timers[timer_id] = (operation, time.monotonic())
        return timer_id

    def stop_timer(self, timer_id: int) -> float:
        """Stop a previously started timer.

        Args:
            timer_id: ID returned by ``start_timer``.

        Returns:
            Elapsed time in seconds.

        Raises:
            KeyError: If the timer_id is unknown or already stopped.
        """
        end_time = time.monotonic()
        with self._lock:
            entry = self._timers.pop(timer_id, None)
            if entry is None:
                raise KeyError(f"Timer ID {timer_id} not found or already stopped")
            operation, start_time = entry

        elapsed = end_time - start_time
        record = TimingRecord(
            operation=operation,
            start_time=start_time,
            end_time=end_time,
            elapsed=elapsed,
        )

        with self._lock:
            store = self._records[operation]
            store.append(record)
            if len(store) > self._max_records:
                excess = len(store) - self._max_records
                del store[:excess]

        return elapsed

    def get_operation_stats(self, operation: str) -> Optional[OperationStats]:
        """Get aggregated statistics for an operation.

        Args:
            operation: Operation name.

        Returns:
            ``OperationStats`` if data exists, else ``None``.
        """
        with self._lock:
            records = list(self._records.get(operation, []))

        if not records:
            return None

        elapsed_values = [r.elapsed for r in records]
        return OperationStats(
            operation=operation,
            count=len(elapsed_values),
            total_time=sum(elapsed_values),
            min_time=min(elapsed_values),
            max_time=max(elapsed_values),
            avg_time=sum(elapsed_values) / len(elapsed_values),
            last_time=elapsed_values[-1],
        )

    def list_operations(self) -> List[str]:
        """Return names of all tracked operations."""
        with self._lock:
            return list(self._records.keys())

    def profile(self, func: Optional[F] = None, *, operation: Optional[str] = None) -> Any:
        """Decorator to automatically time a function.

        Can be used with or without arguments::

            @monitor.profile
            def my_func(): ...

            @monitor.profile(operation="custom_name")
            def my_func(): ...

        Args:
            func: The function to wrap (when used without parentheses).
            operation: Custom operation name (defaults to function qualname).

        Returns:
            Decorated function or decorator.
        """
        def decorator(fn: F) -> F:
            op_name = operation or fn.__qualname__

            @functools.wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                timer_id = self.start_timer(op_name)
                try:
                    result = fn(*args, **kwargs)
                finally:
                    self.stop_timer(timer_id)
                return result

            return wrapper  # type: ignore[return-value]

        if func is not None:
            return decorator(func)
        return decorator

    def clear(self, operation: Optional[str] = None) -> None:
        """Clear performance records.

        Args:
            operation: If provided, only clear that operation's records.
                       Otherwise clear all.
        """
        with self._lock:
            if operation is not None:
                self._records.pop(operation, None)
            else:
                self._records.clear()
                self._timers.clear()
