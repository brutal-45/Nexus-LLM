"""Timing utilities: Timer context manager, ElapsedTime, ETA calculator."""

import time
import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


class Timer:
    """Context manager and standalone timer for measuring elapsed time.

    Usage as context manager:
        with Timer("operation") as t:
            do_something()
        print(f"Took {t.elapsed:.2f}s")

    Usage as standalone:
        timer = Timer()
        timer.start()
        do_something()
        timer.stop()
        print(f"Took {timer.elapsed:.2f}s")
    """

    def __init__(self, name: str = "Timer", log_on_exit: bool = True):
        """Initialize the timer.

        Args:
            name: Timer name for logging.
            log_on_exit: Whether to log elapsed time on context exit.
        """
        self.name = name
        self.log_on_exit = log_on_exit
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self._laps: List[Dict[str, Any]] = []

    def start(self) -> "Timer":
        """Start the timer."""
        self._start_time = time.perf_counter()
        self._end_time = None
        return self

    def stop(self) -> "Timer":
        """Stop the timer."""
        self._end_time = time.perf_counter()
        return self

    def lap(self, label: Optional[str] = None) -> float:
        """Record a lap time.

        Args:
            label: Optional label for this lap.

        Returns:
            Time elapsed since start or last lap.
        """
        if self._start_time is None:
            raise RuntimeError("Timer not started")

        now = time.perf_counter()
        last_lap = self._laps[-1]["time"] if self._laps else self._start_time
        lap_time = now - last_lap

        self._laps.append({
            "label": label or f"lap_{len(self._laps)}",
            "time": now,
            "duration": lap_time,
        })

        return lap_time

    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self._start_time is None:
            return 0.0
        end = self._end_time or time.perf_counter()
        return end - self._start_time

    @property
    def is_running(self) -> bool:
        """Check if the timer is currently running."""
        return self._start_time is not None and self._end_time is None

    def get_laps(self) -> List[Dict[str, Any]]:
        """Get all recorded lap times."""
        return list(self._laps)

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of timing information."""
        return {
            "name": self.name,
            "elapsed": self.elapsed,
            "is_running": self.is_running,
            "num_laps": len(self._laps),
            "laps": self._laps,
        }

    def __enter__(self) -> "Timer":
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
        if self.log_on_exit:
            logger.info(f"{self.name}: {self.elapsed:.4f}s")

    def __repr__(self) -> str:
        return f"Timer(name={self.name!r}, elapsed={self.elapsed:.4f}s)"


class ElapsedTime:
    """Utility class for formatting and working with elapsed time values."""

    def __init__(self, seconds: float):
        self.total_seconds = seconds

    @property
    def microseconds(self) -> float:
        return self.total_seconds * 1_000_000

    @property
    def milliseconds(self) -> float:
        return self.total_seconds * 1_000

    @property
    def minutes(self) -> float:
        return self.total_seconds / 60

    @property
    def hours(self) -> float:
        return self.total_seconds / 3600

    @property
    def days(self) -> float:
        return self.total_seconds / 86400

    def format(self, precision: int = 2) -> str:
        """Format the elapsed time in a human-readable way.

        Args:
            precision: Decimal places for seconds.

        Returns:
            Formatted time string.
        """
        if self.total_seconds < 0.001:
            return f"{self.microseconds:.0f}μs"
        elif self.total_seconds < 1.0:
            return f"{self.milliseconds:.{precision}f}ms"
        elif self.total_seconds < 60:
            return f"{self.total_seconds:.{precision}f}s"
        elif self.total_seconds < 3600:
            minutes = int(self.total_seconds // 60)
            seconds = self.total_seconds % 60
            return f"{minutes}m {seconds:.{precision}f}s"
        elif self.total_seconds < 86400:
            hours = int(self.total_seconds // 3600)
            minutes = int((self.total_seconds % 3600) // 60)
            seconds = self.total_seconds % 60
            return f"{hours}h {minutes}m {seconds:.{precision}f}s"
        else:
            days = int(self.total_seconds // 86400)
            hours = int((self.total_seconds % 86400) // 3600)
            return f"{days}d {hours}h"

    def __repr__(self) -> str:
        return f"ElapsedTime({self.format()})"

    def __str__(self) -> str:
        return self.format()

    @classmethod
    def from_milliseconds(cls, ms: float) -> "ElapsedTime":
        return cls(ms / 1_000)

    @classmethod
    def from_minutes(cls, minutes: float) -> "ElapsedTime":
        return cls(minutes * 60)

    @classmethod
    def from_hours(cls, hours: float) -> "ElapsedTime":
        return cls(hours * 3600)


class ETACalculator:
    """Estimates time remaining for iterative processes."""

    def __init__(
        self,
        total_steps: int,
        smoothing: float = 0.3,
    ):
        """Initialize the ETA calculator.

        Args:
            total_steps: Total number of steps to complete.
            smoothing: Exponential smoothing factor (0-1). Higher values give more weight to recent steps.
        """
        self.total_steps = max(total_steps, 1)
        self.smoothing = smoothing
        self._start_time = time.time()
        self._step_times: List[float] = []
        self._smoothed_step_time: Optional[float] = None
        self._completed_steps = 0

    def update(self, steps: int = 1):
        """Record completed steps.

        Args:
            steps: Number of steps completed in this update.
        """
        now = time.time()
        self._step_times.append(now)
        self._completed_steps += steps

    def step(self):
        """Record a single completed step."""
        self.update(1)

    @property
    def progress(self) -> float:
        """Get progress as a fraction (0.0 to 1.0)."""
        return min(self._completed_steps / self.total_steps, 1.0)

    @property
    def elapsed_seconds(self) -> float:
        """Get total elapsed time in seconds."""
        return time.time() - self._start_time

    @property
    def remaining_steps(self) -> int:
        """Get number of remaining steps."""
        return max(0, self.total_steps - self._completed_steps)

    @property
    def steps_per_second(self) -> float:
        """Get current processing rate in steps per second."""
        if self._completed_steps <= 0 or self.elapsed_seconds <= 0:
            return 0.0
        return self._completed_steps / self.elapsed_seconds

    @property
    def eta_seconds(self) -> float:
        """Get estimated time remaining in seconds."""
        if self._completed_steps <= 0:
            return 0.0

        rate = self.steps_per_second
        if rate <= 0:
            return float("inf")

        return self.remaining_steps / rate

    def get_eta(self) -> ElapsedTime:
        """Get estimated time remaining as an ElapsedTime object."""
        return ElapsedTime(self.eta_seconds)

    def format_progress(self) -> str:
        """Format a progress string with ETA.

        Returns:
            Formatted string like "150/1000 (15.0%) | ETA: 5m 30.00s"
        """
        eta = self.get_eta()
        pct = self.progress * 100
        rate = self.steps_per_second
        return (
            f"{self._completed_steps}/{self.total_steps} ({pct:.1f}%) | "
            f"Rate: {rate:.1f} steps/s | ETA: {eta.format()}"
        )

    def reset(self):
        """Reset the ETA calculator."""
        self._start_time = time.time()
        self._step_times = []
        self._smoothed_step_time = None
        self._completed_steps = 0
