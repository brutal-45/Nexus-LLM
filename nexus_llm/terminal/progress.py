"""
Nexus-LLM Progress Indicators Module

Provides progress bars, spinners, percentage displays, and ETA tracking
for long-running operations in the terminal.
"""

from __future__ import annotations

import shutil
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator


@dataclass
class ProgressState:
    """Current state of a progress tracker."""
    current: float = 0.0
    total: float = 100.0
    description: str = ""
    start_time: float = field(default_factory=time.time)
    unit: str = ""
    completed: bool = False

    @property
    def percentage(self) -> float:
        """Get progress as a percentage (0-100)."""
        if self.total <= 0:
            return 0.0
        return min(100.0, (self.current / self.total) * 100)

    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        return time.time() - self.start_time

    @property
    def rate(self) -> float:
        """Get the processing rate (units per second)."""
        if self.elapsed <= 0:
            return 0.0
        return self.current / self.elapsed

    @property
    def eta(self) -> float | None:
        """Estimate time remaining in seconds."""
        rate = self.rate
        if rate <= 0 or self.current >= self.total:
            return None
        return (self.total - self.current) / rate


class ProgressBar:
    """A terminal progress bar with percentage, ETA, and rate display.

    Renders an animated progress bar with customizable appearance,
    supporting both determinate (known total) and indeterminate modes.
    """

    def __init__(
        self,
        total: float = 100.0,
        description: str = "",
        unit: str = "",
        bar_width: int = 30,
        fill_char: str = "█",
        empty_char: str = "░",
        show_percentage: bool = True,
        show_eta: bool = True,
        show_rate: bool = True,
        show_count: bool = True,
        color: str = "green",
    ) -> None:
        self._state = ProgressState(total=total, description=description, unit=unit)
        self._bar_width = bar_width
        self._fill_char = fill_char
        self._empty_char = empty_char
        self._show_percentage = show_percentage
        self._show_eta = show_eta
        self._show_rate = show_rate
        self._show_count = show_count
        self._color = color
        self._last_render_len = 0

    @property
    def state(self) -> ProgressState:
        """Get the current progress state."""
        return self._state

    def update(self, current: float, description: str | None = None) -> None:
        """Update the progress value.

        Args:
            current: New current value.
            description: Optional new description.
        """
        self._state.current = min(current, self._state.total)
        if description is not None:
            self._state.description = description
        if self._state.current >= self._state.total:
            self._state.completed = True
        self._render()

    def increment(self, amount: float = 1.0) -> None:
        """Increment the progress by a given amount.

        Args:
            amount: Amount to add to the current value.
        """
        self.update(self._state.current + amount)

    def complete(self) -> None:
        """Mark the progress as complete."""
        self._state.current = self._state.total
        self._state.completed = True
        self._render()
        sys.stdout.write("\n")
        sys.stdout.flush()

    def _render(self) -> None:
        """Render the progress bar to the terminal."""
        state = self._state
        pct = state.percentage / 100.0

        # Clear previous render
        if self._last_render_len > 0:
            sys.stdout.write("\r" + " " * self._last_render_len + "\r")

        # Build progress bar
        filled = int(self._bar_width * pct)
        empty = self._bar_width - filled

        # Color the fill
        color_codes = {
            "green": "\033[38;5;84m",
            "cyan": "\033[38;5;117m",
            "yellow": "\033[38;5;186m",
            "red": "\033[38;5;198m",
            "blue": "\033[38;5;117m",
            "magenta": "\033[38;5;198m",
            "white": "\033[38;5;231m",
        }
        reset = "\033[0m"
        color_code = color_codes.get(self._color, color_codes["green"])

        bar = f"{color_code}{self._fill_char * filled}{reset}{self._empty_char * empty}"

        # Build info string
        parts = []
        if state.description:
            parts.append(f"\033[1m{state.description}\033[0m")

        parts.append(f"[{bar}]")

        if self._show_percentage:
            parts.append(f"{state.percentage:5.1f}%")

        if self._show_count and state.total > 0:
            unit_str = state.unit
            parts.append(f"{state.current:.0f}/{state.total:.0f}{unit_str}")

        if self._show_rate and state.rate > 0:
            rate_str = f"{state.rate:.1f}"
            if state.unit:
                rate_str += f"{state.unit}/s"
            else:
                rate_str += " it/s"
            parts.append(rate_str)

        if self._show_eta and state.eta is not None and not state.completed:
            eta = state.eta
            if eta < 60:
                eta_str = f"{eta:.0f}s"
            elif eta < 3600:
                eta_str = f"{eta / 60:.1f}m"
            else:
                eta_str = f"{eta / 3600:.1f}h"
            parts.append(f"ETA {eta_str}")

        if state.completed:
            parts.append("✓")

        line = " ".join(parts)
        self._last_render_len = len(line)
        sys.stdout.write(line)
        sys.stdout.flush()

    def __enter__(self) -> ProgressBar:
        return self

    def __exit__(self, *args: Any) -> None:
        self.complete()


class ProgressTracker:
    """Tracks progress across multiple tasks or stages.

    Supports nested progress, task management, and composite
    progress calculation across multiple concurrent operations.
    """

    def __init__(self, description: str = "Overall") -> None:
        self._description = description
        self._tasks: dict[str, ProgressState] = {}
        self._bar = ProgressBar(total=100.0, description=description)

    def add_task(self, name: str, total: float, unit: str = "") -> str:
        """Add a new task to track.

        Args:
            name: Task name.
            total: Total units for the task.
            unit: Unit label.

        Returns:
            The task name (used as an ID).
        """
        self._tasks[name] = ProgressState(total=total, unit=unit)
        return name

    def update_task(self, name: str, current: float) -> None:
        """Update a task's progress.

        Args:
            name: Task name.
            current: New current value.
        """
        if name in self._tasks:
            self._tasks[name].current = min(current, self._tasks[name].total)
            if self._tasks[name].current >= self._tasks[name].total:
                self._tasks[name].completed = True
            self._update_overall()

    def _update_overall(self) -> None:
        """Recalculate and display overall progress."""
        if not self._tasks:
            return
        total_pct = sum(t.percentage for t in self._tasks.values())
        overall_pct = total_pct / len(self._tasks)
        self._bar.update(overall_pct)

    def complete_task(self, name: str) -> None:
        """Mark a task as complete.

        Args:
            name: Task name.
        """
        if name in self._tasks:
            self._tasks[name].current = self._tasks[name].total
            self._tasks[name].completed = True
            self._update_overall()

    def remove_task(self, name: str) -> None:
        """Remove a task from tracking.

        Args:
            name: Task name.
        """
        self._tasks.pop(name, None)
        self._update_overall()

    def complete_all(self) -> None:
        """Mark all tasks as complete."""
        for task in self._tasks.values():
            task.current = task.total
            task.completed = True
        self._bar.complete()

    @property
    def overall_percentage(self) -> float:
        """Get the overall progress percentage."""
        if not self._tasks:
            return 0.0
        return sum(t.percentage for t in self._tasks.values()) / len(self._tasks)

    @property
    def all_completed(self) -> bool:
        """Check if all tasks are completed."""
        return all(t.completed for t in self._tasks.values()) if self._tasks else False


def track(
    iterable: Any,
    total: float | None = None,
    description: str = "",
    unit: str = "",
) -> Iterator:
    """Wrap an iterable with a progress bar.

    Args:
        iterable: The iterable to wrap.
        total: Total number of items (auto-detected for sequences).
        description: Description for the progress bar.
        unit: Unit label.

    Yields:
        Items from the iterable with progress tracking.
    """
    if total is None:
        try:
            total = float(len(iterable))
        except TypeError:
            total = 0.0

    bar = ProgressBar(total=total, description=description, unit=unit)
    count = 0.0

    for item in iterable:
        yield item
        count += 1
        bar.update(count)

    bar.complete()
