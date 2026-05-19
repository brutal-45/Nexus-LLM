"""
Nexus-LLM Spinner Module

Provides animated loading spinners with multiple styles:
dots, line, arrow, bounce, pulse, and custom animations.
"""

from __future__ import annotations

import sys
import time
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Any


class SpinnerStyle(str, Enum):
    """Built-in spinner animation styles."""
    DOTS = "dots"
    LINE = "line"
    ARROW = "arrow"
    BOUNCE = "bounce"
    PULSE = "pulse"
    WAVE = "wave"
    EARTH = "earth"
    MOON = "moon"
    CLOCK = "clock"
    BRAILLE = "braille"
    CIRCLE = "circle"
    SIMPLER = "simpler"


# Frame definitions for each spinner style
SPINNER_FRAMES: dict[str, list[str]] = {
    SpinnerStyle.DOTS: ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"],
    SpinnerStyle.LINE: ["-", "\\", "|", "/"],
    SpinnerStyle.ARROW: ["←", "↖", "↑", "↗", "→", "↘", "↓", "↙"],
    SpinnerStyle.BOUNCE: ["⠁", "⠂", "⠄", "⡀", "⢀", "⠠", "⠐", "⠈"],
    SpinnerStyle.PULSE: ["●", "◉", "◎", "○", "◎", "◉"],
    SpinnerStyle.WAVE: ["🌊", "🌊", "🌊", "🌊"],
    SpinnerStyle.EARTH: ["🌍", "🌎", "🌏"],
    SpinnerStyle.MOON: ["🌑", "🌒", "🌓", "🌔", "🌕", "🌖", "🌗", "🌘"],
    SpinnerStyle.CLOCK: ["🕐", "🕑", "🕒", "🕓", "🕔", "🕕", "🕖", "🕗", "🕘", "🕙", "🕚", "🕛"],
    SpinnerStyle.BRAILLE: ["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"],
    SpinnerStyle.CIRCLE: ["⚆", "⚇", "⚈", "⚉"],
    SpinnerStyle.SIMPLER: [".", "o", "O", "°"],
}

# Interval (in seconds) between frames for each style
SPINNER_INTERVALS: dict[str, float] = {
    SpinnerStyle.DOTS: 0.08,
    SpinnerStyle.LINE: 0.12,
    SpinnerStyle.ARROW: 0.1,
    SpinnerStyle.BOUNCE: 0.08,
    SpinnerStyle.PULSE: 0.15,
    SpinnerStyle.WAVE: 0.2,
    SpinnerStyle.EARTH: 0.3,
    SpinnerStyle.MOON: 0.15,
    SpinnerStyle.CLOCK: 0.2,
    SpinnerStyle.BRAILLE: 0.08,
    SpinnerStyle.CIRCLE: 0.15,
    SpinnerStyle.SIMPLER: 0.2,
}

# ANSI color codes for spinner text
SPINNER_COLORS: dict[str, str] = {
    "cyan": "\033[38;5;117m",
    "green": "\033[38;5;84m",
    "yellow": "\033[38;5;186m",
    "blue": "\033[38;5;117m",
    "magenta": "\033[38;5;198m",
    "white": "\033[38;5;231m",
    "red": "\033[38;5;198m",
}

RESET = "\033[0m"


@dataclass
class SpinnerConfig:
    """Configuration for a spinner instance."""
    text: str = "Loading"
    style: SpinnerStyle = SpinnerStyle.DOTS
    color: str = "cyan"
    interval: float | None = None
    hide_cursor: bool = True
    final_text: str | None = None


class Spinner:
    """Animated terminal spinner for indicating ongoing operations.

    Runs a spinner animation in a background thread while work proceeds.
    Supports multiple animation styles and customizable appearance.

    Usage:
        with Spinner("Processing", style=SpinnerStyle.DOTS):
            # do work
            time.sleep(2)

    Or manual control:
        spinner = Spinner("Processing")
        spinner.start()
        # do work
        spinner.stop()
    """

    def __init__(
        self,
        text: str = "Loading",
        style: SpinnerStyle | str = SpinnerStyle.DOTS,
        color: str = "cyan",
        interval: float | None = None,
        hide_cursor: bool = True,
        final_text: str | None = None,
    ) -> None:
        if isinstance(style, str):
            style = SpinnerStyle(style)
        self._text = text
        self._style = style
        self._color = color
        self._interval = interval or SPINNER_INTERVALS.get(self._style, 0.1)
        self._hide_cursor = hide_cursor
        self._final_text = final_text
        self._running = False
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    @property
    def text(self) -> str:
        """Get the spinner text."""
        return self._text

    @text.setter
    def text(self, value: str) -> None:
        """Update the spinner text while running."""
        self._text = value

    def start(self) -> None:
        """Start the spinner animation."""
        if self._running:
            return
        self._running = True
        self._stop_event.clear()

        if self._hide_cursor:
            sys.stdout.write("\033[?25l")  # Hide cursor
            sys.stdout.flush()

        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def stop(self, final_text: str | None = None) -> None:
        """Stop the spinner animation.

        Args:
            final_text: Optional text to display after stopping.
        """
        if not self._running:
            return
        self._running = False
        self._stop_event.set()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

        # Clear the spinner line
        sys.stdout.write("\r" + " " * (len(self._text) + 10) + "\r")
        sys.stdout.flush()

        if self._hide_cursor:
            sys.stdout.write("\033[?25h")  # Show cursor
            sys.stdout.flush()

        display_text = final_text or self._final_text
        if display_text:
            color_code = SPINNER_COLORS.get(self._color, "")
            sys.stdout.write(f"\r{color_code}✓{RESET} {display_text}\n")
            sys.stdout.flush()

    def _spin(self) -> None:
        """Background thread function that renders spinner frames."""
        frames = SPINNER_FRAMES.get(self._style, SPINNER_FRAMES[SpinnerStyle.DOTS])
        color_code = SPINNER_COLORS.get(self._color, SPINNER_COLORS["cyan"])
        idx = 0

        while not self._stop_event.is_set():
            frame = frames[idx % len(frames)]
            line = f"\r{color_code}{frame}{RESET} {self._text}"
            sys.stdout.write(line)
            sys.stdout.flush()

            idx += 1
            self._stop_event.wait(self._interval)

        # Clear the line on stop
        clear_len = len(self._text) + 10
        sys.stdout.write("\r" + " " * clear_len + "\r")
        sys.stdout.flush()

    def update_text(self, text: str) -> None:
        """Update the spinner text while running.

        Args:
            text: New text to display.
        """
        self._text = text

    def __enter__(self) -> Spinner:
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop()

    @staticmethod
    def get_available_styles() -> list[str]:
        """Get a list of available spinner style names.

        Returns:
            List of style name strings.
        """
        return [s.value for s in SpinnerStyle]

    @staticmethod
    def preview_styles() -> str:
        """Generate a preview of all spinner styles.

        Returns:
            A string showing one frame of each style.
        """
        lines = []
        for style_name, frames in SPINNER_FRAMES.items():
            first_frame = frames[0]
            lines.append(f"  {first_frame} {style_name}")
        return "\n".join(lines)
