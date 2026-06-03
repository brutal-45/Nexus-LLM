"""Logging utility for Nexus-LLM.

Provides ``setup_logger`` which creates a logger with a Rich console handler
(for colourised, formatted console output) and an optional rotating file
handler (for persistent, plain-text logs).
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

_CONSOLE_FORMAT = "%(message)s"
_FILE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


class _LevelFormatter(logging.Formatter):
    """Formatter that applies different styles per log level for console."""

    _LEVEL_STYLES = {
        logging.DEBUG:    "[dim]%(message)s[/dim]",
        logging.INFO:     "%(message)s",
        logging.WARNING:  "[bold yellow]⚠ %(message)s[/bold yellow]",
        logging.ERROR:    "[bold red]✖ %(message)s[/bold red]",
        logging.CRITICAL: "[bold red on white]✖ %(message)s[/bold red on white]",
    }

    def format(self, record: logging.LogRecord) -> str:
        template = self._LEVEL_STYLES.get(record.levelno, _CONSOLE_FORMAT)
        # Use the base Formatter to handle the timestamp etc.
        self._style._fmt = template  # type: ignore[attr-defined]
        return super().format(record)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def setup_logger(
    name: str = "nexus_llm",
    level: str = "INFO",
    log_file: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,   # 10 MB
    backup_count: int = 5,
    rich_console: bool = True,
) -> logging.Logger:
    """Create and configure a logger with Rich console and/or file output.

    Args:
        name:        Logger name (usually the module or package name).
        level:       Minimum log level as a string (DEBUG, INFO, …).
        log_file:    Optional path to a rotating log file.  When provided the
                     directory is created automatically.
        max_bytes:   Maximum size (bytes) of each log file before rotation.
        backup_count: Number of rotated backup files to keep.
        rich_console: Whether to use a Rich-enhanced console handler.

    Returns:
        A configured ``logging.Logger`` instance.
    """
    logger = logging.getLogger(name)
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    # Avoid duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    # ---- Console handler ----
    if rich_console:
        try:
            from rich.logging import RichHandler

            console_handler = RichHandler(
                level=numeric_level,
                show_time=True,
                show_path=False,
                markup=True,
                rich_tracebacks=True,
                tracebacks_show_locals=False,
            )
            console_handler.setFormatter(logging.Formatter(_CONSOLE_FORMAT))
        except ImportError:
            # Fallback to a plain StreamHandler
            console_handler = logging.StreamHandler(sys.stderr)
            console_handler.setFormatter(
                logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
            )
    else:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(
            logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
        )

    console_handler.setLevel(numeric_level)
    logger.addHandler(console_handler)

    # ---- File handler (optional, rotating) ----
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            filename=str(log_path),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(logging.Formatter(_FILE_FORMAT))
        logger.addHandler(file_handler)

    # Prevent propagation to the root logger to avoid duplicate output
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """Return an existing logger by name (does not configure it).

    Useful for getting a module-level logger after ``setup_logger`` has
    already been called at application startup.
    """
    return logging.getLogger(name)
