"""Logging: file + console logging, rotation, levels, formatting."""

import os
import sys
import logging
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from typing import Optional, Dict, Any


LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_level_map = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "warn": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


def setup_logger(
    name: str = "nexus_llm",
    level: str = "info",
    log_file: Optional[str] = None,
    log_dir: Optional[str] = None,
    console: bool = True,
    rotation: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    format_string: Optional[str] = None,
    propagate: bool = False,
) -> logging.Logger:
    """Set up a logger with file and/or console output.

    Args:
        name: Logger name.
        level: Logging level (debug, info, warning, error, critical).
        log_file: Specific log file path. If provided, overrides log_dir.
        log_dir: Directory for log files. File will be named after the logger.
        console: Whether to add console handler.
        rotation: Rotation type: "size", "time", or None.
        max_bytes: Max file size for size-based rotation.
        backup_count: Number of backup files to keep.
        format_string: Custom format string.
        propagate: Whether to propagate to parent loggers.

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(_level_map.get(level.lower(), logging.INFO))
    logger.propagate = propagate

    formatter = logging.Formatter(
        format_string or LOG_FORMAT, datefmt=DATE_FORMAT
    )

    # Remove existing handlers
    logger.handlers = []

    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(_level_map.get(level.lower(), logging.INFO))
        logger.addHandler(console_handler)

    if log_file or log_dir:
        if log_file is None and log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, f"{name}.log")

        if log_file is not None:
            os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)

            if rotation == "size":
                file_handler = RotatingFileHandler(
                    log_file,
                    maxBytes=max_bytes,
                    backupCount=backup_count,
                    encoding="utf-8",
                )
            elif rotation == "time":
                file_handler = TimedRotatingFileHandler(
                    log_file,
                    when="midnight",
                    interval=1,
                    backupCount=backup_count,
                    encoding="utf-8",
                )
            else:
                file_handler = logging.FileHandler(
                    log_file, encoding="utf-8"
                )

            file_handler.setFormatter(formatter)
            file_handler.setLevel(_level_map.get(level.lower(), logging.INFO))
            logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "nexus_llm") -> logging.Logger:
    """Get an existing logger by name.

    Args:
        name: Logger name.

    Returns:
        Logger instance.
    """
    return logging.getLogger(name)


def set_log_level(logger: logging.Logger, level: str):
    """Set the logging level for a logger and all its handlers.

    Args:
        logger: Logger instance.
        level: New logging level.
    """
    lvl = _level_map.get(level.lower(), logging.INFO)
    logger.setLevel(lvl)
    for handler in logger.handlers:
        handler.setLevel(lvl)


class LoggerContext:
    """Context manager for temporarily changing log level."""

    def __init__(self, logger: logging.Logger, level: str):
        self.logger = logger
        self.new_level = _level_map.get(level.lower(), logging.INFO)
        self.old_level = logger.level

    def __enter__(self):
        self.logger.setLevel(self.new_level)
        return self.logger

    def __exit__(self, *args):
        self.logger.setLevel(self.old_level)


def log_function_call(logger: logging.Logger, level: str = "debug"):
    """Decorator that logs function calls.

    Args:
        logger: Logger to use.
        level: Logging level for the messages.
    """
    log_fn = getattr(logger, level, logger.debug)

    def decorator(func):
        def wrapper(*args, **kwargs):
            log_fn(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
            try:
                result = func(*args, **kwargs)
                log_fn(f"{func.__name__} returned successfully")
                return result
            except Exception as e:
                logger.error(f"{func.__name__} raised {type(e).__name__}: {e}")
                raise
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    return decorator
