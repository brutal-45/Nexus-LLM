"""Tests for logging."""
import pytest
import logging
import os
import tempfile


def test_logger_creation():
    """Test creating a logger."""
    logger = logging.getLogger("test_nexus")
    assert logger is not None
    assert logger.name == "test_nexus"


def test_logger_level():
    """Test setting logger level."""
    logger = logging.getLogger("test_nexus_level")
    logger.setLevel(logging.DEBUG)
    assert logger.level == logging.DEBUG


def test_logger_handler():
    """Test adding a handler to a logger."""
    logger = logging.getLogger("test_nexus_handler")
    handler = logging.StreamHandler()
    logger.addHandler(handler)
    assert len(logger.handlers) >= 1


def test_logger_file_handler(tmp_dir):
    """Test logging to a file."""
    log_path = os.path.join(tmp_dir, "test.log")
    logger = logging.getLogger("test_nexus_file")
    handler = logging.FileHandler(log_path)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.info("Test log message")
    handler.flush()
    with open(log_path) as f:
        content = f.read()
    assert "Test log message" in content


def test_logger_formatter():
    """Test logger formatter."""
    fmt = logging.Formatter("%(levelname)s - %(message)s")
    assert fmt._fmt == "%(levelname)s - %(message)s"
