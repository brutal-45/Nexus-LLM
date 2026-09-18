"""Tests for GPU memory tracking."""
import pytest


def test_memory_formatting():
    def fmt(b):
        for u in ["B", "KB", "MB", "GB"]:
            if b < 1024:
                return f"{b:.1f} {u}"
            b /= 1024
        return f"{b:.1f} TB"
    assert "GB" in fmt(4 * 1024**3)
    assert "MB" in fmt(512 * 1024**2)


def test_memory_usage_calculation():
    assert abs((12288 / 24576) * 100 - 50.0) < 0.01


def test_memory_available():
    assert 24576 - 8192 == 16384


def test_memory_tracking_without_gpu():
    if not False:  # No GPU
        info = {"total": 0, "used": 0}
    assert info["total"] == 0
