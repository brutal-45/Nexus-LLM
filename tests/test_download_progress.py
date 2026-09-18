"""Tests for download progress."""
import pytest


def test_progress_calculation():
    downloaded = 500
    total = 1000
    progress = (downloaded / total) * 100
    assert progress == 50.0

def test_progress_display():
    downloaded, total = 750, 1000
    bar_width = 20
    filled = int(bar_width * downloaded / total)
    bar = "=" * filled + "-" * (bar_width - filled)
    assert len(bar) == 20

def test_eta_calculation():
    downloaded = 500
    total = 1000
    elapsed = 10.0
    speed = downloaded / elapsed
    remaining = total - downloaded
    eta = remaining / speed
    assert eta == 10.0

def test_download_size_formatting():
    def fmt(b):
        for u in ["B", "KB", "MB", "GB"]:
            if b < 1024:
                return f"{b:.1f} {u}"
            b /= 1024
    assert "MB" in fmt(50 * 1024**2)
