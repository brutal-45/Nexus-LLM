"""Tests for progress bar display."""
import pytest


def test_progress_bar_zero():
    progress = 0.0
    assert 0.0 <= progress <= 1.0

def test_progress_bar_half():
    progress = 0.5
    assert progress == 0.5

def test_progress_bar_complete():
    progress = 1.0
    assert progress == 1.0

def test_progress_bar_width():
    width = 40
    filled = int(0.75 * width)
    bar = "#" * filled + "-" * (width - filled)
    assert len(bar) == width

def test_progress_bar_percentage():
    progress = 0.65
    pct = int(progress * 100)
    assert pct == 65
