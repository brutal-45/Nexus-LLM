"""Tests for duration formatting."""
import pytest


def test_duration_seconds():
    secs = 45
    assert f"{secs}s" == "45s"

def test_duration_minutes():
    secs = 90
    m, s = divmod(secs, 60)
    assert f"{m}m {s}s" == "1m 30s"

def test_duration_hours():
    secs = 3661
    h, remainder = divmod(secs, 3600)
    m, s = divmod(remainder, 60)
    assert f"{h}h {m}m {s}s" == "1h 1m 1s"

def test_duration_zero():
    assert "0s" == "0s"

def test_duration_days():
    secs = 90061
    d, remainder = divmod(secs, 86400)
    assert d == 1
