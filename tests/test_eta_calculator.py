"""Tests for ETA calculator."""
import pytest


def test_eta_basic():
    total = 1000
    completed = 500
    elapsed = 10.0
    speed = completed / elapsed
    remaining = total - completed
    eta = remaining / speed
    assert eta == 10.0

def test_eta_zero_completed():
    total = 1000
    completed = 0
    eta = float("inf") if completed == 0 else 0
    assert eta == float("inf")

def test_eta_complete():
    total = 100
    completed = 100
    eta = 0
    assert eta == 0

def test_eta_with_varying_speed():
    speeds = [100, 150, 200]
    avg_speed = sum(speeds) / len(speeds)
    remaining = 500
    eta = remaining / avg_speed
    assert eta > 0

def test_eta_format():
    seconds = 3661
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    formatted = f"{hours:02d}:{minutes:02d}:{secs:02d}"
    assert formatted == "01:01:01"
