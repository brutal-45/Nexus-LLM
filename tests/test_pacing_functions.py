"""Tests for pacing functions."""
import pytest
import math


def test_linear_pacing():
    step, total = 500, 1000
    pace = step / total
    assert abs(pace - 0.5) < 1e-5

def test_root_pacing():
    step, total = 250, 1000
    pace = math.sqrt(step / total)
    assert pace > step / total  # Root pacing is faster early

def test_exponential_pacing():
    step, total = 500, 1000
    pace = (step / total) ** 2
    assert pace < 0.5  # Quadratic pacing is slower

def test_pacing_monotonic():
    values = [i / 100 for i in range(1, 100)]
    for i in range(len(values) - 1):
        assert values[i] < values[i + 1]
