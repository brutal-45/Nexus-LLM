"""Tests for CPU profiler."""
import pytest
import time


def test_cpu_profiler_basic():
    start = time.monotonic()
    _ = sum(range(1000))
    elapsed = time.monotonic() - start
    assert elapsed < 1.0

def test_cpu_profiler_function_timing():
    def profile(func):
        start = time.monotonic()
        result = func()
        elapsed = time.monotonic() - start
        return result, elapsed

    def compute():
        return sum(i**2 for i in range(1000))

    result, elapsed = profile(compute)
    assert result > 0
    assert elapsed >= 0

def test_cpu_profiler_overhead():
    start = time.monotonic()
    for _ in range(1000):
        pass
    overhead = time.monotonic() - start
    assert overhead < 0.1

def test_cpu_profiler_hotspot():
    def slow():
        time.sleep(0.01)
        return "done"

    start = time.monotonic()
    slow()
    assert time.monotonic() - start >= 0.01
