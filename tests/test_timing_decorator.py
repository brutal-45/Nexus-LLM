"""Tests for timing decorator."""
import pytest
import time
import functools


def test_timing_decorator_measures_time():
    def timer(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.monotonic()
            result = func(*args, **kwargs)
            elapsed = time.monotonic() - start
            return result, elapsed
        return wrapper

    @timer
    def slow_func():
        time.sleep(0.01)
        return "done"

    result, elapsed = slow_func()
    assert result == "done"
    assert elapsed >= 0.01

def test_timing_decorator_preserves_result():
    def timer(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.monotonic()
            result = func(*args, **kwargs)
            return result
        return wrapper

    @timer
    def add(a, b):
        return a + b

    assert add(2, 3) == 5

def test_timing_decorator_with_args():
    def timer(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.monotonic()
            result = func(*args, **kwargs)
            return result
        return wrapper

    @timer
    def greet(name, greeting="Hello"):
        return f"{greeting}, {name}"

    assert greet("World") == "Hello, World"
    assert greet("World", greeting="Hi") == "Hi, World"
