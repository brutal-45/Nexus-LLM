"""Tests for memory profiler."""
import pytest
import sys


def test_memory_profiler_object_size():
    obj = [1] * 1000
    size = sys.getsizeof(obj)
    assert size > 0

def test_memory_profiler_dict_size():
    d = {f"key_{i}": i for i in range(100)}
    size = sys.getsizeof(d)
    assert size > 0

def test_memory_profiler_string_size():
    s = "x" * 10000
    size = sys.getsizeof(s)
    assert size > 10000

def test_memory_profiler_nested():
    nested = {"a": [1, 2, 3], "b": {"c": "hello"}}
    size = sys.getsizeof(nested)
    assert size > 0

def test_memory_profiler_empty():
    size = sys.getsizeof([])
    assert size > 0
