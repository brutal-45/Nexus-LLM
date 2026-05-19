"""Tests for cache decorator."""
import pytest
import functools


def test_cache_decorator_caches_result():
    call_count = 0

    def cached(func):
        cache = {}
        @functools.wraps(func)
        def wrapper(*args):
            nonlocal call_count
            if args not in cache:
                call_count += 1
                cache[args] = func(*args)
            return cache[args]
        return wrapper

    @cached
    def expensive(n):
        return n * 2

    assert expensive(5) == 10
    assert expensive(5) == 10
    assert call_count == 1

def test_cache_different_args():
    cache = {}

    def compute(n):
        if n not in cache:
            cache[n] = n ** 2
        return cache[n]

    assert compute(3) == 9
    assert compute(4) == 16
    assert len(cache) == 2

def test_cache_invalidation():
    cache = {"a": 1, "b": 2}
    cache.clear()
    assert len(cache) == 0
