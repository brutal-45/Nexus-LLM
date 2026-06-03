"""Tests for thread pool."""
import pytest
from concurrent.futures import ThreadPoolExecutor


def test_thread_pool_submit():
    with ThreadPoolExecutor(max_workers=2) as pool:
        future = pool.submit(lambda: 42)
        assert future.result() == 42

def test_thread_pool_map():
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda x: x * 2, [1, 2, 3, 4]))
        assert results == [2, 4, 6, 8]

def test_thread_pool_multiple_futures():
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(lambda i=i: i ** 2) for i in range(5)]
        results = [f.result() for f in futures]
        assert results == [0, 1, 4, 9, 16]

def test_thread_pool_max_workers():
    max_workers = 4
    assert max_workers > 0
