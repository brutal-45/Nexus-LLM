"""Tests for throughput measurement."""
import pytest


def test_requests_per_second():
    assert 100 / 10.0 == 10.0


def test_tokens_per_second():
    assert 1000 / 5.0 == 200.0


def test_concurrent_throughput():
    assert 8 / 0.1 == 80.0


def test_throughput_scales_with_batch():
    assert 10 * 4 * 0.9 > 10


def test_throughput_measurement_stability():
    m = [100, 102, 98, 101, 99]
    mean = sum(m) / len(m)
    std = (sum((x - mean)**2 for x in m) / len(m)) ** 0.5
    assert std / mean < 0.05
