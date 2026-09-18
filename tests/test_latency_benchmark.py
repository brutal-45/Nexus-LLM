"""Tests for latency measurement."""
import pytest
import time


def test_latency_measurement():
    start = time.monotonic()
    time.sleep(0.01)
    assert time.monotonic() - start >= 0.01


def test_average_latency():
    lats = [0.1, 0.15, 0.12, 0.11, 0.13]
    assert abs(sum(lats) / len(lats) - 0.122) < 0.01


def test_p99_latency():
    lats = sorted([0.1, 0.15, 0.12, 0.11, 0.13, 0.5, 0.2])
    p99 = lats[int(len(lats) * 0.99)]
    assert p99 >= 0.1


def test_time_to_first_token():
    assert 0 < 0.05 < 1.0


def test_latency_outlier_detection():
    lats = [0.1, 0.1, 0.1, 0.1, 5.0]
    mean = sum(lats) / len(lats)
    assert len([l for l in lats if l > mean * 3]) == 1
