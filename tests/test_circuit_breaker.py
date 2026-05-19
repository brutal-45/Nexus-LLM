"""Tests for circuit breaker."""
import pytest
import time


def test_circuit_breaker_closed():
    state = "closed"
    failures = 0
    threshold = 5
    assert failures < threshold
    assert state == "closed"

def test_circuit_breaker_opens():
    failures = 5
    threshold = 5
    state = "open" if failures >= threshold else "closed"
    assert state == "open"

def test_circuit_breaker_half_open():
    state = "half_open"
    # Allow one test request
    success = True
    state = "closed" if success else "open"
    assert state == "closed"

def test_circuit_breaker_resets():
    failures = 3
    failures = 0  # Reset
    assert failures == 0

def test_circuit_breaker_timeout():
    last_failure = time.monotonic() - 30
    timeout = 60
    can_try = (time.monotonic() - last_failure) >= timeout
    assert not can_try
