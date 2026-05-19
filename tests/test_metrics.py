"""Tests for metrics."""
import pytest
import time
from nexus.inference.server import ServerMetrics


def test_server_metrics_initial():
    """Test initial metrics values."""
    m = ServerMetrics()
    assert m.total_requests == 0
    assert m.total_tokens_generated == 0
    assert m.avg_latency_ms == 0.0


def test_server_metrics_record():
    """Test recording request metrics."""
    m = ServerMetrics()
    m.record_request(latency_s=0.1, prompt_tokens=50, completion_tokens=100)
    assert m.total_requests == 1
    assert m.total_tokens_generated == 100
    assert m.total_prompt_tokens == 50


def test_server_metrics_multiple_requests():
    """Test recording multiple requests."""
    m = ServerMetrics()
    for i in range(10):
        m.record_request(latency_s=0.1 * (i + 1), prompt_tokens=10, completion_tokens=20)
    assert m.total_requests == 10
    assert m.total_tokens_generated == 200


def test_server_metrics_error_count():
    """Test error counting."""
    m = ServerMetrics()
    m.record_error()
    m.record_error()
    assert m.total_errors == 2


def test_server_metrics_latency_rolling():
    """Test that latency uses rolling average."""
    m = ServerMetrics()
    m.record_request(latency_s=1.0, prompt_tokens=10, completion_tokens=10)
    assert m.avg_latency_ms == pytest.approx(1000.0, rel=0.01)
    m.record_request(latency_s=0.0, prompt_tokens=10, completion_tokens=10)
    assert m.avg_latency_ms == pytest.approx(500.0, rel=0.01)
