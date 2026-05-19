"""Tests for FastAPI server endpoints."""
import pytest
import json
from unittest.mock import MagicMock, patch, AsyncMock

from nexus.inference.server import (
    InferenceServer, ServerMetrics, ChatMessage, ChatCompletionRequest,
    CompletionRequest, UsageStats, json_str,
)


def test_server_metrics_defaults():
    """Test ServerMetrics default values."""
    m = ServerMetrics()
    assert m.total_requests == 0
    assert m.total_tokens_generated == 0
    assert m.total_errors == 0
    assert m.avg_latency_ms == 0.0


def test_server_metrics_record_request():
    """Test recording a request in metrics."""
    m = ServerMetrics()
    m.record_request(latency_s=0.5, prompt_tokens=100, completion_tokens=50)
    assert m.total_requests == 1
    assert m.total_tokens_generated == 50
    assert m.total_prompt_tokens == 100
    assert m.total_completion_tokens == 50
    assert m.avg_latency_ms > 0


def test_server_metrics_record_error():
    """Test recording an error."""
    m = ServerMetrics()
    m.record_error()
    assert m.total_errors == 1


def test_server_metrics_latency_tracking():
    """Test that latency is tracked correctly."""
    m = ServerMetrics()
    m.record_request(latency_s=1.0, prompt_tokens=10, completion_tokens=10)
    assert m.avg_latency_ms == pytest.approx(1000.0, rel=0.01)
    m.record_request(latency_s=2.0, prompt_tokens=10, completion_tokens=10)
    assert m.avg_latency_ms == pytest.approx(1500.0, rel=0.01)


def test_chat_message_model():
    """Test ChatMessage Pydantic model."""
    msg = ChatMessage(role="user", content="Hello")
    assert msg.role == "user"
    assert msg.content == "Hello"
    assert msg.name is None


def test_chat_completion_request():
    """Test ChatCompletionRequest model."""
    req = ChatCompletionRequest(
        model="nexus-100b",
        messages=[ChatMessage(role="user", content="Hi")],
        temperature=0.5,
        max_tokens=100,
    )
    assert req.model == "nexus-100b"
    assert req.temperature == 0.5
    assert req.max_tokens == 100


def test_completion_request():
    """Test CompletionRequest model."""
    req = CompletionRequest(
        model="nexus-100b",
        prompt="Once upon a time",
        temperature=0.8,
    )
    assert req.prompt == "Once upon a time"
    assert req.temperature == 0.8


def test_usage_stats():
    """Test UsageStats model."""
    stats = UsageStats(prompt_tokens=10, completion_tokens=20, total_tokens=30)
    assert stats.prompt_tokens == 10
    assert stats.completion_tokens == 20
    assert stats.total_tokens == 30


def test_json_str():
    """Test json_str utility."""
    data = {"key": "value", "num": 42}
    result = json_str(data)
    parsed = json.loads(result)
    assert parsed["key"] == "value"
    assert parsed["num"] == 42
