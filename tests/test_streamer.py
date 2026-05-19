"""Tests for streamer."""
import pytest
import json
from nexus.inference.streaming_inference import TokenOutput, SSEFormatter, WebSocketStreamer


def test_token_output_creation():
    """Test creating a TokenOutput."""
    tok = TokenOutput(token_id=42, token_text="hello", logprob=-1.5, index=0)
    assert tok.token_id == 42
    assert tok.token_text == "hello"
    assert tok.logprob == -1.5
    assert tok.finish_reason is None


def test_token_output_with_finish():
    """Test TokenOutput with finish reason."""
    tok = TokenOutput(token_id=2, token_text="", logprob=0.0, finish_reason="stop")
    assert tok.finish_reason == "stop"


def test_sse_format_token():
    """Test SSE token formatting."""
    tok = TokenOutput(token_id=10, token_text="world", logprob=-2.0, index=1)
    sse = SSEFormatter.format_token(tok)
    assert sse.startswith("data: ")
    data = json.loads(sse[6:].strip())
    assert data["token"] == "world"
    assert data["token_id"] == 10
    assert data["logprob"] == pytest.approx(-2.0, abs=0.01)


def test_sse_format_done():
    """Test SSE done signal."""
    sse = SSEFormatter.format_done()
    assert sse == "data: [DONE]\n\n"


def test_sse_format_error():
    """Test SSE error formatting."""
    sse = SSEFormatter.format_error("test error")
    assert "test error" in sse
    data = json.loads(sse[6:].strip())
    assert data["error"] == "test error"


def test_websocket_format_token():
    """Test WebSocket token formatting."""
    tok = TokenOutput(token_id=5, token_text="test", logprob=-1.0, index=0)
    msg = WebSocketStreamer.format_token(tok)
    assert msg["type"] == "token"
    assert msg["data"]["text"] == "test"


def test_websocket_format_done():
    """Test WebSocket done formatting."""
    msg = WebSocketStreamer.format_done("stop")
    assert msg["type"] == "done"
    assert msg["data"]["reason"] == "stop"


def test_websocket_format_error():
    """Test WebSocket error formatting."""
    msg = WebSocketStreamer.format_error("connection lost")
    assert msg["type"] == "error"
    assert msg["data"]["message"] == "connection lost"
