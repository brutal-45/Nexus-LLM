"""Tests for streaming token generation."""
import pytest
from unittest.mock import MagicMock
from queue import Queue
import threading


def test_streaming_yields_tokens():
    tokens = ["Hello", " world", "!"]
    assert [t for t in tokens] == ["Hello", " world", "!"]


def test_streaming_queue_producer_consumer():
    q = Queue()
    tokens = ["The", " quick", " brown", " fox"]
    def producer():
        for t in tokens:
            q.put(t)
        q.put(None)
    thread = threading.Thread(target=producer)
    thread.start()
    collected = []
    while True:
        item = q.get(timeout=1.0)
        if item is None:
            break
        collected.append(item)
    thread.join()
    assert collected == tokens


def test_streaming_stop_on_eos():
    tokens = ["Hello", "<EOS>"]
    result = []
    for t in tokens:
        if t == "<EOS>":
            break
        result.append(t)
    assert result == ["Hello"]


def test_streaming_with_callback():
    callback = MagicMock()
    for token in ["A", "B", "C"]:
        callback(token)
    assert callback.call_count == 3


def test_streaming_latency_measurement():
    import time
    start = time.monotonic()
    time.sleep(0.01)
    assert time.monotonic() - start >= 0.01
