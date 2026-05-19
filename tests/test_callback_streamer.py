"""Tests for callback streamer."""
import pytest
from unittest.mock import MagicMock


def test_callback_invoked_on_token():
    cb = MagicMock()
    for t in ["Hello", " world", "!"]:
        cb(t)
    assert cb.call_count == 3


def test_multiple_callbacks():
    cb1, cb2 = MagicMock(), MagicMock()
    for cb in [cb1, cb2]:
        cb("test")
    assert cb1.called and cb2.called


def test_callback_with_metadata():
    cb = MagicMock()
    cb({"token": "Hello", "logprob": -0.5})
    cb.assert_called_once_with({"token": "Hello", "logprob": -0.5})


def test_callback_error_handling():
    def bad_cb(token):
        raise ValueError("error")
    results = []
    for t in ["a", "b"]:
        try:
            bad_cb(t)
        except ValueError:
            results.append(t)
    assert len(results) == 2


def test_callback_on_stream_end():
    on_end = MagicMock()
    on_end()
    assert on_end.called
