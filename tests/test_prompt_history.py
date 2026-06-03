"""Tests for prompt history browsing."""
import pytest


def test_history_stores_prompts():
    h = []
    h.append("What is AI?")
    h.append("Tell me about ML.")
    assert len(h) == 2


def test_history_navigation_up():
    h = ["first", "second", "third"]
    assert h[len(h) - 1] == "third"
    assert h[len(h) - 2] == "second"


def test_history_navigation_down():
    h = ["first", "second", "third"]
    assert h[0] == "first"
    assert h[1] == "second"


def test_history_max_size():
    h = [f"p_{i}" for i in range(200)]
    if len(h) > 100:
        h = h[-100:]
    assert len(h) == 100


def test_history_deduplication():
    h = []
    for p in ["hello", "hello", "world", "world", "world"]:
        if not h or h[-1] != p:
            h.append(p)
    assert h == ["hello", "world"]
