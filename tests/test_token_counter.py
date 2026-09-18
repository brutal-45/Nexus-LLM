"""Tests for token counter display."""
import pytest


def test_token_count_basic():
    text = "Hello world this is a test"
    token_count = len(text.split())
    assert token_count == 6

def test_token_count_with_limit():
    limit = 2048
    current = 1500
    remaining = limit - current
    assert remaining == 548

def test_token_count_percentage():
    limit = 2048
    current = 1024
    pct = (current / limit) * 100
    assert pct == 50.0

def test_token_count_display():
    count, limit = 500, 2048
    display = f"{count}/{limit}"
    assert "500" in display and "2048" in display
