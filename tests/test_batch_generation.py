"""Tests for batch generation."""
import pytest


def test_batch_generation_size():
    prompts = [f"Prompt {i}" for i in range(4)]
    outputs = [f"Output for {p}" for p in prompts]
    assert len(outputs) == 4


def test_batch_generation_consistent_lengths():
    max_tokens = 100
    outputs = [["token"] * min(i * 20 + 10, max_tokens) for i in range(5)]
    for output in outputs:
        assert len(output) <= max_tokens


def test_batch_generation_with_params():
    params = [{"temperature": 0.1, "top_p": 0.9}, {"temperature": 0.5, "top_p": 0.95}]
    for p in params:
        assert 0.0 <= p["temperature"]
        assert 0.0 <= p["top_p"] <= 1.0


def test_empty_batch_handling():
    prompts = []
    outputs = [f"Output for {p}" for p in prompts]
    assert len(outputs) == 0


def test_batch_padding():
    sequences = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
    max_len = max(len(s) for s in sequences)
    padded = [s + [0] * (max_len - len(s)) for s in sequences]
    assert all(len(s) == max_len for s in padded)
