"""Tests for data quality scoring."""
import pytest
import math


def test_quality_score_range():
    def compute_quality(text):
        if not text.strip():
            return 0.0
        words = text.split()
        avg_len = sum(len(w) for w in words) / max(len(words), 1)
        return min(avg_len / 10.0, 1.0)
    assert 0.0 <= compute_quality("This is a well-written sentence about AI.") <= 1.0


def test_empty_text_quality():
    def compute_quality(text):
        if not text.strip():
            return 0.0
        return min(len(text.split()) / 100.0, 1.0)
    assert compute_quality("") == 0.0
    assert compute_quality("   ") == 0.0


def test_repetition_ratio():
    text = "hello hello hello world world"
    words = text.split()
    unique = set(words)
    ratio = 1.0 - len(unique) / len(words)
    assert 0.0 <= ratio <= 1.0
    assert ratio > 0


def test_perplexity_quality_proxy():
    high_probs = [0.3, 0.25, 0.2, 0.15, 0.1]
    low_probs = [0.05, 0.05, 0.05, 0.05, 0.8]
    def nll(probs):
        return -sum(math.log(p) for p in probs) / len(probs)
    assert nll(high_probs) < nll(low_probs)


def test_quality_threshold_filter():
    documents = [("Good text about science.", 0.8), ("asdf zxcv", 0.1), ("Decent AI doc.", 0.7)]
    filtered = [d for d, s in documents if s >= 0.5]
    assert len(filtered) == 2
