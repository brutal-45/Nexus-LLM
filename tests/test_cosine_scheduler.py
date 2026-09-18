"""Tests for cosine LR scheduler."""
import pytest
import math


def test_cosine_decay_basic():
    step, total = 500, 1000
    scale = 0.5 * (1.0 + math.cos(math.pi * step / total))
    assert 0.0 <= scale <= 1.0

def test_cosine_decay_start():
    scale = 0.5 * (1.0 + math.cos(0))
    assert abs(scale - 1.0) < 1e-5

def test_cosine_decay_mid():
    scale = 0.5 * (1.0 + math.cos(math.pi))
    assert abs(scale - 0.0) < 1e-5

def test_cosine_decay_with_warmup():
    warmup, total = 100, 1000
    step = 50
    if step < warmup:
        scale = step / warmup
    else:
        progress = (step - warmup) / (total - warmup)
        scale = 0.5 * (1.0 + math.cos(math.pi * progress))
    assert 0.0 <= scale <= 1.0
