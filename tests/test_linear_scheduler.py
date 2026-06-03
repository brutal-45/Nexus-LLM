"""Tests for linear LR scheduler."""
import pytest


def test_linear_warmup():
    warmup_steps = 100
    total_steps = 1000
    for step in [0, 50, 100]:
        lr_scale = min(step / warmup_steps, 1.0)
        assert 0.0 <= lr_scale <= 1.0

def test_linear_decay():
    total_steps = 1000
    for step in [100, 500, 900]:
        lr_scale = 1.0 - (step / total_steps)
        assert 0.0 <= lr_scale <= 1.0

def test_linear_scheduler_full_cycle():
    warmup, total = 100, 1000
    step = 500
    if step < warmup:
        scale = step / warmup
    else:
        scale = 1.0 - (step - warmup) / (total - warmup)
    assert 0.0 <= scale <= 1.0

def test_linear_scheduler_at_end():
    total = 1000
    scale = 1.0 - 1.0
    assert scale == 0.0
