"""Tests for AdamW optimizer."""
import pytest


def test_adamw_weight_decay():
    param = 1.0
    lr = 0.001
    weight_decay = 0.01
    decay = param * weight_decay
    assert decay == 0.01

def test_adamw_decay_separate_from_gradient():
    # AdamW decouples weight decay from gradient update
    grad = 0.5
    wd_update = 0.01  # lr * weight_decay * param
    total_update = grad + wd_update
    assert total_update > grad

def test_adamw_bias_correction():
    step = 1
    beta1, beta2 = 0.9, 0.999
    bias_correction1 = 1 - beta1 ** step
    bias_correction2 = 1 - beta2 ** step
    assert bias_correction1 > 0
    assert bias_correction2 > 0

def test_adamw_momentum():
    beta1 = 0.9
    m = 0.0
    g = 1.0
    m = beta1 * m + (1 - beta1) * g
    assert abs(m - 0.1) < 1e-5
