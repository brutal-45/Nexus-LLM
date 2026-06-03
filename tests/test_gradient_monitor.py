"""Tests for gradient monitoring."""
import pytest
import torch


def test_gradient_norm_calculation():
    params = [torch.randn(10, requires_grad=True) for _ in range(3)]
    total_norm = sum(p.grad.norm().item() if p.grad is not None else 0.0 for p in params)
    assert total_norm == 0.0  # No gradients yet

def test_gradient_clipping():
    grad = torch.tensor([10.0, 20.0, 30.0])
    max_norm = 1.0
    norm = grad.norm().item()
    clipped = grad * (max_norm / max(norm, max_norm))
    assert clipped.norm().item() <= max_norm + 1e-5

def test_gradient_zero():
    param = torch.zeros(5, requires_grad=True)
    assert param.grad is None

def test_gradient_nan_detection():
    values = [1.0, 2.0, float("nan"), 3.0]
    has_nan = any(math.isnan(v) for v in values) if __import__("math") else False
    import math
    has_nan = any(math.isnan(v) for v in values)
    assert has_nan is True
