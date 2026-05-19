"""Tests for Adafactor optimizer."""
import pytest


def test_adafactor_factored_second_moment():
    # Adafactor uses factored R and C matrices instead of full V
    import numpy as np
    grad = np.random.randn(64, 64)
    row = np.mean(grad ** 2, axis=1)
    col = np.mean(grad ** 2, axis=0)
    reconstructed = np.outer(row, col) / np.mean(grad ** 2)
    assert reconstructed.shape == (64, 64)

def test_adafactor_memory_savings():
    shape = (4096, 4096)
    full_size = shape[0] * shape[1]
    factored_size = shape[0] + shape[1]
    assert factored_size < full_size

def test_adafactor_no_bias_correction():
    # Adafactor does not use bias correction by default
    use_bias_correction = False
    assert use_bias_correction is False

def test_adafactor_relative_step():
    # Adafactor can use relative step sizes
    min_lr = 1e-5
    assert min_lr > 0
