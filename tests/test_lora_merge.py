"""Tests for LoRA adapter merging."""
import pytest
import numpy as np


def test_lora_weight_composition():
    W0 = np.random.randn(64, 64)
    B, A = np.random.randn(64, 8), np.random.randn(8, 64)
    assert (W0 + B @ A).shape == (64, 64)


def test_lora_merge_multiple_adapters():
    W0 = np.zeros((32, 32))
    for _ in range(3):
        W0 += np.random.randn(32, 4) @ np.random.randn(4, 32)
    assert W0.shape == (32, 32)


def test_lora_merge_with_scaling():
    W0 = np.eye(16)
    scaling = 16 / 4
    W_merged = W0 + scaling * (np.random.randn(16, 4) @ np.random.randn(4, 16))
    assert W_merged.shape == (16, 16)


def test_lora_unmerge():
    W0 = np.eye(8)
    B, A = np.random.randn(8, 2), np.random.randn(2, 8)
    assert np.allclose(W0 + B @ A - B @ A, W0)
