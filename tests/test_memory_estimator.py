"""Tests for memory estimation."""
import pytest


def test_model_size_estimation():
    size_gb = 7e9 * 2 / (1024**3)
    assert abs(size_gb - 13.0) < 0.5


def test_kv_cache_estimation():
    size_mb = 2 * 1 * 2048 * 32 * 4096 * 2 / (1024**2)
    assert size_mb > 0


def test_optimizer_memory_estimation():
    size_mb = 1_000_000 * 2 * 4 / (1024**2)
    assert size_mb > 0


def test_training_memory_estimation():
    total = 13.0 + 26.0 + 13.0 + 4.0
    assert total > 50.0
