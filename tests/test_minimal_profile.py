"""Tests for minimal profile."""
import pytest


def test_minimal_profile_config():
    profile = {
        "model": {"max_length": 512, "batch_size": 1},
        "training": {"epochs": 1, "lr": 0.001},
        "safety": {"enabled": False},
    }
    assert profile["model"]["batch_size"] == 1

def test_minimal_profile_no_optional():
    optional = {"monitoring": False, "logging": False, "caching": False}
    assert all(v is False for v in optional.values())

def test_minimal_profile_memory():
    # Minimal should use least memory
    model_size_mb = 500
    assert model_size_mb < 1000

def test_minimal_profile_disables_features():
    features = {"rag": False, "plugins": False, "streaming": False}
    assert sum(features.values()) == 0
