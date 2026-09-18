"""Tests for model loading, unloading, and switching."""
import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock


class SimpleModel(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.dim = dim

    def forward(self, x):
        return self.linear(x)


class ModelManager:
    """Simple model manager for testing."""
    def __init__(self):
        self.models = {}
        self.active_model = None

    def load(self, name, model):
        self.models[name] = model
        if self.active_model is None:
            self.active_model = name

    def unload(self, name):
        if name in self.models:
            del self.models[name]
        if self.active_model == name:
            self.active_model = None

    def switch(self, name):
        if name not in self.models:
            raise KeyError(f"Model {name} not loaded")
        self.active_model = name

    def get_active(self):
        if self.active_model is None:
            return None
        return self.models.get(self.active_model)


@pytest.fixture
def manager():
    return ModelManager()


@pytest.fixture
def model_a():
    return SimpleModel(dim=64)


@pytest.fixture
def model_b():
    return SimpleModel(dim=128)


def test_model_manager_load(manager, model_a):
    """Test loading a model into the manager."""
    manager.load("model_a", model_a)
    assert "model_a" in manager.models
    assert manager.active_model == "model_a"


def test_model_manager_unload(manager, model_a):
    """Test unloading a model from the manager."""
    manager.load("model_a", model_a)
    manager.unload("model_a")
    assert "model_a" not in manager.models
    assert manager.active_model is None


def test_model_manager_switch(manager, model_a, model_b):
    """Test switching between loaded models."""
    manager.load("model_a", model_a)
    manager.load("model_b", model_b)
    assert manager.active_model == "model_a"
    manager.switch("model_b")
    assert manager.active_model == "model_b"
    active = manager.get_active()
    assert active is model_b


def test_model_manager_switch_nonexistent(manager):
    """Test switching to a non-existent model raises error."""
    with pytest.raises(KeyError):
        manager.switch("nonexistent")


def test_model_manager_multiple_load(manager, model_a, model_b):
    """Test loading multiple models simultaneously."""
    manager.load("a", model_a)
    manager.load("b", model_b)
    assert len(manager.models) == 2
    assert manager.get_active() is model_a
