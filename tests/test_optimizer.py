"""Tests for optimizer configs."""
import pytest
import torch
import torch.nn as nn
from nexus.training.optimizers import SGD, Adam, AdamW, LION, LAMB


@pytest.fixture
def simple_model():
    return nn.Linear(16, 16)


def test_sgd_creation(simple_model):
    """Test creating SGD optimizer."""
    opt = SGD(simple_model.parameters(), lr=0.01)
    assert opt is not None


def test_sgd_step(simple_model):
    """Test SGD step updates parameters."""
    opt = SGD(simple_model.parameters(), lr=0.1)
    x = torch.randn(4, 16)
    y = simple_model(x).sum()
    y.backward()
    initial_weight = simple_model.weight.data.clone()
    opt.step()
    assert not torch.equal(simple_model.weight.data, initial_weight)


def test_adam_creation(simple_model):
    """Test creating Adam optimizer."""
    opt = Adam(simple_model.parameters(), lr=1e-3)
    assert opt is not None


def test_adamw_creation(simple_model):
    """Test creating AdamW optimizer."""
    opt = AdamW(simple_model.parameters(), lr=1e-4, weight_decay=0.1)
    assert opt is not None


def test_adamw_step(simple_model):
    """Test AdamW step updates parameters."""
    opt = AdamW(simple_model.parameters(), lr=0.01)
    x = torch.randn(4, 16)
    y = simple_model(x).sum()
    y.backward()
    initial_weight = simple_model.weight.data.clone()
    opt.step()
    assert not torch.equal(simple_model.weight.data, initial_weight)


def test_lion_creation(simple_model):
    """Test creating LION optimizer."""
    opt = LION(simple_model.parameters(), lr=1e-4)
    assert opt is not None


def test_lamb_creation(simple_model):
    """Test creating LAMB optimizer."""
    opt = LAMB(simple_model.parameters(), lr=1e-3)
    assert opt is not None


def test_sgd_invalid_lr():
    """Test that SGD rejects negative learning rate."""
    model = nn.Linear(4, 4)
    with pytest.raises(ValueError):
        SGD(model.parameters(), lr=-0.01)


def test_adamw_invalid_lr():
    """Test that AdamW rejects negative learning rate."""
    model = nn.Linear(4, 4)
    with pytest.raises(ValueError):
        AdamW(model.parameters(), lr=-1e-4)
