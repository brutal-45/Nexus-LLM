"""Tests for checkpointing."""
import pytest
import os
import json
import torch
import torch.nn as nn
from nexus.training.checkpoint import CheckpointManager, CheckpointState


@pytest.fixture
def checkpoint_dir(tmp_dir):
    return os.path.join(tmp_dir, "checkpoints")


@pytest.fixture
def simple_model():
    return nn.Linear(32, 32)


def test_checkpoint_state_defaults():
    """Test CheckpointState defaults."""
    state = CheckpointState()
    assert state.step == 0
    assert state.epoch == 0
    assert state.best_metric == float("inf")
    assert state.metrics == {}


def test_checkpoint_manager_creation(checkpoint_dir):
    """Test creating a CheckpointManager."""
    mgr = CheckpointManager(checkpoint_dir=checkpoint_dir)
    assert os.path.isdir(checkpoint_dir)


def test_checkpoint_should_save(checkpoint_dir):
    """Test should_save logic."""
    mgr = CheckpointManager(checkpoint_dir=checkpoint_dir, save_interval=100)
    assert mgr.should_save(0) is False
    assert mgr.should_save(100) is True
    assert mgr.should_save(50) is False


def test_checkpoint_save_and_load(checkpoint_dir, simple_model):
    """Test saving and loading a checkpoint."""
    mgr = CheckpointManager(
        checkpoint_dir=checkpoint_dir, save_interval=100, save_safetensors=False
    )
    path = mgr.save(model=simple_model, step=100, epoch=1, metrics={"loss": 2.5})
    assert os.path.isdir(path)
    # Check metadata
    metadata_path = os.path.join(path, "metadata.json")
    assert os.path.exists(metadata_path)
    with open(metadata_path) as f:
        meta = json.load(f)
    assert meta["step"] == 100
    assert meta["epoch"] == 1


def test_checkpoint_best_tracking(checkpoint_dir, simple_model):
    """Test best checkpoint tracking."""
    mgr = CheckpointManager(
        checkpoint_dir=checkpoint_dir, save_interval=100, save_safetensors=False, metric_mode="min"
    )
    mgr.save(model=simple_model, step=100, metrics={"eval_loss": 3.0})
    assert mgr.best_metric == 3.0
    mgr.save(model=simple_model, step=200, metrics={"eval_loss": 2.0})
    assert mgr.best_metric == 2.0
