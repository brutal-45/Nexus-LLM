"""Tests for CPU offloading."""
import pytest
import torch
import torch.nn as nn


class OffloadManager:
    """Simple CPU offload manager for testing."""
    def __init__(self, device="cpu"):
        self.device = device
        self.offloaded_params = {}

    def offload(self, name, tensor):
        self.offloaded_params[name] = tensor.to("cpu")

    def load(self, name, device=None):
        device = device or self.device
        if name in self.offloaded_params:
            return self.offloaded_params[name].to(device)
        return None

    def is_offloaded(self, name):
        return name in self.offloaded_params


@pytest.fixture
def offload_mgr():
    return OffloadManager()


def test_offload_tensor(offload_mgr):
    """Test offloading a tensor to CPU."""
    t = torch.randn(10, 10)
    offload_mgr.offload("weight", t)
    assert offload_mgr.is_offloaded("weight")


def test_load_tensor(offload_mgr):
    """Test loading a tensor back from CPU."""
    t = torch.randn(5, 5)
    offload_mgr.offload("weight", t)
    loaded = offload_mgr.load("weight")
    assert loaded is not None
    assert torch.equal(loaded, t.cpu())


def test_offload_preserves_values(offload_mgr):
    """Test that offloading preserves tensor values."""
    original = torch.tensor([1.0, 2.0, 3.0])
    offload_mgr.offload("test", original)
    loaded = offload_mgr.load("test")
    assert torch.equal(original.cpu(), loaded)


def test_offload_nonexistent(offload_mgr):
    """Test loading a non-existent tensor returns None."""
    assert offload_mgr.load("nonexistent") is None
