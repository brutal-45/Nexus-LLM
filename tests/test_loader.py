"""Tests for model loader."""
import pytest
import os
import torch
import torch.nn as nn
from nexus.model.config import ModelConfig


class SimpleModel(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.config = ModelConfig(hidden_size=dim, num_attention_heads=4, num_key_value_heads=2)

    def forward(self, x):
        return self.linear(x)


def test_model_save_load(tmp_dir):
    """Test saving and loading model weights."""
    model = SimpleModel()
    path = os.path.join(tmp_dir, "model.pt")
    torch.save(model.state_dict(), path)
    assert os.path.exists(path)
    
    loaded_state = torch.load(path, map_location="cpu")
    new_model = SimpleModel()
    new_model.load_state_dict(loaded_state)
    
    # Verify weights match
    for (n1, p1), (n2, p2) in zip(model.named_parameters(), new_model.named_parameters()):
        assert torch.equal(p1, p2)


def test_model_config_save_load(tmp_dir):
    """Test saving and loading model config."""
    cfg = ModelConfig(name="Test", hidden_size=128, num_attention_heads=4, num_key_value_heads=2)
    path = os.path.join(tmp_dir, "config.yaml")
    cfg.save_yaml(path)
    
    loaded = ModelConfig.from_yaml(path)
    assert loaded.hidden_size == 128
    assert loaded.name == "Test"


def test_model_parameter_count():
    """Test model parameter counting."""
    model = SimpleModel(dim=64)
    total = sum(p.numel() for p in model.parameters())
    assert total > 0
    # Linear(64, 64): weight=64*64=4096, bias=64
    assert total == 4096 + 64
