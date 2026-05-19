"""Tests for model export."""
import pytest
import os
import torch
import torch.nn as nn
import json


class SimpleExportModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 16)

    def forward(self, x):
        return self.linear(x)


def test_export_state_dict(tmp_dir):
    """Test exporting model state dict."""
    model = SimpleExportModel()
    path = os.path.join(tmp_dir, "model.pt")
    torch.save(model.state_dict(), path)
    assert os.path.exists(path)
    assert os.path.getsize(path) > 0


def test_export_load_state_dict(tmp_dir):
    """Test loading exported state dict."""
    model = SimpleExportModel()
    path = os.path.join(tmp_dir, "model.pt")
    torch.save(model.state_dict(), path)
    
    new_model = SimpleExportModel()
    state_dict = torch.load(path, map_location="cpu")
    new_model.load_state_dict(state_dict)
    
    # Verify weights match
    for (n1, p1), (n2, p2) in zip(model.named_parameters(), new_model.named_parameters()):
        assert torch.equal(p1, p2)


def test_export_config_json(tmp_dir):
    """Test exporting config as JSON."""
    config = {"hidden_size": 256, "num_layers": 4, "vocab_size": 32000}
    path = os.path.join(tmp_dir, "config.json")
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    
    with open(path) as f:
        loaded = json.load(f)
    assert loaded["hidden_size"] == 256
    assert loaded["num_layers"] == 4


def test_export_torchscript(tmp_dir):
    """Test exporting model via TorchScript."""
    model = SimpleExportModel()
    model.eval()
    x = torch.randn(1, 16)
    scripted = torch.jit.trace(model, x)
    
    path = os.path.join(tmp_dir, "model_scripted.pt")
    scripted.save(path)
    assert os.path.exists(path)


def test_export_onnx_format(tmp_dir):
    """Test ONNX-style export (using state_dict as proxy)."""
    model = SimpleExportModel()
    path = os.path.join(tmp_dir, "model.onnx.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_shape": [1, 16],
        "output_shape": [1, 16],
    }, path)
    
    data = torch.load(path, map_location="cpu")
    assert "model_state_dict" in data
    assert data["input_shape"] == [1, 16]
