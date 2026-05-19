"""Tests for model adapter."""
import pytest
import torch
import torch.nn as nn
from nexus.training.lora import LoRALinear, LoRAConfig, apply_lora_to_model


class AdapterModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(32, 32)
    def forward(self, x):
        return self.proj(x)


@pytest.fixture
def adapter_model():
    return AdapterModel()


def test_adapter_replaces_linear(adapter_model):
    """Test that LoRA adapter replaces linear layers."""
    config = LoRAConfig(rank=4, alpha=8.0, target_modules=["proj"])
    adapted = apply_lora_to_model(adapter_model, config)
    assert isinstance(adapted.proj, LoRALinear)


def test_adapter_forward_shape(adapter_model):
    """Test that adapted model maintains output shape."""
    config = LoRAConfig(rank=4, alpha=8.0, target_modules=["proj"])
    adapted = apply_lora_to_model(adapter_model, config)
    x = torch.randn(2, 32)
    out = adapted(x)
    assert out.shape == (2, 32)


def test_adapter_base_weights_frozen(adapter_model):
    """Test that base model weights are frozen after adaptation."""
    config = LoRAConfig(rank=4, alpha=8.0, target_modules=["proj"])
    adapted = apply_lora_to_model(adapter_model, config)
    assert adapted.proj.linear.weight.requires_grad is False


def test_adapter_lora_weights_trainable(adapter_model):
    """Test that LoRA weights are trainable."""
    config = LoRAConfig(rank=4, alpha=8.0, target_modules=["proj"])
    adapted = apply_lora_to_model(adapter_model, config)
    assert adapted.proj.lora_A.weight.requires_grad is True
    assert adapted.proj.lora_B.weight.requires_grad is True
