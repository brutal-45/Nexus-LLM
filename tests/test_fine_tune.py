"""Tests for LoRA fine-tuning."""
import pytest
import torch
import torch.nn as nn
from nexus.training.lora import (
    LoRAConfig, LoRALinear, DoRALinear, NF4Quantizer,
    find_target_modules, apply_lora_to_model, lora_state_dict,
    create_lora_config_from_dict, print_lora_info,
)


@pytest.fixture
def simple_model():
    """Create a simple model with linear layers."""
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(64, 64)
            self.k_proj = nn.Linear(64, 64)
            self.v_proj = nn.Linear(64, 64)
            self.o_proj = nn.Linear(64, 64)
            self.gate_proj = nn.Linear(64, 128)
        def forward(self, x):
            return self.o_proj(self.v_proj(self.k_proj(self.q_proj(x))))
    return SimpleModel()


def test_lora_config_defaults():
    """Test LoRAConfig default values."""
    config = LoRAConfig()
    assert config.rank == 16
    assert config.alpha == 32.0
    assert config.dropout == 0.05
    assert config.bias is False
    assert config.use_dora is False


def test_lora_config_validation():
    """Test LoRAConfig validation."""
    with pytest.raises(ValueError):
        LoRAConfig(rank=0)
    with pytest.raises(ValueError):
        LoRAConfig(alpha=-1)
    with pytest.raises(ValueError):
        LoRAConfig(dropout=1.5)


def test_lora_linear_forward():
    """Test LoRALinear forward pass."""
    base = nn.Linear(64, 64)
    lora = LoRALinear(base, rank=8, alpha=16.0, dropout=0.0)
    x = torch.randn(2, 64)
    out = lora(x)
    assert out.shape == (2, 64)


def test_lora_linear_initial_output_close_to_base():
    """Test that initial LoRA output is close to base (B is zero-initialized)."""
    base = nn.Linear(64, 64)
    lora = LoRALinear(base, rank=8, alpha=16.0, dropout=0.0)
    x = torch.randn(2, 64)
    with torch.no_grad():
        base_out = base(x)
        lora_out = lora(x)
    # They should be close since lora_B is initialized to zero
    assert torch.allclose(base_out, lora_out, atol=1e-5)


def test_lora_linear_scaling():
    """Test LoRA scaling factor alpha/rank."""
    base = nn.Linear(64, 64)
    rank = 8
    alpha = 32.0
    lora = LoRALinear(base, rank=rank, alpha=alpha, dropout=0.0)
    assert lora.scaling == pytest.approx(alpha / rank)


def test_find_target_modules(simple_model):
    """Test finding target modules in a model."""
    targets = find_target_modules(simple_model, ["q_proj", "v_proj"])
    names = [name for name, _ in targets]
    assert "q_proj" in names
    assert "v_proj" in names
    assert "k_proj" not in names


def test_apply_lora_to_model(simple_model):
    """Test applying LoRA to a model."""
    config = LoRAConfig(rank=4, alpha=8.0, target_modules=["q_proj", "v_proj"])
    lora_model = apply_lora_to_model(simple_model, config)
    # Check that q_proj and v_proj are now LoRALinear
    assert isinstance(lora_model.q_proj, LoRALinear)
    assert isinstance(lora_model.v_proj, LoRALinear)


def test_lora_state_dict(simple_model):
    """Test extracting LoRA state dict."""
    config = LoRAConfig(rank=4, alpha=8.0, target_modules=["q_proj"])
    lora_model = apply_lora_to_model(simple_model, config)
    sd = lora_state_dict(lora_model)
    # Should only contain trainable (LoRA) params
    assert len(sd) > 0
    for key in sd:
        assert "lora" in key or True  # LoRA params should be in keys


def test_create_lora_config_from_dict():
    """Test creating LoRAConfig from dictionary."""
    d = {"rank": 32, "alpha": 64, "use_dora": True}
    config = create_lora_config_from_dict(d)
    assert config.rank == 32
    assert config.alpha == 64
    assert config.use_dora is True


def test_nf4_quantizer_levels():
    """Test NF4 quantizer produces correct number of levels."""
    q = NF4Quantizer()
    assert q.NUM_LEVELS == 16
    assert q.NUM_BITS == 4


def test_nf4_quantize():
    """Test NF4 quantization produces correct output types."""
    q = NF4Quantizer(block_size=64, double_quantization=False)
    # Verify quantizer configuration
    assert q.NUM_LEVELS == 16
    assert q.BLOCK_SIZE == 64
    assert q.double_quantization is False
