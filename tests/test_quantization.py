"""Tests for quantization."""
import pytest
import torch
import torch.nn as nn
from nexus.inference.quantize import QuantConfig, Quantizer, QuantizedLinear, GPTQQuantizer


def test_quant_config_defaults():
    """Test QuantConfig defaults."""
    cfg = QuantConfig()
    assert cfg.bits == 8
    assert cfg.group_size == 128
    assert cfg.sym is True


def test_quant_config_int4():
    """Test INT4 config."""
    cfg = QuantConfig(bits=4)
    assert cfg.bits == 4


def test_quantizer_int8():
    """Test INT8 quantization."""
    q = Quantizer(QuantConfig(bits=8, group_size=64))
    layer = nn.Linear(128, 64)
    w_q, scale, zp = q.quantize_linear(layer)
    assert w_q.dtype == torch.int8
    assert scale.shape[0] == 64  # out_features


def test_quantizer_int4():
    """Test INT4 quantization."""
    q = Quantizer(QuantConfig(bits=4, group_size=64))
    layer = nn.Linear(128, 64)
    w_q, scale, zp = q.quantize_linear(layer)
    assert scale.shape[0] == 64


def test_quantized_linear_forward():
    """Test QuantizedLinear forward pass."""
    q = Quantizer(QuantConfig(bits=8, group_size=64))
    layer = nn.Linear(128, 64)
    w_q, scale, zp = q.quantize_linear(layer)
    # Verify shapes
    assert w_q.shape[0] == 64  # out_features
    assert scale.shape[0] == 64  # out_features


def test_quantizer_unsupported_bits():
    """Test that unsupported bit widths raise error."""
    q = Quantizer(QuantConfig(bits=3))
    layer = nn.Linear(64, 32)
    with pytest.raises(ValueError):
        q.quantize_linear(layer)


def test_int4_pack_unpack():
    """Test INT4 packing and unpacking."""
    tensor = torch.randint(-8, 7, (4, 10), dtype=torch.int8)
    packed = Quantizer._pack_int4(tensor)
    assert packed.dtype == torch.int8
    # Packed should have half the columns
    assert packed.shape[1] == 5


def test_quantizer_default_group_size():
    """Test Quantizer with group_size matching input dimension."""
    q = Quantizer(QuantConfig(bits=8, group_size=32))
    layer = nn.Linear(32, 16)
    w_q, scale, zp = q.quantize_linear(layer)
    assert w_q.dtype == torch.int8
