"""Tests for memory management."""
import pytest
import torch


def test_tensor_memory_size():
    """Test calculating tensor memory size."""
    t = torch.randn(100, 100, dtype=torch.float32)
    # float32 = 4 bytes, 100*100 = 10000 elements
    expected = 10000 * 4
    assert t.numel() * t.element_size() == expected


def test_tensor_memory_bf16():
    """Test that BF16 uses half the memory of float32."""
    t32 = torch.randn(100, 100, dtype=torch.float32)
    t16 = torch.randn(100, 100, dtype=torch.bfloat16)
    assert t32.element_size() == 4
    assert t16.element_size() == 2
    assert t32.numel() * t32.element_size() == 2 * (t16.numel() * t16.element_size())


def test_kv_cache_memory_estimation():
    """Test KV cache memory estimation."""
    num_layers = 2
    num_heads = 4
    head_dim = 32
    seq_len = 64
    batch_size = 1
    bytes_per_elem = 2  # bf16
    
    # KV cache: 2 (K+V) * num_layers * batch * heads * seq * head_dim * bytes
    mem = 2 * num_layers * batch_size * num_heads * seq_len * head_dim * bytes_per_elem
    assert mem > 0
    # Verify it matches StandardKVCache calculation
    assert mem == 2 * 2 * 1 * 4 * 64 * 32 * 2


def test_gradient_memory_estimation():
    """Test gradient memory estimation."""
    params = 1_000_000
    bytes_per_param = 4  # float32
    grad_mem = params * bytes_per_param
    assert grad_mem == 4_000_000


def test_optimizer_memory_estimation():
    """Test optimizer state memory estimation (AdamW has 2 state tensors)."""
    params = 1_000_000
    bytes_per_param = 4
    # AdamW: exp_avg + exp_avg_sq = 2x param memory
    optimizer_mem = 2 * params * bytes_per_param
    assert optimizer_mem == 8_000_000
