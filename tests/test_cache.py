"""Tests for KV cache."""
import pytest
import torch
from nexus.inference.kv_cache import StandardKVCache, KVCacheQuantizer, SlidingWindowKVCache, CrossLayerKVCache


@pytest.fixture
def standard_cache():
    return StandardKVCache(
        num_layers=2, num_heads=4, head_dim=32,
        max_seq_len=64, batch_size=1, device="cpu", pin_memory=False,
    )


def test_standard_cache_creation(standard_cache):
    """Test creating a StandardKVCache."""
    assert standard_cache.num_layers == 2
    assert standard_cache.num_heads == 4
    assert standard_cache.max_seq_len == 64


def test_standard_cache_update_and_get(standard_cache):
    """Test updating and retrieving from cache."""
    new_k = torch.randn(1, 4, 3, 32)
    new_v = torch.randn(1, 4, 3, 32)
    standard_cache.update(0, new_k, new_v)
    k, v = standard_cache.get(0, seq_len=3)
    assert k.shape == (1, 4, 3, 32)
    assert v.shape == (1, 4, 3, 32)


def test_standard_cache_current_seq_len(standard_cache):
    """Test current_seq_len tracking."""
    assert standard_cache.current_seq_len == 0
    new_k = torch.randn(1, 4, 5, 32)
    new_v = torch.randn(1, 4, 5, 32)
    standard_cache.update(0, new_k, new_v)
    assert standard_cache.current_seq_len == 5


def test_standard_cache_clear(standard_cache):
    """Test clearing the cache."""
    new_k = torch.randn(1, 4, 3, 32)
    new_v = torch.randn(1, 4, 3, 32)
    standard_cache.update(0, new_k, new_v)
    standard_cache.clear()
    assert standard_cache.current_seq_len == 0


def test_standard_cache_memory_usage(standard_cache):
    """Test memory usage calculation."""
    mem = standard_cache.memory_usage_bytes()
    assert mem > 0
    expected = 2 * 2 * 1 * 4 * 64 * 32 * 2  # bf16 = 2 bytes
    assert mem == expected


def test_standard_cache_copy(standard_cache):
    """Test deep copy for beam search."""
    new_k = torch.randn(1, 4, 3, 32)
    new_v = torch.randn(1, 4, 3, 32)
    standard_cache.update(0, new_k, new_v)
    copied = standard_cache.copy()
    assert copied.current_seq_len == standard_cache.current_seq_len


def test_standard_cache_attn_mask(standard_cache):
    """Test attention mask generation."""
    mask = standard_cache.get_attn_mask(5, dtype=torch.float32)
    assert mask.shape == (1, 1, 5, 5)
    # Causal: lower triangle is True
    assert mask[0, 0, 0, 0] == True
    assert mask[0, 0, 0, 4] == False  # Cannot attend to future


def test_standard_cache_overflow():
    """Test that cache overflow raises ValueError."""
    cache = StandardKVCache(
        num_layers=1, num_heads=2, head_dim=8,
        max_seq_len=4, batch_size=1, device="cpu",
    )
    new_k = torch.randn(1, 2, 4, 8)
    new_v = torch.randn(1, 2, 4, 8)
    cache.update(0, new_k, new_v)
    # Try to write beyond max_seq_len
    with pytest.raises(ValueError):
        cache.update(0, torch.randn(1, 2, 1, 8), torch.randn(1, 2, 1, 8))


def test_sliding_window_cache_creation():
    """Test creating a SlidingWindowKVCache."""
    cache = SlidingWindowKVCache(
        num_layers=2, num_heads=4, head_dim=32,
        window_size=16, sink_size=4, batch_size=1, device="cpu",
    )
    assert cache.window_size == 16
    assert cache.sink_size == 4
    assert cache.effective_size == 20


def test_sliding_window_cache_memory():
    """Test SlidingWindowKVCache memory usage."""
    cache = SlidingWindowKVCache(
        num_layers=2, num_heads=4, head_dim=32,
        window_size=16, sink_size=4, batch_size=1, device="cpu",
    )
    mem = cache.memory_usage_bytes()
    assert mem > 0


def test_cross_layer_cache_creation():
    """Test creating a CrossLayerKVCache."""
    cache = CrossLayerKVCache(
        num_layers=8, num_heads=4, head_dim=32,
        max_seq_len=64, batch_size=1, sharing_strategy="uniform",
        share_ratio=4, device="cpu",
    )
    assert len(cache.stored_layers) == 2  # 8 / 4 = 2


def test_cross_layer_cache_memory_savings():
    """Test CrossLayerKVCache memory savings ratio."""
    cache = CrossLayerKVCache(
        num_layers=8, num_heads=4, head_dim=32,
        max_seq_len=64, batch_size=1, sharing_strategy="uniform",
        share_ratio=4, device="cpu",
    )
    savings = cache.memory_savings_ratio()
    assert savings == pytest.approx(0.75)  # 6/8 = 75% savings


def test_kv_cache_quantizer_int8():
    """Test KV cache quantizer with INT8."""
    q = KVCacheQuantizer(quant_type="int8")
    x = torch.randn(2, 4, 32)
    x_q, scale = q.quantize(x)
    x_dq = q.dequantize(x_q, scale)
    assert x_dq.shape == x.shape
    # Should be approximately the same
    assert torch.nn.functional.cosine_similarity(
        x.flatten().unsqueeze(0), x_dq.flatten().unsqueeze(0)
    ).item() > 0.5
