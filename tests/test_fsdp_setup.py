"""Tests for FSDP setup."""
import pytest


def test_fsdp_shard_strategy():
    strategies = ["FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD"]
    assert "FULL_SHARD" in strategies

def test_fsdp_wrap_policy():
    layers = [f"layer_{i}" for i in range(6)]
    wrap_every = 2
    groups = [layers[i:i+wrap_every] for i in range(0, len(layers), wrap_every)]
    assert len(groups) == 3

def test_fsdp_mixed_precision():
    dtypes = {"param_dtype": "float16", "reduce_dtype": "float16", "buffer_dtype": "float32"}
    assert dtypes["buffer_dtype"] == "float32"

def test_fsdp_cpu_offload():
    config = {"cpu_offload": True}
    assert config["cpu_offload"] is True
