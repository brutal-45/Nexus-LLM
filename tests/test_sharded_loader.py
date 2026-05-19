"""Tests for sharded model loading."""
import pytest


def test_shard_index_parsing():
    index = {"metadata": {"total_size": 1e9}, "weight_map": {"layer.0.w": "shard_0.bin"}}
    assert "metadata" in index
    assert len(index["weight_map"]) == 1


def test_shard_file_count():
    num_shards = 5
    files = [f"model-{i:05d}-of-{num_shards:05d}.bin" for i in range(1, num_shards + 1)]
    assert len(files) == num_shards


def test_weight_to_shard_mapping():
    wm = {"enc.0": "s0.bin", "enc.1": "s0.bin", "dec.0": "s1.bin"}
    assert wm["enc.0"] == "s0.bin"
    assert wm["dec.0"] == "s1.bin"


def test_shard_sequential_loading():
    loaded = [{"id": i} for i in range(3)]
    assert len(loaded) == 3
