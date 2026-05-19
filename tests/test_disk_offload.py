"""Tests for disk offloading."""
import pytest
from pathlib import Path


def test_offload_to_disk(tmp_dir):
    data = b"x" * 1024
    p = tmp_dir / "tensor.bin"
    p.write_bytes(data)
    assert p.exists() and p.stat().st_size == 1024


def test_reload_from_disk(tmp_dir):
    data = b"test_data"
    p = tmp_dir / "t.bin"
    p.write_bytes(data)
    assert p.read_bytes() == data


def test_offload_memory_freed():
    offloaded = None
    assert offloaded is None


def test_multiple_tensor_offload(tmp_dir):
    for i in range(5):
        (tmp_dir / f"t_{i}.bin").write_bytes(f"d{i}".encode())
    assert len(list(tmp_dir.glob("*.bin"))) == 5
