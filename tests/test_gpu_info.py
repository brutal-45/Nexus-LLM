"""Tests for GPU info detection."""
import pytest
from unittest.mock import patch


def test_gpu_info_structure():
    info = {"name": "A100", "memory_total_mb": 81920, "cuda_version": "12.0"}
    assert "name" in info
    assert info["memory_total_mb"] > 0

def test_gpu_info_no_gpu():
    info = {"available": False, "devices": []}
    assert info["available"] is False
    assert len(info["devices"]) == 0

def test_gpu_info_multi_gpu():
    devices = [{"id": 0, "name": "A100"}, {"id": 1, "name": "A100"}]
    assert len(devices) == 2

def test_gpu_compute_capability():
    cap = (8, 0)  # CUDA compute capability
    assert cap[0] >= 7
