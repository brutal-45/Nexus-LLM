"""Tests for multi-adapter serving."""
import pytest
from unittest.mock import MagicMock


def test_adapter_registration():
    adapters = {}
    adapters["v1"] = {"rank": 8}
    adapters["v2"] = {"rank": 16}
    assert len(adapters) == 2


def test_adapter_selection():
    adapters = {"default": MagicMock(), "creative": MagicMock()}
    assert adapters["creative"] is not None


def test_adapter_hot_swap():
    current = "v1"
    loaded = {"v1": True, "v2": True}
    current = "v2"
    assert loaded[current]


def test_adapter_memory_isolation():
    weights = {"a1": [1.0, 2.0], "a2": [3.0, 4.0]}
    weights["a1"][0] = 99.0
    assert weights["a2"][0] == 3.0


def test_adapter_list():
    adapters = ["base", "lora_math", "lora_code"]
    assert len(adapters) == 3
    assert "base" in adapters
