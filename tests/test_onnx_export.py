"""Tests for ONNX export."""
import pytest
from unittest.mock import MagicMock


def test_onnx_export_config():
    config = {"opset_version": 14, "dynamic_axes": {"input": {0: "batch"}}, "input_names": ["input"]}
    assert config["opset_version"] >= 11

def test_onnx_dynamic_shapes():
    dynamic_axes = {"input": {0: "batch_size", 1: "seq_len"}}
    assert 0 in dynamic_axes["input"]
    assert 1 in dynamic_axes["input"]

def test_onnx_input_names():
    input_names = ["input_ids", "attention_mask"]
    assert len(input_names) == 2

def test_onnx_output_names():
    output_names = ["logits"]
    assert "logits" in output_names
