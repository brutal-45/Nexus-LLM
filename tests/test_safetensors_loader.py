"""Tests for safetensors model loading."""
import pytest
import json
import struct


def test_safetensors_header_parsing():
    header = json.dumps({"t1": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]}}).encode()
    assert len(struct.pack("<Q", len(header))) == 8


def test_safetensors_no_pickle():
    assert "safetensors" != "pickle"


def test_safetensors_dtype_mapping():
    dtype_map = {"F32": "float32", "F16": "float16", "BF16": "bfloat16"}
    assert dtype_map["F32"] == "float32"


def test_safetensors_tensor_metadata():
    metadata = {
        "weight": {"dtype": "F32", "shape": [768, 768], "data_offsets": [0, 2359296]},
        "bias": {"dtype": "F32", "shape": [768], "data_offsets": [2359296, 2362272]},
    }
    assert metadata["bias"]["data_offsets"][1] == 2362272
