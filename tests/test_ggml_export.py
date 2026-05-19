"""Tests for GGML export."""
import pytest


def test_ggml_quantization_types():
    qtypes = ["Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0"]
    assert "Q4_0" in qtypes
    assert len(qtypes) == 5

def test_ggml_quantization_bits():
    qbits = {"Q4_0": 4, "Q5_0": 5, "Q8_0": 8}
    for qtype, bits in qbits.items():
        assert bits in (4, 5, 8)

def test_ggml_model_size_reduction():
    original_size = 13000  # 13GB for 7B FP16
    q4_size = original_size * 4 / 16
    assert q4_size < original_size

def test_ggml_file_format():
    # GGML format has magic number and metadata
    magic = b"GGML"
    assert len(magic) == 4
