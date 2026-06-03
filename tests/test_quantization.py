"""Tests for the quantization module.

Covers Quantizer, QuantConfig, and QuantizationBenchmark.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from nexus_llm.quantization.quantizer import Quantizer
from nexus_llm.quantization.config import QuantConfig
from nexus_llm.quantization.benchmark import QuantizationBenchmark


# ---------------------------------------------------------------------------
# QuantConfig
# ---------------------------------------------------------------------------

class TestQuantConfig:
    """Tests for QuantConfig."""

    def test_defaults(self):
        config = QuantConfig()
        assert config is not None

    def test_custom_config(self):
        config = QuantConfig(bits=4, method="gptq")
        assert config.bits == 4
        assert config.method == "gptq"

    def test_to_dict(self):
        config = QuantConfig(bits=8)
        d = config.to_dict()
        assert isinstance(d, dict)

    def test_from_dict(self):
        data = {"bits": 4, "method": "awq"}
        config = QuantConfig.from_dict(data)
        assert config.bits == 4


# ---------------------------------------------------------------------------
# Quantizer
# ---------------------------------------------------------------------------

class TestQuantizer:
    """Tests for Quantizer."""

    def test_create_quantizer(self):
        q = Quantizer()
        assert q is not None

    def test_quantize_with_config(self):
        config = QuantConfig(bits=8)
        q = Quantizer(config=config)
        assert q.config is not None

    def test_get_info(self):
        q = Quantizer()
        info = q.get_info()
        assert isinstance(info, dict)


# ---------------------------------------------------------------------------
# QuantizationBenchmark
# ---------------------------------------------------------------------------

class TestQuantizationBenchmark:
    """Tests for QuantizationBenchmark."""

    def test_create_benchmark(self):
        bench = QuantizationBenchmark()
        assert bench is not None

    def test_run_benchmark(self):
        bench = QuantizationBenchmark()
        # Should have a run method
        assert hasattr(bench, "run") or hasattr(bench, "benchmark")

    def test_compare_quantizations(self):
        bench = QuantizationBenchmark()
        configs = [QuantConfig(bits=8), QuantConfig(bits=4)]
        # Compare should work without actual models
        assert hasattr(bench, "compare") or hasattr(bench, "run")
