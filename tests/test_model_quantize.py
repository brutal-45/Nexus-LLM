"""Test model quantization for Nexus-LLM."""
import pytest
from dataclasses import dataclass
from typing import Dict, Any, List, Optional


class QuantizationError(Exception):
    pass


SUPPORTED_METHODS = ("int8", "int4", "gptq", "awq", "fp8", "fp4", "gguf_q4", "gguf_q8")


@dataclass
class QuantizationConfig:
    method: str = "int8"
    group_size: int = 128
    bits: int = 8
    calibration_dataset: str = ""
    optimize: bool = True

    def __post_init__(self):
        if self.method not in SUPPORTED_METHODS:
            raise QuantizationError(f"Unsupported method: {self.method}")


@dataclass
class QuantizationResult:
    original_size_mb: float
    quantized_size_mb: float
    method: str
    bits: int
    success: bool

    @property
    def compression_ratio(self) -> float:
        if self.quantized_size_mb == 0:
            return float("inf")
        return self.original_size_mb / self.quantized_size_mb

    @property
    def size_reduction_percent(self) -> float:
        if self.original_size_mb == 0:
            return 0.0
        return (1 - self.quantized_size_mb / self.original_size_mb) * 100


class ModelQuantizer:
    def __init__(self, config: QuantizationConfig = None):
        self._config = config or QuantizationConfig()

    @property
    def config(self):
        return self._config

    def estimate_quantized_size(self, original_size_mb: float, bits: int = None) -> float:
        target_bits = bits or self._config.bits
        ratio = target_bits / 32  # relative to fp32
        return original_size_mb * ratio

    def validate_method(self, method: str) -> bool:
        return method in SUPPORTED_METHODS

    def get_supported_methods(self) -> List[str]:
        return list(SUPPORTED_METHODS)

    def quantize(self, model_size_mb: float, method: str = None) -> QuantizationResult:
        target_method = method or self._config.method
        if not self.validate_method(target_method):
            raise QuantizationError(f"Unsupported method: {target_method}")

        bits_map = {"int8": 8, "int4": 4, "gptq": 4, "awq": 4, "fp8": 8, "fp4": 4, "gguf_q4": 4, "gguf_q8": 8}
        bits = bits_map.get(target_method, 8)
        quantized_size = self.estimate_quantized_size(model_size_mb, bits)

        return QuantizationResult(
            original_size_mb=model_size_mb,
            quantized_size_mb=quantized_size,
            method=target_method,
            bits=bits,
            success=True,
        )

    def compare_methods(self, model_size_mb: float) -> List[Dict[str, Any]]:
        results = []
        for method in SUPPORTED_METHODS:
            result = self.quantize(model_size_mb, method)
            results.append({
                "method": method,
                "quantized_size_mb": result.quantized_size_mb,
                "compression_ratio": result.compression_ratio,
                "size_reduction_percent": result.size_reduction_percent,
            })
        return results


class TestQuantizationConfig:
    def test_defaults(self):
        config = QuantizationConfig()
        assert config.method == "int8"
        assert config.group_size == 128
        assert config.bits == 8

    def test_custom(self):
        config = QuantizationConfig(method="gptq", bits=4)
        assert config.method == "gptq"

    def test_invalid_method(self):
        with pytest.raises(QuantizationError, match="Unsupported"):
            QuantizationConfig(method="invalid")


class TestQuantizationResult:
    def test_compression_ratio(self):
        result = QuantizationResult(original_size_mb=100, quantized_size_mb=25, method="int4", bits=4, success=True)
        assert result.compression_ratio == 4.0

    def test_size_reduction_percent(self):
        result = QuantizationResult(original_size_mb=100, quantized_size_mb=25, method="int4", bits=4, success=True)
        assert abs(result.size_reduction_percent - 75.0) < 0.01

    def test_zero_quantized_size(self):
        result = QuantizationResult(original_size_mb=100, quantized_size_mb=0, method="int4", bits=4, success=True)
        assert result.compression_ratio == float("inf")


class TestModelQuantizer:
    def test_estimate_quantized_size_int8(self):
        quantizer = ModelQuantizer(QuantizationConfig(method="int8"))
        size = quantizer.estimate_quantized_size(100.0)
        assert abs(size - 25.0) < 0.01  # 8/32 = 0.25

    def test_estimate_quantized_size_int4(self):
        quantizer = ModelQuantizer(QuantizationConfig(method="int4", bits=4))
        size = quantizer.estimate_quantized_size(100.0)
        assert abs(size - 12.5) < 0.01  # 4/32 = 0.125

    def test_validate_method(self):
        quantizer = ModelQuantizer()
        for method in SUPPORTED_METHODS:
            assert quantizer.validate_method(method) is True
        assert quantizer.validate_method("invalid") is False

    def test_get_supported_methods(self):
        quantizer = ModelQuantizer()
        methods = quantizer.get_supported_methods()
        assert "int8" in methods
        assert "gptq" in methods

    def test_quantize_int8(self):
        quantizer = ModelQuantizer(QuantizationConfig(method="int8"))
        result = quantizer.quantize(100.0)
        assert result.success is True
        assert result.method == "int8"
        assert result.bits == 8
        assert result.quantized_size_mb < result.original_size_mb

    def test_quantize_int4(self):
        quantizer = ModelQuantizer(QuantizationConfig(method="int4", bits=4))
        result = quantizer.quantize(100.0)
        assert result.bits == 4
        assert result.compression_ratio > 1.0

    def test_quantize_unsupported(self):
        quantizer = ModelQuantizer(QuantizationConfig(method="int8"))
        with pytest.raises(QuantizationError):
            quantizer.quantize(100.0, method="invalid")

    def test_compare_methods(self):
        quantizer = ModelQuantizer(QuantizationConfig(method="int8"))
        comparisons = quantizer.compare_methods(100.0)
        assert len(comparisons) == len(SUPPORTED_METHODS)
        assert all("method" in c for c in comparisons)
        assert all("compression_ratio" in c for c in comparisons)
