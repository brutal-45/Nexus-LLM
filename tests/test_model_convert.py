"""Test model conversion for Nexus-LLM."""
import pytest
from dataclasses import dataclass
from typing import Optional, Dict, Any


class ConversionError(Exception):
    pass


@dataclass
class ConversionConfig:
    source_format: str = "pytorch"
    target_format: str = "onnx"
    quantize: bool = False
    optimize: bool = True
    output_dir: str = "/tmp/converted"


SUPPORTED_FORMATS = ("pytorch", "onnx", "tensorflow", "gguf", "safetensors")


class ModelConverter:
    def __init__(self, config: ConversionConfig = None):
        self._config = config or ConversionConfig()

    @property
    def config(self):
        return self._config

    def validate_format(self, fmt: str) -> bool:
        return fmt in SUPPORTED_FORMATS

    def validate_conversion(self, source: str, target: str) -> bool:
        if not self.validate_format(source):
            raise ConversionError(f"Unsupported source format: {source}")
        if not self.validate_format(target):
            raise ConversionError(f"Unsupported target format: {target}")
        if source == target:
            raise ConversionError("Source and target formats must differ")
        return True

    def get_conversion_path(self, source: str, target: str) -> list:
        self.validate_conversion(source, target)
        if source == "pytorch" and target == "onnx":
            return ["pytorch", "onnx"]
        if source == "pytorch" and target == "gguf":
            return ["pytorch", "safetensors", "gguf"]
        if source == "tensorflow" and target == "onnx":
            return ["tensorflow", "pytorch", "onnx"]
        return [source, target]

    def estimate_output_size(self, input_size_mb: float, target_format: str) -> float:
        multipliers = {
            "pytorch": 1.0,
            "onnx": 1.1,
            "tensorflow": 1.05,
            "gguf": 0.5,
            "safetensors": 0.95,
        }
        if target_format not in multipliers:
            raise ConversionError(f"Unknown target format: {target_format}")
        return input_size_mb * multipliers[target_format]

    def convert(self, model_path: str, source: str = None, target: str = None) -> dict:
        src = source or self._config.source_format
        tgt = target or self._config.target_format
        self.validate_conversion(src, tgt)
        return {
            "source_format": src,
            "target_format": tgt,
            "input_path": model_path,
            "output_path": f"{model_path}.{tgt}",
            "success": True,
        }


class TestConversionConfig:
    def test_defaults(self):
        config = ConversionConfig()
        assert config.source_format == "pytorch"
        assert config.target_format == "onnx"
        assert config.optimize is True

    def test_custom(self):
        config = ConversionConfig(source_format="pytorch", target_format="gguf", quantize=True)
        assert config.quantize is True


class TestModelConverter:
    def test_validate_format_supported(self):
        conv = ModelConverter()
        for fmt in SUPPORTED_FORMATS:
            assert conv.validate_format(fmt) is True

    def test_validate_format_unsupported(self):
        conv = ModelConverter()
        assert conv.validate_format("unsupported") is False

    def test_validate_conversion_valid(self):
        conv = ModelConverter()
        assert conv.validate_conversion("pytorch", "onnx") is True

    def test_validate_conversion_same_format(self):
        conv = ModelConverter()
        with pytest.raises(ConversionError, match="must differ"):
            conv.validate_conversion("pytorch", "pytorch")

    def test_validate_conversion_unsupported_source(self):
        conv = ModelConverter()
        with pytest.raises(ConversionError, match="source"):
            conv.validate_conversion("unsupported", "onnx")

    def test_get_conversion_path_direct(self):
        conv = ModelConverter()
        path = conv.get_conversion_path("pytorch", "onnx")
        assert path[0] == "pytorch"
        assert path[-1] == "onnx"

    def test_get_conversion_path_multi_step(self):
        conv = ModelConverter()
        path = conv.get_conversion_path("pytorch", "gguf")
        assert len(path) >= 2
        assert "gguf" in path

    def test_estimate_output_size(self):
        conv = ModelConverter()
        assert conv.estimate_output_size(100, "pytorch") == 100.0
        assert conv.estimate_output_size(100, "gguf") == 50.0
        assert conv.estimate_output_size(100, "onnx") == 110.0

    def test_estimate_output_size_unknown_format(self):
        conv = ModelConverter()
        with pytest.raises(ConversionError):
            conv.estimate_output_size(100, "unknown")

    def test_convert(self):
        conv = ModelConverter()
        result = conv.convert("/models/test", "pytorch", "onnx")
        assert result["success"] is True
        assert result["source_format"] == "pytorch"
        assert result["target_format"] == "onnx"

    def test_convert_uses_config_defaults(self):
        conv = ModelConverter(ConversionConfig(source_format="pytorch", target_format="gguf"))
        result = conv.convert("/models/test")
        assert result["target_format"] == "gguf"
