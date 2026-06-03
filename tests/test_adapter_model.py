"""Test adapter model for Nexus-LLM."""
import pytest
from dataclasses import dataclass
from typing import Dict, Any, Optional, List


class AdapterError(Exception):
    pass


@dataclass
class AdapterConfig:
    name: str = "adapter"
    base_model: str = "nexus-llm-base"
    adapter_type: str = "lora"
    rank: int = 8
    alpha: float = 16.0
    target_modules: List[str] = None

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj"]


class AdapterModel:
    def __init__(self, config: AdapterConfig = None):
        self._config = config or AdapterConfig()
        self._loaded = False
        self._adapter_weights = {}

    @property
    def config(self):
        return self._config

    @property
    def is_loaded(self):
        return self._loaded

    def load(self):
        self._loaded = True

    def unload(self):
        self._loaded = False
        self._adapter_weights.clear()

    def load_adapter(self, path: str) -> None:
        if not path:
            raise AdapterError("Adapter path cannot be empty")
        self._adapter_weights[path] = {"rank": self._config.rank, "alpha": self._config.alpha}

    def merge_adapter(self) -> None:
        if not self._adapter_weights:
            raise AdapterError("No adapter loaded to merge")
        self._adapter_weights.clear()

    def set_adapter(self, name: str) -> None:
        if name not in self._adapter_weights and name != "base":
            raise AdapterError(f"Adapter '{name}' not found")

    def generate(self, prompt: str, **kwargs) -> str:
        if not self._loaded:
            raise AdapterError("Model not loaded")
        return f"[ADAPTER:{self._config.adapter_type}] {prompt}"

    def get_adapter_info(self) -> Dict[str, Any]:
        return {
            "name": self._config.name,
            "base_model": self._config.base_model,
            "adapter_type": self._config.adapter_type,
            "rank": self._config.rank,
            "alpha": self._config.alpha,
            "loaded_adapters": list(self._adapter_weights.keys()),
        }

    def get_info(self) -> Dict[str, Any]:
        return {
            "name": self._config.name,
            "type": "adapter",
            "is_loaded": self._loaded,
        }


class TestAdapterConfig:
    def test_defaults(self):
        config = AdapterConfig()
        assert config.adapter_type == "lora"
        assert config.rank == 8
        assert config.alpha == 16.0
        assert "q_proj" in config.target_modules

    def test_custom(self):
        config = AdapterConfig(adapter_type="qlora", rank=16, alpha=32.0)
        assert config.adapter_type == "qlora"
        assert config.rank == 16


class TestAdapterModel:
    def test_init(self):
        model = AdapterModel()
        assert model.is_loaded is False

    def test_load_unload(self):
        model = AdapterModel()
        model.load()
        assert model.is_loaded is True
        model.unload()
        assert model.is_loaded is False

    def test_load_adapter(self):
        model = AdapterModel()
        model.load_adapter("/path/to/adapter")
        info = model.get_adapter_info()
        assert "/path/to/adapter" in info["loaded_adapters"]

    def test_load_adapter_empty_path(self):
        model = AdapterModel()
        with pytest.raises(AdapterError, match="empty"):
            model.load_adapter("")

    def test_merge_adapter(self):
        model = AdapterModel()
        model.load_adapter("/path/to/adapter")
        model.merge_adapter()
        info = model.get_adapter_info()
        assert len(info["loaded_adapters"]) == 0

    def test_merge_without_adapter(self):
        model = AdapterModel()
        with pytest.raises(AdapterError, match="No adapter"):
            model.merge_adapter()

    def test_set_adapter(self):
        model = AdapterModel()
        model.load_adapter("my_adapter")
        model.set_adapter("my_adapter")

    def test_set_nonexistent_adapter(self):
        model = AdapterModel()
        with pytest.raises(AdapterError, match="not found"):
            model.set_adapter("nonexistent")

    def test_generate(self):
        model = AdapterModel()
        model.load()
        result = model.generate("test prompt")
        assert "[ADAPTER:lora]" in result

    def test_generate_not_loaded(self):
        model = AdapterModel()
        with pytest.raises(AdapterError):
            model.generate("test")

    def test_get_adapter_info(self):
        model = AdapterModel()
        info = model.get_adapter_info()
        assert info["adapter_type"] == "lora"
        assert info["rank"] == 8

    def test_get_info(self):
        model = AdapterModel()
        info = model.get_info()
        assert info["type"] == "adapter"
