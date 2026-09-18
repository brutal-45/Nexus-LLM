"""Test base model class for Nexus-LLM."""
import pytest
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List


class ModelError(Exception):
    pass


@dataclass
class ModelConfig:
    name: str = "base-model"
    model_type: str = "base"
    max_length: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    device: str = "auto"


class BaseModel(ABC):
    def __init__(self, config: ModelConfig = None):
        self._config = config or ModelConfig()
        self._loaded = False
        self._model = None

    @property
    def config(self) -> ModelConfig:
        return self._config

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def model_type(self) -> str:
        return self._config.model_type

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False
        self._model = None

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        pass

    def get_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.model_type,
            "max_length": self._config.max_length,
            "is_loaded": self.is_loaded,
        }

    def validate_generate_params(self, **kwargs) -> None:
        if "max_length" in kwargs and kwargs["max_length"] > self._config.max_length:
            raise ModelError(f"max_length exceeds model limit of {self._config.max_length}")
        if "temperature" in kwargs and not (0 <= kwargs["temperature"] <= 2):
            raise ModelError("temperature must be between 0 and 2")


class DummyModel(BaseModel):
    def generate(self, prompt: str, **kwargs) -> str:
        if not self.is_loaded:
            raise ModelError("Model not loaded")
        return f"Generated response for: {prompt[:50]}"


class TestModelConfig:
    def test_default_config(self):
        config = ModelConfig()
        assert config.name == "base-model"
        assert config.model_type == "base"
        assert config.max_length == 2048

    def test_custom_config(self):
        config = ModelConfig(name="custom", temperature=1.5)
        assert config.name == "custom"
        assert config.temperature == 1.5


class TestBaseModel:
    def test_init_with_defaults(self):
        model = DummyModel()
        assert model.name == "base-model"
        assert model.model_type == "base"
        assert model.is_loaded is False

    def test_init_with_config(self):
        config = ModelConfig(name="test-model")
        model = DummyModel(config)
        assert model.name == "test-model"

    def test_load(self):
        model = DummyModel()
        model.load()
        assert model.is_loaded is True

    def test_unload(self):
        model = DummyModel()
        model.load()
        model.unload()
        assert model.is_loaded is False

    def test_generate_without_load_raises(self):
        model = DummyModel()
        with pytest.raises(ModelError, match="not loaded"):
            model.generate("test")

    def test_generate_after_load(self):
        model = DummyModel()
        model.load()
        result = model.generate("hello")
        assert "hello" in result

    def test_get_info(self):
        model = DummyModel()
        info = model.get_info()
        assert info["name"] == "base-model"
        assert info["is_loaded"] is False

    def test_get_info_after_load(self):
        model = DummyModel()
        model.load()
        info = model.get_info()
        assert info["is_loaded"] is True

    def test_validate_generate_params_ok(self):
        model = DummyModel()
        model.validate_generate_params(temperature=0.5, max_length=100)

    def test_validate_max_length_exceeded(self):
        model = DummyModel()
        with pytest.raises(ModelError, match="max_length"):
            model.validate_generate_params(max_length=99999)

    def test_validate_temperature_out_of_range(self):
        model = DummyModel()
        with pytest.raises(ModelError, match="temperature"):
            model.validate_generate_params(temperature=5.0)

    def test_config_property(self):
        config = ModelConfig(name="test")
        model = DummyModel(config)
        assert model.config is config
