"""Test causal LM model for Nexus-LLM."""
import pytest
from dataclasses import dataclass
from typing import Optional, Dict, Any, List


class ModelError(Exception):
    pass


@dataclass
class CausalLMConfig:
    name: str = "causal-lm"
    max_length: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0
    do_sample: bool = True
    num_return_sequences: int = 1


class CausalLMModel:
    def __init__(self, config: CausalLMConfig = None):
        self._config = config or CausalLMConfig()
        self._loaded = False

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

    def generate(self, prompt: str, **kwargs) -> str:
        if not self._loaded:
            raise ModelError("Model not loaded")
        if not prompt:
            raise ModelError("Prompt cannot be empty")
        max_len = kwargs.get("max_length", self._config.max_length)
        temp = kwargs.get("temperature", self._config.temperature)
        if not (0 <= temp <= 2):
            raise ModelError("temperature must be between 0 and 2")
        return f"[CAUSAL_LM] {prompt}"

    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        return [self.generate(p, **kwargs) for p in prompts]

    def compute_loss(self, input_ids, labels):
        if not self._loaded:
            raise ModelError("Model not loaded")
        return 2.718

    def get_info(self) -> Dict[str, Any]:
        return {
            "name": self._config.name,
            "type": "causal_lm",
            "max_length": self._config.max_length,
            "is_loaded": self._loaded,
        }


class TestCausalLMConfig:
    def test_defaults(self):
        config = CausalLMConfig()
        assert config.name == "causal-lm"
        assert config.temperature == 0.7
        assert config.do_sample is True

    def test_custom(self):
        config = CausalLMConfig(name="gpt2", temperature=1.0)
        assert config.name == "gpt2"
        assert config.temperature == 1.0


class TestCausalLMModel:
    def test_init(self):
        model = CausalLMModel()
        assert model.is_loaded is False
        assert model.config.name == "causal-lm"

    def test_load_unload(self):
        model = CausalLMModel()
        model.load()
        assert model.is_loaded is True
        model.unload()
        assert model.is_loaded is False

    def test_generate_not_loaded(self):
        model = CausalLMModel()
        with pytest.raises(ModelError, match="not loaded"):
            model.generate("test")

    def test_generate_empty_prompt(self):
        model = CausalLMModel()
        model.load()
        with pytest.raises(ModelError, match="empty"):
            model.generate("")

    def test_generate_success(self):
        model = CausalLMModel()
        model.load()
        result = model.generate("Hello world")
        assert "[CAUSAL_LM]" in result
        assert "Hello world" in result

    def test_generate_with_kwargs(self):
        model = CausalLMModel()
        model.load()
        result = model.generate("test", temperature=0.5, max_length=100)
        assert "[CAUSAL_LM]" in result

    def test_generate_invalid_temperature(self):
        model = CausalLMModel()
        model.load()
        with pytest.raises(ModelError, match="temperature"):
            model.generate("test", temperature=5.0)

    def test_generate_batch(self):
        model = CausalLMModel()
        model.load()
        results = model.generate_batch(["hello", "world"])
        assert len(results) == 2
        assert "[CAUSAL_LM]" in results[0]

    def test_compute_loss(self):
        model = CausalLMModel()
        model.load()
        loss = model.compute_loss([1, 2, 3], [1, 2, 3])
        assert isinstance(loss, float)
        assert loss > 0

    def test_compute_loss_not_loaded(self):
        model = CausalLMModel()
        with pytest.raises(ModelError):
            model.compute_loss([], [])

    def test_get_info(self):
        model = CausalLMModel()
        info = model.get_info()
        assert info["type"] == "causal_lm"
        assert info["is_loaded"] is False
