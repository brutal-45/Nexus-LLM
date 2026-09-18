"""Tests for model registry."""
import pytest

from nexus_llm.registry import Registry, GlobalRegistry
from nexus_llm.enums import ModelType, DeviceType, PrecisionType
from nexus_llm.types import ModelInfo


class TestModelRegistry:
    """Test model-specific registry operations."""

    @pytest.fixture
    def model_registry(self):
        return Registry[ModelInfo](name="models", allow_overwrite=True)

    def test_register_model(self, model_registry):
        info = ModelInfo(name="gpt2", model_type=ModelType.CAUSAL_LM, size="124M")
        model_registry.register("gpt2", info)
        assert model_registry.has("gpt2")

    def test_get_model(self, model_registry):
        info = ModelInfo(name="gpt2", model_type=ModelType.CAUSAL_LM)
        model_registry.register("gpt2", info)
        retrieved = model_registry.get("gpt2")
        assert retrieved.name == "gpt2"
        assert retrieved.model_type == ModelType.CAUSAL_LM

    def test_list_models(self, model_registry):
        model_registry.register("gpt2", ModelInfo(name="gpt2"))
        model_registry.register("llama7b", ModelInfo(name="llama7b"))
        model_registry.register("mistral7b", ModelInfo(name="mistral7b"))
        assert len(model_registry.list()) == 3

    def test_search_models(self, model_registry):
        model_registry.register("gpt2_small", ModelInfo(name="gpt2_small"))
        model_registry.register("gpt2_medium", ModelInfo(name="gpt2_medium"))
        model_registry.register("llama7b", ModelInfo(name="llama7b"))
        results = model_registry.search("gpt2")
        assert len(results) == 2

    def test_model_with_tags(self, model_registry):
        model_registry.register("gpt2", ModelInfo(name="gpt2"),
                                tags={"causal", "small"})
        model_registry.register("llama", ModelInfo(name="llama"),
                                tags={"causal", "large"})
        small_models = model_registry.list_by_tag("small")
        assert len(small_models) == 1

    def test_model_metadata(self, model_registry):
        info = ModelInfo(
            name="gpt2",
            model_type=ModelType.CHAT,
            parameter_count=124000000,
            context_length=1024,
            device=DeviceType.CUDA,
            precision=PrecisionType.FP16,
        )
        model_registry.register("gpt2", info, metadata={"source": "huggingface"})
        entry = model_registry.get_entry("gpt2")
        assert entry.metadata["source"] == "huggingface"
        assert entry.component.parameter_count == 124000000

    def test_unregister_model(self, model_registry):
        model_registry.register("gpt2", ModelInfo(name="gpt2"))
        model_registry.unregister("gpt2")
        assert not model_registry.has("gpt2")

    def test_overwrite_model(self, model_registry):
        model_registry.register("gpt2", ModelInfo(name="gpt2", size="124M"))
        model_registry.register("gpt2", ModelInfo(name="gpt2", size="355M"))
        assert model_registry.get("gpt2").size == "355M"


class TestGlobalModelRegistry:
    """Test models in GlobalRegistry."""

    @pytest.fixture(autouse=True)
    def reset(self):
        GlobalRegistry.reset()
        yield
        GlobalRegistry.reset()

    def test_register_in_global(self):
        g = GlobalRegistry()
        g.models.register("gpt2", ModelInfo(name="gpt2"))
        assert g.models.has("gpt2")

    def test_multiple_model_types(self):
        g = GlobalRegistry()
        g.models.register("gpt2", ModelInfo(name="gpt2", model_type=ModelType.CAUSAL_LM))
        g.models.register("t5", ModelInfo(name="t5", model_type=ModelType.SEQ2SEQ_LM))
        g.models.register("flan-t5", ModelInfo(name="flan-t5", model_type=ModelType.INSTRUCTION))
        assert g.models.size() == 3
