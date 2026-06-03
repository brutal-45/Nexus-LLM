"""Test model catalog for Nexus-LLM."""
import pytest
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ModelEntry:
    name: str
    model_type: str
    description: str
    parameters: str
    context_length: int
    license: str = "unknown"
    tags: List[str] = field(default_factory=list)
    url: Optional[str] = None

    def matches_tag(self, tag: str) -> bool:
        return tag.lower() in [t.lower() for t in self.tags]

    def matches_type(self, model_type: str) -> bool:
        return self.model_type == model_type


BUILTIN_MODELS = {
    "nexus-llm-base": ModelEntry(
        name="nexus-llm-base",
        model_type="causal_lm",
        description="Base language model",
        parameters="7B",
        context_length=2048,
        license="apache-2.0",
        tags=["general", "base"],
    ),
    "nexus-llm-chat": ModelEntry(
        name="nexus-llm-chat",
        model_type="chat",
        description="Chat-tuned model",
        parameters="7B",
        context_length=4096,
        license="apache-2.0",
        tags=["chat", "instruct"],
    ),
    "nexus-llm-code": ModelEntry(
        name="nexus-llm-code",
        model_type="code",
        description="Code generation model",
        parameters="7B",
        context_length=4096,
        license="apache-2.0",
        tags=["code", "programming"],
    ),
    "nexus-llm-large": ModelEntry(
        name="nexus-llm-large",
        model_type="causal_lm",
        description="Large language model",
        parameters="13B",
        context_length=4096,
        license="apache-2.0",
        tags=["general", "large"],
    ),
    "nexus-llm-small": ModelEntry(
        name="nexus-llm-small",
        model_type="causal_lm",
        description="Small fast model",
        parameters="1.3B",
        context_length=1024,
        license="apache-2.0",
        tags=["general", "small", "fast"],
    ),
}


class ModelCatalog:
    def __init__(self):
        self._models = dict(BUILTIN_MODELS)

    def get(self, name: str) -> ModelEntry:
        if name not in self._models:
            raise KeyError(f"Model '{name}' not found in catalog")
        return self._models[name]

    def list_models(self) -> List[str]:
        return sorted(self._models.keys())

    def search_by_tag(self, tag: str) -> List[ModelEntry]:
        return [m for m in self._models.values() if m.matches_tag(tag)]

    def search_by_type(self, model_type: str) -> List[ModelEntry]:
        return [m for m in self._models.values() if m.matches_type(model_type)]

    def add_model(self, entry: ModelEntry):
        self._models[entry.name] = entry

    def remove_model(self, name: str):
        if name not in self._models:
            raise KeyError(f"Model '{name}' not found")
        del self._models[name]

    def count(self) -> int:
        return len(self._models)


class TestModelEntry:
    def test_creation(self):
        entry = ModelEntry(name="test", model_type="causal_lm", description="Test", parameters="7B", context_length=2048)
        assert entry.name == "test"
        assert entry.model_type == "causal_lm"

    def test_default_license(self):
        entry = ModelEntry(name="t", model_type="causal_lm", description="", parameters="7B", context_length=2048)
        assert entry.license == "unknown"

    def test_matches_tag(self):
        entry = BUILTIN_MODELS["nexus-llm-chat"]
        assert entry.matches_tag("chat") is True
        assert entry.matches_tag("code") is False

    def test_matches_tag_case_insensitive(self):
        entry = BUILTIN_MODELS["nexus-llm-chat"]
        assert entry.matches_tag("CHAT") is True

    def test_matches_type(self):
        entry = BUILTIN_MODELS["nexus-llm-code"]
        assert entry.matches_type("code") is True
        assert entry.matches_type("causal_lm") is False


class TestBuiltinModels:
    def test_base_exists(self):
        assert "nexus-llm-base" in BUILTIN_MODELS

    def test_chat_exists(self):
        assert "nexus-llm-chat" in BUILTIN_MODELS

    def test_code_exists(self):
        assert "nexus-llm-code" in BUILTIN_MODELS

    def test_all_have_descriptions(self):
        for entry in BUILTIN_MODELS.values():
            assert entry.description

    def test_all_have_parameters(self):
        for entry in BUILTIN_MODELS.values():
            assert entry.parameters

    def test_all_context_lengths_positive(self):
        for entry in BUILTIN_MODELS.values():
            assert entry.context_length > 0


class TestModelCatalog:
    def test_get_model(self):
        catalog = ModelCatalog()
        model = catalog.get("nexus-llm-base")
        assert model.name == "nexus-llm-base"

    def test_get_nonexistent(self):
        catalog = ModelCatalog()
        with pytest.raises(KeyError, match="not found"):
            catalog.get("nonexistent-model")

    def test_list_models(self):
        catalog = ModelCatalog()
        models = catalog.list_models()
        assert "nexus-llm-base" in models
        assert len(models) >= 5

    def test_search_by_tag(self):
        catalog = ModelCatalog()
        chat_models = catalog.search_by_tag("chat")
        assert len(chat_models) >= 1
        assert any(m.name == "nexus-llm-chat" for m in chat_models)

    def test_search_by_type(self):
        catalog = ModelCatalog()
        causal_models = catalog.search_by_type("causal_lm")
        assert len(causal_models) >= 2

    def test_add_model(self):
        catalog = ModelCatalog()
        new_model = ModelEntry(
            name="custom-model", model_type="causal_lm",
            description="Custom", parameters="3B", context_length=2048,
        )
        catalog.add_model(new_model)
        assert catalog.get("custom-model").name == "custom-model"

    def test_remove_model(self):
        catalog = ModelCatalog()
        catalog.remove_model("nexus-llm-small")
        with pytest.raises(KeyError):
            catalog.get("nexus-llm-small")

    def test_count(self):
        catalog = ModelCatalog()
        assert catalog.count() >= 5

    def test_search_nonexistent_tag(self):
        catalog = ModelCatalog()
        result = catalog.search_by_tag("nonexistent")
        assert result == []
