"""Tests for component registry (register, unregister, lookup, factory)."""
import pytest

from nexus_llm.registry import Registry, RegistryEntry, GlobalRegistry


class TestRegistryEntry:
    """Test RegistryEntry."""

    def test_create_entry(self):
        entry = RegistryEntry(name="test", component="value")
        assert entry.name == "test"
        assert entry.component == "value"
        assert entry.metadata == {}
        assert entry.tags == set()

    def test_entry_with_metadata_and_tags(self):
        entry = RegistryEntry(
            name="test",
            component="value",
            metadata={"version": "1.0"},
            tags={"core", "main"},
        )
        assert entry.metadata["version"] == "1.0"
        assert "core" in entry.tags

    def test_repr(self):
        entry = RegistryEntry(name="test", component=42)
        r = repr(entry)
        assert "test" in r
        assert "int" in r


class TestRegistry:
    """Test Registry class."""

    def test_register_and_get(self):
        reg = Registry(name="test")
        reg.register("item1", "value1")
        assert reg.get("item1") == "value1"

    def test_register_duplicate_raises(self):
        reg = Registry(name="test")
        reg.register("item1", "value1")
        with pytest.raises(ValueError, match="already registered"):
            reg.register("item1", "value2")

    def test_register_overwrite_allowed(self):
        reg = Registry(name="test", allow_overwrite=True)
        reg.register("item1", "value1")
        reg.register("item1", "value2")
        assert reg.get("item1") == "value2"

    def test_unregister(self):
        reg = Registry(name="test")
        reg.register("item1", "value1")
        removed = reg.unregister("item1")
        assert removed.component == "value1"
        assert reg.size() == 0

    def test_unregister_nonexistent_raises(self):
        reg = Registry(name="test")
        with pytest.raises(KeyError, match="not found"):
            reg.unregister("missing")

    def test_get_nonexistent_raises(self):
        reg = Registry(name="test")
        with pytest.raises(KeyError, match="not found"):
            reg.get("missing")

    def test_has(self):
        reg = Registry(name="test")
        reg.register("item1", "value1")
        assert reg.has("item1") is True
        assert reg.has("missing") is False

    def test_list(self):
        reg = Registry(name="test")
        reg.register("a", 1)
        reg.register("b", 2)
        names = reg.list()
        assert "a" in names
        assert "b" in names

    def test_size(self):
        reg = Registry(name="test")
        assert reg.size() == 0
        reg.register("a", 1)
        assert reg.size() == 1

    def test_clear(self):
        reg = Registry(name="test")
        reg.register("a", 1)
        reg.register("b", 2)
        reg.clear()
        assert reg.size() == 0

    def test_list_by_tag(self):
        reg = Registry(name="test")
        reg.register("a", 1, tags={"model"})
        reg.register("b", 2, tags={"tool"})
        reg.register("c", 3, tags={"model"})
        models = reg.list_by_tag("model")
        assert len(models) == 2

    def test_search(self):
        reg = Registry(name="test")
        reg.register("gpt2_small", "model1")
        reg.register("gpt2_medium", "model2")
        reg.register("llama7b", "model3")
        results = reg.search("gpt2")
        assert len(results) == 2

    def test_get_entry(self):
        reg = Registry(name="test")
        reg.register("a", 1, metadata={"v": "1.0"}, tags={"core"})
        entry = reg.get_entry("a")
        assert entry.metadata["v"] == "1.0"
        assert "core" in entry.tags

    def test_dunder_contains(self):
        reg = Registry(name="test")
        reg.register("a", 1)
        assert "a" in reg

    def test_dunder_getitem(self):
        reg = Registry(name="test")
        reg.register("a", 42)
        assert reg["a"] == 42

    def test_dunder_setitem(self):
        reg = Registry(name="test", allow_overwrite=True)
        reg["a"] = 1
        assert reg.get("a") == 1

    def test_dunder_delitem(self):
        reg = Registry(name="test")
        reg.register("a", 1)
        del reg["a"]
        assert reg.size() == 0

    def test_dunder_len(self):
        reg = Registry(name="test")
        reg.register("a", 1)
        assert len(reg) == 1

    def test_dunder_iter(self):
        reg = Registry(name="test")
        reg.register("a", 1)
        reg.register("b", 2)
        names = list(reg)
        assert "a" in names
        assert "b" in names

    def test_repr(self):
        reg = Registry(name="test")
        r = repr(reg)
        assert "test" in r
        assert "entries=0" in r


class TestGlobalRegistry:
    """Test GlobalRegistry singleton."""

    @pytest.fixture(autouse=True)
    def reset_global(self):
        GlobalRegistry.reset()
        yield
        GlobalRegistry.reset()

    def test_singleton(self):
        g1 = GlobalRegistry()
        g2 = GlobalRegistry()
        assert g1 is g2

    def test_default_registries(self):
        g = GlobalRegistry()
        names = g.list_registries()
        assert "models" in names
        assert "plugins" in names
        assert "commands" in names

    def test_get_registry(self):
        g = GlobalRegistry()
        models = g.get_registry("models")
        assert isinstance(models, Registry)

    def test_get_nonexistent_registry_raises(self):
        g = GlobalRegistry()
        with pytest.raises(KeyError, match="not found"):
            g.get_registry("nonexistent")

    def test_create_registry(self):
        g = GlobalRegistry()
        reg = g.create_registry("custom")
        assert isinstance(reg, Registry)
        assert "custom" in g.list_registries()

    def test_create_duplicate_registry_raises(self):
        g = GlobalRegistry()
        g.create_registry("custom")
        with pytest.raises(ValueError, match="already exists"):
            g.create_registry("custom")

    def test_models_property(self):
        g = GlobalRegistry()
        assert isinstance(g.models, Registry)

    def test_plugins_property(self):
        g = GlobalRegistry()
        assert isinstance(g.plugins, Registry)

    def test_reset(self):
        g = GlobalRegistry()
        g.models.register("test", "value")
        GlobalRegistry.reset()
        g2 = GlobalRegistry()
        assert g2.models.size() == 0
