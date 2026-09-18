"""Tests for the model_catalog module."""

import pytest

from nexus_llm.core.model_catalog import (
    MODEL_CATALOG,
    ModelInfo,
    get_model_info,
    list_models,
    list_categories,
    get_recommended_models,
)


# ---------------------------------------------------------------------------
# ModelInfo dataclass
# ---------------------------------------------------------------------------

class TestModelInfo:
    """Tests for the ModelInfo dataclass."""

    def test_create_model_info(self):
        info = ModelInfo(
            id="test-model",
            name="Test Model",
            hf_id="org/test-model",
            category="test",
            size="100M",
            params="100M",
            description="A test model",
        )
        assert info.id == "test-model"
        assert info.name == "Test Model"
        assert info.hf_id == "org/test-model"
        assert info.category == "test"
        assert info.size == "100M"
        assert info.params == "100M"
        assert info.description == "A test model"
        assert info.model_type == "causal"  # default
        assert info.recommended is False  # default
        assert info.min_ram_gb == 4  # default

    def test_custom_model_info(self):
        info = ModelInfo(
            id="test-seq2seq",
            name="Test Seq2Seq",
            hf_id="org/test-seq2seq",
            category="test",
            size="200M",
            params="200M",
            description="A seq2seq test model",
            model_type="seq2seq",
            recommended=True,
            min_ram_gb=8,
        )
        assert info.model_type == "seq2seq"
        assert info.recommended is True
        assert info.min_ram_gb == 8


# ---------------------------------------------------------------------------
# MODEL_CATALOG
# ---------------------------------------------------------------------------

class TestModelCatalog:
    """Tests for the MODEL_CATALOG dictionary."""

    def test_catalog_not_empty(self):
        assert len(MODEL_CATALOG) > 0

    def test_catalog_has_expected_model_count(self):
        # The catalog contains 40 models
        assert len(MODEL_CATALOG) == 40

    def test_catalog_has_gpt2_family(self):
        for mid in ("gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"):
            assert mid in MODEL_CATALOG, f"Missing model: {mid}"

    def test_catalog_has_dialogpt_family(self):
        for mid in ("dialogpt-small", "dialogpt-medium", "dialogpt-large"):
            assert mid in MODEL_CATALOG, f"Missing model: {mid}"

    def test_catalog_has_phi_family(self):
        for mid in ("phi-1", "phi-1.5", "phi-2"):
            assert mid in MODEL_CATALOG, f"Missing model: {mid}"

    def test_catalog_has_pythia_family(self):
        for mid in ("pythia-70m", "pythia-160m", "pythia-410m",
                     "pythia-1b", "pythia-1.4b", "pythia-2.8b"):
            assert mid in MODEL_CATALOG, f"Missing model: {mid}"

    def test_catalog_has_opt_family(self):
        for mid in ("opt-125m", "opt-350m", "opt-1.3b", "opt-2.7b"):
            assert mid in MODEL_CATALOG, f"Missing model: {mid}"

    def test_catalog_has_tinyllama(self):
        assert "tinyllama" in MODEL_CATALOG

    def test_catalog_has_qwen_family(self):
        for mid in ("qwen2.5-0.5b", "qwen2.5-1.5b", "qwen2.5-3b"):
            assert mid in MODEL_CATALOG, f"Missing model: {mid}"

    def test_catalog_has_smollm_family(self):
        for mid in ("smollm-135m", "smollm-360m", "smollm-1.7b"):
            assert mid in MODEL_CATALOG, f"Missing model: {mid}"

    def test_catalog_has_gemma_family(self):
        for mid in ("gemma-2b", "gemma-2b-it"):
            assert mid in MODEL_CATALOG, f"Missing model: {mid}"

    def test_catalog_has_mamba_family(self):
        for mid in ("mamba-130m", "mamba-370m", "mamba-790m"):
            assert mid in MODEL_CATALOG, f"Missing model: {mid}"

    def test_catalog_has_stablelm_family(self):
        for mid in ("stablelm-2-1.6b", "stablelm-2-zephyr"):
            assert mid in MODEL_CATALOG, f"Missing model: {mid}"

    def test_catalog_has_bloom_family(self):
        for mid in ("bloom-560m", "bloom-1b1", "bloom-1b7"):
            assert mid in MODEL_CATALOG, f"Missing model: {mid}"

    def test_catalog_has_flan_t5_family(self):
        for mid in ("flan-t5-small", "flan-t5-base", "flan-t5-large"):
            assert mid in MODEL_CATALOG, f"Missing model: {mid}"

    def test_all_entries_are_model_info(self):
        for key, value in MODEL_CATALOG.items():
            assert isinstance(value, ModelInfo), f"{key} is not ModelInfo"
            assert value.id == key, f"Key {key} doesn't match id {value.id}"

    def test_all_entries_have_hf_id(self):
        for key, value in MODEL_CATALOG.items():
            assert value.hf_id, f"{key} has empty hf_id"
            assert "/" in value.hf_id, f"{key} hf_id should be org/model format"


# ---------------------------------------------------------------------------
# get_model_info
# ---------------------------------------------------------------------------

class TestGetModelInfo:
    """Tests for the get_model_info function."""

    def test_valid_model_id(self):
        info = get_model_info("gpt2-medium")
        assert info.id == "gpt2-medium"
        assert info.name == "GPT-2 Medium"

    def test_invalid_model_id_raises(self):
        from nexus_llm.core.exceptions import ModelNotFoundError
        with pytest.raises(ModelNotFoundError):
            get_model_info("nonexistent-model")

    def test_error_message_contains_available(self):
        from nexus_llm.core.exceptions import ModelNotFoundError
        with pytest.raises(ModelNotFoundError, match="Available models"):
            get_model_info("fake-model")


# ---------------------------------------------------------------------------
# list_models
# ---------------------------------------------------------------------------

class TestListModels:
    """Tests for the list_models function."""

    def test_list_all_models(self):
        models = list_models()
        assert len(models) == len(MODEL_CATALOG)

    def test_list_models_returns_model_info(self):
        models = list_models()
        for m in models:
            assert isinstance(m, ModelInfo)

    def test_list_models_sorted_by_name(self):
        models = list_models()
        names = [m.name for m in models]
        assert names == sorted(names)

    def test_list_models_category_filter(self):
        models = list_models(category="gpt2")
        for m in models:
            assert m.category == "gpt2"
        assert len(models) == 4  # gpt2, gpt2-medium, gpt2-large, gpt2-xl

    def test_list_models_category_filter_phi(self):
        models = list_models(category="phi")
        for m in models:
            assert m.category == "phi"
        assert len(models) == 3

    def test_list_models_nonexistent_category(self):
        models = list_models(category="nonexistent")
        assert models == []


# ---------------------------------------------------------------------------
# list_categories
# ---------------------------------------------------------------------------

class TestListCategories:
    """Tests for the list_categories function."""

    def test_returns_list(self):
        cats = list_categories()
        assert isinstance(cats, list)
        assert len(cats) > 0

    def test_sorted(self):
        cats = list_categories()
        assert cats == sorted(cats)

    def test_contains_known_categories(self):
        cats = list_categories()
        expected = {"gpt2", "dialogpt", "phi", "pythia", "opt", "llama",
                    "qwen", "smollm", "gemma", "mamba", "stablelm",
                    "bloom", "flan-t5"}
        assert expected.issubset(set(cats))

    def test_no_duplicates(self):
        cats = list_categories()
        assert len(cats) == len(set(cats))


# ---------------------------------------------------------------------------
# get_recommended_models
# ---------------------------------------------------------------------------

class TestGetRecommendedModels:
    """Tests for the get_recommended_models function."""

    def test_returns_list(self):
        recs = get_recommended_models()
        assert isinstance(recs, list)
        assert len(recs) > 0

    def test_all_recommended(self):
        recs = get_recommended_models()
        for m in recs:
            assert m.recommended is True

    def test_known_recommended_models(self):
        recs = get_recommended_models()
        rec_ids = {m.id for m in recs}
        # Based on the catalog: gpt2-medium, phi-2, tinyllama, gemma-2b-it,
        # stablelm-2-zephyr, flan-t5-base
        assert "gpt2-medium" in rec_ids
        assert "phi-2" in rec_ids
        assert "tinyllama" in rec_ids
        assert "gemma-2b-it" in rec_ids
        assert "flan-t5-base" in rec_ids
