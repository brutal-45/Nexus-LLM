"""Tests for model catalog with 39+ models."""
import pytest

from nexus_llm.types import ModelInfo
from nexus_llm.enums import ModelType, PrecisionType, DeviceType


# Define a comprehensive model catalog similar to what the project supports
MODEL_CATALOG = {
    # OpenAI-family models
    "gpt2": {"full_name": "openai-community/gpt2", "size": "124M", "type": ModelType.CAUSAL_LM, "license": "MIT"},
    "gpt2-medium": {"full_name": "openai-community/gpt2-medium", "size": "355M", "type": ModelType.CAUSAL_LM, "license": "MIT"},
    "gpt2-large": {"full_name": "openai-community/gpt2-large", "size": "774M", "type": ModelType.CAUSAL_LM, "license": "MIT"},
    "gpt2-xl": {"full_name": "openai-community/gpt2-xl", "size": "1.5B", "type": ModelType.CAUSAL_LM, "license": "MIT"},
    # Meta LLaMA family
    "llama-7b": {"full_name": "meta-llama/Llama-2-7b-hf", "size": "7B", "type": ModelType.CAUSAL_LM, "license": "LLAMA2"},
    "llama-13b": {"full_name": "meta-llama/Llama-2-13b-hf", "size": "13B", "type": ModelType.CAUSAL_LM, "license": "LLAMA2"},
    "llama-70b": {"full_name": "meta-llama/Llama-2-70b-hf", "size": "70B", "type": ModelType.CAUSAL_LM, "license": "LLAMA2"},
    "llama3-8b": {"full_name": "meta-llama/Meta-Llama-3-8B", "size": "8B", "type": ModelType.CAUSAL_LM, "license": "LLAMA3"},
    "llama3-70b": {"full_name": "meta-llama/Meta-Llama-3-70B", "size": "70B", "type": ModelType.CAUSAL_LM, "license": "LLAMA3"},
    # Mistral family
    "mistral-7b": {"full_name": "mistralai/Mistral-7B-v0.1", "size": "7B", "type": ModelType.CAUSAL_LM, "license": "Apache-2.0"},
    "mistral-7b-instruct": {"full_name": "mistralai/Mistral-7B-Instruct-v0.2", "size": "7B", "type": ModelType.INSTRUCTION, "license": "Apache-2.0"},
    "mixtral-8x7b": {"full_name": "mistralai/Mixtral-8x7B-v0.1", "size": "47B", "type": ModelType.CAUSAL_LM, "license": "Apache-2.0"},
    # Falcon family
    "falcon-7b": {"full_name": "tiiuae/falcon-7b", "size": "7B", "type": ModelType.CAUSAL_LM, "license": "Apache-2.0"},
    "falcon-40b": {"full_name": "tiiuae/falcon-40b", "size": "40B", "type": ModelType.CAUSAL_LM, "license": "Apache-2.0"},
    # Google family
    "gemma-2b": {"full_name": "google/gemma-2b", "size": "2B", "type": ModelType.CAUSAL_LM, "license": "Gemma"},
    "gemma-7b": {"full_name": "google/gemma-7b", "size": "7B", "type": ModelType.CAUSAL_LM, "license": "Gemma"},
    # Microsoft family
    "phi-2": {"full_name": "microsoft/phi-2", "size": "2.7B", "type": ModelType.CAUSAL_LM, "license": "MIT"},
    "phi-3-mini": {"full_name": "microsoft/Phi-3-mini-4k-instruct", "size": "3.8B", "type": ModelType.INSTRUCTION, "license": "MIT"},
    # Qwen family
    "qwen-7b": {"full_name": "Qwen/Qwen-7B", "size": "7B", "type": ModelType.CAUSAL_LM, "license": "Apache-2.0"},
    "qwen-14b": {"full_name": "Qwen/Qwen-14B", "size": "14B", "type": ModelType.CAUSAL_LM, "license": "Apache-2.0"},
    "qwen-72b": {"full_name": "Qwen/Qwen-72B", "size": "72B", "type": ModelType.CAUSAL_LM, "license": "Apache-2.0"},
    # Yi family
    "yi-6b": {"full_name": "01-ai/Yi-6B", "size": "6B", "type": ModelType.CAUSAL_LM, "license": "Apache-2.0"},
    "yi-34b": {"full_name": "01-ai/Yi-34B", "size": "34B", "type": ModelType.CAUSAL_LM, "license": "Apache-2.0"},
    # Chat models
    "vicuna-7b": {"full_name": "lmsys/vicuna-7b-v1.5", "size": "7B", "type": ModelType.CHAT, "license": "Apache-2.0"},
    "vicuna-13b": {"full_name": "lmsys/vicuna-13b-v1.5", "size": "13B", "type": ModelType.CHAT, "license": "Apache-2.0"},
    "wizardlm-7b": {"full_name": "WizardLM/WizardLM-7B-V1.0", "size": "7B", "type": ModelType.CHAT, "license": "Apache-2.0"},
    "openhermes-7b": {"full_name": "teknium/OpenHermes-2.5-Mistral-7B", "size": "7B", "type": ModelType.CHAT, "license": "Apache-2.0"},
    "zephyr-7b": {"full_name": "HuggingFaceH4/zephyr-7b-beta", "size": "7B", "type": ModelType.CHAT, "license": "Apache-2.0"},
    # Code models
    "codellama-7b": {"full_name": "codellama/CodeLlama-7b-hf", "size": "7B", "type": ModelType.CODE, "license": "LLAMA2"},
    "codellama-13b": {"full_name": "codellama/CodeLlama-13b-hf", "size": "13B", "type": ModelType.CODE, "license": "LLAMA2"},
    "starcoder": {"full_name": "bigcode/starcoder", "size": "15.5B", "type": ModelType.CODE, "license": "BigCode"},
    "deepseek-coder-7b": {"full_name": "deepseek-ai/deepseek-coder-6.7b-base", "size": "6.7B", "type": ModelType.CODE, "license": "Apache-2.0"},
    # Seq2Seq models
    "t5-small": {"full_name": "google-t5/t5-small", "size": "60M", "type": ModelType.SEQ2SEQ_LM, "license": "Apache-2.0"},
    "t5-base": {"full_name": "google-t5/t5-base", "size": "220M", "type": ModelType.SEQ2SEQ_LM, "license": "Apache-2.0"},
    "flan-t5-large": {"full_name": "google/flan-t5-large", "size": "770M", "type": ModelType.INSTRUCTION, "license": "Apache-2.0"},
    # Embedding models
    "all-minilm-l6": {"full_name": "sentence-transformers/all-MiniLM-L6-v2", "size": "22M", "type": ModelType.EMBEDDING, "license": "Apache-2.0"},
    "bge-small": {"full_name": "BAAI/bge-small-en-v1.5", "size": "33M", "type": ModelType.EMBEDDING, "license": "MIT"},
    "e5-base": {"full_name": "intfloat/e5-base-v2", "size": "110M", "type": ModelType.EMBEDDING, "license": "Apache-2.0"},
    # Multimodal
    "llava-7b": {"full_name": "llava-hf/llava-1.5-7b-hf", "size": "7B", "type": ModelType.MULTIMODAL, "license": "Apache-2.0"},
    # RLHF/DPO
    "rlhf-llama-7b": {"full_name": "meta-llama/Llama-2-7b-chat-hf", "size": "7B", "type": ModelType.RLHF, "license": "LLAMA2"},
}


class TestModelCatalogCompleteness:
    """Test that the model catalog has sufficient coverage."""

    def test_at_least_39_models(self):
        assert len(MODEL_CATALOG) >= 39, f"Only {len(MODEL_CATALOG)} models in catalog"

    def test_covers_all_model_types(self):
        types = {info["type"] for info in MODEL_CATALOG.values()}
        assert ModelType.CAUSAL_LM in types
        assert ModelType.SEQ2SEQ_LM in types
        assert ModelType.CHAT in types
        assert ModelType.CODE in types
        assert ModelType.INSTRUCTION in types
        assert ModelType.EMBEDDING in types
        assert ModelType.MULTIMODAL in types
        assert ModelType.RLHF in types


class TestModelCatalogRegistration:
    """Test registering catalog models into registry."""

    def test_register_all_models(self):
        from nexus_llm.registry import Registry
        registry = Registry[ModelInfo](name="catalog_test", allow_overwrite=True)
        for short_name, info in MODEL_CATALOG.items():
            model_info = ModelInfo(
                name=short_name,
                full_name=info["full_name"],
                size=info["size"],
                model_type=info["type"],
                license=info["license"],
            )
            registry.register(short_name, model_info)
        assert registry.size() >= 39

    def test_search_by_name(self):
        from nexus_llm.registry import Registry
        registry = Registry[ModelInfo](name="search_test", allow_overwrite=True)
        for short_name, info in MODEL_CATALOG.items():
            registry.register(short_name, ModelInfo(
                name=short_name, full_name=info["full_name"], size=info["size"],
                model_type=info["type"], license=info["license"],
            ))
        results = registry.search("llama")
        assert len(results) >= 5

    def test_filter_by_type_tag(self):
        from nexus_llm.registry import Registry
        registry = Registry[ModelInfo](name="tag_test", allow_overwrite=True)
        for short_name, info in MODEL_CATALOG.items():
            registry.register(short_name, ModelInfo(
                name=short_name, model_type=info["type"],
            ), tags={info["type"].value})
        code_models = registry.list_by_tag("code")
        assert len(code_models) >= 3

    def test_model_info_to_dict(self):
        for short_name, info in MODEL_CATALOG.items():
            model_info = ModelInfo(
                name=short_name,
                full_name=info["full_name"],
                size=info["size"],
                model_type=info["type"],
                license=info["license"],
            )
            d = model_info.to_dict()
            assert d["name"] == short_name
            assert d["model_type"] == info["type"].value
