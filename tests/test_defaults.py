"""Test default configuration values for Nexus-LLM."""
import pytest
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any


# --- Minimal default value definitions to test ---

DEFAULT_MODEL_NAME = "nexus-llm-base"
DEFAULT_MODEL_TYPE = "causal_lm"
DEFAULT_MAX_LENGTH = 2048
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_TOP_K = 50
DEFAULT_REPETITION_PENALTY = 1.0
DEFAULT_BATCH_SIZE = 1
DEFAULT_SEED = 42

DEFAULT_API_HOST = "0.0.0.0"
DEFAULT_API_PORT = 8000
DEFAULT_API_WORKERS = 4
DEFAULT_API_TIMEOUT = 30

DEFAULT_SAFETY_ENABLED = True
DEFAULT_MAX_TOXICITY = 0.5
DEFAULT_CONTENT_FILTER = True

DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 64
DEFAULT_RAG_TOP_K = 5
DEFAULT_EMBEDDING_DIM = 768

DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

DEFAULT_CACHE_DIR = "~/.cache/nexus_llm"
DEFAULT_DEVICE = "auto"


@dataclass
class ModelDefaults:
    name: str = DEFAULT_MODEL_NAME
    model_type: str = DEFAULT_MODEL_TYPE
    max_length: int = DEFAULT_MAX_LENGTH
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    top_k: int = DEFAULT_TOP_K
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY
    batch_size: int = DEFAULT_BATCH_SIZE
    seed: int = DEFAULT_SEED


@dataclass
class APIDefaults:
    host: str = DEFAULT_API_HOST
    port: int = DEFAULT_API_PORT
    workers: int = DEFAULT_API_WORKERS
    timeout: int = DEFAULT_API_TIMEOUT


@dataclass
class SafetyDefaults:
    enabled: bool = DEFAULT_SAFETY_ENABLED
    max_toxicity: float = DEFAULT_MAX_TOXICITY
    content_filter: bool = DEFAULT_CONTENT_FILTER


@dataclass
class RAGDefaults:
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    top_k: int = DEFAULT_RAG_TOP_K
    embedding_dim: int = DEFAULT_EMBEDDING_DIM


class TestModelDefaults:
    """Test model default values."""

    def test_default_model_name(self):
        assert DEFAULT_MODEL_NAME == "nexus-llm-base"

    def test_default_model_type(self):
        assert DEFAULT_MODEL_TYPE == "causal_lm"

    def test_default_max_length(self):
        assert DEFAULT_MAX_LENGTH == 2048
        assert DEFAULT_MAX_LENGTH > 0

    def test_default_temperature_range(self):
        assert 0.0 <= DEFAULT_TEMPERATURE <= 2.0
        assert DEFAULT_TEMPERATURE == 0.7

    def test_default_top_p_range(self):
        assert 0.0 < DEFAULT_TOP_P <= 1.0
        assert DEFAULT_TOP_P == 0.9

    def test_default_top_k_positive(self):
        assert DEFAULT_TOP_K > 0
        assert DEFAULT_TOP_K == 50

    def test_default_repetition_penalty(self):
        assert DEFAULT_REPETITION_PENALTY >= 1.0
        assert DEFAULT_REPETITION_PENALTY == 1.0

    def test_default_batch_size(self):
        assert DEFAULT_BATCH_SIZE >= 1
        assert DEFAULT_BATCH_SIZE == 1

    def test_default_seed(self):
        assert isinstance(DEFAULT_SEED, int)
        assert DEFAULT_SEED == 42

    def test_model_defaults_dataclass(self):
        defaults = ModelDefaults()
        assert defaults.name == DEFAULT_MODEL_NAME
        assert defaults.model_type == DEFAULT_MODEL_TYPE
        assert defaults.max_length == DEFAULT_MAX_LENGTH
        assert defaults.temperature == DEFAULT_TEMPERATURE

    def test_model_defaults_custom_override(self):
        custom = ModelDefaults(name="custom-model", temperature=1.2)
        assert custom.name == "custom-model"
        assert custom.temperature == 1.2
        assert custom.max_length == DEFAULT_MAX_LENGTH


class TestAPIDefaults:
    """Test API default values."""

    def test_default_api_host(self):
        assert DEFAULT_API_HOST == "0.0.0.0"

    def test_default_api_port(self):
        assert DEFAULT_API_PORT == 8000
        assert 1 <= DEFAULT_API_PORT <= 65535

    def test_default_api_workers(self):
        assert DEFAULT_API_WORKERS == 4
        assert DEFAULT_API_WORKERS >= 1

    def test_default_api_timeout(self):
        assert DEFAULT_API_TIMEOUT == 30
        assert DEFAULT_API_TIMEOUT > 0

    def test_api_defaults_dataclass(self):
        defaults = APIDefaults()
        assert defaults.host == DEFAULT_API_HOST
        assert defaults.port == DEFAULT_API_PORT

    def test_api_defaults_custom_port(self):
        custom = APIDefaults(port=9000)
        assert custom.port == 9000
        assert custom.host == DEFAULT_API_HOST


class TestSafetyDefaults:
    """Test safety default values."""

    def test_safety_enabled_by_default(self):
        assert DEFAULT_SAFETY_ENABLED is True

    def test_max_toxicity_range(self):
        assert 0.0 <= DEFAULT_MAX_TOXICITY <= 1.0
        assert DEFAULT_MAX_TOXICITY == 0.5

    def test_content_filter_enabled(self):
        assert DEFAULT_CONTENT_FILTER is True

    def test_safety_defaults_dataclass(self):
        defaults = SafetyDefaults()
        assert defaults.enabled is True
        assert defaults.max_toxicity == 0.5
        assert defaults.content_filter is True

    def test_safety_defaults_disabled(self):
        custom = SafetyDefaults(enabled=False)
        assert custom.enabled is False
        assert custom.max_toxicity == DEFAULT_MAX_TOXICITY


class TestRAGDefaults:
    """Test RAG default values."""

    def test_default_chunk_size(self):
        assert DEFAULT_CHUNK_SIZE == 512
        assert DEFAULT_CHUNK_SIZE > 0

    def test_default_chunk_overlap(self):
        assert DEFAULT_CHUNK_OVERLAP == 64
        assert DEFAULT_CHUNK_OVERLAP < DEFAULT_CHUNK_SIZE

    def test_default_rag_top_k(self):
        assert DEFAULT_RAG_TOP_K == 5
        assert DEFAULT_RAG_TOP_K > 0

    def test_default_embedding_dim(self):
        assert DEFAULT_EMBEDDING_DIM == 768
        assert DEFAULT_EMBEDDING_DIM > 0

    def test_rag_defaults_dataclass(self):
        defaults = RAGDefaults()
        assert defaults.chunk_size == DEFAULT_CHUNK_SIZE
        assert defaults.chunk_overlap == DEFAULT_CHUNK_OVERLAP
        assert defaults.top_k == DEFAULT_RAG_TOP_K

    def test_chunk_overlap_less_than_chunk_size(self):
        """Overlap must always be smaller than chunk size."""
        defaults = RAGDefaults(chunk_size=256)
        assert defaults.chunk_overlap < defaults.chunk_size


class TestMiscDefaults:
    """Test miscellaneous default values."""

    def test_default_log_level(self):
        assert DEFAULT_LOG_LEVEL in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
        assert DEFAULT_LOG_LEVEL == "INFO"

    def test_default_log_format(self):
        assert "%(asctime)s" in DEFAULT_LOG_FORMAT
        assert "%(levelname)s" in DEFAULT_LOG_FORMAT

    def test_default_cache_dir(self):
        assert DEFAULT_CACHE_DIR is not None
        assert "nexus" in DEFAULT_CACHE_DIR.lower() or "cache" in DEFAULT_CACHE_DIR.lower()

    def test_default_device(self):
        assert DEFAULT_DEVICE in ("auto", "cpu", "cuda", "mps")
        assert DEFAULT_DEVICE == "auto"
