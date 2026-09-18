"""Test configuration schema for Nexus-LLM."""
import pytest
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any


class SchemaError(Exception):
    """Raised when schema validation fails."""
    pass


@dataclass
class ModelSchema:
    name: str = "nexus-llm-base"
    model_type: str = "causal_lm"
    max_length: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0
    device: str = "auto"

    def validate(self):
        if not self.name:
            raise SchemaError("model.name is required")
        if self.model_type not in ("causal_lm", "seq2seq", "chat", "code"):
            raise SchemaError(f"Invalid model_type: {self.model_type}")
        if not (0 < self.max_length):
            raise SchemaError("max_length must be positive")
        if not (0.0 <= self.temperature <= 2.0):
            raise SchemaError("temperature must be between 0.0 and 2.0")
        if not (0.0 < self.top_p <= 1.0):
            raise SchemaError("top_p must be between 0.0 and 1.0")
        if self.top_k < 1:
            raise SchemaError("top_k must be >= 1")
        if self.repetition_penalty < 1.0:
            raise SchemaError("repetition_penalty must be >= 1.0")
        if self.device not in ("auto", "cpu", "cuda", "mps"):
            raise SchemaError(f"Invalid device: {self.device}")
        return True


@dataclass
class APISchema:
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    timeout: int = 30
    cors_origins: List[str] = field(default_factory=lambda: ["*"])

    def validate(self):
        if not self.host:
            raise SchemaError("api.host is required")
        if not (1 <= self.port <= 65535):
            raise SchemaError(f"Invalid port: {self.port}")
        if self.workers < 1:
            raise SchemaError("workers must be >= 1")
        if self.timeout < 1:
            raise SchemaError("timeout must be >= 1")
        return True


@dataclass
class SafetySchema:
    enabled: bool = True
    content_filter: bool = True
    max_toxicity: float = 0.5
    blocked_patterns: List[str] = field(default_factory=list)

    def validate(self):
        if not (0.0 <= self.max_toxicity <= 1.0):
            raise SchemaError("max_toxicity must be between 0.0 and 1.0")
        return True


@dataclass
class RAGSchema:
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 5
    embedding_model: str = "text-embedding-base"
    embedding_dim: int = 768

    def validate(self):
        if self.chunk_size < 1:
            raise SchemaError("chunk_size must be >= 1")
        if self.chunk_overlap >= self.chunk_size:
            raise SchemaError("chunk_overlap must be less than chunk_size")
        if self.top_k < 1:
            raise SchemaError("top_k must be >= 1")
        if self.embedding_dim < 1:
            raise SchemaError("embedding_dim must be >= 1")
        return True


@dataclass
class ConfigSchema:
    model: ModelSchema = field(default_factory=ModelSchema)
    api: APISchema = field(default_factory=APISchema)
    safety: SafetySchema = field(default_factory=SafetySchema)
    rag: RAGSchema = field(default_factory=RAGSchema)

    def validate(self):
        self.model.validate()
        self.api.validate()
        self.safety.validate()
        self.rag.validate()
        return True

    def to_dict(self):
        return asdict(self)


class TestModelSchema:
    def test_valid_defaults(self):
        schema = ModelSchema()
        assert schema.validate() is True

    def test_valid_custom(self):
        schema = ModelSchema(name="gpt2", model_type="causal_lm", temperature=1.0)
        assert schema.validate() is True

    def test_empty_name_fails(self):
        schema = ModelSchema(name="")
        with pytest.raises(SchemaError, match="name is required"):
            schema.validate()

    def test_invalid_model_type(self):
        schema = ModelSchema(model_type="invalid")
        with pytest.raises(SchemaError, match="Invalid model_type"):
            schema.validate()

    def test_all_valid_model_types(self):
        for mt in ("causal_lm", "seq2seq", "chat", "code"):
            schema = ModelSchema(model_type=mt)
            assert schema.validate() is True

    def test_negative_max_length_fails(self):
        schema = ModelSchema(max_length=-1)
        with pytest.raises(SchemaError, match="max_length"):
            schema.validate()

    def test_temperature_too_high(self):
        schema = ModelSchema(temperature=3.0)
        with pytest.raises(SchemaError, match="temperature"):
            schema.validate()

    def test_top_p_zero_fails(self):
        schema = ModelSchema(top_p=0.0)
        with pytest.raises(SchemaError, match="top_p"):
            schema.validate()

    def test_top_k_zero_fails(self):
        schema = ModelSchema(top_k=0)
        with pytest.raises(SchemaError, match="top_k"):
            schema.validate()

    def test_repetition_penalty_below_one(self):
        schema = ModelSchema(repetition_penalty=0.5)
        with pytest.raises(SchemaError, match="repetition_penalty"):
            schema.validate()

    def test_invalid_device(self):
        schema = ModelSchema(device="tpu")
        with pytest.raises(SchemaError, match="device"):
            schema.validate()


class TestAPISchema:
    def test_valid_defaults(self):
        assert APISchema().validate() is True

    def test_port_zero_fails(self):
        with pytest.raises(SchemaError, match="port"):
            APISchema(port=0).validate()

    def test_port_too_high_fails(self):
        with pytest.raises(SchemaError, match="port"):
            APISchema(port=70000).validate()

    def test_workers_zero_fails(self):
        with pytest.raises(SchemaError, match="workers"):
            APISchema(workers=0).validate()

    def test_timeout_zero_fails(self):
        with pytest.raises(SchemaError, match="timeout"):
            APISchema(timeout=0).validate()


class TestSafetySchema:
    def test_valid_defaults(self):
        assert SafetySchema().validate() is True

    def test_toxicity_above_one(self):
        with pytest.raises(SchemaError, match="max_toxicity"):
            SafetySchema(max_toxicity=1.5).validate()

    def test_toxicity_negative(self):
        with pytest.raises(SchemaError, match="max_toxicity"):
            SafetySchema(max_toxicity=-0.1).validate()

    def test_safety_disabled(self):
        schema = SafetySchema(enabled=False)
        assert schema.validate() is True


class TestRAGSchema:
    def test_valid_defaults(self):
        assert RAGSchema().validate() is True

    def test_overlap_exceeds_chunk_size(self):
        with pytest.raises(SchemaError, match="chunk_overlap"):
            RAGSchema(chunk_overlap=600).validate()

    def test_zero_chunk_size(self):
        with pytest.raises(SchemaError, match="chunk_size"):
            RAGSchema(chunk_size=0).validate()

    def test_zero_top_k(self):
        with pytest.raises(SchemaError, match="top_k"):
            RAGSchema(top_k=0).validate()


class TestConfigSchema:
    def test_valid_defaults(self):
        assert ConfigSchema().validate() is True

    def test_to_dict(self):
        config = ConfigSchema()
        d = config.to_dict()
        assert "model" in d
        assert "api" in d
        assert "safety" in d
        assert "rag" in d
        assert d["model"]["name"] == "nexus-llm-base"
        assert d["api"]["port"] == 8000

    def test_nested_validation_propagates(self):
        config = ConfigSchema(model=ModelSchema(temperature=5.0))
        with pytest.raises(SchemaError, match="temperature"):
            config.validate()

    def test_custom_nested_config(self):
        config = ConfigSchema(
            model=ModelSchema(name="custom"),
            api=APISchema(port=9000),
        )
        assert config.validate() is True
        d = config.to_dict()
        assert d["model"]["name"] == "custom"
        assert d["api"]["port"] == 9000
