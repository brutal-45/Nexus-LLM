"""Test request/response schemas for Nexus-LLM."""
import pytest
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


class SchemaError(Exception):
    pass


@dataclass
class GenerateRequest:
    prompt: str
    max_length: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    stop: List[str] = field(default_factory=list)
    stream: bool = False

    def validate(self):
        if not self.prompt:
            raise SchemaError("prompt is required")
        if not (0 < self.max_length <= 8192):
            raise SchemaError("max_length must be between 1 and 8192")
        if not (0 <= self.temperature <= 2):
            raise SchemaError("temperature must be between 0 and 2")
        if not (0 < self.top_p <= 1):
            raise SchemaError("top_p must be between 0 and 1")
        if self.top_k < 1:
            raise SchemaError("top_k must be >= 1")
        return True


@dataclass
class GenerateResponse:
    generated_text: str
    finish_reason: str = "stop"
    usage: Dict[str, int] = field(default_factory=dict)

    def validate(self):
        if not self.generated_text:
            raise SchemaError("generated_text is required")
        return True


@dataclass
class ChatRequest:
    messages: List[Dict[str, str]]
    model: str = "nexus-llm-chat"
    temperature: float = 0.7
    max_tokens: int = 4096

    def validate(self):
        if not self.messages:
            raise SchemaError("messages is required")
        for msg in self.messages:
            if "role" not in msg:
                raise SchemaError("Each message must have a 'role'")
            if "content" not in msg:
                raise SchemaError("Each message must have 'content'")
            if msg["role"] not in ("system", "user", "assistant"):
                raise SchemaError(f"Invalid role: {msg['role']}")
        return True


@dataclass
class ChatResponse:
    message: Dict[str, str]
    finish_reason: str = "stop"
    usage: Dict[str, int] = field(default_factory=dict)


@dataclass
class EmbeddingRequest:
    input: List[str]
    model: str = "nexus-llm-embed"

    def validate(self):
        if not self.input:
            raise SchemaError("input is required")
        return True


@dataclass
class EmbeddingResponse:
    embeddings: List[List[float]]
    model: str = ""
    usage: Dict[str, int] = field(default_factory=dict)


@dataclass
class ErrorResponse:
    error: str
    code: int = 500
    details: Optional[str] = None


class TestGenerateRequest:
    def test_valid_request(self):
        req = GenerateRequest(prompt="hello")
        assert req.validate() is True

    def test_empty_prompt(self):
        req = GenerateRequest(prompt="")
        with pytest.raises(SchemaError, match="prompt"):
            req.validate()

    def test_invalid_max_length(self):
        req = GenerateRequest(prompt="test", max_length=0)
        with pytest.raises(SchemaError, match="max_length"):
            req.validate()

    def test_invalid_temperature(self):
        req = GenerateRequest(prompt="test", temperature=3.0)
        with pytest.raises(SchemaError, match="temperature"):
            req.validate()

    def test_invalid_top_p(self):
        req = GenerateRequest(prompt="test", top_p=0.0)
        with pytest.raises(SchemaError, match="top_p"):
            req.validate()

    def test_invalid_top_k(self):
        req = GenerateRequest(prompt="test", top_k=0)
        with pytest.raises(SchemaError, match="top_k"):
            req.validate()

    def test_defaults(self):
        req = GenerateRequest(prompt="test")
        assert req.temperature == 0.7
        assert req.stream is False


class TestGenerateResponse:
    def test_valid(self):
        resp = GenerateResponse(generated_text="hello")
        assert resp.validate() is True

    def test_empty_text(self):
        resp = GenerateResponse(generated_text="")
        with pytest.raises(SchemaError):
            resp.validate()


class TestChatRequest:
    def test_valid(self):
        req = ChatRequest(messages=[{"role": "user", "content": "hello"}])
        assert req.validate() is True

    def test_empty_messages(self):
        req = ChatRequest(messages=[])
        with pytest.raises(SchemaError, match="messages"):
            req.validate()

    def test_missing_role(self):
        req = ChatRequest(messages=[{"content": "hello"}])
        with pytest.raises(SchemaError, match="role"):
            req.validate()

    def test_missing_content(self):
        req = ChatRequest(messages=[{"role": "user"}])
        with pytest.raises(SchemaError, match="content"):
            req.validate()

    def test_invalid_role(self):
        req = ChatRequest(messages=[{"role": "invalid", "content": "hello"}])
        with pytest.raises(SchemaError, match="Invalid role"):
            req.validate()

    def test_all_valid_roles(self):
        for role in ("system", "user", "assistant"):
            req = ChatRequest(messages=[{"role": role, "content": "test"}])
            assert req.validate() is True


class TestChatResponse:
    def test_creation(self):
        resp = ChatResponse(message={"role": "assistant", "content": "Hi"})
        assert resp.message["role"] == "assistant"


class TestEmbeddingRequest:
    def test_valid(self):
        req = EmbeddingRequest(input=["hello world"])
        assert req.validate() is True

    def test_empty_input(self):
        req = EmbeddingRequest(input=[])
        with pytest.raises(SchemaError):
            req.validate()


class TestEmbeddingResponse:
    def test_creation(self):
        resp = EmbeddingResponse(embeddings=[[0.1, 0.2, 0.3]])
        assert len(resp.embeddings) == 1


class TestErrorResponse:
    def test_creation(self):
        err = ErrorResponse(error="Not found", code=404)
        assert err.code == 404
        assert err.details is None

    def test_with_details(self):
        err = ErrorResponse(error="Validation failed", code=422, details="Missing field")
        assert err.details == "Missing field"
