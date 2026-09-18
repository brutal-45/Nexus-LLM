"""Tests for type definitions (Message, Conversation, GenerationConfig, etc.)."""
from datetime import datetime

import pytest

from nexus_llm.enums import ChatRole, DeviceType, MessageType, ModelType, PrecisionType
from nexus_llm.types import (
    Message,
    Conversation,
    GenerationConfig,
    ModelInfo,
    TrainingConfig,
    EvalConfig,
    BenchmarkConfig,
    ServerConfig,
    DownloadConfig,
    ChatConfig,
    BenchmarkResult,
    EvalResult,
)


class TestMessage:
    """Test Message dataclass."""

    def test_create_basic_message(self):
        msg = Message(role=ChatRole.USER, content="Hello")
        assert msg.role == ChatRole.USER
        assert msg.content == "Hello"

    def test_default_message_type(self):
        msg = Message(role=ChatRole.USER, content="test")
        assert msg.message_type == MessageType.TEXT

    def test_to_dict(self):
        msg = Message(role=ChatRole.USER, content="Hello")
        d = msg.to_dict()
        assert d["role"] == "user"
        assert d["content"] == "Hello"
        assert d["message_type"] == "text"
        assert "timestamp" in d

    def test_from_dict(self):
        data = {"role": "assistant", "content": "Hi there", "message_type": "text",
                "timestamp": datetime.now().isoformat()}
        msg = Message.from_dict(data)
        assert msg.role == ChatRole.ASSISTANT
        assert msg.content == "Hi there"

    def test_message_with_metadata(self):
        msg = Message(role=ChatRole.USER, content="test", metadata={"key": "val"})
        assert msg.metadata["key"] == "val"

    def test_system_message(self):
        msg = Message(role=ChatRole.SYSTEM, content="You are helpful")
        assert msg.role == ChatRole.SYSTEM


class TestConversation:
    """Test Conversation dataclass."""

    def test_create_conversation(self):
        conv = Conversation()
        assert conv.id  # auto-generated
        assert conv.messages == []

    def test_add_message(self):
        conv = Conversation()
        msg = conv.add_message(ChatRole.USER, "Hello")
        assert len(conv.messages) == 1
        assert msg.content == "Hello"

    def test_get_history(self):
        conv = Conversation()
        conv.add_message(ChatRole.USER, "msg1")
        conv.add_message(ChatRole.ASSISTANT, "msg2")
        conv.add_message(ChatRole.USER, "msg3")
        history = conv.get_history(limit=2)
        assert len(history) == 2
        assert history[0].content == "msg2"

    def test_clear_history(self):
        conv = Conversation()
        conv.add_message(ChatRole.USER, "Hello")
        conv.clear_history()
        assert len(conv.messages) == 0

    def test_to_dict_and_from_dict(self):
        conv = Conversation(model="gpt2")
        conv.add_message(ChatRole.USER, "Hi")
        d = conv.to_dict()
        conv2 = Conversation.from_dict(d)
        assert conv2.model == "gpt2"
        assert len(conv2.messages) == 1

    def test_auto_id_generation(self):
        conv = Conversation()
        assert conv.id != ""
        assert len(conv.id) > 0


class TestGenerationConfig:
    """Test GenerationConfig dataclass."""

    def test_defaults(self):
        config = GenerationConfig()
        assert config.max_tokens == 2048
        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.do_sample is True

    def test_to_dict(self):
        config = GenerationConfig(max_tokens=1024, temperature=0.5)
        d = config.to_dict()
        assert d["max_tokens"] == 1024
        assert d["temperature"] == 0.5

    def test_custom_values(self):
        config = GenerationConfig(max_tokens=512, temperature=0.0, top_k=10)
        assert config.max_tokens == 512
        assert config.temperature == 0.0
        assert config.top_k == 10


class TestModelInfo:
    """Test ModelInfo dataclass."""

    def test_defaults(self):
        info = ModelInfo()
        assert info.model_type == ModelType.CAUSAL_LM
        assert info.precision == PrecisionType.FP16
        assert info.device == DeviceType.AUTO
        assert info.is_loaded is False

    def test_to_dict(self):
        info = ModelInfo(name="gpt2", model_type=ModelType.CHAT)
        d = info.to_dict()
        assert d["name"] == "gpt2"
        assert d["model_type"] == "chat"


class TestTrainingConfig:
    """Test TrainingConfig dataclass."""

    def test_defaults(self):
        config = TrainingConfig()
        assert config.epochs == 3
        assert config.batch_size == 8
        assert config.learning_rate == 2e-5
        assert config.lora_rank == 8

    def test_to_dict(self):
        config = TrainingConfig(model="gpt2", dataset="data.jsonl")
        d = config.to_dict()
        assert d["model"] == "gpt2"
        assert d["dataset"] == "data.jsonl"


class TestServerConfig:
    """Test ServerConfig dataclass."""

    def test_defaults(self):
        config = ServerConfig()
        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert config.cors is False

    def test_api_key_masked_in_dict(self):
        config = ServerConfig(api_key="secret123")
        d = config.to_dict()
        assert d["api_key"] == "***"


class TestBenchmarkResult:
    """Test BenchmarkResult dataclass."""

    def test_defaults(self):
        result = BenchmarkResult()
        assert result.avg_latency_ms == 0.0
        assert result.batch_size == 1

    def test_to_dict(self):
        result = BenchmarkResult(model="gpt2", avg_latency_ms=50.0)
        d = result.to_dict()
        assert d["model"] == "gpt2"
        assert d["avg_latency_ms"] == 50.0
