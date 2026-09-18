"""Test chat model wrapper for Nexus-LLM."""
import pytest
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


class ModelError(Exception):
    pass


@dataclass
class ChatMessage:
    role: str
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}

    @classmethod
    def from_dict(cls, data: dict) -> "ChatMessage":
        return cls(role=data["role"], content=data["content"])


@dataclass
class ChatConfig:
    name: str = "chat-model"
    system_prompt: str = "You are a helpful assistant."
    max_length: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9


class ChatModel:
    def __init__(self, config: ChatConfig = None):
        self._config = config or ChatConfig()
        self._loaded = False
        self._history: List[ChatMessage] = []

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
        self._history.clear()

    def chat(self, message: str, system: str = None, history: List[ChatMessage] = None, **kwargs) -> str:
        if not self._loaded:
            raise ModelError("Model not loaded")
        if not message:
            raise ModelError("Message cannot be empty")
        sys_prompt = system or self._config.system_prompt
        response = f"[CHAT] Response to: {message}"
        self._history.append(ChatMessage(role="user", content=message))
        self._history.append(ChatMessage(role="assistant", content=response))
        return response

    def chat_with_history(self, messages: List[ChatMessage], **kwargs) -> str:
        if not self._loaded:
            raise ModelError("Model not loaded")
        if not messages:
            raise ModelError("Messages cannot be empty")
        last_msg = messages[-1]
        return f"[CHAT] Response to: {last_msg.content}"

    def get_history(self) -> List[ChatMessage]:
        return list(self._history)

    def clear_history(self):
        self._history.clear()

    def format_messages(self, messages: List[ChatMessage]) -> str:
        parts = []
        for msg in messages:
            parts.append(f"{msg.role}: {msg.content}")
        return "\n".join(parts)

    def get_info(self) -> Dict[str, Any]:
        return {
            "name": self._config.name,
            "type": "chat",
            "system_prompt": self._config.system_prompt,
            "is_loaded": self._loaded,
            "history_length": len(self._history),
        }


class TestChatMessage:
    def test_creation(self):
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_to_dict(self):
        msg = ChatMessage(role="user", content="Hello")
        d = msg.to_dict()
        assert d == {"role": "user", "content": "Hello"}

    def test_from_dict(self):
        d = {"role": "assistant", "content": "Hi"}
        msg = ChatMessage.from_dict(d)
        assert msg.role == "assistant"
        assert msg.content == "Hi"

    def test_roundtrip(self):
        original = ChatMessage(role="user", content="test")
        restored = ChatMessage.from_dict(original.to_dict())
        assert restored.role == original.role
        assert restored.content == original.content


class TestChatConfig:
    def test_defaults(self):
        config = ChatConfig()
        assert config.system_prompt == "You are a helpful assistant."
        assert config.max_length == 4096

    def test_custom_system_prompt(self):
        config = ChatConfig(system_prompt="You are a code assistant.")
        assert "code" in config.system_prompt


class TestChatModel:
    def test_init(self):
        model = ChatModel()
        assert model.is_loaded is False

    def test_load_unload(self):
        model = ChatModel()
        model.load()
        assert model.is_loaded is True
        model.unload()
        assert model.is_loaded is False

    def test_chat(self):
        model = ChatModel()
        model.load()
        response = model.chat("Hello")
        assert "[CHAT]" in response
        assert "Hello" in response

    def test_chat_not_loaded(self):
        model = ChatModel()
        with pytest.raises(ModelError, match="not loaded"):
            model.chat("test")

    def test_chat_empty_message(self):
        model = ChatModel()
        model.load()
        with pytest.raises(ModelError, match="empty"):
            model.chat("")

    def test_chat_history_stored(self):
        model = ChatModel()
        model.load()
        model.chat("msg1")
        model.chat("msg2")
        history = model.get_history()
        assert len(history) == 4  # 2 user + 2 assistant

    def test_clear_history(self):
        model = ChatModel()
        model.load()
        model.chat("msg1")
        model.clear_history()
        assert len(model.get_history()) == 0

    def test_chat_with_custom_system(self):
        model = ChatModel()
        model.load()
        response = model.chat("Hello", system="You are a pirate.")
        assert "[CHAT]" in response

    def test_chat_with_history(self):
        model = ChatModel()
        model.load()
        messages = [
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="assistant", content="Hi"),
            ChatMessage(role="user", content="How are you?"),
        ]
        response = model.chat_with_history(messages)
        assert "How are you?" in response

    def test_format_messages(self):
        model = ChatModel()
        messages = [
            ChatMessage(role="user", content="Hi"),
            ChatMessage(role="assistant", content="Hello"),
        ]
        formatted = model.format_messages(messages)
        assert "user: Hi" in formatted
        assert "assistant: Hello" in formatted

    def test_get_info(self):
        model = ChatModel()
        model.load()
        info = model.get_info()
        assert info["type"] == "chat"
        assert info["history_length"] == 0
