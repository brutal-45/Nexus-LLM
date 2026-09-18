"""Test chat agent for Nexus-LLM."""
import pytest
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class ChatAgentConfig:
    name: str = "chat-agent"
    system_prompt: str = "You are a helpful assistant."
    max_history: int = 50
    model: str = "nexus-llm-chat"
    temperature: float = 0.7


@dataclass
class ChatMessage:
    role: str
    content: str


class ChatAgent:
    def __init__(self, config: ChatAgentConfig = None):
        self._config = config or ChatAgentConfig()
        self._history: List[ChatMessage] = []
        self._system_prompt = self._config.system_prompt

    @property
    def config(self):
        return self._config

    @property
    def system_prompt(self):
        return self._system_prompt

    @system_prompt.setter
    def system_prompt(self, value: str):
        if not value:
            raise ValueError("System prompt cannot be empty")
        self._system_prompt = value

    def chat(self, message: str) -> str:
        if not message:
            raise ValueError("Message cannot be empty")
        self._history.append(ChatMessage(role="user", content=message))
        response = f"[ASSISTANT] Response to: {message}"
        self._history.append(ChatMessage(role="assistant", content=response))
        if len(self._history) > self._config.max_history * 2:
            self._history = self._history[-(self._config.max_history * 2):]
        return response

    def get_history(self) -> List[ChatMessage]:
        return list(self._history)

    def clear_history(self):
        self._history.clear()

    def get_context_window(self) -> List[ChatMessage]:
        system = ChatMessage(role="system", content=self._system_prompt)
        return [system] + self._history

    def summarize_history(self) -> str:
        if not self._history:
            return "No conversation history."
        user_msgs = sum(1 for m in self._history if m.role == "user")
        assistant_msgs = sum(1 for m in self._history if m.role == "assistant")
        return f"Conversation: {user_msgs} user messages, {assistant_msgs} assistant messages."

    def export_history(self) -> List[Dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in self._history]

    def import_history(self, messages: List[Dict[str, str]]):
        self._history = [ChatMessage(role=m["role"], content=m["content"]) for m in messages]


class TestChatAgentConfig:
    def test_defaults(self):
        config = ChatAgentConfig()
        assert config.name == "chat-agent"
        assert config.max_history == 50

    def test_custom(self):
        config = ChatAgentConfig(system_prompt="You are a pirate.", temperature=1.0)
        assert "pirate" in config.system_prompt


class TestChatAgent:
    def test_chat(self):
        agent = ChatAgent()
        response = agent.chat("Hello")
        assert "[ASSISTANT]" in response
        assert "Hello" in response

    def test_empty_message(self):
        agent = ChatAgent()
        with pytest.raises(ValueError, match="empty"):
            agent.chat("")

    def test_history_stored(self):
        agent = ChatAgent()
        agent.chat("msg1")
        agent.chat("msg2")
        history = agent.get_history()
        assert len(history) == 4  # 2 user + 2 assistant

    def test_clear_history(self):
        agent = ChatAgent()
        agent.chat("msg1")
        agent.clear_history()
        assert len(agent.get_history()) == 0

    def test_max_history(self):
        agent = ChatAgent(ChatAgentConfig(max_history=2))
        for i in range(10):
            agent.chat(f"msg{i}")
        history = agent.get_history()
        assert len(history) <= 4  # 2 user + 2 assistant

    def test_system_prompt(self):
        agent = ChatAgent()
        assert agent.system_prompt == "You are a helpful assistant."

    def test_set_system_prompt(self):
        agent = ChatAgent()
        agent.system_prompt = "New prompt"
        assert agent.system_prompt == "New prompt"

    def test_set_empty_system_prompt(self):
        agent = ChatAgent()
        with pytest.raises(ValueError, match="empty"):
            agent.system_prompt = ""

    def test_get_context_window(self):
        agent = ChatAgent()
        agent.chat("hello")
        context = agent.get_context_window()
        assert context[0].role == "system"
        assert len(context) > 1

    def test_summarize_history(self):
        agent = ChatAgent()
        agent.chat("hello")
        agent.chat("how are you")
        summary = agent.summarize_history()
        assert "2 user messages" in summary

    def test_export_import(self):
        agent = ChatAgent()
        agent.chat("hello")
        exported = agent.export_history()
        agent2 = ChatAgent()
        agent2.import_history(exported)
        assert len(agent2.get_history()) == len(agent.get_history())
