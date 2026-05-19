"""Tests for chat interface."""
import pytest
from nexus.chat.conversation import Conversation, Message, Role, ConversationStats


def test_conversation_creation():
    """Test creating a new conversation."""
    conv = Conversation(system_prompt="You are helpful.")
    assert conv.system_prompt == "You are helpful."
    assert len(conv.messages) == 1
    assert conv.messages[0].role == Role.SYSTEM


def test_conversation_add_user_message():
    """Test adding a user message."""
    conv = Conversation()
    msg = conv.add_user_message("Hello!")
    assert msg.role == Role.USER
    assert msg.content == "Hello!"
    assert len(conv.messages) == 1


def test_conversation_add_assistant_message():
    """Test adding an assistant message."""
    conv = Conversation()
    conv.add_user_message("Hi")
    msg = conv.add_assistant_message("Hello!", model="nexus-7b", token_count=5)
    assert msg.role == Role.ASSISTANT
    assert msg.content == "Hello!"
    assert msg.model == "nexus-7b"
    assert msg.token_count == 5


def test_conversation_get_api_messages():
    """Test getting API-formatted messages."""
    conv = Conversation(system_prompt="Be helpful.")
    conv.add_user_message("Hi")
    conv.add_assistant_message("Hello!", model="test")
    api_msgs = conv.get_api_messages()
    assert len(api_msgs) == 3
    assert api_msgs[0]["role"] == "system"
    assert api_msgs[1]["role"] == "user"


def test_conversation_clear():
    """Test clearing conversation while keeping system prompt."""
    conv = Conversation(system_prompt="Test")
    conv.add_user_message("Hi")
    conv.add_assistant_message("Hello!")
    conv.clear(keep_system=True)
    assert len(conv.messages) == 1
    assert conv.messages[0].role == Role.SYSTEM


def test_conversation_clear_all():
    """Test clearing all messages including system."""
    conv = Conversation(system_prompt="Test")
    conv.add_user_message("Hi")
    conv.clear(keep_system=False)
    assert len(conv.messages) == 0


def test_conversation_get_stats():
    """Test conversation statistics."""
    conv = Conversation()
    conv.add_user_message("Hello")
    conv.add_assistant_message("Hi there!", token_count=10, model="nexus")
    stats = conv.get_stats()
    assert stats.total_messages == 2
    assert stats.user_messages == 1
    assert stats.assistant_messages == 1
    assert stats.total_tokens == 10
    assert "nexus" in stats.models_used


def test_conversation_get_last_assistant_message():
    """Test getting the last assistant message."""
    conv = Conversation()
    assert conv.get_last_assistant_message() is None
    conv.add_user_message("Hi")
    conv.add_assistant_message("Hello!")
    last = conv.get_last_assistant_message()
    assert last is not None
    assert last.content == "Hello!"


def test_message_to_dict():
    """Test Message serialization."""
    msg = Message(role=Role.USER, content="test")
    d = msg.to_dict()
    assert d["role"] == "user"
    assert d["content"] == "test"


def test_message_from_dict():
    """Test Message deserialization."""
    d = {"role": "assistant", "content": "response", "token_count": 5}
    msg = Message.from_dict(d)
    assert msg.role == Role.ASSISTANT
    assert msg.content == "response"
    assert msg.token_count == 5


def test_message_format_for_api():
    """Test Message API formatting."""
    msg = Message(role=Role.USER, content="Hello")
    api = msg.format_for_api()
    assert api == {"role": "user", "content": "Hello"}


def test_conversation_len():
    """Test conversation length."""
    conv = Conversation()
    assert len(conv) == 0
    conv.add_user_message("Hi")
    assert len(conv) == 1


def test_conversation_repr():
    """Test conversation string representation."""
    conv = Conversation()
    conv.add_user_message("Hi")
    r = repr(conv)
    assert "Conversation" in r
    assert "messages=1" in r
