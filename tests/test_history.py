"""Tests for chat history management."""
import pytest
import json
import os
import tempfile
from nexus.chat.conversation import Conversation, Message, Role
from nexus.chat.config import HistoryConfig


def test_history_config_defaults():
    """Test HistoryConfig defaults."""
    cfg = HistoryConfig()
    assert cfg.max_context_messages == 50
    assert cfg.max_context_tokens == 32000
    assert cfg.save_history is True
    assert cfg.auto_save is True


def test_conversation_save_load(tmp_dir):
    """Test saving and loading conversation history."""
    conv = Conversation(system_prompt="Be helpful")
    conv.add_user_message("Hello")
    conv.add_assistant_message("Hi there!", token_count=10)
    
    filepath = os.path.join(tmp_dir, "test_conv.json")
    conv.save(filepath)
    assert os.path.exists(filepath)
    
    loaded = Conversation.load(filepath)
    assert len(loaded.messages) == 3
    assert loaded.messages[1].content == "Hello"


def test_conversation_save_contains_all_messages(tmp_dir):
    """Test that saved conversation preserves all messages."""
    conv = Conversation()
    for i in range(5):
        conv.add_user_message(f"Message {i}")
        conv.add_assistant_message(f"Reply {i}", token_count=10)
    
    filepath = os.path.join(tmp_dir, "multi_msg.json")
    conv.save(filepath)
    loaded = Conversation.load(filepath)
    assert len(loaded.messages) == 10


def test_conversation_export_markdown():
    """Test exporting conversation as markdown."""
    conv = Conversation(system_prompt="Test system")
    conv.add_user_message("Hello")
    conv.add_assistant_message("Hi there!", token_count=5)
    
    md = conv.export_markdown()
    assert "Nexus Chat" in md
    assert "Hello" in md
    assert "Hi there!" in md
    assert "Stats" in md


def test_conversation_context_window_limit():
    """Test that context window limits are respected."""
    conv = Conversation(max_context_messages=5)
    for i in range(10):
        conv.add_user_message(f"Msg {i}")
    
    api_msgs = conv.get_api_messages()
    assert len(api_msgs) <= 5
