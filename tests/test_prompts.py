"""Tests for input prompts."""
import pytest
from nexus.chat.config import ChatConfig


def test_default_system_prompt():
    """Test default system prompt is set."""
    cfg = ChatConfig()
    assert cfg.system_prompt is not None
    assert len(cfg.system_prompt) > 0
    assert "Nexus" in cfg.system_prompt


def test_custom_system_prompt():
    """Test setting a custom system prompt."""
    cfg = ChatConfig(system_prompt="You are a coding assistant.")
    assert cfg.system_prompt == "You are a coding assistant."


def test_system_prompt_in_conversation():
    """Test that system prompt is added to conversation."""
    from nexus.chat.conversation import Conversation, Role
    conv = Conversation(system_prompt="Be helpful")
    assert len(conv.messages) == 1
    assert conv.messages[0].role == Role.SYSTEM
    assert conv.messages[0].content == "Be helpful"


def test_prompt_template_formatting():
    """Test prompt template formatting."""
    template = "### Instruction:\n{instruction}\n\n### Response:\n{response}"
    result = template.format(instruction="Write code", response="print('hello')")
    assert "Write code" in result
    assert "print" in result
