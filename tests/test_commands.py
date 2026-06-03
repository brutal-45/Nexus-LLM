"""Tests for slash commands."""
import pytest
from unittest.mock import MagicMock
from nexus.chat.commands import Command, CommandHandler


class MockChat:
    """Mock chat interface for testing commands."""
    def __init__(self):
        self.config = MagicMock()
        self.config.model_name = "nexus-7b"
        self.config.backend = "local"
        self.config.history.history_dir = "/tmp/nexus_history"
        self.config.generation.temperature = 0.7
        self.config.generation.top_p = 0.9
        self.config.generation.top_k = 50
        self.config.generation.max_tokens = 4096
        self.config.generation.stream = True
        self.config.ui.theme = "dark"
        self.conversation = MagicMock()
        self.renderer = MagicMock()


@pytest.fixture
def handler():
    chat = MockChat()
    return CommandHandler(chat)


def test_command_handler_creation(handler):
    """Test that CommandHandler is created with built-in commands."""
    assert len(handler.commands) > 0
    assert "help" in handler.commands
    assert "clear" in handler.commands
    assert "quit" in handler.commands


def test_command_handler_aliases(handler):
    """Test that command aliases work."""
    assert "h" in handler.commands
    assert "?" in handler.commands
    assert handler.commands["h"].name == handler.commands["help"].name


def test_command_handler_non_command(handler):
    """Test that non-commands return False."""
    was_cmd, result = handler.handle("Hello there")
    assert was_cmd is False
    assert result is None


def test_command_handler_unknown_command(handler):
    """Test handling an unknown command."""
    was_cmd, result = handler.handle("/unknown_cmd")
    assert was_cmd is True


def test_command_dataclass():
    """Test Command dataclass."""
    cmd = Command(
        name="test",
        description="A test command",
        usage="/test",
        aliases=["t"],
    )
    assert cmd.name == "test"
    assert cmd.description == "A test command"
    assert cmd.aliases == ["t"]
    assert cmd.handler is None


def test_command_handler_help(handler):
    """Test that help command is registered."""
    help_cmd = handler.commands.get("help")
    assert help_cmd is not None
    assert help_cmd.name == "help"


def test_command_handler_model(handler):
    """Test model command registration."""
    model_cmd = handler.commands.get("model")
    assert model_cmd is not None
    assert model_cmd.name == "model"


def test_command_handler_all_commands_have_handlers(handler):
    """Test that all primary commands have handlers."""
    seen = set()
    for name, cmd in handler.commands.items():
        if cmd.name not in seen:
            seen.add(cmd.name)
            assert cmd.handler is not None, f"Command {cmd.name} has no handler"
