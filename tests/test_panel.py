"""Tests for panel display."""
import pytest
from nexus.chat.renderer import TerminalRenderer


@pytest.fixture
def renderer():
    return TerminalRenderer(theme_name="dark")


def test_panel_divider(renderer, capsys):
    """Test printing a divider."""
    renderer.print_divider()
    captured = capsys.readouterr()
    assert len(captured.out) > 0


def test_panel_key_value(renderer, capsys):
    """Test printing key-value pairs."""
    renderer.print_key_value("Model", "nexus-7b")
    captured = capsys.readouterr()
    assert "Model" in captured.out
    assert "nexus-7b" in captured.out


def test_panel_code_block(renderer, capsys):
    """Test printing a code block."""
    renderer.print_code_block("print('hello')", language="python")
    captured = capsys.readouterr()
    assert "print" in captured.out
