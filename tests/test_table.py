"""Tests for table display."""
import pytest
from nexus.chat.renderer import TerminalRenderer


@pytest.fixture
def renderer():
    return TerminalRenderer(theme_name="dark")


def test_table_rendering(renderer, capsys):
    """Test rendering a table."""
    headers = ["Name", "Value", "Status"]
    rows = [["model", "nexus-7b", "loaded"], ["temp", "0.7", "active"]]
    renderer.print_table(headers, rows)
    captured = capsys.readouterr()
    assert "model" in captured.out
    assert "nexus-7b" in captured.out


def test_table_column_alignment(renderer, capsys):
    """Test table column alignment."""
    headers = ["Short", "Medium Length", "X"]
    rows = [["a", "hello", "1"], ["bb", "world!", "22"]]
    renderer.print_table(headers, rows)
    captured = capsys.readouterr()
    assert "Short" in captured.out


def test_table_empty():
    """Test rendering an empty table does not crash."""
    r = TerminalRenderer()
    # Should not raise
    r.print_table([], [])
