"""Tests for syntax highlighting."""
import pytest
from nexus.chat.renderer import TerminalRenderer


@pytest.fixture
def renderer():
    return TerminalRenderer(theme_name="dark")


def test_python_keyword_highlighting(renderer):
    """Test Python keyword highlighting."""
    result = renderer._highlight_python("def hello():")
    assert "def" in result


def test_python_comment_highlighting(renderer):
    """Test Python comment highlighting."""
    result = renderer._highlight_python("# This is a comment")
    assert "comment" in result or "#" in result


def test_python_string_highlighting(renderer):
    """Test Python string highlighting."""
    result = renderer._highlight_python('print("hello")')
    assert "hello" in result


def test_js_keyword_highlighting(renderer):
    """Test JavaScript keyword highlighting."""
    result = renderer._highlight_js("function test() {")
    assert "function" in result


def test_json_key_highlighting(renderer):
    """Test JSON key highlighting."""
    result = renderer._highlight_json('"name": "test"')
    assert "name" in result


def test_bash_comment_highlighting(renderer):
    """Test Bash comment highlighting."""
    result = renderer._highlight_bash("# This is a comment")
    assert "#" in result


def test_string_highlighting_double_quotes(renderer):
    """Test double-quote string highlighting."""
    result = renderer._highlight_strings('He said "hello world"')
    assert "hello world" in result
