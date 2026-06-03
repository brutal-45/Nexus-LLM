"""Tests for markdown extensions."""
import pytest
from nexus.chat.renderer import TerminalRenderer


@pytest.fixture
def renderer():
    return TerminalRenderer(theme_name="dark")


def test_markdown_inline_code(renderer):
    """Test inline code rendering."""
    text = "Use `pip install nexus` to install"
    result = renderer.render_markdown(text)
    assert "pip install nexus" in result


def test_markdown_bold(renderer):
    """Test bold text rendering."""
    text = "This is **important** text"
    result = renderer.render_markdown(text)
    assert "important" in result


def test_markdown_italic(renderer):
    """Test italic text rendering."""
    text = "This is *emphasized* text"
    result = renderer.render_markdown(text)
    assert "emphasized" in result


def test_markdown_heading(renderer):
    """Test heading rendering."""
    text = "# Main Title"
    result = renderer.render_markdown(text)
    assert "Main Title" in result


def test_markdown_link(renderer):
    """Test link rendering."""
    text = "[Click here](https://example.com)"
    result = renderer.render_markdown(text)
    assert "Click here" in result


def test_markdown_code_block(renderer):
    """Test fenced code block rendering."""
    text = "```python\nprint('hello')\n```"
    result = renderer.render_markdown(text)
    assert "python" in result
