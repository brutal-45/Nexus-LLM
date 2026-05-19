"""Tests for output formatting."""
import pytest
from nexus.chat.renderer import Colors, Theme, TerminalRenderer, THEMES


def test_colors_constants():
    """Test that color constants are valid ANSI codes."""
    assert Colors.RESET == "\033[0m"
    assert Colors.BOLD == "\033[1m"
    assert Colors.RED == "\033[31m"
    assert Colors.GREEN == "\033[32m"


def test_theme_creation():
    """Test creating a custom theme."""
    theme = Theme(name="test")
    assert theme.name == "test"
    assert theme.user_color is not None
    assert theme.assistant_color is not None


def test_builtin_themes():
    """Test that built-in themes are available."""
    assert "dark" in THEMES
    assert "light" in THEMES
    assert "dracula" in THEMES
    assert "monokai" in THEMES


def test_renderer_creation():
    """Test creating a terminal renderer."""
    renderer = TerminalRenderer(theme_name="dark")
    assert renderer.theme.name == "dark"


def test_renderer_markdown_code_block():
    """Test rendering a code block."""
    renderer = TerminalRenderer(theme_name="dark")
    text = "```python\nprint('hello')\n```"
    result = renderer.render_markdown(text)
    assert "python" in result
    assert "print" in result


def test_renderer_inline_code():
    """Test rendering inline code."""
    renderer = TerminalRenderer(theme_name="dark")
    text = "Use `pip install` to install"
    result = renderer.render_markdown(text)
    assert "pip install" in result


def test_renderer_inline_bold():
    """Test rendering bold text."""
    renderer = TerminalRenderer(theme_name="dark")
    text = "This is **important** text"
    result = renderer.render_markdown(text)
    assert "important" in result


def test_renderer_inline_italic():
    """Test rendering italic text."""
    renderer = TerminalRenderer(theme_name="dark")
    text = "This is *emphasized* text"
    result = renderer.render_markdown(text)
    assert "emphasized" in result
