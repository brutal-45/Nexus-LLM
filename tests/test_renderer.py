"""Tests for text renderer."""
import pytest
from nexus.chat.renderer import TerminalRenderer, Theme, THEMES, StreamState


def test_renderer_default_theme():
    """Test renderer uses dark theme by default."""
    r = TerminalRenderer()
    assert r.theme.name == "dark"


def test_renderer_custom_theme():
    """Test renderer with custom theme."""
    r = TerminalRenderer(theme_name="monokai")
    assert r.theme.name == "monokai"


def test_renderer_unknown_theme_fallback():
    """Test that unknown theme falls back to dark."""
    r = TerminalRenderer(theme_name="nonexistent")
    assert r.theme.name == "dark"


def test_renderer_max_width():
    """Test renderer max width setting."""
    r = TerminalRenderer(max_width=80)
    assert r.max_width <= 80


def test_stream_state_defaults():
    """Test StreamState default values."""
    state = StreamState()
    assert state.is_in_code_block is False
    assert state.token_count == 0
    assert state.code_buffer is not None
    assert state.start_time > 0


def test_renderer_render_plain_text():
    """Test rendering plain text passes through."""
    r = TerminalRenderer()
    result = r.render_markdown("Hello world")
    assert "Hello world" in result


def test_renderer_render_code_block():
    """Test rendering a code block."""
    r = TerminalRenderer()
    text = "```python\nprint('hi')\n```"
    result = r.render_markdown(text)
    assert "python" in result
