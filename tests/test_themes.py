"""Tests for terminal themes."""
import pytest
from nexus.chat.renderer import Theme, THEMES, Colors


def test_themes_available():
    """Test that all expected themes are available."""
    assert "dark" in THEMES
    assert "light" in THEMES
    assert "dracula" in THEMES
    assert "monokai" in THEMES


def test_theme_attributes():
    """Test that themes have required attributes."""
    for name, theme in THEMES.items():
        assert hasattr(theme, "name")
        assert hasattr(theme, "user_color")
        assert hasattr(theme, "assistant_color")
        assert hasattr(theme, "error")
        assert hasattr(theme, "success")
        assert hasattr(theme, "heading")


def test_theme_name_matches_key():
    """Test that theme name matches dict key."""
    for key, theme in THEMES.items():
        assert theme.name == key


def test_custom_theme_creation():
    """Test creating a custom theme."""
    custom = Theme(
        name="custom",
        user_color=Colors.BRIGHT_CYAN,
        assistant_color=Colors.BRIGHT_GREEN,
    )
    assert custom.name == "custom"
    assert custom.user_color == Colors.BRIGHT_CYAN


def test_colors_are_ansi():
    """Test that color constants are ANSI escape codes."""
    assert Colors.RESET.startswith("\033[")
    assert Colors.BOLD.startswith("\033[")
    assert Colors.RED.startswith("\033[")
