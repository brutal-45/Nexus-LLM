"""Tests for ANSI utilities."""
import pytest
from nexus.chat.renderer import Colors


def test_ansi_reset():
    """Test ANSI reset code."""
    assert Colors.RESET == "\033[0m"


def test_ansi_bold():
    """Test ANSI bold code."""
    assert Colors.BOLD == "\033[1m"


def test_ansi_colors_foreground():
    """Test ANSI foreground color codes."""
    assert Colors.RED == "\033[31m"
    assert Colors.GREEN == "\033[32m"
    assert Colors.BLUE == "\033[34m"
    assert Colors.YELLOW == "\033[33m"


def test_ansi_colors_bright():
    """Test ANSI bright color codes."""
    assert Colors.BRIGHT_RED == "\033[91m"
    assert Colors.BRIGHT_GREEN == "\033[92m"
    assert Colors.BRIGHT_BLUE == "\033[94m"


def test_ansi_styles():
    """Test ANSI style codes."""
    assert Colors.ITALIC == "\033[3m"
    assert Colors.UNDERLINE == "\033[4m"
    assert Colors.DIM == "\033[2m"


def test_ansi_background():
    """Test ANSI background color codes."""
    assert Colors.BG_RED == "\033[41m"
    assert Colors.BG_GREEN == "\033[42m"


def test_ansi_code_format():
    """Test that all ANSI codes follow the escape sequence format."""
    for attr_name in dir(Colors):
        if attr_name.startswith("_"):
            continue
        value = getattr(Colors, attr_name)
        if isinstance(value, str) and value.startswith("\033["):
            assert value.endswith("m")
