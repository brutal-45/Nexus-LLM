"""Tests for :mod:`nexus_llm.terminal.ansi`."""

from __future__ import annotations

import pytest

from nexus_llm.terminal.ansi import (
    AnsiColor,
    AnsiFormatter,
    AnsiStyle,
    bg_color256,
    color256,
    rgb,
)


@pytest.fixture
def formatter() -> AnsiFormatter:
    return AnsiFormatter(support_color=True)


class TestAnsiCodes:
    """Escape sequences must be exact — terminals are unforgiving."""

    def test_supported_colors(self):
        assert {c.value for c in AnsiColor} >= {"red", "green", "blue", "yellow", "white", "black"}

    def test_supported_styles(self):
        assert {s.value for s in AnsiStyle} >= {"reset", "bold", "dim", "italic", "underline"}

    def test_foreground_code(self, formatter):
        assert formatter.fg("red") == "\x1b[31m"

    def test_background_code(self, formatter):
        assert formatter.bg("red") == "\x1b[41m"

    def test_color256(self):
        assert color256(9) == "\x1b[38;5;9m"

    def test_color256_out_of_range_is_clamped(self):
        assert color256(-1) == "\x1b[38;5;0m"
        assert color256(999) == "\x1b[38;5;255m"

    def test_bg_color256_uses_background_slot(self):
        assert bg_color256(4).startswith("\x1b[48;5;")

    def test_truecolor_rgb(self):
        assert rgb(10, 20, 30) == "\x1b[38;2;10;20;30m"

    def test_clear_screen_and_cursor(self, formatter):
        assert formatter.clear_screen() == "\x1b[2J"
        assert formatter.cursor_home() == "\x1b[H"
        assert formatter.hide_cursor() == "\x1b[?25l"
        assert formatter.show_cursor() == "\x1b[?25h"

    def test_cursor_position(self, formatter):
        assert formatter.cursor_pos(3, 7) == "\x1b[3;7H"

    def test_alt_screen_switch(self, formatter):
        assert formatter.enter_alt_screen() != formatter.exit_alt_screen()


class TestAnsiRendering:
    def test_style_wraps_and_resets(self, formatter):
        assert formatter.style("x", "bold") == "\x1b[1mx\x1b[0m"

    def test_style_with_unknown_style_is_a_noop(self, formatter):
        assert formatter.style("plain", "not-a-style") == "plain"

    def test_color_paints_text(self, formatter):
        painted = formatter.color("hi", fg="red")
        assert "hi" in painted and painted.endswith("\x1b[0m")

    def test_hyperlink(self, formatter):
        link = formatter.hyperlink("https://example.com", "docs")
        assert "https://example.com" in link and "docs" in link

    def test_set_title(self, formatter):
        assert formatter.set_title("nexus") == "\x1b]0;nexus\x07"


class TestAnsiMeasurement:
    def test_strip_ansi(self, formatter):
        assert formatter.strip_ansi("\x1b[31mhi\x1b[0m") == "hi"

    def test_visible_length_ignores_escapes(self, formatter):
        assert formatter.visible_length("\x1b[31mhello\x1b[0m") == 5

    def test_visible_length_of_plain_text(self, formatter):
        assert formatter.visible_length("plain") == 5

    def test_strip_handles_empty(self, formatter):
        assert formatter.strip_ansi("") == ""


class TestAnsiNoColorMode:
    def test_color_disabled_strips_formatting(self):
        quiet = AnsiFormatter(support_color=False)
        assert quiet.fg("red") == ""
        assert quiet.style("text", "bold") == "text"
        assert quiet.color("text", fg="red") == "text"

    def test_supports_color_property(self):
        assert AnsiFormatter(support_color=True).supports_color is True
        assert AnsiFormatter(support_color=False).supports_color is False
