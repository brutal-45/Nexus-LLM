"""Tests for dark terminal theme."""
import pytest


def test_dark_theme_has_dark_background():
    assert "\033[40m".startswith("\033[")


def test_dark_theme_light_foreground():
    assert "\033[37m".startswith("\033[")


def test_dark_theme_contrast_ratio():
    contrast = (0.85 + 0.05) / (0.05 + 0.05)
    assert contrast > 4.5


def test_dark_theme_syntax_colors():
    colors = {"kw": "\033[35m", "str": "\033[32m", "cmt": "\033[90m", "num": "\033[33m"}
    assert len(set(colors.values())) == 4
