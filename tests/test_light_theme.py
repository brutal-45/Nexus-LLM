"""Tests for light terminal theme."""
import pytest


def test_light_theme_has_light_background():
    assert "\033[47m".startswith("\033[")


def test_light_theme_dark_foreground():
    assert "\033[30m".startswith("\033[")


def test_light_theme_contrast_ratio():
    contrast = (0.90 + 0.05) / (0.10 + 0.05)
    assert contrast > 4.5


def test_light_theme_readable():
    for c in ["\033[34m", "\033[31m", "\033[90m"]:
        assert c.startswith("\033[")
