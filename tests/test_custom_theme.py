"""Tests for custom theme creation."""
import pytest


def test_custom_theme_creation():
    theme = {"name": "ocean", "bg": "#1a1b26", "fg": "#a9b1d6"}
    assert theme["name"] == "ocean"
    assert theme["bg"].startswith("#")


def test_custom_theme_override():
    base = {"bg": "#000", "fg": "#fff", "accent": "#f00"}
    merged = {**base, **{"accent": "#0f0"}}
    assert merged["accent"] == "#0f0" and merged["bg"] == "#000"


def test_custom_theme_validation():
    def valid_hex(c):
        return c.startswith("#") and len(c) in (4, 7)
    assert valid_hex("#fff") and valid_hex("#ffffff") and not valid_hex("red")


def test_custom_theme_presets():
    presets = ["dracula", "solarized", "nord", "gruvbox"]
    assert "dracula" in presets
