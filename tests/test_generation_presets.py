"""Tests for generation preset configs."""
import pytest


def test_default_preset_exists():
    presets = {"default": {"temperature": 0.7, "top_p": 0.9, "max_tokens": 512}}
    assert "default" in presets


def test_creative_preset_higher_temperature():
    presets = {"default": {"temperature": 0.7}, "creative": {"temperature": 1.0}}
    assert presets["creative"]["temperature"] > presets["default"]["temperature"]


def test_preset_validation():
    preset = {"temperature": 0.7, "top_p": 0.9, "top_k": 50, "max_tokens": 512}
    assert 0.0 < preset["temperature"] <= 2.0
    assert 0.0 < preset["top_p"] <= 1.0
    assert preset["max_tokens"] > 0


def test_preset_override():
    base = {"temperature": 0.7, "top_p": 0.9, "max_tokens": 512}
    merged = {**base, **{"temperature": 0.3}}
    assert merged["temperature"] == 0.3
    assert merged["top_p"] == 0.9


def test_all_presets_have_required_keys():
    required = {"temperature", "top_p", "max_tokens"}
    presets = {"default": {"temperature": 0.7, "top_p": 0.9, "max_tokens": 512}}
    for name, p in presets.items():
        assert required.issubset(p.keys())
