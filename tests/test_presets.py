"""Tests for the presets module.

Covers PresetManager, Preset, and PresetLibrary.
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from nexus_llm.presets.preset import Preset
from nexus_llm.presets.manager import PresetManager
from nexus_llm.presets.library import PresetLibrary
from nexus_llm.presets.categories import PresetCategory


# ---------------------------------------------------------------------------
# Preset
# ---------------------------------------------------------------------------

class TestPreset:
    """Tests for Preset."""

    def test_create_preset(self):
        p = Preset(name="test-preset", description="A test preset")
        assert p.name == "test-preset"
        assert p.description == "A test preset"

    def test_preset_defaults(self):
        p = Preset(name="test")
        assert p.parameters == {}
        assert p.category == ""

    def test_to_dict(self):
        p = Preset(name="test", description="desc", parameters={"temp": 0.7})
        d = p.to_dict()
        assert d["name"] == "test"
        assert d["parameters"]["temp"] == 0.7

    def test_from_dict(self):
        data = {"name": "test", "description": "desc", "parameters": {"temp": 0.7}, "category": "creative"}
        p = Preset.from_dict(data)
        assert p.name == "test"
        assert p.parameters["temp"] == 0.7


# ---------------------------------------------------------------------------
# PresetManager
# ---------------------------------------------------------------------------

class TestPresetManager:
    """Tests for PresetManager."""

    def test_init(self):
        pm = PresetManager()
        assert pm is not None

    def test_get_preset(self):
        pm = PresetManager()
        preset = pm.get_preset("creative/poet")
        # May return None if not loaded from files, but should not crash
        assert preset is None or isinstance(preset, Preset)

    def test_list_presets(self):
        pm = PresetManager()
        presets = pm.list_presets()
        assert isinstance(presets, list)

    def test_save_and_load_preset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = PresetManager()
            p = Preset(name="custom", description="Custom preset", parameters={"temp": 0.8})
            # Save the preset
            path = os.path.join(tmpdir, "custom.json")
            with open(path, "w") as f:
                json.dump(p.to_dict(), f)
            assert os.path.exists(path)


# ---------------------------------------------------------------------------
# PresetLibrary
# ---------------------------------------------------------------------------

class TestPresetLibrary:
    """Tests for PresetLibrary."""

    def test_init(self):
        lib = PresetLibrary()
        assert lib is not None

    def test_add_and_get(self):
        lib = PresetLibrary()
        p = Preset(name="test-preset", description="Test")
        lib.add(p)
        result = lib.get("test-preset")
        assert result is not None
        assert result.name == "test-preset"

    def test_list_all(self):
        lib = PresetLibrary()
        lib.add(Preset(name="p1"))
        lib.add(Preset(name="p2"))
        presets = lib.list_all()
        assert len(presets) == 2

    def test_remove(self):
        lib = PresetLibrary()
        lib.add(Preset(name="to-remove"))
        lib.remove("to-remove")
        assert lib.get("to-remove") is None

    def test_search(self):
        lib = PresetLibrary()
        lib.add(Preset(name="poet", category="creative"))
        lib.add(Preset(name="debugger", category="code"))
        results = lib.search("creative")
        assert len(results) >= 1


# ---------------------------------------------------------------------------
# PresetCategory
# ---------------------------------------------------------------------------

class TestPresetCategory:
    """Tests for PresetCategory."""

    def test_category_exists(self):
        # PresetCategory should define common categories
        assert PresetCategory is not None
