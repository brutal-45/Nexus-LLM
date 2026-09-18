"""Tests for evaluation presets loading."""

import pytest
from nexus_llm.presets.preset_manager import PresetManager


class TestEvaluationPresets:
    def test_load_presets(self):
        manager = PresetManager()
        presets = manager.load("evaluation_presets.yaml")
        assert isinstance(presets, dict)

    def test_has_default_preset(self):
        manager = PresetManager()
        presets = manager.load("evaluation_presets.yaml")
        assert len(presets) > 0

    def test_get_preset_by_name(self):
        manager = PresetManager()
        presets = manager.load("evaluation_presets.yaml")
        # Try to get any preset
        if presets:
            name = list(presets.keys())[0]
            preset = manager.get_preset(name)
            assert preset is not None
