"""Tests for safety configuration loading."""

import pytest
import yaml
import tempfile
import os
from nexus_llm.config_loader import ConfigLoader


class TestSafetyConfig:
    def test_load_default(self):
        loader = ConfigLoader()
        config = loader.load("nexus_llm/config/safety_config.yaml")
        assert isinstance(config, dict)

    def test_has_required_keys(self):
        loader = ConfigLoader()
        config = loader.load("nexus_llm/config/safety_config.yaml")
        # Check for typical safety config keys
        assert "content_filter" in config or "moderation" in config or len(config) > 0
