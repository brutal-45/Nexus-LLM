"""Tests for agents configuration loading."""

import pytest
from nexus_llm.config_loader import ConfigLoader


class TestAgentsConfig:
    def test_load_default(self):
        loader = ConfigLoader()
        config = loader.load("nexus_llm/config/agents_config.yaml")
        assert isinstance(config, dict)

    def test_has_agent_definitions(self):
        loader = ConfigLoader()
        config = loader.load("nexus_llm/config/agents_config.yaml")
        assert len(config) > 0
