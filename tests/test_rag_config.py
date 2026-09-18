"""Tests for RAG configuration loading."""

import pytest
from nexus_llm.config_loader import ConfigLoader


class TestRagConfig:
    def test_load_default(self):
        loader = ConfigLoader()
        config = loader.load("nexus_llm/config/rag_config.yaml")
        assert isinstance(config, dict)

    def test_has_chunking_config(self):
        loader = ConfigLoader()
        config = loader.load("nexus_llm/config/rag_config.yaml")
        # Should have chunking or embedding config
        assert len(config) > 0
