"""Tests for nexus_llm.pipeline.cache module."""

import pytest
from nexus_llm.pipeline.cache import PipelineCache


class TestPipelineCache:
    """Tests for the PipelineCache class."""

    def test_init(self):
        cache = PipelineCache()
        assert cache is not None

    def test_put_get(self):
        cache = PipelineCache()
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_get_missing(self):
        cache = PipelineCache()
        assert cache.get("missing") is None

    def test_get_with_default(self):
        cache = PipelineCache()
        assert cache.get("missing", default="default") == "default"

    def test_has(self):
        cache = PipelineCache()
        cache.put("key1", "value1")
        assert cache.has("key1") is True
        assert cache.has("missing") is False

    def test_invalidate(self):
        cache = PipelineCache()
        cache.put("key1", "value1")
        cache.invalidate("key1")
        assert cache.has("key1") is False

    def test_clear(self):
        cache = PipelineCache()
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.clear()
        assert cache.has("key1") is False
        assert cache.has("key2") is False

    def test_size(self):
        cache = PipelineCache()
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        assert cache.size() == 2
