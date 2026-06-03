"""Tests for the embeddings module.

Covers EmbeddingEngine, EmbeddingStore, and EmbeddingCache.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from nexus_llm.embeddings.engine import EmbeddingEngine
from nexus_llm.embeddings.store import EmbeddingStore
from nexus_llm.embeddings.cache import EmbeddingCache


# ---------------------------------------------------------------------------
# EmbeddingEngine
# ---------------------------------------------------------------------------

class TestEmbeddingEngine:
    """Tests for EmbeddingEngine."""

    def test_create_engine(self):
        engine = EmbeddingEngine()
        assert engine is not None

    @patch.object(EmbeddingEngine, 'embed', return_value=[0.1, 0.2, 0.3])
    def test_embed_single_text(self, mock_embed):
        engine = EmbeddingEngine()
        result = engine.embed("Hello world")
        assert isinstance(result, list)
        assert len(result) == 3

    @patch.object(EmbeddingEngine, 'embed_batch', return_value=[[0.1, 0.2], [0.3, 0.4]])
    def test_embed_batch(self, mock_batch):
        engine = EmbeddingEngine()
        result = engine.embed_batch(["Hello", "World"])
        assert len(result) == 2

    def test_get_dimension(self):
        engine = EmbeddingEngine()
        dim = engine.get_dimension()
        assert isinstance(dim, int)
        assert dim > 0


# ---------------------------------------------------------------------------
# EmbeddingStore
# ---------------------------------------------------------------------------

class TestEmbeddingStore:
    """Tests for EmbeddingStore."""

    def test_create_store(self):
        store = EmbeddingStore()
        assert store is not None

    def test_add_and_search(self):
        store = EmbeddingStore()
        store.add("doc1", [0.1, 0.2, 0.3])
        store.add("doc2", [0.4, 0.5, 0.6])
        results = store.search([0.1, 0.2, 0.3], top_k=1)
        assert len(results) >= 1

    def test_delete(self):
        store = EmbeddingStore()
        store.add("doc1", [0.1, 0.2, 0.3])
        store.delete("doc1")
        # After deletion, search should not return doc1
        assert store.count() == 0

    def test_count(self):
        store = EmbeddingStore()
        store.add("doc1", [0.1, 0.2])
        store.add("doc2", [0.3, 0.4])
        assert store.count() == 2


# ---------------------------------------------------------------------------
# EmbeddingCache
# ---------------------------------------------------------------------------

class TestEmbeddingCache:
    """Tests for EmbeddingCache."""

    def test_create_cache(self):
        cache = EmbeddingCache()
        assert cache is not None

    def test_get_miss(self):
        cache = EmbeddingCache()
        result = cache.get("nonexistent text")
        assert result is None

    def test_put_and_get(self):
        cache = EmbeddingCache()
        cache.put("hello", [0.1, 0.2, 0.3])
        result = cache.get("hello")
        assert result == [0.1, 0.2, 0.3]

    def test_cache_size(self):
        cache = EmbeddingCache()
        cache.put("a", [0.1])
        cache.put("b", [0.2])
        assert cache.size() == 2

    def test_clear(self):
        cache = EmbeddingCache()
        cache.put("a", [0.1])
        cache.clear()
        assert cache.size() == 0
