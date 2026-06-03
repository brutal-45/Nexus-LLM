"""Tests for the retrieval module.

Covers RetrievalEngine, VectorIndex, and HybridRetriever.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from nexus_llm.retrieval.engine import RetrievalEngine
from nexus_llm.retrieval.vector_index import VectorIndex
from nexus_llm.retrieval.hybrid import HybridRetriever


# ---------------------------------------------------------------------------
# VectorIndex
# ---------------------------------------------------------------------------

class TestVectorIndex:
    """Tests for VectorIndex."""

    def test_create_index(self):
        idx = VectorIndex()
        assert idx is not None

    def test_add_and_search(self):
        idx = VectorIndex()
        idx.add("doc1", [0.1, 0.2, 0.3])
        idx.add("doc2", [0.4, 0.5, 0.6])
        idx.add("doc3", [0.7, 0.8, 0.9])
        results = idx.search([0.1, 0.2, 0.3], top_k=2)
        assert len(results) >= 1

    def test_count(self):
        idx = VectorIndex()
        idx.add("doc1", [0.1, 0.2])
        idx.add("doc2", [0.3, 0.4])
        assert idx.count() == 2

    def test_delete(self):
        idx = VectorIndex()
        idx.add("doc1", [0.1, 0.2])
        idx.delete("doc1")
        assert idx.count() == 0


# ---------------------------------------------------------------------------
# RetrievalEngine
# ---------------------------------------------------------------------------

class TestRetrievalEngine:
    """Tests for RetrievalEngine."""

    def test_create_engine(self):
        engine = RetrievalEngine()
        assert engine is not None

    def test_add_documents(self):
        engine = RetrievalEngine()
        engine.add_documents([
            {"id": "d1", "content": "Python is great", "embedding": [0.1, 0.2]},
        ])
        # Should not crash

    def test_retrieve(self):
        engine = RetrievalEngine()
        engine.add_documents([
            {"id": "d1", "content": "Python programming", "embedding": [0.1, 0.2]},
            {"id": "d2", "content": "Java programming", "embedding": [0.3, 0.4]},
        ])
        results = engine.retrieve(query="programming", top_k=2)
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# HybridRetriever
# ---------------------------------------------------------------------------

class TestHybridRetriever:
    """Tests for HybridRetriever."""

    def test_create_retriever(self):
        retriever = HybridRetriever()
        assert retriever is not None

    def test_add_and_retrieve(self):
        retriever = HybridRetriever()
        retriever.add_document("d1", "Python is a programming language", [0.1, 0.2])
        retriever.add_document("d2", "Cooking Italian food", [0.5, 0.6])
        results = retriever.retrieve("programming", top_k=1)
        assert isinstance(results, list)
