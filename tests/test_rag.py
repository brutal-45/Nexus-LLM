"""Tests for the RAG module.

Covers RAGEngine, DocumentStore, Retriever, Indexer, Chunker, and RAGPipeline.
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from nexus_llm.rag.document_store import Document, DocumentStore
from nexus_llm.rag.retriever import Retriever, RetrievalStrategy
from nexus_llm.rag.indexer import Indexer, InvertedIndex
from nexus_llm.rag.chunker import Chunker, Chunk, ChunkStrategy
from nexus_llm.rag.pipeline import RAGPipeline, RAGResult
from nexus_llm.rag.engine import RAGEngine, QueryResult


# ---------------------------------------------------------------------------
# Document & DocumentStore
# ---------------------------------------------------------------------------

class TestDocument:
    """Tests for Document dataclass."""

    def test_create_document(self):
        doc = Document(content="Hello world")
        assert doc.content == "Hello world"
        assert doc.id  # auto-generated
        assert doc.metadata == {}

    def test_create_document_with_id(self):
        doc = Document(content="Test", id="doc1")
        assert doc.id == "doc1"

    def test_to_dict(self):
        doc = Document(content="Test", id="d1", metadata={"key": "val"})
        d = doc.to_dict()
        assert d["content"] == "Test"
        assert d["id"] == "d1"
        assert d["metadata"] == {"key": "val"}

    def test_from_dict(self):
        data = {"content": "Hello", "id": "d2", "metadata": {}}
        doc = Document.from_dict(data)
        assert doc.content == "Hello"
        assert doc.id == "d2"


class TestDocumentStore:
    """Tests for DocumentStore."""

    def test_add_and_get(self):
        store = DocumentStore()
        doc = Document(content="Test content")
        doc_id = store.add(doc)
        retrieved = store.get(doc_id)
        assert retrieved is not None
        assert retrieved.content == "Test content"

    def test_get_nonexistent(self):
        store = DocumentStore()
        assert store.get("nonexistent") is None

    def test_delete(self):
        store = DocumentStore()
        doc = Document(content="Delete me", id="del1")
        store.add(doc)
        assert store.delete("del1") is True
        assert store.get("del1") is None

    def test_delete_nonexistent(self):
        store = DocumentStore()
        assert store.delete("nope") is False

    def test_count(self):
        store = DocumentStore()
        store.add(Document(content="A"))
        store.add(Document(content="B"))
        assert store.count() == 2

    def test_list_ids(self):
        store = DocumentStore()
        store.add(Document(content="A", id="id1"))
        store.add(Document(content="B", id="id2"))
        assert set(store.list_ids()) == {"id1", "id2"}

    def test_search(self):
        store = DocumentStore()
        store.add(Document(content="Python programming language", id="d1"))
        store.add(Document(content="Java programming language", id="d2"))
        store.add(Document(content="Cooking recipes", id="d3"))
        results = store.search("programming", top_k=2)
        assert len(results) == 2

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "docs.json")
            store = DocumentStore(persist_path=path)
            store.add(Document(content="Persist me", id="p1"))
            # Load a new store from the same path
            store2 = DocumentStore(persist_path=path)
            assert store2.count() == 1


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class TestRetriever:
    """Tests for Retriever."""

    def test_keyword_retrieve(self):
        retriever = Retriever(strategy="keyword", top_k=5)
        docs = [
            Document(content="Python is a programming language", id="d1"),
            Document(content="Java is also a programming language", id="d2"),
            Document(content="The sky is blue", id="d3"),
        ]
        retriever.index_documents(docs)
        results = retriever.retrieve("programming")
        assert len(results) > 0
        # Top result should be about programming
        top_doc, score = results[0]
        assert "programming" in top_doc.content.lower()
        assert score > 0

    def test_semantic_retrieve(self):
        retriever = Retriever(strategy="semantic", top_k=3)
        docs = [
            Document(content="Machine learning models", id="d1"),
            Document(content="Deep learning neural networks", id="d2"),
            Document(content="Cooking Italian food", id="d3"),
        ]
        retriever.index_documents(docs)
        results = retriever.retrieve("learning models")
        assert len(results) > 0

    def test_hybrid_retrieve(self):
        retriever = Retriever(strategy="hybrid", top_k=3)
        docs = [
            Document(content="Neural network architectures", id="d1"),
            Document(content="Transformer models for NLP", id="d2"),
        ]
        retriever.index_documents(docs)
        results = retriever.retrieve("neural network")
        assert len(results) > 0

    def test_invalid_strategy(self):
        with pytest.raises(ValueError):
            Retriever(strategy="invalid")


# ---------------------------------------------------------------------------
# Indexer & InvertedIndex
# ---------------------------------------------------------------------------

class TestInvertedIndex:
    """Tests for InvertedIndex."""

    def test_add_and_search(self):
        idx = InvertedIndex()
        doc = Document(content="hello world", id="d1")
        idx.add_document(doc)
        result = idx.search("hello")
        assert "d1" in result

    def test_search_nonexistent_term(self):
        idx = InvertedIndex()
        result = idx.search("nonexistent")
        assert result == set()

    def test_search_multi(self):
        idx = InvertedIndex()
        doc = Document(content="hello world python", id="d1")
        idx.add_document(doc)
        result = idx.search_multi(["hello", "world"])
        assert "d1" in result

    def test_doc_count(self):
        idx = InvertedIndex()
        idx.add_document(Document(content="a", id="d1"))
        idx.add_document(Document(content="b", id="d2"))
        assert idx.doc_count == 2

    def test_to_dict_and_from_dict(self):
        idx = InvertedIndex()
        idx.add_document(Document(content="test content", id="d1"))
        d = idx.to_dict()
        idx2 = InvertedIndex.from_dict(d)
        assert idx2.doc_count == 1


class TestIndexer:
    """Tests for Indexer."""

    def test_index_documents(self):
        indexer = Indexer()
        docs = [
            Document(content="hello world", id="d1"),
            Document(content="python programming", id="d2"),
        ]
        inv = indexer.index(docs)
        assert inv.doc_count == 2

    def test_build_index_from_store(self):
        store = DocumentStore()
        store.add(Document(content="test document", id="d1"))
        indexer = Indexer()
        inv = indexer.build_index(store)
        assert inv.doc_count == 1

    def test_save_and_load_index(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "index.json")
            indexer = Indexer()
            docs = [Document(content="save test", id="d1")]
            indexer.index(docs)
            indexer.save_index(path)
            indexer2 = Indexer()
            loaded = indexer2.load_index(path)
            assert loaded.doc_count == 1

    def test_save_without_index_raises(self):
        indexer = Indexer()
        with pytest.raises(RuntimeError):
            indexer.save_index("/tmp/test.json")

    def test_save_without_path_raises(self):
        indexer = Indexer()
        docs = [Document(content="test", id="d1")]
        indexer.index(docs)
        with pytest.raises(ValueError):
            indexer.save_index()


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------

class TestChunker:
    """Tests for Chunker."""

    def test_fixed_size_chunking(self):
        chunker = Chunker(strategy="fixed_size", chunk_size=20, overlap=5)
        chunks = chunker.chunk("A" * 100)
        assert len(chunks) > 1
        for c in chunks:
            assert isinstance(c, Chunk)
            assert c.content

    def test_sentence_chunking(self):
        chunker = Chunker(strategy="sentence", chunk_size=100)
        text = "This is sentence one. This is sentence two. This is sentence three."
        chunks = chunker.chunk(text, strategy="sentence")
        assert len(chunks) >= 1

    def test_paragraph_chunking(self):
        chunker = Chunker(strategy="paragraph", chunk_size=500)
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = chunker.chunk(text, strategy="paragraph")
        assert len(chunks) >= 1

    def test_semantic_chunking(self):
        chunker = Chunker(strategy="semantic", chunk_size=500)
        text = (
            "First topic is about AI.\n\n"
            "However, second topic is about biology.\n\n"
            "Furthermore, third topic is about chemistry."
        )
        chunks = chunker.chunk(text, strategy="semantic")
        assert len(chunks) >= 1

    def test_chunk_metadata(self):
        chunker = Chunker(strategy="fixed_size", chunk_size=20)
        chunks = chunker.chunk("Hello world this is a test")
        assert chunks[0].metadata is not None
        assert chunks[0].metadata.get("strategy") == "fixed_size"

    def test_empty_text(self):
        chunker = Chunker(chunk_size=50)
        chunks = chunker.chunk("")
        assert chunks == []

    def test_invalid_strategy(self):
        chunker = Chunker()
        with pytest.raises(ValueError):
            chunker.chunk("text", strategy="nonexistent")


# ---------------------------------------------------------------------------
# RAGPipeline
# ---------------------------------------------------------------------------

class TestRAGPipeline:
    """Tests for RAGPipeline."""

    def test_run_with_documents(self):
        pipeline = RAGPipeline(top_k=3)
        docs = [
            Document(content="Python is a programming language", id="d1"),
            Document(content="The Eiffel Tower is in Paris", id="d2"),
        ]
        pipeline.add_documents(docs)
        result = pipeline.run("What programming language?")
        assert isinstance(result, RAGResult)
        assert result.query == "What programming language?"

    def test_run_no_documents(self):
        pipeline = RAGPipeline()
        result = pipeline.run("What is Python?")
        assert isinstance(result, RAGResult)
        assert "could not find" in result.answer.lower() or result.answer

    def test_add_documents_returns_ids(self):
        pipeline = RAGPipeline()
        docs = [Document(content="Test content")]
        ids = pipeline.add_documents(docs)
        assert len(ids) == 1

    def test_rag_result_summary(self):
        result = RAGResult(
            query="test",
            answer="answer text",
            sources=[],
            scores=[],
        )
        assert "test" in result.summary()


# ---------------------------------------------------------------------------
# RAGEngine
# ---------------------------------------------------------------------------

class TestRAGEngine:
    """Tests for RAGEngine."""

    def test_init_defaults(self):
        engine = RAGEngine()
        assert engine.store is not None
        assert engine.retriever is not None
        assert engine.indexer is not None
        assert engine.chunker is not None
        assert engine.pipeline is not None

    def test_add_and_query(self):
        engine = RAGEngine()
        engine.add_documents([
            {"content": "Python is a popular programming language"},
            {"content": "The Eiffel Tower is in Paris, France"},
        ])
        result = engine.query("programming language")
        assert isinstance(result, QueryResult)
        assert isinstance(result.answer, str)

    def test_query_result_from_rag_result(self):
        rag_result = RAGResult(
            query="test",
            answer="test answer",
            sources=[Document(content="source")],
            scores=[0.9],
            metadata={"key": "value"},
        )
        qr = QueryResult.from_rag_result(rag_result)
        assert qr.answer == "test answer"
        assert len(qr.sources) == 1
        assert 0.9 in qr.scores

    def test_custom_params(self):
        engine = RAGEngine(
            retrieval_strategy="semantic",
            chunk_size=256,
            chunk_overlap=20,
            top_k=3,
        )
        assert engine._retrieval_strategy == "semantic"
