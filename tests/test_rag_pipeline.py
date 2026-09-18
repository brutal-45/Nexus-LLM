"""Test RAG pipeline for Nexus-LLM."""
import pytest
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class RAGConfig:
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 5
    temperature: float = 0.7
    max_length: int = 2048


@dataclass
class RAGResponse:
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float = 0.0


class SimpleDocumentStore:
    def __init__(self):
        self._docs: Dict[str, str] = {}

    def add(self, doc_id: str, text: str):
        self._docs[doc_id] = text

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        query_terms = set(query.lower().split())
        scored = []
        for doc_id, text in self._docs.items():
            doc_terms = set(text.lower().split())
            overlap = len(query_terms & doc_terms)
            if overlap > 0:
                score = overlap / max(len(query_terms), 1)
                scored.append({"id": doc_id, "text": text, "score": score})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def count(self):
        return len(self._docs)


class RAGPipeline:
    def __init__(self, config: RAGConfig = None):
        self._config = config or RAGConfig()
        self._store = SimpleDocumentStore()
        self._chunker = SimpleChunker(chunk_size=self._config.chunk_size)

    @property
    def config(self):
        return self._config

    def index_documents(self, documents: List[Dict]):
        for doc in documents:
            doc_id = doc.get("id", str(hash(doc.get("text", ""))))
            self._store.add(doc_id, doc.get("text", ""))
        return len(documents)

    def query(self, question: str) -> RAGResponse:
        if not question:
            raise ValueError("Question cannot be empty")

        results = self._store.search(question, top_k=self._config.top_k)
        if not results:
            return RAGResponse(answer="I don't have information about that.", sources=[], confidence=0.0)

        context = "\n".join(r["text"] for r in results)
        answer = f"Based on the retrieved context, here is the answer to: {question}"
        confidence = min(r["score"] for r in results) if results else 0.0

        return RAGResponse(
            answer=answer,
            sources=[{"id": r["id"], "text": r["text"], "score": r["score"]} for r in results],
            confidence=confidence,
        )

    def query_with_context(self, question: str, context: str) -> RAGResponse:
        if not question:
            raise ValueError("Question cannot be empty")
        answer = f"Given the context, the answer to '{question}' is provided."
        return RAGResponse(answer=answer, sources=[{"text": context}], confidence=0.8)

    @property
    def document_count(self):
        return self._store.count()


class SimpleChunker:
    def __init__(self, chunk_size=512):
        self._chunk_size = chunk_size

    def chunk(self, text: str) -> List[str]:
        if len(text) <= self._chunk_size:
            return [text]
        chunks = []
        for i in range(0, len(text), self._chunk_size):
            chunks.append(text[i:i + self._chunk_size])
        return chunks


class TestRAGConfig:
    def test_defaults(self):
        config = RAGConfig()
        assert config.chunk_size == 512
        assert config.top_k == 5

    def test_custom(self):
        config = RAGConfig(chunk_size=256, top_k=10)
        assert config.chunk_size == 256


class TestSimpleDocumentStore:
    def test_add_and_search(self):
        store = SimpleDocumentStore()
        store.add("doc1", "Machine learning uses algorithms")
        store.add("doc2", "Cooking recipes for dinner")
        results = store.search("machine learning")
        assert len(results) >= 1
        assert results[0]["id"] == "doc1"

    def test_no_results(self):
        store = SimpleDocumentStore()
        store.add("doc1", "hello world")
        assert store.search("quantum") == []

    def test_count(self):
        store = SimpleDocumentStore()
        store.add("1", "a")
        store.add("2", "b")
        assert store.count() == 2


class TestRAGPipeline:
    def test_index_documents(self):
        pipeline = RAGPipeline()
        count = pipeline.index_documents([
            {"id": "doc1", "text": "Python is a programming language."},
            {"id": "doc2", "text": "Machine learning uses data."},
        ])
        assert count == 2
        assert pipeline.document_count == 2

    def test_query(self):
        pipeline = RAGPipeline()
        pipeline.index_documents([
            {"id": "doc1", "text": "Python is a programming language."},
            {"id": "doc2", "text": "Machine learning uses data and algorithms."},
        ])
        response = pipeline.query("What is machine learning?")
        assert response.answer
        assert len(response.sources) > 0

    def test_query_empty_raises(self):
        pipeline = RAGPipeline()
        with pytest.raises(ValueError, match="empty"):
            pipeline.query("")

    def test_query_no_results(self):
        pipeline = RAGPipeline()
        pipeline.index_documents([{"id": "1", "text": "cooking recipes"}])
        response = pipeline.query("quantum physics")
        assert response.confidence == 0.0

    def test_query_with_context(self):
        pipeline = RAGPipeline()
        response = pipeline.query_with_context("What is AI?", "AI is artificial intelligence.")
        assert response.answer
        assert response.confidence > 0

    def test_config_property(self):
        config = RAGConfig(top_k=3)
        pipeline = RAGPipeline(config)
        assert pipeline.config.top_k == 3


class TestSimpleChunker:
    def test_short_text(self):
        chunker = SimpleChunker(chunk_size=100)
        chunks = chunker.chunk("short text")
        assert len(chunks) == 1

    def test_long_text(self):
        chunker = SimpleChunker(chunk_size=10)
        chunks = chunker.chunk("a" * 50)
        assert len(chunks) == 5
