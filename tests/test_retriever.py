"""Test document retrieval for Nexus-LLM."""
import math
import pytest
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class Document:
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    if len(v1) != len(v2):
        raise ValueError("Vectors must have same length")
    if not v1:
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


class Retriever:
    def __init__(self, top_k: int = 5, min_score: float = 0.0):
        self._top_k = top_k
        self._min_score = min_score
        self._documents: List[Document] = []

    def add_documents(self, documents: List[Document]):
        self._documents.extend(documents)

    def clear(self):
        self._documents.clear()

    def retrieve_by_keyword(self, query: str, top_k: int = None) -> List[Document]:
        k = top_k or self._top_k
        query_terms = set(query.lower().split())
        scored = []
        for doc in self._documents:
            doc_terms = set(doc.text.lower().split())
            overlap = len(query_terms & doc_terms)
            if overlap > 0:
                score = overlap / len(query_terms)
                scored.append(Document(id=doc.id, text=doc.text, metadata=doc.metadata, score=score))
        scored.sort(key=lambda d: d.score, reverse=True)
        results = [d for d in scored[:k] if d.score >= self._min_score]
        return results

    def retrieve_by_embedding(self, query_embedding: List[float], doc_embeddings: Dict[str, List[float]], top_k: int = None) -> List[Document]:
        k = top_k or self._top_k
        scored = []
        for doc_id, embedding in doc_embeddings.items():
            score = cosine_similarity(query_embedding, embedding)
            doc = next((d for d in self._documents if d.id == doc_id), None)
            if doc:
                scored.append(Document(id=doc.id, text=doc.text, metadata=doc.metadata, score=score))
        scored.sort(key=lambda d: d.score, reverse=True)
        return [d for d in scored[:k] if d.score >= self._min_score]

    @property
    def document_count(self):
        return len(self._documents)


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 0.0, 0.0]
        assert abs(cosine_similarity(v, v) - 1.0) < 0.001

    def test_orthogonal_vectors(self):
        v1 = [1.0, 0.0]
        v2 = [0.0, 1.0]
        assert abs(cosine_similarity(v1, v2)) < 0.001

    def test_opposite_vectors(self):
        v1 = [1.0, 0.0]
        v2 = [-1.0, 0.0]
        assert abs(cosine_similarity(v1, v2) + 1.0) < 0.001

    def test_different_lengths_raises(self):
        with pytest.raises(ValueError):
            cosine_similarity([1.0], [1.0, 2.0])

    def test_empty_vectors(self):
        assert cosine_similarity([], []) == 0.0


class TestDocument:
    def test_creation(self):
        doc = Document(id="1", text="hello world", metadata={"source": "test"})
        assert doc.id == "1"
        assert doc.score == 0.0

    def test_default_metadata(self):
        doc = Document(id="1", text="test")
        assert doc.metadata == {}


class TestRetriever:
    def test_add_and_count(self):
        retriever = Retriever()
        retriever.add_documents([Document(id="1", text="hello")])
        assert retriever.document_count == 1

    def test_keyword_retrieval(self):
        retriever = Retriever(top_k=3)
        retriever.add_documents([
            Document(id="1", text="machine learning algorithms"),
            Document(id="2", text="deep learning neural networks"),
            Document(id="3", text="cooking recipes for dinner"),
        ])
        results = retriever.retrieve_by_keyword("machine learning")
        assert len(results) >= 1
        assert results[0].id == "1"

    def test_keyword_retrieval_scoring(self):
        retriever = Retriever()
        retriever.add_documents([
            Document(id="1", text="python programming"),
            Document(id="2", text="python python programming"),
        ])
        results = retriever.retrieve_by_keyword("python")
        assert all(r.score > 0 for r in results)

    def test_no_results(self):
        retriever = Retriever()
        retriever.add_documents([Document(id="1", text="hello world")])
        results = retriever.retrieve_by_keyword("quantum physics")
        assert len(results) == 0

    def test_embedding_retrieval(self):
        retriever = Retriever(top_k=2)
        retriever.add_documents([
            Document(id="1", text="cat"),
            Document(id="2", text="dog"),
            Document(id="3", text="car"),
        ])
        query_emb = [1.0, 0.0]
        doc_embs = {"1": [0.9, 0.1], "2": [0.1, 0.9], "3": [0.5, 0.5]}
        results = retriever.retrieve_by_embedding(query_emb, doc_embs)
        assert len(results) <= 2
        assert results[0].id == "1"  # closest to query

    def test_min_score_filter(self):
        retriever = Retriever(min_score=0.5)
        retriever.add_documents([
            Document(id="1", text="machine learning"),
            Document(id="2", text="cooking"),
        ])
        results = retriever.retrieve_by_keyword("machine learning")
        for r in results:
            assert r.score >= 0.5

    def test_clear(self):
        retriever = Retriever()
        retriever.add_documents([Document(id="1", text="test")])
        retriever.clear()
        assert retriever.document_count == 0
