"""Test document indexing for Nexus-LLM."""
import pytest
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set
from collections import defaultdict


@dataclass
class IndexEntry:
    doc_id: str
    position: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class InvertedIndex:
    def __init__(self):
        self._index: Dict[str, List[IndexEntry]] = defaultdict(list)
        self._doc_count = 0
        self._doc_lengths: Dict[str, int] = {}

    def index_document(self, doc_id: str, text: str, metadata: Dict = None):
        words = text.lower().split()
        self._doc_lengths[doc_id] = len(words)
        self._doc_count += 1
        for position, word in enumerate(words):
            self._index[word].append(IndexEntry(doc_id=doc_id, position=position, metadata=metadata or {}))

    def search(self, term: str) -> List[IndexEntry]:
        return self._index.get(term.lower(), [])

    def get_document_frequency(self, term: str) -> int:
        entries = self._index.get(term.lower(), [])
        return len(set(e.doc_id for e in entries))

    def get_term_frequency(self, term: str, doc_id: str) -> int:
        entries = self._index.get(term.lower(), [])
        return sum(1 for e in entries if e.doc_id == doc_id)

    def get_vocabulary_size(self) -> int:
        return len(self._index)

    def get_document_count(self) -> int:
        return self._doc_count

    def remove_document(self, doc_id: str):
        for term in list(self._index.keys()):
            self._index[term] = [e for e in self._index[term] if e.doc_id != doc_id]
            if not self._index[term]:
                del self._index[term]
        self._doc_lengths.pop(doc_id, None)
        self._doc_count = max(0, self._doc_count - 1)


class DocumentIndexer:
    def __init__(self):
        self._index = InvertedIndex()
        self._indexed_docs: Set[str] = set()

    def index(self, doc_id: str, text: str, metadata: Dict = None):
        if doc_id in self._indexed_docs:
            self._index.remove_document(doc_id)
        self._index.index_document(doc_id, text, metadata)
        self._indexed_docs.add(doc_id)

    def search(self, query: str) -> List[Dict]:
        terms = query.lower().split()
        results = defaultdict(lambda: {"score": 0.0, "matches": 0})
        for term in terms:
            entries = self._index.search(term)
            for entry in entries:
                results[entry.doc_id]["score"] += 1.0
                results[entry.doc_id]["matches"] += 1
        return [
            {"doc_id": doc_id, "score": data["score"], "matches": data["matches"]}
            for doc_id, data in sorted(results.items(), key=lambda x: x[1]["score"], reverse=True)
        ]

    def get_stats(self) -> Dict[str, Any]:
        return {
            "document_count": self._index.get_document_count(),
            "vocabulary_size": self._index.get_vocabulary_size(),
            "indexed_docs": len(self._indexed_docs),
        }

    def is_indexed(self, doc_id: str) -> bool:
        return doc_id in self._indexed_docs


class TestInvertedIndex:
    def test_index_and_search(self):
        idx = InvertedIndex()
        idx.index_document("doc1", "hello world")
        results = idx.search("hello")
        assert len(results) == 1
        assert results[0].doc_id == "doc1"

    def test_multiple_docs(self):
        idx = InvertedIndex()
        idx.index_document("doc1", "machine learning")
        idx.index_document("doc2", "deep learning")
        results = idx.search("learning")
        assert len(results) == 2

    def test_document_frequency(self):
        idx = InvertedIndex()
        idx.index_document("doc1", "python code")
        idx.index_document("doc2", "python script")
        assert idx.get_document_frequency("python") == 2
        assert idx.get_document_frequency("code") == 1

    def test_term_frequency(self):
        idx = InvertedIndex()
        idx.index_document("doc1", "python python python")
        assert idx.get_term_frequency("python", "doc1") == 3

    def test_vocabulary_size(self):
        idx = InvertedIndex()
        idx.index_document("doc1", "hello world foo")
        assert idx.get_vocabulary_size() == 3

    def test_remove_document(self):
        idx = InvertedIndex()
        idx.index_document("doc1", "hello world")
        idx.remove_document("doc1")
        assert idx.search("hello") == []
        assert idx.get_document_count() == 0


class TestDocumentIndexer:
    def test_index_and_search(self):
        indexer = DocumentIndexer()
        indexer.index("doc1", "machine learning algorithms")
        indexer.index("doc2", "deep learning models")
        results = indexer.search("learning")
        assert len(results) == 2

    def test_reindex_document(self):
        indexer = DocumentIndexer()
        indexer.index("doc1", "old content")
        indexer.index("doc1", "new content updated")
        results = indexer.search("updated")
        assert len(results) == 1

    def test_multi_term_search(self):
        indexer = DocumentIndexer()
        indexer.index("doc1", "machine learning is great")
        indexer.index("doc2", "machine learning and deep learning")
        results = indexer.search("machine learning")
        assert len(results) >= 1

    def test_stats(self):
        indexer = DocumentIndexer()
        indexer.index("doc1", "hello world")
        indexer.index("doc2", "foo bar")
        stats = indexer.get_stats()
        assert stats["document_count"] == 2
        assert stats["vocabulary_size"] >= 4

    def test_is_indexed(self):
        indexer = DocumentIndexer()
        indexer.index("doc1", "test")
        assert indexer.is_indexed("doc1") is True
        assert indexer.is_indexed("doc2") is False

    def test_no_results(self):
        indexer = DocumentIndexer()
        indexer.index("doc1", "hello world")
        results = indexer.search("nonexistent")
        assert len(results) == 0
