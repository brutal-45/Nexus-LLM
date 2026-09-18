"""Test vector storage for Nexus-LLM."""
import math
import pytest
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple


@dataclass
class VectorEntry:
    id: str
    vector: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class VectorStore:
    def __init__(self, dimension: int = 128):
        self._dimension = dimension
        self._entries: Dict[str, VectorEntry] = {}

    @property
    def dimension(self):
        return self._dimension

    @property
    def count(self):
        return len(self._entries)

    def add(self, entry: VectorEntry):
        if len(entry.vector) != self._dimension:
            raise ValueError(f"Vector dimension mismatch: expected {self._dimension}, got {len(entry.vector)}")
        self._entries[entry.id] = entry

    def get(self, id: str) -> Optional[VectorEntry]:
        return self._entries.get(id)

    def delete(self, id: str) -> bool:
        if id in self._entries:
            del self._entries[id]
            return True
        return False

    def update(self, id: str, vector: List[float] = None, metadata: Dict = None):
        if id not in self._entries:
            raise KeyError(f"Entry '{id}' not found")
        entry = self._entries[id]
        if vector is not None:
            if len(vector) != self._dimension:
                raise ValueError("Vector dimension mismatch")
            entry.vector = vector
        if metadata is not None:
            entry.metadata.update(metadata)

    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    def search(self, query_vector: List[float], top_k: int = 5) -> List[Tuple[str, float]]:
        if len(query_vector) != self._dimension:
            raise ValueError("Query vector dimension mismatch")
        scored = []
        for id, entry in self._entries.items():
            sim = self._cosine_similarity(query_vector, entry.vector)
            scored.append((id, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def search_with_filter(self, query_vector: List[float], filter_fn, top_k: int = 5) -> List[Tuple[str, float]]:
        scored = []
        for id, entry in self._entries.items():
            if filter_fn(entry):
                sim = self._cosine_similarity(query_vector, entry.vector)
                scored.append((id, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def clear(self):
        self._entries.clear()


class TestVectorEntry:
    def test_creation(self):
        entry = VectorEntry(id="1", vector=[1.0, 2.0, 3.0], metadata={"source": "test"})
        assert entry.id == "1"
        assert len(entry.vector) == 3

    def test_default_metadata(self):
        entry = VectorEntry(id="1", vector=[1.0])
        assert entry.metadata == {}


class TestVectorStore:
    def test_add_and_count(self):
        store = VectorStore(dimension=3)
        store.add(VectorEntry(id="1", vector=[1.0, 0.0, 0.0]))
        assert store.count == 1

    def test_dimension_mismatch(self):
        store = VectorStore(dimension=3)
        with pytest.raises(ValueError, match="dimension mismatch"):
            store.add(VectorEntry(id="1", vector=[1.0, 0.0]))

    def test_get(self):
        store = VectorStore(dimension=3)
        store.add(VectorEntry(id="1", vector=[1.0, 0.0, 0.0]))
        entry = store.get("1")
        assert entry is not None
        assert entry.id == "1"

    def test_get_nonexistent(self):
        store = VectorStore(dimension=3)
        assert store.get("nonexistent") is None

    def test_delete(self):
        store = VectorStore(dimension=3)
        store.add(VectorEntry(id="1", vector=[1.0, 0.0, 0.0]))
        assert store.delete("1") is True
        assert store.count == 0

    def test_delete_nonexistent(self):
        store = VectorStore(dimension=3)
        assert store.delete("nonexistent") is False

    def test_update_vector(self):
        store = VectorStore(dimension=3)
        store.add(VectorEntry(id="1", vector=[1.0, 0.0, 0.0]))
        store.update("1", vector=[0.0, 1.0, 0.0])
        assert store.get("1").vector == [0.0, 1.0, 0.0]

    def test_update_metadata(self):
        store = VectorStore(dimension=3)
        store.add(VectorEntry(id="1", vector=[1.0, 0.0, 0.0], metadata={"a": 1}))
        store.update("1", metadata={"b": 2})
        assert store.get("1").metadata["b"] == 2

    def test_update_nonexistent(self):
        store = VectorStore(dimension=3)
        with pytest.raises(KeyError):
            store.update("nonexistent", vector=[0.0, 0.0, 0.0])

    def test_search(self):
        store = VectorStore(dimension=3)
        store.add(VectorEntry(id="1", vector=[1.0, 0.0, 0.0]))
        store.add(VectorEntry(id="2", vector=[0.0, 1.0, 0.0]))
        store.add(VectorEntry(id="3", vector=[0.0, 0.0, 1.0]))
        results = store.search([1.0, 0.0, 0.0], top_k=2)
        assert len(results) == 2
        assert results[0][0] == "1"

    def test_search_dimension_mismatch(self):
        store = VectorStore(dimension=3)
        with pytest.raises(ValueError):
            store.search([1.0, 0.0], top_k=1)

    def test_search_with_filter(self):
        store = VectorStore(dimension=3)
        store.add(VectorEntry(id="1", vector=[1.0, 0.0, 0.0], metadata={"type": "a"}))
        store.add(VectorEntry(id="2", vector=[1.0, 0.1, 0.0], metadata={"type": "b"}))
        results = store.search_with_filter(
            [1.0, 0.0, 0.0],
            filter_fn=lambda e: e.metadata.get("type") == "a",
            top_k=5,
        )
        assert len(results) == 1
        assert results[0][0] == "1"

    def test_clear(self):
        store = VectorStore(dimension=3)
        store.add(VectorEntry(id="1", vector=[1.0, 0.0, 0.0]))
        store.clear()
        assert store.count == 0
