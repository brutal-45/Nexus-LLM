"""Embedding store for Nexus-LLM.

In-memory vector store with metadata support, similarity search,
and optional JSON-based persistence.
"""

import json
import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple

from nexus_llm.embeddings.engine import EmbeddingEngine

logger = logging.getLogger(__name__)

# Type alias for search results
SearchResult = Tuple[str, float, Dict[str, Any]]


class EmbeddingStore:
    """In-memory embedding store with similarity search.

    Example::

        store = EmbeddingStore()
        store.add("doc1", [0.1, 0.2, 0.3], {"title": "Hello"})
        results = store.search([0.1, 0.2, 0.3], top_k=5)
    """

    def __init__(
        self,
        persistence_path: Optional[str] = None,
    ) -> None:
        """Initialise the store.

        Args:
            persistence_path: Optional file path for JSON persistence.
                              When provided, data is loaded on init and
                              saved after every mutation.
        """
        self._vectors: Dict[str, List[float]] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._persistence_path = persistence_path

        if persistence_path and os.path.exists(persistence_path):
            self._load(persistence_path)

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------

    def add(
        self,
        id: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add or update an embedding in the store.

        Args:
            id: Unique identifier.
            embedding: The embedding vector.
            metadata: Optional metadata dictionary.
        """
        self._vectors[id] = list(embedding)
        self._metadata[id] = dict(metadata) if metadata else {}
        logger.debug("Added/updated embedding %r (dim=%d)", id, len(embedding))
        self._persist()

    def delete(self, id: str) -> bool:
        """Remove an embedding from the store.

        Args:
            id: Unique identifier.

        Returns:
            ``True`` if the item existed and was removed, ``False`` otherwise.
        """
        if id in self._vectors:
            del self._vectors[id]
            del self._metadata[id]
            logger.debug("Deleted embedding %r", id)
            self._persist()
            return True
        return False

    def get(self, id: str) -> Optional[Tuple[List[float], Dict[str, Any]]]:
        """Retrieve an embedding and its metadata by ID.

        Args:
            id: Unique identifier.

        Returns:
            Tuple of (embedding, metadata) or ``None`` if not found.
        """
        if id not in self._vectors:
            return None
        return (list(self._vectors[id]), dict(self._metadata[id]))

    def size(self) -> int:
        """Return the number of stored embeddings."""
        return len(self._vectors)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
    ) -> List[SearchResult]:
        """Find the most similar embeddings using cosine similarity.

        Args:
            query_embedding: The query vector.
            top_k: Number of results to return.

        Returns:
            List of ``(id, score, metadata)`` tuples, sorted by
            descending similarity.
        """
        if not self._vectors:
            return []

        results: List[SearchResult] = []
        for id, vec in self._vectors.items():
            score = self._cosine_similarity(query_embedding, vec)
            results.append((id, score, dict(self._metadata[id])))

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Optional[str] = None) -> str:
        """Persist the store to a JSON file.

        Args:
            path: Override path (defaults to *persistence_path*).

        Returns:
            The path written to.
        """
        target = path or self._persistence_path
        if not target:
            raise ValueError("No persistence path configured")

        data = {
            "vectors": self._vectors,
            "metadata": self._metadata,
        }
        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        with open(target, "w", encoding="utf-8") as fh:
            json.dump(data, fh, default=str)
        logger.info("Saved embedding store to %s (%d items)", target, self.size())
        return target

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self, path: str) -> None:
        """Load the store from a JSON file."""
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        self._vectors = data.get("vectors", {})
        self._metadata = data.get("metadata", {})
        logger.info("Loaded embedding store from %s (%d items)", path, self.size())

    def _persist(self) -> None:
        """Auto-persist if a persistence path is configured."""
        if self._persistence_path:
            try:
                self.save(self._persistence_path)
            except Exception as exc:
                logger.warning("Auto-persist failed: %s", exc)

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(a) != len(b) or not a:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)
