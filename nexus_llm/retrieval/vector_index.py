"""Vector index for Nexus-LLM.

In-memory vector index using cosine similarity for nearest-neighbour
search with metadata support.
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Type alias for search results
IndexSearchResult = Tuple[str, float]


class VectorIndex:
    """In-memory vector index with cosine similarity search.

    Example::

        idx = VectorIndex()
        idx.add("vec1", [0.1, 0.2, 0.3], {"label": "A"})
        idx.add("vec2", [0.4, 0.5, 0.6], {"label": "B"})
        results = idx.search([0.1, 0.2, 0.3], top_k=5)
    """

    def __init__(self) -> None:
        self._vectors: Dict[str, List[float]] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._norms: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------

    def add(
        self,
        id: str,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add or update a vector in the index.

        Args:
            id: Unique identifier.
            vector: The embedding vector.
            metadata: Optional metadata dictionary.
        """
        self._vectors[id] = list(vector)
        self._metadata[id] = dict(metadata) if metadata else {}
        self._norms[id] = math.sqrt(sum(v * v for v in vector))
        logger.debug("Indexed vector %r (dim=%d)", id, len(vector))

    def delete(self, id: str) -> bool:
        """Remove a vector from the index.

        Args:
            id: Unique identifier.

        Returns:
            ``True`` if the item existed and was removed.
        """
        if id in self._vectors:
            del self._vectors[id]
            del self._metadata[id]
            del self._norms[id]
            logger.debug("Removed vector %r from index", id)
            return True
        return False

    def get(self, id: str) -> Optional[Tuple[List[float], Dict[str, Any]]]:
        """Retrieve a vector and its metadata by ID.

        Args:
            id: Unique identifier.

        Returns:
            Tuple of (vector, metadata) or ``None`` if not found.
        """
        if id not in self._vectors:
            return None
        return (list(self._vectors[id]), dict(self._metadata[id]))

    def size(self) -> int:
        """Return the number of indexed vectors."""
        return len(self._vectors)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
    ) -> List[IndexSearchResult]:
        """Find the most similar vectors using cosine similarity.

        Args:
            query_vector: The query embedding.
            top_k: Number of results to return.

        Returns:
            List of ``(id, score)`` tuples sorted by descending similarity.
        """
        if not self._vectors:
            return []

        query_norm = math.sqrt(sum(v * v for v in query_vector))
        if query_norm == 0.0:
            return []

        results: List[IndexSearchResult] = []
        for id, vec in self._vectors.items():
            vec_norm = self._norms[id]
            if vec_norm == 0.0:
                continue

            dot = sum(q * v for q, v in zip(query_vector, vec))
            score = dot / (query_norm * vec_norm)
            results.append((id, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def search_with_metadata(
        self,
        query_vector: List[float],
        top_k: int = 10,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search and return results with metadata.

        Args:
            query_vector: The query embedding.
            top_k: Number of results to return.

        Returns:
            List of ``(id, score, metadata)`` tuples.
        """
        base_results = self.search(query_vector, top_k)
        return [
            (id, score, dict(self._metadata[id]))
            for id, score in base_results
        ]
