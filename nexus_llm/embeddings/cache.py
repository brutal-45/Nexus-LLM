"""Embedding cache for Nexus-LLM.

LRU cache for text-to-embedding mappings with hit/miss statistics
and configurable capacity.
"""

import hashlib
import logging
from collections import OrderedDict
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default maximum cache entries
DEFAULT_CAPACITY = 1024


class EmbeddingCache:
    """LRU cache for embedding look-ups.

    Avoids redundant embedding computation by caching results keyed
    on the input text hash.

    Example::

        cache = EmbeddingCache(capacity=500)
        cache.put("hello world", [0.1, 0.2, 0.3])
        vec = cache.get("hello world")   # hit
        vec = cache.get("goodbye")       # miss → None
        print(cache.stats())
    """

    def __init__(self, capacity: int = DEFAULT_CAPACITY) -> None:
        """Initialise the cache.

        Args:
            capacity: Maximum number of entries.  When exceeded, the
                      least-recently-used entry is evicted.

        Raises:
            ValueError: If *capacity* is less than 1.
        """
        if capacity < 1:
            raise ValueError(f"Cache capacity must be >= 1, got {capacity}")
        self._capacity = capacity
        self._cache: OrderedDict[str, List[float]] = OrderedDict()
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def get(self, text: str) -> Optional[List[float]]:
        """Look up an embedding by text.

        Args:
            text: The input text.

        Returns:
            The cached embedding vector, or ``None`` on cache miss.
        """
        key = self._make_key(text)
        if key in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            logger.debug("Cache hit for text (key=%s…)", key[:12])
            return list(self._cache[key])

        self._misses += 1
        logger.debug("Cache miss for text (key=%s…)", key[:12])
        return None

    def put(self, text: str, embedding: List[float]) -> None:
        """Store an embedding in the cache.

        If the text is already cached its value is updated and it is
        moved to the most-recently-used position.

        Args:
            text: The input text.
            embedding: The embedding vector.
        """
        key = self._make_key(text)

        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key] = list(embedding)
        else:
            # Evict LRU if at capacity
            while len(self._cache) >= self._capacity:
                evicted_key, _ = self._cache.popitem(last=False)
                logger.debug("Evicted cache entry (key=%s…)", evicted_key[:12])
            self._cache[key] = list(embedding)

        logger.debug("Cached embedding for text (key=%s…)", key[:12])

    def clear(self) -> None:
        """Remove all entries from the cache and reset statistics."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
        logger.info("Cache cleared")

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        """Return cache statistics.

        Returns:
            Dict with ``size``, ``capacity``, ``hits``, ``misses``,
            ``hit_rate``, and ``eviction_count``.
        """
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            "size": len(self._cache),
            "capacity": self._capacity,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(hit_rate, 4),
        }

    @property
    def size(self) -> int:
        """Current number of cached entries."""
        return len(self._cache)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_key(text: str) -> str:
        """Produce a deterministic cache key from text.

        Uses SHA-256 for collision resistance.
        """
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
