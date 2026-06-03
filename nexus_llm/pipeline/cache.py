"""Nexus-LLM Pipeline Caching.

Provides the PipelineCache for caching intermediate pipeline results
to avoid redundant computation.
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A single cache entry.

    Attributes:
        key: Cache key (hash).
        value: Cached value.
        created_at: When the entry was created.
        expires_at: When the entry expires (None = no expiry).
        hit_count: Number of cache hits.
        size_bytes: Approximate size in bytes.
    """

    key: str
    value: Any
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    hit_count: int = 0
    size_bytes: int = 0

    def is_expired(self) -> bool:
        """Check whether the entry has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "hit_count": self.hit_count,
            "is_expired": self.is_expired(),
            "size_bytes": self.size_bytes,
        }


class PipelineCache:
    """In-memory cache for pipeline intermediate results.

    The PipelineCache stores results keyed by input hash, supporting
    TTL-based expiration and LRU eviction.

    Example::

        cache = PipelineCache(max_size=1000, default_ttl=300)
        cache.put("step1", "input_data_hash", result_data)
        cached = cache.get("step1", "input_data_hash")
    """

    def __init__(self, max_size: int = 1000, default_ttl: float = 300.0) -> None:
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._cache: Dict[str, CacheEntry] = {}
        self._stats = {"hits": 0, "misses": 0, "evictions": 0}
        logger.debug("PipelineCache initialized (max_size=%d, ttl=%.0fs)", max_size, default_ttl)

    def put(
        self,
        stage: str,
        input_data: Any,
        value: Any,
        ttl: Optional[float] = None,
    ) -> str:
        """Store a value in the cache.

        Args:
            stage: Pipeline stage name.
            input_data: Input data (used to compute cache key).
            value: The value to cache.
            ttl: Time-to-live in seconds (None uses default).

        Returns:
            The cache key.
        """
        key = self._make_key(stage, input_data)
        actual_ttl = ttl if ttl is not None else self._default_ttl
        expires_at = time.time() + actual_ttl if actual_ttl > 0 else None

        # Estimate size
        size_bytes = len(str(value).encode("utf-8")) if isinstance(value, str) else 0

        # Evict if at capacity
        if len(self._cache) >= self._max_size and key not in self._cache:
            self._evict_one()

        self._cache[key] = CacheEntry(
            key=key,
            value=value,
            expires_at=expires_at,
            size_bytes=size_bytes,
        )
        logger.debug("Cached entry: %s (ttl=%.0fs)", key, actual_ttl)
        return key

    def get(self, stage: str, input_data: Any) -> Optional[Any]:
        """Retrieve a cached value.

        Args:
            stage: Pipeline stage name.
            input_data: Input data (used to compute cache key).

        Returns:
            The cached value, or None if not found or expired.
        """
        key = self._make_key(stage, input_data)
        entry = self._cache.get(key)

        if entry is None:
            self._stats["misses"] += 1
            return None

        if entry.is_expired():
            del self._cache[key]
            self._stats["misses"] += 1
            return None

        entry.hit_count += 1
        self._stats["hits"] += 1
        return entry.value

    def invalidate(self, stage: str, input_data: Any) -> bool:
        """Invalidate a specific cache entry.

        Args:
            stage: Pipeline stage name.
            input_data: Input data.

        Returns:
            True if the entry was found and removed.
        """
        key = self._make_key(stage, input_data)
        return self._cache.pop(key, None) is not None

    def invalidate_stage(self, stage: str) -> int:
        """Invalidate all entries for a pipeline stage.

        Args:
            stage: Pipeline stage name.

        Returns:
            Number of entries invalidated.
        """
        prefix = f"{stage}:"
        keys_to_remove = [k for k in self._cache if k.startswith(prefix)]
        for key in keys_to_remove:
            del self._cache[key]
        return len(keys_to_remove)

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        logger.info("Cache cleared")

    def cleanup_expired(self) -> int:
        """Remove all expired entries.

        Returns:
            Number of entries removed.
        """
        expired_keys = [k for k, v in self._cache.items() if v.is_expired()]
        for key in expired_keys:
            del self._cache[key]
        return len(expired_keys)

    def stats(self) -> Dict[str, Any]:
        """Return cache statistics.

        Returns:
            Dictionary with hits, misses, evictions, size, and hit rate.
        """
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total if total > 0 else 0.0
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "evictions": self._stats["evictions"],
            "hit_rate": round(hit_rate, 4),
        }

    def _make_key(self, stage: str, input_data: Any) -> str:
        """Compute a cache key from stage and input data."""
        data_str = json.dumps(input_data, sort_keys=True, default=str)
        data_hash = hashlib.md5(data_str.encode("utf-8")).hexdigest()
        return f"{stage}:{data_hash}"

    def _evict_one(self) -> None:
        """Evict the least recently used (lowest hit_count) entry."""
        if not self._cache:
            return
        # Find entry with lowest hit count (LRU approximation)
        lru_key = min(self._cache, key=lambda k: self._cache[k].hit_count)
        del self._cache[lru_key]
        self._stats["evictions"] += 1
