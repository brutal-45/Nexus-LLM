"""Nexus-LLM Response Cache Manager.

Provides caching capabilities for LLM responses, supporting
in-memory and file-based caching with TTL, LRU eviction,
and cache statistics.
"""

import hashlib
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A single cache entry.

    Attributes:
        key: Cache key.
        value: Cached value.
        created_at: Creation timestamp.
        expires_at: Expiration timestamp (0 = never).
        hit_count: Number of cache hits.
        size_bytes: Approximate size in bytes.
    """

    key: str
    value: Any
    created_at: float = field(default_factory=time.time)
    expires_at: float = 0.0
    hit_count: int = 0
    size_bytes: int = 0


class CacheManager:
    """Manages response caching with TTL and LRU eviction.

    Example::

        cache = CacheManager(max_size=1000, default_ttl=3600)
        cache.put("key1", {"response": "hello"})
        result = cache.get("key1")
    """

    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: float = 3600.0,
        cache_dir: Optional[str] = None,
    ) -> None:
        """Initialize the CacheManager.

        Args:
            max_size: Maximum number of entries.
            default_ttl: Default time-to-live in seconds (0 = never expires).
            cache_dir: Optional directory for persistent cache storage.
        """
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._cache_dir = cache_dir
        self._entries: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = {"hits": 0, "misses": 0, "evictions": 0}
        logger.debug("CacheManager initialized: max_size=%d, ttl=%.0f", max_size, default_ttl)

    @property
    def size(self) -> int:
        """Current number of cache entries."""
        return len(self._entries)

    @property
    def max_size(self) -> int:
        """Maximum cache size."""
        return self._max_size

    @property
    def stats(self) -> Dict[str, int]:
        """Cache statistics."""
        return dict(self._stats)

    def _make_key(self, *args: Any, **kwargs: Any) -> str:
        """Generate a cache key from arguments.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Hash string as cache key.
        """
        key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str)
        return hashlib.sha256(key_data.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """Retrieve a cached value.

        Args:
            key: Cache key.

        Returns:
            The cached value, or None if not found or expired.
        """
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self._stats["misses"] += 1
                return None

            # Check expiration
            if entry.expires_at > 0 and time.time() > entry.expires_at:
                del self._entries[key]
                self._stats["misses"] += 1
                return None

            # Update access (LRU)
            self._entries.move_to_end(key)
            entry.hit_count += 1
            self._stats["hits"] += 1
            return entry.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Store a value in the cache.

        Args:
            key: Cache key.
            value: Value to cache.
            ttl: Time-to-live in seconds (uses default if not specified).
        """
        with self._lock:
            # Evict if at capacity
            while len(self._entries) >= self._max_size:
                self._evict_one()

            effective_ttl = ttl if ttl is not None else self._default_ttl
            expires_at = (time.time() + effective_ttl) if effective_ttl > 0 else 0.0

            # Estimate size
            try:
                size = len(json.dumps(value, default=str).encode())
            except Exception:
                size = 0

            entry = CacheEntry(
                key=key,
                value=value,
                expires_at=expires_at,
                size_bytes=size,
            )

            # Remove old entry if overwriting
            if key in self._entries:
                del self._entries[key]

            self._entries[key] = entry

    def delete(self, key: str) -> bool:
        """Delete a cache entry.

        Args:
            key: Cache key.

        Returns:
            True if the entry was found and deleted.
        """
        with self._lock:
            if key in self._entries:
                del self._entries[key]
                return True
            return False

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._entries.clear()
            logger.info("Cache cleared")

    def has(self, key: str) -> bool:
        """Check if a key exists and is not expired.

        Args:
            key: Cache key.

        Returns:
            True if the key exists and is valid.
        """
        return self.get(key) is not None

    def cleanup_expired(self) -> int:
        """Remove all expired entries.

        Returns:
            Number of entries removed.
        """
        now = time.time()
        expired_keys = [
            k for k, v in self._entries.items()
            if v.expires_at > 0 and now > v.expires_at
        ]
        with self._lock:
            for key in expired_keys:
                del self._entries[key]
        if expired_keys:
            logger.info("Cleaned up %d expired entries", len(expired_keys))
        return len(expired_keys)

    def _evict_one(self) -> None:
        """Evict the least recently used entry."""
        if self._entries:
            key, _ = self._entries.popitem(last=False)
            self._stats["evictions"] += 1
            logger.debug("Evicted cache entry: %s", key)

    def get_or_compute(self, key: str, compute_fn: callable, ttl: Optional[float] = None) -> Any:
        """Get from cache or compute and cache the result.

        Args:
            key: Cache key.
            compute_fn: Function to compute the value if not cached.
            ttl: Time-to-live for the computed value.

        Returns:
            The cached or computed value.
        """
        value = self.get(key)
        if value is not None:
            return value
        value = compute_fn()
        self.put(key, value, ttl=ttl)
        return value

    def get_stats_summary(self) -> Dict[str, Any]:
        """Get a summary of cache statistics.

        Returns:
            Dictionary with hit rate, sizes, and eviction count.
        """
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total if total > 0 else 0.0
        total_size = sum(e.size_bytes for e in self._entries.values())
        return {
            "size": len(self._entries),
            "max_size": self._max_size,
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "hit_rate": hit_rate,
            "evictions": self._stats["evictions"],
            "total_size_bytes": total_size,
        }
