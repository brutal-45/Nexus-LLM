"""Nexus-LLM Response Cache Storage.

Implements a persistent response cache backed by SQLite, providing
efficient caching of LLM responses with TTL, LRU eviction, cache
invalidation, and usage statistics.
"""

import hashlib
import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from nexus_llm.storage.database import DatabaseManager

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A cached response entry.

    Attributes:
        cache_key: SHA256 hash of the cache key.
        original_key: The original cache key string.
        model_name: Model that generated the response.
        prompt_hash: Hash of the input prompt.
        prompt_preview: First 200 chars of the prompt.
        response: The cached response text.
        generation_config: Generation parameters used.
        token_count: Number of tokens in the response.
        hit_count: Number of times this entry was retrieved.
        created_at: Creation timestamp.
        accessed_at: Last access timestamp.
        expires_at: Expiration timestamp (None = no expiry).
        metadata: Additional metadata as JSON.
    """

    cache_key: str = ""
    original_key: str = ""
    model_name: str = ""
    prompt_hash: str = ""
    prompt_preview: str = ""
    response: str = ""
    generation_config: Dict[str, Any] = field(default_factory=dict)
    token_count: int = 0
    hit_count: int = 0
    created_at: Optional[str] = None
    accessed_at: Optional[str] = None
    expires_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheStats:
    """Statistics about the response cache.

    Attributes:
        total_entries: Total number of cached entries.
        total_hits: Total cache hit count.
        total_misses: Total cache miss count.
        hit_rate: Cache hit rate as a fraction (0-1).
        total_size_mb: Estimated cache size in MB.
        expired_entries: Number of expired entries.
    """

    total_entries: int = 0
    total_hits: int = 0
    total_misses: int = 0
    hit_rate: float = 0.0
    total_size_mb: float = 0.0
    expired_entries: int = 0


class CacheStore:
    """Persistent response cache for LLM inference results.

    Stores generated responses keyed by a combination of model name,
    prompt, and generation parameters. Supports TTL-based expiration,
    LRU eviction, and namespace-based cache organization.

    Attributes:
        db: Database manager instance.
        default_ttl: Default time-to-live in seconds (None = no expiry).
        max_entries: Maximum number of cache entries before eviction.
    """

    def __init__(
        self,
        db: DatabaseManager,
        default_ttl: Optional[int] = None,
        max_entries: int = 10000,
    ) -> None:
        """Initialize the cache store.

        Args:
            db: DatabaseManager instance for database access.
            default_ttl: Default TTL in seconds. None means no expiry.
            max_entries: Maximum entries before LRU eviction.
        """
        self.db = db
        self.default_ttl = default_ttl
        self.max_entries = max_entries
        self._hits = 0
        self._misses = 0
        self._ensure_table()

    def _ensure_table(self) -> None:
        """Create the cache table if it doesn't exist."""
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS response_cache (
                cache_key TEXT PRIMARY KEY,
                original_key TEXT,
                model_name TEXT NOT NULL,
                prompt_hash TEXT,
                prompt_preview TEXT,
                response TEXT NOT NULL,
                generation_config TEXT,
                token_count INTEGER DEFAULT 0,
                hit_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                namespace TEXT DEFAULT 'default',
                metadata TEXT
            )
        """)
        self.db.execute("""
            CREATE INDEX IF NOT EXISTS idx_cache_model ON response_cache(model_name)
        """)
        self.db.execute("""
            CREATE INDEX IF NOT EXISTS idx_cache_expires ON response_cache(expires_at)
        """)
        self.db.execute("""
            CREATE INDEX IF NOT EXISTS idx_cache_accessed ON response_cache(accessed_at)
        """)
        self.db.execute("""
            CREATE INDEX IF NOT EXISTS idx_cache_namespace ON response_cache(namespace)
        """)

    @staticmethod
    def compute_cache_key(
        model_name: str,
        prompt: str,
        generation_config: Optional[Dict[str, Any]] = None,
        namespace: str = "default",
    ) -> str:
        """Compute a deterministic cache key from model, prompt, and config.

        Args:
            model_name: Model identifier.
            prompt: Input prompt text.
            generation_config: Generation parameters affecting output.
            namespace: Cache namespace for isolation.

        Returns:
            Hex-encoded SHA256 hash as the cache key.
        """
        config_str = json.dumps(generation_config or {}, sort_keys=True)
        key_input = f"{namespace}:{model_name}:{prompt}:{config_str}"
        return hashlib.sha256(key_input.encode("utf-8")).hexdigest()

    def get(
        self,
        model_name: str,
        prompt: str,
        generation_config: Optional[Dict[str, Any]] = None,
        namespace: str = "default",
    ) -> Optional[str]:
        """Retrieve a cached response.

        Args:
            model_name: Model identifier.
            prompt: Input prompt text.
            generation_config: Generation parameters used.
            namespace: Cache namespace.

        Returns:
            Cached response text if found and not expired, None otherwise.
        """
        cache_key = self.compute_cache_key(model_name, prompt, generation_config, namespace)

        row = self.db.fetch_one(
            "SELECT * FROM response_cache WHERE cache_key = ?",
            (cache_key,),
        )

        if row is None:
            self._misses += 1
            return None

        # Check expiration
        expires_at = row.get("expires_at")
        if expires_at:
            try:
                exp_time = datetime.fromisoformat(expires_at)
                if datetime.now() > exp_time:
                    self.db.execute("DELETE FROM response_cache WHERE cache_key = ?", (cache_key,))
                    self._misses += 1
                    return None
            except (ValueError, TypeError):
                pass

        # Update access statistics
        now = datetime.now().isoformat()
        self.db.execute(
            "UPDATE response_cache SET hit_count = hit_count + 1, accessed_at = ? WHERE cache_key = ?",
            (now, cache_key),
        )

        self._hits += 1
        return row.get("response", "")

    def put(
        self,
        model_name: str,
        prompt: str,
        response: str,
        generation_config: Optional[Dict[str, Any]] = None,
        token_count: int = 0,
        ttl: Optional[int] = None,
        namespace: str = "default",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Store a response in the cache.

        Args:
            model_name: Model that generated the response.
            prompt: Input prompt text.
            response: Generated response to cache.
            generation_config: Generation parameters used.
            token_count: Number of tokens in the response.
            ttl: Time-to-live in seconds (overrides default).
            namespace: Cache namespace.
            metadata: Additional metadata.

        Returns:
            The cache key for the stored entry.
        """
        cache_key = self.compute_cache_key(model_name, prompt, generation_config, namespace)
        now = datetime.now().isoformat()

        effective_ttl = ttl if ttl is not None else self.default_ttl
        expires_at = None
        if effective_ttl is not None:
            expires_at = (datetime.now() + timedelta(seconds=effective_ttl)).isoformat()

        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        prompt_preview = prompt[:200]
        config_json = json.dumps(generation_config or {}, sort_keys=True)
        metadata_json = json.dumps(metadata or {})

        self.db.execute(
            """INSERT OR REPLACE INTO response_cache
               (cache_key, original_key, model_name, prompt_hash, prompt_preview,
                response, generation_config, token_count, hit_count, created_at,
                accessed_at, expires_at, namespace, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                cache_key,
                f"{namespace}:{model_name}",
                model_name,
                prompt_hash,
                prompt_preview,
                response,
                config_json,
                token_count,
                0,
                now,
                now,
                expires_at,
                namespace,
                metadata_json,
            ),
        )

        # Enforce max entries with LRU eviction
        self._evict_if_needed()

        return cache_key

    def invalidate(
        self,
        model_name: Optional[str] = None,
        prompt: Optional[str] = None,
        namespace: Optional[str] = None,
    ) -> int:
        """Invalidate cache entries matching given criteria.

        Args:
            model_name: Invalidate entries for this model.
            prompt: Invalidate entries matching this prompt.
            namespace: Invalidate entries in this namespace.

        Returns:
            Number of entries invalidated.
        """
        conditions = []
        params: List[Any] = []

        if model_name:
            conditions.append("model_name = ?")
            params.append(model_name)
        if prompt:
            prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            conditions.append("prompt_hash = ?")
            params.append(prompt_hash)
        if namespace:
            conditions.append("namespace = ?")
            params.append(namespace)

        if not conditions:
            return 0

        where = " AND ".join(conditions)
        cursor = self.db.execute(
            f"DELETE FROM response_cache WHERE {where}",
            tuple(params),
        )
        count = cursor.rowcount
        if count > 0:
            logger.info(f"Invalidated {count} cache entries")
        return count

    def clear(self, namespace: Optional[str] = None) -> int:
        """Clear cache entries.

        Args:
            namespace: If set, only clear entries in this namespace.
                Otherwise clears all entries.

        Returns:
            Number of entries cleared.
        """
        if namespace:
            cursor = self.db.execute(
                "DELETE FROM response_cache WHERE namespace = ?",
                (namespace,),
            )
        else:
            cursor = self.db.execute("DELETE FROM response_cache")

        count = cursor.rowcount
        logger.info(f"Cleared {count} cache entries" + (f" in namespace '{namespace}'" if namespace else ""))
        return count

    def cleanup_expired(self) -> int:
        """Remove all expired cache entries.

        Returns:
            Number of expired entries removed.
        """
        now = datetime.now().isoformat()
        cursor = self.db.execute(
            "DELETE FROM response_cache WHERE expires_at IS NOT NULL AND expires_at < ?",
            (now,),
        )
        count = cursor.rowcount
        if count > 0:
            logger.info(f"Cleaned up {count} expired cache entries")
        return count

    def get_stats(self) -> CacheStats:
        """Get cache statistics.

        Returns:
            CacheStats with current metrics.
        """
        total = self.db.fetch_value("SELECT COUNT(*) FROM response_cache") or 0
        now = datetime.now().isoformat()
        expired = self.db.fetch_value(
            "SELECT COUNT(*) FROM response_cache WHERE expires_at IS NOT NULL AND expires_at < ?",
            (now,),
        ) or 0

        total_size = self.db.fetch_value(
            "SELECT COALESCE(SUM(LENGTH(response)), 0) FROM response_cache"
        ) or 0

        total_hits = self._hits + self._misses
        hit_rate = self._hits / total_hits if total_hits > 0 else 0.0

        return CacheStats(
            total_entries=total,
            total_hits=self._hits,
            total_misses=self._misses,
            hit_rate=round(hit_rate, 4),
            total_size_mb=total_size / (1024 * 1024),
            expired_entries=expired,
        )

    def _evict_if_needed(self) -> None:
        """Evict least-recently-used entries if over max_entries limit."""
        count = self.db.fetch_value("SELECT COUNT(*) FROM response_cache") or 0
        if count <= self.max_entries:
            return

        excess = count - self.max_entries
        # Remove the least recently accessed entries
        self.db.execute(
            """DELETE FROM response_cache WHERE cache_key IN (
                SELECT cache_key FROM response_cache
                ORDER BY accessed_at ASC LIMIT ?
            )""",
            (excess,),
        )
        logger.debug(f"Evicted {excess} LRU cache entries")

    def get_entries_by_model(self, model_name: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get cache entries for a specific model.

        Args:
            model_name: Model to filter by.
            limit: Maximum entries to return.

        Returns:
            List of cache entry dictionaries (without response content).
        """
        rows = self.db.fetch_all(
            """SELECT cache_key, original_key, model_name, prompt_preview,
                      token_count, hit_count, created_at, accessed_at, expires_at, namespace
               FROM response_cache WHERE model_name = ?
               ORDER BY accessed_at DESC LIMIT ?""",
            (model_name, limit),
        )
        return rows

    def warm_up(
        self,
        entries: List[Dict[str, Any]],
        namespace: str = "default",
    ) -> int:
        """Bulk-insert cache entries for warming up the cache.

        Args:
            entries: List of dicts with keys: model_name, prompt, response,
                generation_config, token_count, ttl.
            namespace: Cache namespace.

        Returns:
            Number of entries inserted.
        """
        count = 0
        for entry in entries:
            try:
                self.put(
                    model_name=entry["model_name"],
                    prompt=entry["prompt"],
                    response=entry["response"],
                    generation_config=entry.get("generation_config"),
                    token_count=entry.get("token_count", 0),
                    ttl=entry.get("ttl"),
                    namespace=namespace,
                    metadata=entry.get("metadata"),
                )
                count += 1
            except Exception as e:
                logger.warning(f"Failed to warm cache entry: {e}")

        logger.info(f"Warmed cache with {count} entries")
        return count
