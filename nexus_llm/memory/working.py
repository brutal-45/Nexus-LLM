"""WorkingMemory — ephemeral key-value scratch pad for agent work."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class WorkingMemory:
    """Light-weight scratch pad that agents use for temporary storage.

    Data is stored as key-value pairs with optional per-key metadata such as
    a TTL (time-to-live) expressed in seconds.  Expired entries are lazily
    evicted on access.

    Parameters
    ----------
    id:
        Optional explicit identifier; auto-generated if not supplied.
    default_ttl:
        Default time-to-live in seconds for entries.  ``None`` means no
        expiration.
    """

    def __init__(
        self,
        id: Optional[str] = None,
        default_ttl: Optional[float] = None,
    ) -> None:
        self.id: str = id or uuid.uuid4().hex[:12]
        self.default_ttl = default_ttl
        self._store: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def store(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Store *value* under *key*.

        Parameters
        ----------
        key:
            Lookup key.
        value:
            Arbitrary Python object.
        ttl:
            Time-to-live in seconds.  Overrides ``default_ttl`` for this
            entry.  ``None`` falls back to the instance default.
        """
        if not isinstance(key, str):
            raise TypeError(f"key must be a string, got {type(key)!r}")
        effective_ttl = ttl if ttl is not None else self.default_ttl
        now = datetime.now(timezone.utc)
        expires_at = (
            datetime.fromtimestamp(now.timestamp() + effective_ttl, tz=timezone.utc)
            if effective_ttl is not None
            else None
        )
        self._store[key] = {
            "value": value,
            "stored_at": now,
            "expires_at": expires_at,
        }
        logger.debug("WorkingMemory %s: stored %r", self.id, key)

    def retrieve(self, key: str, default: Any = None) -> Any:
        """Retrieve the value for *key*, or *default* if missing / expired.

        Expired entries are removed on access (lazy eviction).
        """
        entry = self._store.get(key)
        if entry is None:
            return default
        if self._is_expired(entry):
            del self._store[key]
            logger.debug("WorkingMemory %s: evicted expired key %r", self.id, key)
            return default
        return entry["value"]

    def list_keys(self) -> List[str]:
        """Return a list of all non-expired keys."""
        self._evict_expired()
        return sorted(self._store.keys())

    def has_key(self, key: str) -> bool:
        """Return ``True`` if *key* exists and is not expired."""
        entry = self._store.get(key)
        if entry is None:
            return False
        if self._is_expired(entry):
            del self._store[key]
            return False
        return True

    def delete(self, key: str) -> bool:
        """Delete *key*.  Returns ``True`` if the key existed."""
        return self._store.pop(key, None) is not None

    def clear(self) -> None:
        """Remove all entries."""
        self._store.clear()
        logger.debug("WorkingMemory %s: cleared", self.id)

    # ------------------------------------------------------------------
    # Bulk access
    # ------------------------------------------------------------------

    def as_dict(self) -> Dict[str, Any]:
        """Return all non-expired key-value pairs as a plain dictionary."""
        self._evict_expired()
        return {k: v["value"] for k, v in self._store.items()}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_expired(entry: Dict[str, Any]) -> bool:
        expires_at = entry.get("expires_at")
        if expires_at is None:
            return False
        return datetime.now(timezone.utc) >= expires_at

    def _evict_expired(self) -> None:
        """Remove all expired entries."""
        expired = [k for k, v in self._store.items() if self._is_expired(v)]
        for key in expired:
            del self._store[key]
        if expired:
            logger.debug(
                "WorkingMemory %s: evicted %d expired key(s)", self.id, len(expired),
            )

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        self._evict_expired()
        return len(self._store)

    def __contains__(self, key: str) -> bool:
        return self.has_key(key)

    def __repr__(self) -> str:
        return f"WorkingMemory(id={self.id!r}, keys={len(self)})"
