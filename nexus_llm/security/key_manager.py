"""Nexus-LLM Key Management.

Provides the KeyManager class for managing encryption keys,
API keys, and other secrets with rotation and access tracking.
"""

import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class KeyInfo:
    """Metadata about a managed key.

    Attributes:
        key_id: Unique key identifier.
        key_type: Type of key (encryption, api, signing, etc.).
        created_at: When the key was created.
        expires_at: When the key expires (None = no expiry).
        rotated_from: Previous key ID if this is a rotation.
        access_count: Number of times the key has been accessed.
        last_accessed: Last access timestamp.
        tags: Optional tags for categorization.
    """

    key_id: str
    key_type: str = "encryption"
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    rotated_from: Optional[str] = None
    access_count: int = 0
    last_accessed: Optional[float] = None
    tags: List[str] = field(default_factory=list)

    def is_expired(self) -> bool:
        """Check whether the key has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key_id": self.key_id,
            "key_type": self.key_type,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "rotated_from": self.rotated_from,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
            "is_expired": self.is_expired(),
            "tags": self.tags,
        }


class KeyManager:
    """Manages encryption keys and secrets.

    The KeyManager stores keys securely in memory, supports key
    rotation, expiration, and access tracking.

    Example::

        km = KeyManager()
        key_id = km.generate_key("encryption", ttl_days=90)
        key = km.get_key(key_id)
        km.rotate_key(key_id)
    """

    def __init__(self) -> None:
        self._keys: Dict[str, bytes] = {}
        self._key_info: Dict[str, KeyInfo] = {}
        logger.debug("KeyManager initialized")

    def generate_key(
        self,
        key_type: str = "encryption",
        ttl_days: Optional[int] = None,
        tags: Optional[List[str]] = None,
    ) -> str:
        """Generate and store a new key.

        Args:
            key_type: Type of key.
            ttl_days: Time-to-live in days (None = no expiry).
            tags: Optional tags.

        Returns:
            The key ID.
        """
        raw_key = os.urandom(32)
        key_id = hashlib.sha256(raw_key).hexdigest()[:16]

        self._keys[key_id] = raw_key
        expires_at = None
        if ttl_days is not None:
            expires_at = time.time() + (ttl_days * 86400)

        self._key_info[key_id] = KeyInfo(
            key_id=key_id,
            key_type=key_type,
            expires_at=expires_at,
            tags=tags or [],
        )
        logger.info("Generated key %s (type=%s, ttl=%s)", key_id, key_type, ttl_days)
        return key_id

    def get_key(self, key_id: str) -> Optional[bytes]:
        """Retrieve a key by ID.

        Args:
            key_id: The key identifier.

        Returns:
            The raw key bytes, or None if not found or expired.
        """
        info = self._key_info.get(key_id)
        if info is None:
            return None
        if info.is_expired():
            logger.warning("Key %s has expired", key_id)
            return None

        info.access_count += 1
        info.last_accessed = time.time()
        return self._keys.get(key_id)

    def get_key_info(self, key_id: str) -> Optional[KeyInfo]:
        """Retrieve key metadata without accessing the key.

        Args:
            key_id: The key identifier.

        Returns:
            KeyInfo or None.
        """
        return self._key_info.get(key_id)

    def rotate_key(self, key_id: str, ttl_days: Optional[int] = None) -> Optional[str]:
        """Rotate a key, creating a new one that replaces it.

        Args:
            key_id: The key to rotate.
            ttl_days: Optional TTL for the new key.

        Returns:
            The new key ID, or None if the original key was not found.
        """
        info = self._key_info.get(key_id)
        if info is None:
            return None

        new_key_id = self.generate_key(
            key_type=info.key_type,
            ttl_days=ttl_days,
            tags=info.tags,
        )
        self._key_info[new_key_id].rotated_from = key_id
        self.delete_key(key_id)
        logger.info("Rotated key %s -> %s", key_id, new_key_id)
        return new_key_id

    def delete_key(self, key_id: str) -> bool:
        """Delete a key.

        Args:
            key_id: The key identifier.

        Returns:
            True if the key was found and deleted.
        """
        self._keys.pop(key_id, None)
        info = self._key_info.pop(key_id, None)
        if info:
            logger.info("Deleted key %s", key_id)
        return info is not None

    def list_keys(self, key_type: Optional[str] = None) -> List[KeyInfo]:
        """List all keys, optionally filtered by type.

        Args:
            key_type: Optional filter by key type.

        Returns:
            List of KeyInfo objects.
        """
        infos = list(self._key_info.values())
        if key_type:
            infos = [i for i in infos if i.key_type == key_type]
        return infos

    def cleanup_expired(self) -> int:
        """Remove all expired keys.

        Returns:
            Number of keys removed.
        """
        expired = [kid for kid, info in self._key_info.items() if info.is_expired()]
        for kid in expired:
            self.delete_key(kid)
        return len(expired)
