"""API key generation, validation, and revocation."""

from __future__ import annotations

import hashlib
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from nexus_llm.auth.permissions import Permission


# Key format: nxl-xxxxxxxx-xxxx-xxxx-xxxx
_KEY_PREFIX = "nxl"
_KEY_PATTERN = f"{_KEY_PREFIX}-{{}}-{{}}-{{}}-{{}}"


def _generate_key_string() -> str:
    """Generate a random API key in the project format.

    Format: ``nxl-xxxxxxxx-xxxx-xxxx-xxxx``
    """
    part1 = secrets.token_hex(4)  # 8 hex chars
    part2 = secrets.token_hex(2)  # 4 hex chars
    part3 = secrets.token_hex(2)  # 4 hex chars
    part4 = secrets.token_hex(2)  # 4 hex chars
    return _KEY_PATTERN.format(part1, part2, part3, part4)


def _hash_key(key: str) -> str:
    """Return a SHA-256 hash of the key for secure storage."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


@dataclass
class APIKeyInfo:
    """Metadata associated with an API key.

    Attributes:
        name: Human-readable name (often the username).
        key_hash: SHA-256 hash of the raw key.
        permissions: Permissions granted by this key.
        created_at: Unix timestamp when the key was created.
        revoked: Whether the key has been revoked.
    """

    name: str
    key_hash: str
    permissions: Set[Permission] = field(default_factory=set)
    created_at: float = field(default_factory=time.time)
    revoked: bool = False


class APIKeyManager:
    """Manages the lifecycle of API keys.

    Keys are stored internally by their SHA-256 hash so that the raw
    key is never persisted in plaintext beyond the initial creation
    response.

    Example::

        km = APIKeyManager()
        raw_key = km.generate_key("alice", {Permission.read, Permission.chat})
        valid, info = km.validate_key(raw_key)   # (True, {...})
        km.revoke_key(raw_key)
    """

    def __init__(self) -> None:
        # key_hash → APIKeyInfo
        self._keys: Dict[str, APIKeyInfo] = {}
        # raw_key → key_hash  (transient, for lookup before revocation)
        self._raw_to_hash: Dict[str, str] = {}
        # Shared mutable metadata dict (also used by AuthManager)
        self._key_metadata: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Key generation
    # ------------------------------------------------------------------

    def generate_key(
        self,
        name: str,
        permissions: Optional[Set[Permission]] = None,
    ) -> str:
        """Generate a new API key.

        Args:
            name: A human-readable label (typically the username).
            permissions: Permissions this key grants.

        Returns:
            The raw API key string.  **This is the only time the raw
            key is available.**
        """
        permissions = permissions or set()
        raw_key = _generate_key_string()
        key_hash = _hash_key(raw_key)

        info = APIKeyInfo(
            name=name,
            key_hash=key_hash,
            permissions=permissions,
        )

        with self._lock:
            self._keys[key_hash] = info
            self._raw_to_hash[raw_key] = key_hash
            self._key_metadata[raw_key] = {
                "name": name,
                "permissions": permissions,
                "created_at": info.created_at,
            }

        return raw_key

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_key(self, key: str) -> Tuple[bool, Dict[str, Any]]:
        """Validate an API key and return associated metadata.

        Args:
            key: The raw API key to validate.

        Returns:
            A ``(valid, user_info)`` tuple.  *valid* is ``True`` if the
            key exists and has not been revoked.  *user_info* contains
            ``name``, ``permissions``, and ``created_at``.
        """
        with self._lock:
            # Fast path: raw key still in transient map
            key_hash = self._raw_to_hash.get(key) or _hash_key(key)
            info = self._keys.get(key_hash)

            if info is None:
                return (False, {})

            if info.revoked:
                return (False, {"name": info.name, "revoked": True})

            user_info: Dict[str, Any] = {
                "name": info.name,
                "permissions": info.permissions,
                "created_at": info.created_at,
            }

            # Merge any extra metadata
            meta = self._key_metadata.get(key, {})
            user_info.update(meta)

            return (True, user_info)

    # ------------------------------------------------------------------
    # Revocation
    # ------------------------------------------------------------------

    def revoke_key(self, key: str) -> None:
        """Revoke an API key.

        Args:
            key: The raw API key to revoke.

        Raises:
            KeyError: If the key does not exist.
        """
        with self._lock:
            key_hash = self._raw_to_hash.get(key) or _hash_key(key)
            info = self._keys.get(key_hash)
            if info is None:
                raise KeyError(f"API key not found.")
            info.revoked = True
            self._raw_to_hash.pop(key, None)

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------

    def list_keys(self) -> List[Dict[str, Any]]:
        """Return metadata for all non-revoked keys.

        Returns:
            A list of dicts with ``name``, ``permissions``,
            ``created_at``, and ``revoked`` fields.
        """
        with self._lock:
            result = []
            for info in self._keys.values():
                result.append(
                    {
                        "name": info.name,
                        "permissions": info.permissions,
                        "created_at": info.created_at,
                        "revoked": info.revoked,
                    }
                )
            return result
