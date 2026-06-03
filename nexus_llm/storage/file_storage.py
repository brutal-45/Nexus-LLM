"""File-based storage backend for Nexus-LLM.

Persists data as JSON files in a directory hierarchy with atomic writes.
"""

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, List, Optional

from nexus_llm.storage.backend import StorageBackend

logger = logging.getLogger(__name__)


class StorageError(Exception):
    """Raised when a storage operation fails."""


class FileStorage(StorageBackend):
    """File-system storage backend using JSON files.

    Each key maps to a single ``<key>.json`` file inside *base_dir*.
    Writes are atomic (write to temp file, then rename) to avoid
    corruption on crash.

    Example::

        store = FileStorage(base_dir="/tmp/nexus_store")
        store.save("config", {"theme": "dark"})
        print(store.load("config"))
    """

    def __init__(self, base_dir: str = ".nexus_storage") -> None:
        self._base_dir = Path(base_dir).resolve()
        os.makedirs(self._base_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def save(self, key: str, value: Any) -> None:
        """Persist *value* as a JSON file under *key*.

        The write is performed atomically via a temporary file and
        ``os.replace``.
        """
        file_path = self._key_to_path(key)
        os.makedirs(file_path.parent, exist_ok=True)

        try:
            payload = json.dumps(value, indent=2, sort_keys=True, default=str)
        except (TypeError, ValueError) as exc:
            raise StorageError(
                f"Cannot serialise value for key {key!r}: {exc}"
            ) from exc

        # Atomic write: write to temp file in same dir, then replace
        fd, tmp_path = tempfile.mkstemp(
            dir=str(file_path.parent), suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w") as tmp_file:
                tmp_file.write(payload)
                tmp_file.flush()
                os.fsync(tmp_file.fileno())
            os.replace(tmp_path, str(file_path))
        except BaseException:
            # Clean up temp file on any failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        logger.debug("Saved key %r to %s", key, file_path)

    def load(self, key: str) -> Any:
        """Load the value stored under *key*.

        Raises:
            KeyError: If *key* does not exist.
            StorageError: If the file cannot be parsed.
        """
        file_path = self._key_to_path(key)
        if not file_path.exists():
            raise KeyError(f"Key {key!r} not found")

        try:
            with open(file_path, "r") as fh:
                return json.load(fh)
        except json.JSONDecodeError as exc:
            raise StorageError(
                f"Corrupt JSON for key {key!r}: {exc}"
            ) from exc

    def delete(self, key: str) -> bool:
        """Delete the file for *key*.

        Returns:
            True if the file existed and was deleted, False otherwise.
        """
        file_path = self._key_to_path(key)
        if file_path.exists():
            file_path.unlink()
            logger.debug("Deleted key %r", key)
            return True
        return False

    def exists(self, key: str) -> bool:
        """Return True if a file exists for *key*."""
        return self._key_to_path(key).exists()

    def list_keys(self) -> List[str]:
        """Return all stored keys (derived from ``.json`` filenames)."""
        keys: List[str] = []
        for path in sorted(self._base_dir.rglob("*.json")):
            rel = path.relative_to(self._base_dir)
            # Strip .json suffix and convert OS path separator to /
            key = str(rel.with_suffix("")).replace(os.sep, "/")
            keys.append(key)
        return keys

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _key_to_path(self, key: str) -> Path:
        """Convert a storage key to a filesystem path.

        Keys may contain ``/`` to denote sub-directories.
        """
        # Sanitise key to prevent directory traversal
        safe_key = key.replace("..", "_").lstrip("/")
        return self._base_dir / f"{safe_key}.json"

    @property
    def base_dir(self) -> str:
        """Return the base directory path as a string."""
        return str(self._base_dir)
