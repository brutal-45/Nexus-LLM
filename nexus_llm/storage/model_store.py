"""Model metadata store for Nexus-LLM.

Manages persistent metadata about downloaded / cached models.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from nexus_llm.storage.backend import StorageBackend

logger = logging.getLogger(__name__)


class ModelStore:
    """Persistent store for model metadata.

    Delegates actual persistence to a :class:`StorageBackend`
    implementation.  Metadata entries are stored under keys of the form
    ``models/<model_id>``.

    Example::

        from nexus_llm.storage.sqlite_storage import SQLiteStorage

        backend = SQLiteStorage(db_path="nexus.db")
        store = ModelStore(backend)
        store.save_model_metadata("gpt2-medium", {
            "path": "/models/gpt2-medium",
            "size_bytes": 1437344000,
        })
        meta = store.get_model_metadata("gpt2-medium")
    """

    KEY_PREFIX = "models"

    def __init__(self, backend: StorageBackend) -> None:
        self._backend = backend

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def save_model_metadata(self, model_id: str, metadata: Dict[str, Any]) -> None:
        """Create or update metadata for *model_id*.

        If metadata already exists for the model, the new values are
        merged into the existing record.

        Args:
            model_id: Unique model identifier (e.g. ``"gpt2-medium"``).
            metadata: Dict of model properties (path, size, format, etc.).
        """
        key = self._make_key(model_id)

        # Merge with existing metadata
        existing: Dict[str, Any] = {}
        if self._backend.exists(key):
            try:
                existing = self._backend.load(key)
            except Exception:
                existing = {}

        existing.update(metadata)
        # Ensure model_id is stored inside the record
        existing["model_id"] = model_id

        self._backend.save(key, existing)
        logger.info("Saved metadata for model %r", model_id)

    def get_model_metadata(self, model_id: str) -> Dict[str, Any]:
        """Retrieve metadata for *model_id*.

        Args:
            model_id: Unique model identifier.

        Returns:
            Dict of model metadata.

        Raises:
            KeyError: If no metadata exists for *model_id*.
        """
        key = self._make_key(model_id)
        return self._backend.load(key)

    def list_cached_models(self) -> List[Dict[str, Any]]:
        """Return a summary list of all cached models.

        Each entry contains at least ``model_id`` plus whatever
        metadata was stored.
        """
        models: List[Dict[str, Any]] = []
        for key in self._backend.list_keys():
            if not key.startswith(self.KEY_PREFIX):
                continue
            try:
                data = self._backend.load(key)
                models.append(data)
            except Exception:
                logger.warning("Failed to load model metadata at key %r", key)
        return models

    def delete_model(self, model_id: str) -> bool:
        """Delete metadata for *model_id*.

        Args:
            model_id: Unique model identifier.

        Returns:
            True if the metadata existed and was deleted.
        """
        key = self._make_key(model_id)
        deleted = self._backend.delete(key)
        if deleted:
            logger.info("Deleted metadata for model %r", model_id)
        return deleted

    def get_model_size(self, model_id: str) -> int:
        """Return the size of a cached model in bytes.

        The size is read from the ``size_bytes`` metadata field.  If
        the model is not found or the field is missing, ``0`` is
        returned.

        Args:
            model_id: Unique model identifier.

        Returns:
            Model size in bytes, or 0 if unavailable.
        """
        try:
            metadata = self.get_model_metadata(model_id)
            return int(metadata.get("size_bytes", 0))
        except (KeyError, ValueError, TypeError):
            return 0

    # ------------------------------------------------------------------
    # Convenience queries
    # ------------------------------------------------------------------

    def model_exists(self, model_id: str) -> bool:
        """Return True if metadata exists for *model_id*."""
        return self._backend.exists(self._make_key(model_id))

    def get_models_by_format(self, fmt: str) -> List[Dict[str, Any]]:
        """Return all models whose ``format`` metadata field matches *fmt*.

        Args:
            fmt: Model format string (e.g. ``"safetensors"``, ``"pytorch"``).

        Returns:
            List of matching model metadata dicts.
        """
        results: List[Dict[str, Any]] = []
        for model in self.list_cached_models():
            if model.get("format") == fmt:
                results.append(model)
        return results

    def total_cache_size(self) -> int:
        """Return the aggregate size (bytes) of all cached models."""
        total = 0
        for model in self.list_cached_models():
            total += int(model.get("size_bytes", 0))
        return total

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_key(self, model_id: str) -> str:
        return f"{self.KEY_PREFIX}/{model_id}"
