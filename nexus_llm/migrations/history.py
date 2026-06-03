"""Migration history tracker for Nexus-LLM.

Persists applied-migration records to a JSON file so that state
survives across process restarts.
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MigrationHistory:
    """Track which migrations have been applied, persisting to a JSON file.

    Example::

        hist = MigrationHistory(path=".migrations.json")
        hist.record("20250101_000001", direction="up")
        print(hist.is_applied("20250101_000001"))  # True
    """

    def __init__(self, path: Optional[str] = None) -> None:
        """Initialise migration history.

        Args:
            path: Filesystem path for the JSON history file.  Defaults to
                  ``.nexus_migrations.json`` in the current directory.
        """
        self._path = path or ".nexus_migrations.json"
        self._records: List[Dict[str, Any]] = self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> List[Dict[str, Any]]:
        """Load history from the JSON file, returning an empty list on miss."""
        if os.path.isfile(self._path):
            try:
                with open(self._path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                    if isinstance(data, list):
                        return data
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to load migration history: %s", exc)
        return []

    def _save(self) -> None:
        """Persist the current records to disk."""
        try:
            with open(self._path, "w", encoding="utf-8") as fh:
                json.dump(self._records, fh, indent=2)
        except OSError as exc:
            logger.error("Failed to save migration history: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, migration_id: str, direction: str = "up") -> None:
        """Record that a migration was applied or rolled back.

        Args:
            migration_id: The migration version/identifier.
            direction: ``"up"`` for apply, ``"down"`` for rollback.
        """
        entry = {
            "migration_id": migration_id,
            "direction": direction,
            "timestamp": time.time(),
        }
        self._records.append(entry)
        self._save()
        logger.info("Recorded migration %s %s", migration_id, direction)

    def get_applied(self) -> List[str]:
        """Return a list of migration IDs currently applied.

        A migration is considered applied if its last recorded direction
        is ``"up"``.
        """
        applied: Dict[str, str] = {}
        for rec in self._records:
            applied[rec["migration_id"]] = rec["direction"]
        return [mid for mid, d in applied.items() if d == "up"]

    def get_last(self) -> str:
        """Return the migration ID of the most recently applied migration.

        Returns:
            The migration ID string.

        Raises:
            ValueError: If no migrations have been applied.
        """
        applied = self.get_applied()
        if not applied:
            raise ValueError("No migrations have been applied")
        # Return the last one by order of insertion in the records
        return applied[-1]

    def is_applied(self, migration_id: str) -> bool:
        """Check whether a specific migration is currently applied.

        Args:
            migration_id: The migration version/identifier.

        Returns:
            True if the migration's last recorded direction is ``"up"``.
        """
        return migration_id in self.get_applied()

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"<MigrationHistory path={self._path!r} "
            f"applied={len(self.get_applied())}>"
        )
