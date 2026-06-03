"""Migration manager for Nexus-LLM.

Orchestrates migration registration, execution, rollback, and
status queries using :class:`MigrationHistory` for persistence.
"""

import logging
from typing import Any, Dict, List, Optional

from nexus_llm.migrations.migration import Migration
from nexus_llm.migrations.history import MigrationHistory

logger = logging.getLogger(__name__)


class MigrationManager:
    """Register and run versioned migrations.

    The manager keeps an ordered list of registered migrations and
    delegates history persistence to :class:`MigrationHistory`.

    Example::

        mgr = MigrationManager()
        mgr.register(V1_InitialSetup())
        mgr.register(V2_AddTrainingConfig())
        applied = mgr.run_pending(context=my_config)
        print(mgr.get_status())
    """

    def __init__(
        self,
        history: Optional[MigrationHistory] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._migrations: Dict[str, Migration] = {}
        self._ordered_versions: List[str] = []
        self._history = history or MigrationHistory()
        self._context: Dict[str, Any] = context or {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, migration: Migration) -> None:
        """Register a migration for future execution.

        Migrations are stored in version-order so that ``run_pending()``
        always applies them sequentially.

        Args:
            migration: A :class:`Migration` instance.

        Raises:
            ValueError: If a migration with the same version is already
                        registered.
        """
        if migration.version in self._migrations:
            raise ValueError(
                f"Migration version {migration.version!r} is already registered"
            )
        self._migrations[migration.version] = migration
        self._ordered_versions.append(migration.version)
        self._ordered_versions.sort()
        logger.debug("Registered migration %s: %s", migration.version, migration.description)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run_pending(self) -> List[str]:
        """Apply all pending (not yet applied) migrations in order.

        Returns:
            List of migration versions that were applied.
        """
        applied: List[str] = []
        for version in self._ordered_versions:
            if not self._history.is_applied(version):
                migration = self._migrations[version]
                try:
                    migration.up(self._context)
                    self._history.record(version, direction="up")
                    applied.append(version)
                    logger.info(
                        "Applied migration %s: %s",
                        version, migration.description,
                    )
                except Exception as exc:
                    logger.error(
                        "Migration %s failed: %s", version, exc,
                    )
                    raise RuntimeError(
                        f"Migration {version} failed: {exc}"
                    ) from exc
        if not applied:
            logger.info("No pending migrations to apply.")
        return applied

    def rollback(self, steps: int = 1) -> List[str]:
        """Roll back the most recent *steps* migrations.

        Migrations are rolled back in reverse order of their version.

        Args:
            steps: Number of migrations to roll back (default 1).

        Returns:
            List of migration versions that were rolled back.

        Raises:
            ValueError: If *steps* is not positive or exceeds the
                        number of applied migrations.
        """
        if steps < 1:
            raise ValueError("steps must be a positive integer")

        applied_versions = [
            v for v in reversed(self._ordered_versions)
            if self._history.is_applied(v)
        ]

        if steps > len(applied_versions):
            raise ValueError(
                f"Cannot roll back {steps} migration(s); "
                f"only {len(applied_versions)} are applied."
            )

        rolled_back: List[str] = []
        for version in applied_versions[:steps]:
            migration = self._migrations[version]
            try:
                migration.down(self._context)
                self._history.record(version, direction="down")
                rolled_back.append(version)
                logger.info(
                    "Rolled back migration %s: %s",
                    version, migration.description,
                )
            except Exception as exc:
                logger.error(
                    "Rollback of migration %s failed: %s", version, exc,
                )
                raise RuntimeError(
                    f"Rollback of migration {version} failed: {exc}"
                ) from exc
        return rolled_back

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Return a status summary of all registered migrations.

        Returns:
            Dict with keys:

            - ``"total"``: total number of registered migrations
            - ``"applied"``: number of applied migrations
            - ``"pending"``: number of pending migrations
            - ``"migrations"``: list of dicts with version, description,
              and applied status
        """
        applied_set = set(self._history.get_applied())
        migration_details = []
        for version in self._ordered_versions:
            migration = self._migrations[version]
            migration_details.append({
                "version": version,
                "description": migration.description,
                "applied": version in applied_set,
            })

        return {
            "total": len(self._ordered_versions),
            "applied": len(applied_set & set(self._ordered_versions)),
            "pending": len(set(self._ordered_versions) - applied_set),
            "migrations": migration_details,
        }

    def get_pending(self) -> List[str]:
        """Return the versions of migrations not yet applied.

        Returns:
            List of version strings in application order.
        """
        return [
            v for v in self._ordered_versions
            if not self._history.is_applied(v)
        ]

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"<MigrationManager registered={len(self._migrations)} "
            f"pending={len(self.get_pending())}>"
        )
