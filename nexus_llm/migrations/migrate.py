"""Nexus-LLM Database Migration Runner.

Discovers, orders, and executes database migrations. Supports applying
pending migrations, rolling back to specific versions, inspecting migration
status, and generating migration checksums for integrity verification.
"""

import hashlib
import importlib
import logging
import os
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class MigrationRecord:
    """Record of an applied migration in the database.

    Attributes:
        migration_id: Unique identifier for the migration.
        version: Numeric version number.
        description: Human-readable description.
        applied_at: Timestamp when the migration was applied.
        checksum: SHA256 checksum of the migration content.
        execution_time_ms: How long the migration took to apply.
    """

    migration_id: str
    version: int
    description: str = ""
    applied_at: Optional[str] = None
    checksum: Optional[str] = None
    execution_time_ms: Optional[int] = None


@dataclass
class MigrationInfo:
    """Information about a discovered migration module.

    Attributes:
        module_name: Python module name for the migration.
        file_path: Path to the migration file.
        migration_id: Unique migration identifier.
        version: Numeric version number.
        description: Human-readable description.
        is_applied: Whether this migration has been applied.
    """

    module_name: str
    file_path: str
    migration_id: str = ""
    version: int = 0
    description: str = ""
    is_applied: bool = False


class MigrationRunner:
    """Discovers and executes database migrations for Nexus-LLM.

    The migration runner scans the migrations package for migration modules,
    tracks applied migrations in the database, and provides methods to
    apply or rollback migrations.

    Attributes:
        db_path: Path to the SQLite database file.
        migrations_package: Python package path containing migration modules.
    """

    def __init__(
        self,
        db_path: str = "nexus_llm.db",
        migrations_package: str = "nexus_llm.migrations",
    ) -> None:
        """Initialize the migration runner.

        Args:
            db_path: Path to the SQLite database file.
            migrations_package: Dotted path to the migrations package.
        """
        self.db_path = db_path
        self.migrations_package = migrations_package
        self._connection: Optional[sqlite3.Connection] = None

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create a database connection.

        Returns:
            Active SQLite connection with foreign keys enabled.
        """
        if self._connection is None:
            self._connection = sqlite3.connect(self.db_path)
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.execute("PRAGMA journal_mode = WAL")
        return self._connection

    def _ensure_migration_table(self) -> None:
        """Ensure the migration_history table exists.

        This is called before any migration operations to guarantee
        the tracking table is available.
        """
        conn = self._get_connection()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS migration_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                migration_id TEXT NOT NULL UNIQUE,
                version INTEGER NOT NULL,
                description TEXT,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                checksum TEXT,
                execution_time_ms INTEGER
            )
        """)
        conn.commit()

    def _compute_checksum(self, file_path: str) -> str:
        """Compute SHA256 checksum of a migration file.

        Args:
            file_path: Path to the migration Python file.

        Returns:
            Hex-encoded SHA256 hash of the file contents.
        """
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def discover_migrations(self) -> List[MigrationInfo]:
        """Discover all migration modules in the migrations package.

        Scans the migrations directory for Python files matching the
        pattern 'vNNN_description.py', loads their metadata, and
        returns them sorted by version number.

        Returns:
            List of MigrationInfo objects sorted by version.
        """
        migrations: List[MigrationInfo] = []

        try:
            package_spec = importlib.util.find_spec(self.migrations_package)
            if package_spec is None or package_spec.submodule_search_locations is None:
                logger.warning(f"Migrations package not found: {self.migrations_package}")
                return migrations

            package_dir = Path(package_spec.submodule_search_locations[0])
        except (ModuleNotFoundError, ValueError) as e:
            logger.warning(f"Failed to locate migrations package: {e}")
            return migrations

        for file_path in sorted(package_dir.glob("v*.py")):
            if file_path.name.startswith("__"):
                continue

            module_name = f"{self.migrations_package}.{file_path.stem}"
            info = MigrationInfo(
                module_name=module_name,
                file_path=str(file_path),
            )

            try:
                module = importlib.import_module(module_name)
                info.migration_id = getattr(module, "MIGRATION_ID", file_path.stem)
                info.version = getattr(module, "MIGRATION_VERSION", 0)
                info.description = getattr(module, "MIGRATION_DESCRIPTION", "")
            except (ImportError, AttributeError) as e:
                logger.warning(f"Failed to load migration module {module_name}: {e}")
                continue

            migrations.append(info)

        migrations.sort(key=lambda m: m.version)
        return migrations

    def get_applied_migrations(self) -> List[MigrationRecord]:
        """Get list of already applied migrations from the database.

        Returns:
            List of MigrationRecord objects for applied migrations.
        """
        self._ensure_migration_table()
        conn = self._get_connection()

        cursor = conn.execute(
            "SELECT migration_id, version, description, applied_at, checksum, execution_time_ms "
            "FROM migration_history ORDER BY version"
        )

        records = []
        for row in cursor.fetchall():
            records.append(MigrationRecord(
                migration_id=row[0],
                version=row[1],
                description=row[2],
                applied_at=row[3],
                checksum=row[4],
                execution_time_ms=row[5],
            ))

        return records

    def get_pending_migrations(self) -> List[MigrationInfo]:
        """Get list of migrations that have not yet been applied.

        Returns:
            List of MigrationInfo for pending migrations.
        """
        all_migrations = self.discover_migrations()
        applied = {r.migration_id for r in self.get_applied_migrations()}

        for migration in all_migrations:
            migration.is_applied = migration.migration_id in applied

        return [m for m in all_migrations if not m.is_applied]

    def apply_migration(self, migration_info: MigrationInfo) -> None:
        """Apply a single migration.

        Imports the migration module, calls its up() function, and
        records the migration in the history table.

        Args:
            migration_info: Information about the migration to apply.

        Raises:
            RuntimeError: If the migration fails to apply.
        """
        self._ensure_migration_table()
        conn = self._get_connection()

        try:
            module = importlib.import_module(migration_info.module_name)
            up_func: Optional[Callable] = getattr(module, "up", None)

            if up_func is None:
                raise RuntimeError(f"Migration {migration_info.module_name} has no up() function")

            checksum = self._compute_checksum(migration_info.file_path)
            start_time = time.monotonic()

            up_func(conn)

            elapsed_ms = int((time.monotonic() - start_time) * 1000)

            conn.execute(
                "INSERT INTO migration_history (migration_id, version, description, checksum, execution_time_ms) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    migration_info.migration_id,
                    migration_info.version,
                    migration_info.description,
                    checksum,
                    elapsed_ms,
                ),
            )
            conn.commit()

            logger.info(
                f"Applied migration {migration_info.migration_id} "
                f"(v{migration_info.version}) in {elapsed_ms}ms"
            )

        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to apply migration {migration_info.migration_id}: {e}")
            raise RuntimeError(f"Migration {migration_info.migration_id} failed: {e}") from e

    def rollback_migration(self, migration_info: MigrationInfo) -> None:
        """Rollback a single migration.

        Imports the migration module, calls its down() function, and
        removes the migration record from the history table.

        Args:
            migration_info: Information about the migration to rollback.

        Raises:
            RuntimeError: If the rollback fails.
        """
        self._ensure_migration_table()
        conn = self._get_connection()

        try:
            module = importlib.import_module(migration_info.module_name)
            down_func: Optional[Callable] = getattr(module, "down", None)

            if down_func is None:
                raise RuntimeError(f"Migration {migration_info.module_name} has no down() function")

            down_func(conn)

            conn.execute(
                "DELETE FROM migration_history WHERE migration_id = ?",
                (migration_info.migration_id,),
            )
            conn.commit()

            logger.info(f"Rolled back migration {migration_info.migration_id}")

        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to rollback migration {migration_info.migration_id}: {e}")
            raise RuntimeError(f"Rollback of {migration_info.migration_id} failed: {e}") from e

    def upgrade(self, target_version: Optional[int] = None) -> List[str]:
        """Apply pending migrations up to a target version.

        Args:
            target_version: Maximum version to apply. If None, applies all pending.

        Returns:
            List of applied migration IDs.
        """
        pending = self.get_pending_migrations()

        if target_version is not None:
            pending = [m for m in pending if m.version <= target_version]

        applied = []
        for migration in pending:
            self.apply_migration(migration)
            applied.append(migration.migration_id)

        if not applied:
            logger.info("No pending migrations to apply")
        else:
            logger.info(f"Applied {len(applied)} migration(s)")

        return applied

    def downgrade(self, target_version: int = 0) -> List[str]:
        """Rollback migrations down to a target version.

        Args:
            target_version: Minimum version to keep. All versions above
                this will be rolled back.

        Returns:
            List of rolled-back migration IDs.
        """
        all_migrations = self.discover_migrations()
        applied_records = self.get_applied_migrations()
        applied_ids = {r.migration_id for r in applied_records}

        # Get applied migrations in reverse order
        applied_migrations = [
            m for m in reversed(all_migrations)
            if m.migration_id in applied_ids and m.version > target_version
        ]

        rolled_back = []
        for migration in applied_migrations:
            self.rollback_migration(migration)
            rolled_back.append(migration.migration_id)

        if not rolled_back:
            logger.info("No migrations to rollback")
        else:
            logger.info(f"Rolled back {len(rolled_back)} migration(s)")

        return rolled_back

    def get_status(self) -> Dict[str, Any]:
        """Get the current migration status.

        Returns:
            Dictionary with current version, applied count, pending count,
            and lists of applied and pending migrations.
        """
        all_migrations = self.discover_migrations()
        applied_records = self.get_applied_migrations()
        applied_ids = {r.migration_id for r in applied_records}

        for migration in all_migrations:
            migration.is_applied = migration.migration_id in applied_ids

        current_version = max((r.version for r in applied_records), default=0)

        return {
            "current_version": current_version,
            "total_migrations": len(all_migrations),
            "applied_count": len(applied_records),
            "pending_count": len(all_migrations) - len(applied_records),
            "applied": [
                {
                    "migration_id": m.migration_id,
                    "version": m.version,
                    "description": m.description,
                }
                for m in all_migrations if m.is_applied
            ],
            "pending": [
                {
                    "migration_id": m.migration_id,
                    "version": m.version,
                    "description": m.description,
                }
                for m in all_migrations if not m.is_applied
            ],
        }

    def close(self) -> None:
        """Close the database connection."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None
