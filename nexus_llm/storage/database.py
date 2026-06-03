"""Nexus-LLM SQLite Database Manager.

Manages SQLite database connections, initialization, and lifecycle for
the Nexus-LLM framework. Provides connection pooling, transaction support,
and database health monitoring.
"""

import json
import logging
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Configuration for the database manager.

    Attributes:
        db_path: Path to the SQLite database file.
        wal_mode: Whether to enable WAL journal mode.
        busy_timeout: Timeout in ms for database lock contention.
        max_connections: Maximum number of connections in the pool.
        auto_migrate: Whether to run migrations on initialization.
        enable_foreign_keys: Whether to enforce foreign key constraints.
    """

    db_path: str = "nexus_llm.db"
    wal_mode: bool = True
    busy_timeout: int = 5000
    max_connections: int = 5
    auto_migrate: bool = True
    enable_foreign_keys: bool = True


@dataclass
class DatabaseStats:
    """Statistics about the database.

    Attributes:
        total_tables: Number of tables in the database.
        total_indices: Number of indices.
        database_size_mb: Size of the database file in MB.
        wal_size_mb: Size of the WAL file in MB.
        page_count: Number of database pages.
        free_pages: Number of free pages.
    """

    total_tables: int = 0
    total_indices: int = 0
    database_size_mb: float = 0.0
    wal_size_mb: float = 0.0
    page_count: int = 0
    free_pages: int = 0


class DatabaseManager:
    """Manages SQLite database connections and lifecycle for Nexus-LLM.

    Provides connection pooling, transaction management, database
    initialization, and health monitoring capabilities.

    Attributes:
        config: Database configuration.
    """

    def __init__(self, config: Optional[DatabaseConfig] = None) -> None:
        """Initialize the database manager.

        Args:
            config: Database configuration. Uses defaults if None.
        """
        self.config = config or DatabaseConfig()
        self._local = threading.local()
        self._lock = threading.RLock()
        self._initialized = False
        self._connections: List[sqlite3.Connection] = []

    def _create_connection(self) -> sqlite3.Connection:
        """Create a new SQLite connection with configured settings.

        Returns:
            Configured SQLite connection.
        """
        db_dir = os.path.dirname(self.config.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        conn = sqlite3.connect(self.config.db_path, timeout=self.config.busy_timeout / 1000.0)
        conn.row_factory = sqlite3.Row

        if self.config.enable_foreign_keys:
            conn.execute("PRAGMA foreign_keys = ON")

        if self.config.wal_mode:
            conn.execute("PRAGMA journal_mode = WAL")

        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA temp_store = MEMORY")
        conn.execute("PRAGMA mmap_size = 268435456")  # 256 MB
        conn.execute("PRAGMA cache_size = -64000")  # 64 MB

        return conn

    @property
    def connection(self) -> sqlite3.Connection:
        """Get a thread-local database connection.

        Returns:
            Thread-local SQLite connection.
        """
        if not hasattr(self._local, "connection") or self._local.connection is None:
            self._local.connection = self._create_connection()
            with self._lock:
                self._connections.append(self._local.connection)
        return self._local.connection

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for database transactions.

        Provides automatic commit on success and rollback on failure.

        Yields:
            Database connection within a transaction.

        Example:
            with db.transaction() as conn:
                conn.execute("INSERT INTO models (name) VALUES (?)", ("gpt2",))
        """
        conn = self.connection
        try:
            conn.execute("BEGIN")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    @contextmanager
    def cursor(self) -> Generator[sqlite3.Cursor, None, None]:
        """Context manager for a database cursor.

        Yields:
            Database cursor that is automatically closed.

        Example:
            with db.cursor() as cur:
                cur.execute("SELECT * FROM models")
                rows = cur.fetchall()
        """
        conn = self.connection
        cur = conn.cursor()
        try:
            yield cur
        finally:
            cur.close()

    def initialize(self) -> None:
        """Initialize the database and run migrations.

        Ensures all required tables exist by running the migration system.
        """
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            if self.config.auto_migrate:
                from nexus_llm.migrations.migrate import MigrationRunner

                runner = MigrationRunner(db_path=self.config.db_path)
                runner.upgrade()
                runner.close()

            self._initialized = True
            logger.info(f"Database initialized at {self.config.db_path}")

    def execute(self, sql: str, params: Tuple = ()) -> sqlite3.Cursor:
        """Execute a SQL statement.

        Args:
            sql: SQL statement to execute.
            params: Parameters for parameterized queries.

        Returns:
            Cursor with results.
        """
        return self.connection.execute(sql, params)

    def execute_many(self, sql: str, params_list: List[Tuple]) -> sqlite3.Cursor:
        """Execute a SQL statement with multiple parameter sets.

        Args:
            sql: SQL statement to execute.
            params_list: List of parameter tuples.

        Returns:
            Cursor.
        """
        return self.connection.executemany(sql, params_list)

    def fetch_one(self, sql: str, params: Tuple = ()) -> Optional[Dict[str, Any]]:
        """Execute a query and return the first row as a dictionary.

        Args:
            sql: SQL query to execute.
            params: Parameters for the query.

        Returns:
            Dictionary representing the row, or None if no results.
        """
        cursor = self.connection.execute(sql, params)
        row = cursor.fetchone()
        if row is None:
            return None
        return dict(row)

    def fetch_all(self, sql: str, params: Tuple = ()) -> List[Dict[str, Any]]:
        """Execute a query and return all rows as dictionaries.

        Args:
            sql: SQL query to execute.
            params: Parameters for the query.

        Returns:
            List of dictionaries representing rows.
        """
        cursor = self.connection.execute(sql, params)
        return [dict(row) for row in cursor.fetchall()]

    def fetch_value(self, sql: str, params: Tuple = ()) -> Any:
        """Execute a query and return the first column of the first row.

        Args:
            sql: SQL query to execute.
            params: Parameters for the query.

        Returns:
            Single value, or None if no results.
        """
        cursor = self.connection.execute(sql, params)
        row = cursor.fetchone()
        return row[0] if row else None

    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database.

        Args:
            table_name: Name of the table to check.

        Returns:
            True if the table exists.
        """
        result = self.fetch_value(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        )
        return result is not None and result > 0

    def get_table_names(self) -> List[str]:
        """Get names of all user tables in the database.

        Returns:
            List of table names.
        """
        rows = self.fetch_all(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
        return [row["name"] for row in rows]

    def get_table_info(self, table_name: str) -> List[Dict[str, Any]]:
        """Get column information for a table.

        Args:
            table_name: Name of the table.

        Returns:
            List of column info dictionaries.
        """
        return self.fetch_all(f"PRAGMA table_info({table_name})")

    def get_stats(self) -> DatabaseStats:
        """Get database statistics.

        Returns:
            DatabaseStats with current database metrics.
        """
        stats = DatabaseStats()

        try:
            tables = self.fetch_all(
                "SELECT COUNT(*) as cnt FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
            stats.total_tables = tables[0]["cnt"] if tables else 0

            indices = self.fetch_all(
                "SELECT COUNT(*) as cnt FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'"
            )
            stats.total_indices = indices[0]["cnt"] if indices else 0

            db_path = Path(self.config.db_path)
            if db_path.exists():
                stats.database_size_mb = db_path.stat().st_size / (1024 * 1024)

            wal_path = db_path.with_suffix(".db-wal")
            if wal_path.exists():
                stats.wal_size_mb = wal_path.stat().st_size / (1024 * 1024)

            page_info = self.fetch_one("PRAGMA page_count")
            if page_info:
                stats.page_count = list(page_info.values())[0]

            free_info = self.fetch_one("PRAGMA freelist_count")
            if free_info:
                stats.free_pages = list(free_info.values())[0]

        except sqlite3.Error as e:
            logger.warning(f"Failed to collect database stats: {e}")

        return stats

    def vacuum(self) -> None:
        """Run VACUUM to compact the database and reclaim space."""
        self.connection.execute("VACUUM")
        logger.info("Database VACUUM completed")

    def checkpoint(self) -> None:
        """Run a WAL checkpoint to flush changes to the main database file."""
        self.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        logger.info("WAL checkpoint completed")

    def backup(self, backup_path: str) -> None:
        """Create a backup of the database.

        Args:
            backup_path: Path for the backup database file.
        """
        backup_dir = os.path.dirname(backup_path)
        if backup_dir:
            os.makedirs(backup_dir, exist_ok=True)

        dest_conn = sqlite3.connect(backup_path)
        with self._lock:
            self.connection.backup(dest_conn)
        dest_conn.close()
        logger.info(f"Database backed up to {backup_path}")

    def health_check(self) -> Dict[str, Any]:
        """Perform a database health check.

        Returns:
            Dictionary with health status information.
        """
        try:
            start = time.monotonic()
            self.fetch_value("SELECT 1")
            latency_ms = (time.monotonic() - start) * 1000

            integrity = self.fetch_value("PRAGMA integrity_check")

            return {
                "healthy": True,
                "latency_ms": round(latency_ms, 2),
                "integrity": integrity,
                "db_path": self.config.db_path,
            }
        except sqlite3.Error as e:
            return {
                "healthy": False,
                "error": str(e),
                "db_path": self.config.db_path,
            }

    def close(self) -> None:
        """Close all database connections."""
        with self._lock:
            for conn in self._connections:
                try:
                    conn.close()
                except sqlite3.Error:
                    pass
            self._connections.clear()

        if hasattr(self._local, "connection"):
            self._local.connection = None

        self._initialized = False
        logger.info("Database connections closed")

    def __enter__(self) -> "DatabaseManager":
        """Context manager entry - initialize the database."""
        self.initialize()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - close connections."""
        self.close()
