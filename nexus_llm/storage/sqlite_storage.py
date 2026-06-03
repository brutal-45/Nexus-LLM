"""SQLite storage backend for Nexus-LLM.

Provides a durable, transactional key-value store backed by SQLite.
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from nexus_llm.storage.backend import StorageBackend

logger = logging.getLogger(__name__)


class StorageError(Exception):
    """Raised when a storage operation fails."""


class SQLiteStorage(StorageBackend):
    """SQLite-backed key-value storage.

    Data is stored in a single ``kv_store`` table with columns
    ``key`` (TEXT PRIMARY KEY) and ``value`` (TEXT holding JSON).

    All mutating operations are wrapped in transactions.

    Example::

        store = SQLiteStorage(db_path="nexus.db")
        store.save("config", {"theme": "dark"})
        store.save("count", 42)
        print(store.load("config"))  # {'theme': 'dark'}
    """

    def __init__(self, db_path: str = "nexus_storage.db") -> None:
        self._db_path = str(Path(db_path).resolve())
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    # ------------------------------------------------------------------
    # Database lifecycle
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        """Create the key-value table if it does not already exist."""
        conn = self._get_connection()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS kv_store (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_kv_key ON kv_store (key)
            """
        )
        conn.commit()
        logger.debug("Initialised SQLite store at %s", self._db_path)

    def _get_connection(self) -> sqlite3.Connection:
        """Return the current connection, creating one if needed."""
        if self._conn is None:
            self._conn = sqlite3.connect(self._db_path)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            logger.debug("Closed SQLite store at %s", self._db_path)

    # ------------------------------------------------------------------
    # Core CRUD operations
    # ------------------------------------------------------------------

    def save(self, key: str, value: Any) -> None:
        """Persist *value* under *key* (upsert semantics).

        Args:
            key: Unique identifier.
            value: Any JSON-serialisable value.

        Raises:
            StorageError: If serialisation or the database write fails.
        """
        try:
            payload = json.dumps(value, sort_keys=True, default=str)
        except (TypeError, ValueError) as exc:
            raise StorageError(
                f"Cannot serialise value for key {key!r}: {exc}"
            ) from exc

        conn = self._get_connection()
        try:
            conn.execute(
                """
                INSERT INTO kv_store (key, value)
                VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (key, payload),
            )
            conn.commit()
        except sqlite3.Error as exc:
            conn.rollback()
            raise StorageError(
                f"Failed to save key {key!r}: {exc}"
            ) from exc

        logger.debug("Saved key %r", key)

    def load(self, key: str) -> Any:
        """Retrieve the value stored under *key*.

        Raises:
            KeyError: If *key* does not exist.
            StorageError: If the stored JSON cannot be parsed.
        """
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT value FROM kv_store WHERE key = ?", (key,)
        )
        row = cursor.fetchone()
        if row is None:
            raise KeyError(f"Key {key!r} not found")

        try:
            return json.loads(row["value"])
        except json.JSONDecodeError as exc:
            raise StorageError(
                f"Corrupt JSON for key {key!r}: {exc}"
            ) from exc

    def delete(self, key: str) -> bool:
        """Remove *key* from the store.

        Returns:
            True if the key existed and was deleted, False otherwise.
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "DELETE FROM kv_store WHERE key = ?", (key,)
            )
            conn.commit()
            deleted = cursor.rowcount > 0
            if deleted:
                logger.debug("Deleted key %r", key)
            return deleted
        except sqlite3.Error as exc:
            conn.rollback()
            raise StorageError(
                f"Failed to delete key {key!r}: {exc}"
            ) from exc

    def exists(self, key: str) -> bool:
        """Return True if *key* is present in the store."""
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT 1 FROM kv_store WHERE key = ?", (key,)
        )
        return cursor.fetchone() is not None

    def list_keys(self) -> List[str]:
        """Return all keys in the store, sorted alphabetically."""
        conn = self._get_connection()
        cursor = conn.execute("SELECT key FROM kv_store ORDER BY key")
        return [row["key"] for row in cursor.fetchall()]

    # ------------------------------------------------------------------
    # Transaction support
    # ------------------------------------------------------------------

    def begin_transaction(self) -> None:
        """Start an explicit transaction."""
        conn = self._get_connection()
        conn.execute("BEGIN")
        logger.debug("Transaction started")

    def commit_transaction(self) -> None:
        """Commit the current transaction."""
        conn = self._get_connection()
        conn.commit()
        logger.debug("Transaction committed")

    def rollback_transaction(self) -> None:
        """Roll back the current transaction."""
        conn = self._get_connection()
        conn.rollback()
        logger.debug("Transaction rolled back")

    # ------------------------------------------------------------------
    # Query support
    # ------------------------------------------------------------------

    def query(self, prefix: str) -> Dict[str, Any]:
        """Return all key-value pairs whose key starts with *prefix*.

        Args:
            prefix: Key prefix to match.

        Returns:
            Dict of matching key → value pairs.
        """
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT key, value FROM kv_store WHERE key LIKE ? ORDER BY key",
            (prefix + "%",),
        )
        results: Dict[str, Any] = {}
        for row in cursor.fetchall():
            try:
                results[row["key"]] = json.loads(row["value"])
            except json.JSONDecodeError:
                results[row["key"]] = row["value"]
        return results

    def count(self) -> int:
        """Return the total number of stored keys."""
        conn = self._get_connection()
        cursor = conn.execute("SELECT COUNT(*) AS cnt FROM kv_store")
        row = cursor.fetchone()
        return row["cnt"] if row else 0

    # ------------------------------------------------------------------
    # Override batch save for atomicity
    # ------------------------------------------------------------------

    def save_many(self, items: dict) -> None:
        """Atomically persist multiple key-value pairs in a single transaction.

        Args:
            items: Dict of key → value pairs.
        """
        conn = self._get_connection()
        try:
            conn.execute("BEGIN")
            for key, value in items.items():
                payload = json.dumps(value, sort_keys=True, default=str)
                conn.execute(
                    """
                    INSERT INTO kv_store (key, value)
                    VALUES (?, ?)
                    ON CONFLICT(key) DO UPDATE SET value = excluded.value
                    """,
                    (key, payload),
                )
            conn.commit()
            logger.debug("Batch-saved %d key(s)", len(items))
        except (sqlite3.Error, TypeError, ValueError) as exc:
            conn.rollback()
            raise StorageError(f"Batch save failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __del__(self) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"<SQLiteStorage db_path={self._db_path!r} keys={self.count()}>"
