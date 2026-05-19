"""Nexus-LLM Initial Schema Migration (v001).

Creates the core database tables for the Nexus-LLM framework including
the migration history tracker, model registry, configuration store,
and system settings.
"""

from typing import Any, Dict, List, Optional

import sqlite3
import logging

logger = logging.getLogger(__name__)

# Migration metadata
MIGRATION_ID = "v001_initial"
MIGRATION_VERSION = 1
MIGRATION_DESCRIPTION = "Initial schema with core tables"


def up(connection: sqlite3.Connection) -> None:
    """Apply the initial schema migration.

    Creates the following tables:
        - migration_history: Tracks applied migrations
        - models: Registered model metadata
        - configurations: Stored configuration profiles
        - system_settings: Global system settings

    Args:
        connection: Active SQLite database connection.
    """
    cursor = connection.cursor()

    # Migration history table
    cursor.execute("""
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

    # Model registry table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            full_name TEXT,
            model_type TEXT NOT NULL DEFAULT 'causal_lm',
            size TEXT,
            parameter_count INTEGER,
            context_length INTEGER DEFAULT 2048,
            device TEXT DEFAULT 'auto',
            precision TEXT DEFAULT 'fp16',
            description TEXT,
            license TEXT,
            local_path TEXT,
            is_loaded INTEGER DEFAULT 0,
            tags TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT
        )
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_models_name ON models(name)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_models_type ON models(model_type)
    """)

    # Configuration profiles table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS configurations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            profile_type TEXT NOT NULL DEFAULT 'custom',
            config_data TEXT NOT NULL,
            description TEXT,
            is_active INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_configurations_name ON configurations(name)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_configurations_type ON configurations(profile_type)
    """)

    # System settings table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS system_settings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT NOT NULL UNIQUE,
            value TEXT NOT NULL,
            value_type TEXT NOT NULL DEFAULT 'string',
            description TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_system_settings_key ON system_settings(key)
    """)

    # Insert default system settings
    default_settings: List[Dict[str, Any]] = [
        {"key": "default_model", "value": "", "value_type": "string", "description": "Default model name"},
        {"key": "default_device", "value": "auto", "value_type": "string", "description": "Default compute device"},
        {"key": "default_precision", "value": "fp16", "value_type": "string", "description": "Default model precision"},
        {"key": "max_concurrent_requests", "value": "10", "value_type": "integer", "description": "Max concurrent inference requests"},
        {"key": "cache_enabled", "value": "true", "value_type": "boolean", "description": "Enable response caching"},
        {"key": "log_level", "value": "info", "value_type": "string", "description": "Logging level"},
        {"key": "data_directory", "value": "./data", "value_type": "string", "description": "Data storage directory"},
    ]

    for setting in default_settings:
        cursor.execute(
            """
            INSERT OR IGNORE INTO system_settings (key, value, value_type, description)
            VALUES (?, ?, ?, ?)
            """,
            (setting["key"], setting["value"], setting["value_type"], setting["description"]),
        )

    connection.commit()
    logger.info(f"Migration {MIGRATION_ID} applied successfully")


def down(connection: sqlite3.Connection) -> None:
    """Rollback the initial schema migration.

    Drops all tables created by this migration.

    Args:
        connection: Active SQLite database connection.
    """
    cursor = connection.cursor()

    tables = ["system_settings", "configurations", "models", "migration_history"]
    for table in tables:
        cursor.execute(f"DROP TABLE IF EXISTS {table}")

    connection.commit()
    logger.info(f"Migration {MIGRATION_ID} rolled back successfully")


def get_statements() -> List[str]:
    """Return the SQL statements for this migration.

    Returns:
        List of SQL CREATE/INSERT statements.
    """
    return [
        """CREATE TABLE IF NOT EXISTS migration_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            migration_id TEXT NOT NULL UNIQUE,
            version INTEGER NOT NULL,
            description TEXT,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            checksum TEXT,
            execution_time_ms INTEGER
        )""",
        """CREATE TABLE IF NOT EXISTS models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            full_name TEXT,
            model_type TEXT NOT NULL DEFAULT 'causal_lm',
            size TEXT,
            parameter_count INTEGER,
            context_length INTEGER DEFAULT 2048,
            device TEXT DEFAULT 'auto',
            precision TEXT DEFAULT 'fp16',
            description TEXT,
            license TEXT,
            local_path TEXT,
            is_loaded INTEGER DEFAULT 0,
            tags TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT
        )""",
        """CREATE TABLE IF NOT EXISTS configurations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            profile_type TEXT NOT NULL DEFAULT 'custom',
            config_data TEXT NOT NULL,
            description TEXT,
            is_active INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""",
        """CREATE TABLE IF NOT EXISTS system_settings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT NOT NULL UNIQUE,
            value TEXT NOT NULL,
            value_type TEXT NOT NULL DEFAULT 'string',
            description TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""",
    ]
