"""Nexus-LLM Training Jobs Migration (v002).

Creates database tables for tracking training jobs, training metrics,
checkpoints, and dataset registry for the Nexus-LLM training subsystem.
"""

from typing import Any, Dict, List, Optional

import sqlite3
import logging

logger = logging.getLogger(__name__)

# Migration metadata
MIGRATION_ID = "v002_training_jobs"
MIGRATION_VERSION = 2
MIGRATION_DESCRIPTION = "Training jobs, metrics, checkpoints, and dataset registry"


def up(connection: sqlite3.Connection) -> None:
    """Apply the training jobs schema migration.

    Creates the following tables:
        - training_jobs: Training job metadata and status tracking
        - training_metrics: Time-series metrics recorded during training
        - checkpoints: Saved model checkpoints
        - datasets: Registered training and evaluation datasets

    Args:
        connection: Active SQLite database connection.
    """
    cursor = connection.cursor()

    # Training jobs table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS training_jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT NOT NULL UNIQUE,
            job_name TEXT,
            model_name TEXT NOT NULL,
            dataset_name TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            progress REAL DEFAULT 0.0,
            config_json TEXT NOT NULL,
            base_model TEXT,
            output_dir TEXT,
            total_epochs INTEGER,
            current_epoch REAL DEFAULT 0.0,
            current_step INTEGER DEFAULT 0,
            total_steps INTEGER,
            best_metric_name TEXT,
            best_metric_value REAL,
            error_message TEXT,
            started_at TIMESTAMP,
            completed_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT
        )
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_training_jobs_job_id ON training_jobs(job_id)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_training_jobs_status ON training_jobs(status)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_training_jobs_model ON training_jobs(model_name)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_training_jobs_created ON training_jobs(created_at)
    """)

    # Training metrics table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS training_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT NOT NULL,
            metric_name TEXT NOT NULL,
            metric_value REAL NOT NULL,
            step INTEGER NOT NULL,
            epoch REAL,
            phase TEXT NOT NULL DEFAULT 'train',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (job_id) REFERENCES training_jobs(job_id) ON DELETE CASCADE
        )
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_metrics_job_id ON training_metrics(job_id)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_metrics_name ON training_metrics(metric_name)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_metrics_step ON training_metrics(job_id, step)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_metrics_phase ON training_metrics(phase)
    """)

    # Checkpoints table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS checkpoints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT NOT NULL,
            checkpoint_path TEXT NOT NULL,
            step INTEGER NOT NULL,
            epoch REAL NOT NULL,
            is_best INTEGER DEFAULT 0,
            metric_name TEXT,
            metric_value REAL,
            file_size_mb REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT,
            FOREIGN KEY (job_id) REFERENCES training_jobs(job_id) ON DELETE CASCADE
        )
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_checkpoints_job_id ON checkpoints(job_id)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_checkpoints_best ON checkpoints(job_id, is_best)
    """)

    # Datasets registry table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS datasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            dataset_type TEXT NOT NULL DEFAULT 'training',
            format TEXT NOT NULL DEFAULT 'jsonl',
            path TEXT NOT NULL,
            description TEXT,
            num_examples INTEGER,
            num_tokens INTEGER,
            size_mb REAL,
            is_preprocessed INTEGER DEFAULT 0,
            split_info TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT
        )
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_datasets_name ON datasets(name)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_datasets_type ON datasets(dataset_type)
    """)

    connection.commit()
    logger.info(f"Migration {MIGRATION_ID} applied successfully")


def down(connection: sqlite3.Connection) -> None:
    """Rollback the training jobs schema migration.

    Args:
        connection: Active SQLite database connection.
    """
    cursor = connection.cursor()

    tables = ["checkpoints", "training_metrics", "datasets", "training_jobs"]
    for table in tables:
        cursor.execute(f"DROP TABLE IF EXISTS {table}")

    connection.commit()
    logger.info(f"Migration {MIGRATION_ID} rolled back successfully")


def get_statements() -> List[str]:
    """Return the SQL statements for this migration.

    Returns:
        List of SQL CREATE statements.
    """
    return [
        """CREATE TABLE IF NOT EXISTS training_jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT NOT NULL UNIQUE,
            job_name TEXT,
            model_name TEXT NOT NULL,
            dataset_name TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            progress REAL DEFAULT 0.0,
            config_json TEXT NOT NULL,
            base_model TEXT,
            output_dir TEXT,
            total_epochs INTEGER,
            current_epoch REAL DEFAULT 0.0,
            current_step INTEGER DEFAULT 0,
            total_steps INTEGER,
            best_metric_name TEXT,
            best_metric_value REAL,
            error_message TEXT,
            started_at TIMESTAMP,
            completed_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT
        )""",
        """CREATE TABLE IF NOT EXISTS training_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT NOT NULL,
            metric_name TEXT NOT NULL,
            metric_value REAL NOT NULL,
            step INTEGER NOT NULL,
            epoch REAL,
            phase TEXT NOT NULL DEFAULT 'train',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (job_id) REFERENCES training_jobs(job_id) ON DELETE CASCADE
        )""",
        """CREATE TABLE IF NOT EXISTS checkpoints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT NOT NULL,
            checkpoint_path TEXT NOT NULL,
            step INTEGER NOT NULL,
            epoch REAL NOT NULL,
            is_best INTEGER DEFAULT 0,
            metric_name TEXT,
            metric_value REAL,
            file_size_mb REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT,
            FOREIGN KEY (job_id) REFERENCES training_jobs(job_id) ON DELETE CASCADE
        )""",
        """CREATE TABLE IF NOT EXISTS datasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            dataset_type TEXT NOT NULL DEFAULT 'training',
            format TEXT NOT NULL DEFAULT 'jsonl',
            path TEXT NOT NULL,
            description TEXT,
            num_examples INTEGER,
            num_tokens INTEGER,
            size_mb REAL,
            is_preprocessed INTEGER DEFAULT 0,
            split_info TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT
        )""",
    ]
