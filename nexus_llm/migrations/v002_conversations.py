"""Nexus-LLM Conversations Table Migration (v002).

Creates database tables for conversation persistence, message history,
conversation tags, and conversation sharing functionality.
"""

from typing import Any, Dict, List, Optional

import sqlite3
import logging

logger = logging.getLogger(__name__)

# Migration metadata
MIGRATION_ID = "v002_conversations"
MIGRATION_VERSION = 2
MIGRATION_DESCRIPTION = "Conversations and message history tables"


def up(connection: sqlite3.Connection) -> None:
    """Apply the conversations schema migration.

    Creates the following tables:
        - conversations: Conversation session metadata
        - messages: Individual messages within conversations
        - conversation_tags: Tags for organizing conversations
        - conversation_tag_map: Many-to-many mapping between conversations and tags
        - conversation_shares: Shared conversation records

    Args:
        connection: Active SQLite database connection.
    """
    cursor = connection.cursor()

    # Conversations table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL UNIQUE,
            title TEXT,
            model_name TEXT NOT NULL,
            system_prompt TEXT,
            message_count INTEGER DEFAULT 0,
            total_tokens INTEGER DEFAULT 0,
            is_archived INTEGER DEFAULT 0,
            is_pinned INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT
        )
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_conversations_id ON conversations(conversation_id)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_conversations_model ON conversations(model_name)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_conversations_updated ON conversations(updated_at)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_conversations_archived ON conversations(is_archived)
    """)

    # Messages table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            name TEXT,
            message_type TEXT DEFAULT 'text',
            token_count INTEGER DEFAULT 0,
            latency_ms INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT,
            FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id) ON DELETE CASCADE
        )
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_messages_role ON messages(role)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_messages_created ON messages(created_at)
    """)

    # Conversation tags table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversation_tags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            color TEXT DEFAULT '#6366f1',
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_tags_name ON conversation_tags(name)
    """)

    # Conversation-tag mapping table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversation_tag_map (
            conversation_id TEXT NOT NULL,
            tag_id INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (conversation_id, tag_id),
            FOREIGN KEY (tag_id) REFERENCES conversation_tags(id) ON DELETE CASCADE
        )
    """)

    # Conversation sharing table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversation_shares (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            share_token TEXT NOT NULL UNIQUE,
            is_public INTEGER DEFAULT 0,
            expires_at TIMESTAMP,
            access_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id) ON DELETE CASCADE
        )
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_shares_token ON conversation_shares(share_token)
    """)

    # Insert default tags
    default_tags = [
        {"name": "important", "color": "#ef4444", "description": "Important conversations"},
        {"name": "work", "color": "#3b82f6", "description": "Work-related conversations"},
        {"name": "personal", "color": "#10b981", "description": "Personal conversations"},
        {"name": "research", "color": "#8b5cf6", "description": "Research discussions"},
        {"name": "coding", "color": "#f59e0b", "description": "Code-related conversations"},
    ]

    for tag in default_tags:
        cursor.execute(
            "INSERT OR IGNORE INTO conversation_tags (name, color, description) VALUES (?, ?, ?)",
            (tag["name"], tag["color"], tag["description"]),
        )

    connection.commit()
    logger.info(f"Migration {MIGRATION_ID} applied successfully")


def down(connection: sqlite3.Connection) -> None:
    """Rollback the conversations schema migration.

    Args:
        connection: Active SQLite database connection.
    """
    cursor = connection.cursor()

    tables = [
        "conversation_shares",
        "conversation_tag_map",
        "conversation_tags",
        "messages",
        "conversations",
    ]
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
        """CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL UNIQUE,
            title TEXT,
            model_name TEXT NOT NULL,
            system_prompt TEXT,
            message_count INTEGER DEFAULT 0,
            total_tokens INTEGER DEFAULT 0,
            is_archived INTEGER DEFAULT 0,
            is_pinned INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT
        )""",
        """CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            name TEXT,
            message_type TEXT DEFAULT 'text',
            token_count INTEGER DEFAULT 0,
            latency_ms INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT,
            FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id) ON DELETE CASCADE
        )""",
        """CREATE TABLE IF NOT EXISTS conversation_tags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            color TEXT DEFAULT '#6366f1',
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""",
        """CREATE TABLE IF NOT EXISTS conversation_tag_map (
            conversation_id TEXT NOT NULL,
            tag_id INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (conversation_id, tag_id),
            FOREIGN KEY (tag_id) REFERENCES conversation_tags(id) ON DELETE CASCADE
        )""",
        """CREATE TABLE IF NOT EXISTS conversation_shares (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            share_token TEXT NOT NULL UNIQUE,
            is_public INTEGER DEFAULT 0,
            expires_at TIMESTAMP,
            access_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id) ON DELETE CASCADE
        )""",
    ]
