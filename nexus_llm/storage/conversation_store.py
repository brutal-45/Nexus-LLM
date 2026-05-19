"""Nexus-LLM Conversation Persistence Store.

Manages the storage, retrieval, search, and lifecycle of conversations
and their messages in the SQLite database. Provides efficient querying
with pagination, filtering, and full-text search capabilities.
"""

import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from nexus_llm.storage.database import DatabaseManager

logger = logging.getLogger(__name__)


@dataclass
class ConversationRecord:
    """A persisted conversation record.

    Attributes:
        conversation_id: Unique conversation identifier.
        title: Display title for the conversation.
        model_name: Model used for this conversation.
        system_prompt: System prompt if set.
        message_count: Number of messages in the conversation.
        total_tokens: Cumulative token count across all messages.
        is_archived: Whether the conversation is archived.
        is_pinned: Whether the conversation is pinned.
        created_at: Creation timestamp.
        updated_at: Last update timestamp.
        metadata: Additional metadata as JSON.
    """

    conversation_id: str = ""
    title: Optional[str] = None
    model_name: str = ""
    system_prompt: Optional[str] = None
    message_count: int = 0
    total_tokens: int = 0
    is_archived: bool = False
    is_pinned: bool = False
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.conversation_id:
            self.conversation_id = str(uuid.uuid4())


@dataclass
class MessageRecord:
    """A persisted message record.

    Attributes:
        id: Auto-incremented message ID.
        conversation_id: Parent conversation ID.
        role: Message role (system/user/assistant).
        content: Message text content.
        name: Optional sender name.
        message_type: Type of message (text/code/image).
        token_count: Number of tokens in the message.
        latency_ms: Generation latency in milliseconds.
        created_at: Creation timestamp.
        metadata: Additional metadata as JSON.
    """

    id: Optional[int] = None
    conversation_id: str = ""
    role: str = "user"
    content: str = ""
    name: Optional[str] = None
    message_type: str = "text"
    token_count: int = 0
    latency_ms: int = 0
    created_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConversationStore:
    """Persistent storage for conversations and messages.

    Provides CRUD operations, search, pagination, and tag management
    for conversations backed by SQLite.

    Attributes:
        db: Database manager instance.
    """

    def __init__(self, db: DatabaseManager) -> None:
        """Initialize the conversation store.

        Args:
            db: DatabaseManager instance for database access.
        """
        self.db = db

    def create_conversation(
        self,
        model_name: str,
        title: Optional[str] = None,
        system_prompt: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ConversationRecord:
        """Create a new conversation.

        Args:
            model_name: Name of the model to use.
            title: Optional conversation title.
            system_prompt: Optional system prompt.
            metadata: Optional additional metadata.

        Returns:
            The newly created ConversationRecord.
        """
        record = ConversationRecord(
            model_name=model_name,
            title=title,
            system_prompt=system_prompt,
            metadata=metadata or {},
        )

        now = datetime.now().isoformat()
        self.db.execute(
            """INSERT INTO conversations
               (conversation_id, title, model_name, system_prompt, message_count,
                total_tokens, is_archived, is_pinned, created_at, updated_at, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record.conversation_id,
                record.title,
                record.model_name,
                record.system_prompt,
                0,
                0,
                0,
                0,
                now,
                now,
                json.dumps(record.metadata),
            ),
        )

        record.created_at = now
        record.updated_at = now
        logger.debug(f"Created conversation {record.conversation_id}")
        return record

    def get_conversation(self, conversation_id: str) -> Optional[ConversationRecord]:
        """Retrieve a conversation by ID.

        Args:
            conversation_id: The unique conversation identifier.

        Returns:
            ConversationRecord if found, None otherwise.
        """
        row = self.db.fetch_one(
            "SELECT * FROM conversations WHERE conversation_id = ?",
            (conversation_id,),
        )
        if row is None:
            return None
        return self._row_to_conversation(row)

    def list_conversations(
        self,
        limit: int = 50,
        offset: int = 0,
        model_name: Optional[str] = None,
        include_archived: bool = False,
        pinned_only: bool = False,
    ) -> Tuple[List[ConversationRecord], int]:
        """List conversations with optional filtering and pagination.

        Args:
            limit: Maximum number of conversations to return.
            offset: Number of conversations to skip.
            model_name: Filter by model name.
            include_archived: Include archived conversations.
            pinned_only: Only return pinned conversations.

        Returns:
            Tuple of (list of ConversationRecord, total count).
        """
        conditions = []
        params: list = []

        if not include_archived:
            conditions.append("is_archived = 0")
        if pinned_only:
            conditions.append("is_pinned = 1")
        if model_name:
            conditions.append("model_name = ?")
            params.append(model_name)

        where = " AND ".join(conditions) if conditions else "1=1"

        count = self.db.fetch_value(
            f"SELECT COUNT(*) FROM conversations WHERE {where}",
            tuple(params),
        ) or 0

        rows = self.db.fetch_all(
            f"SELECT * FROM conversations WHERE {where} ORDER BY updated_at DESC LIMIT ? OFFSET ?",
            tuple(params + [limit, offset]),
        )

        conversations = [self._row_to_conversation(row) for row in rows]
        return conversations, count

    def update_conversation(
        self,
        conversation_id: str,
        title: Optional[str] = None,
        system_prompt: Optional[str] = None,
        is_archived: Optional[bool] = None,
        is_pinned: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update a conversation's attributes.

        Args:
            conversation_id: The conversation to update.
            title: New title.
            system_prompt: New system prompt.
            is_archived: Archive status.
            is_pinned: Pin status.
            metadata: New metadata (merged with existing).

        Returns:
            True if the conversation was updated.
        """
        updates: List[str] = []
        params: List[Any] = []

        if title is not None:
            updates.append("title = ?")
            params.append(title)
        if system_prompt is not None:
            updates.append("system_prompt = ?")
            params.append(system_prompt)
        if is_archived is not None:
            updates.append("is_archived = ?")
            params.append(1 if is_archived else 0)
        if is_pinned is not None:
            updates.append("is_pinned = ?")
            params.append(1 if is_pinned else 0)
        if metadata is not None:
            existing = self.get_conversation(conversation_id)
            if existing:
                merged = {**existing.metadata, **metadata}
                updates.append("metadata = ?")
                params.append(json.dumps(merged))

        if not updates:
            return False

        updates.append("updated_at = ?")
        params.append(datetime.now().isoformat())
        params.append(conversation_id)

        self.db.execute(
            f"UPDATE conversations SET {', '.join(updates)} WHERE conversation_id = ?",
            tuple(params),
        )
        return True

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation and all its messages.

        Args:
            conversation_id: The conversation to delete.

        Returns:
            True if the conversation was deleted.
        """
        cursor = self.db.execute(
            "DELETE FROM conversations WHERE conversation_id = ?",
            (conversation_id,),
        )
        return cursor.rowcount > 0

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        name: Optional[str] = None,
        message_type: str = "text",
        token_count: int = 0,
        latency_ms: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MessageRecord:
        """Add a message to a conversation.

        Also updates the parent conversation's message_count, total_tokens,
        and updated_at timestamp.

        Args:
            conversation_id: Target conversation.
            role: Message role.
            content: Message content.
            name: Optional sender name.
            message_type: Message type.
            token_count: Token count for the message.
            latency_ms: Generation latency.
            metadata: Additional metadata.

        Returns:
            The newly created MessageRecord.
        """
        now = datetime.now().isoformat()
        record = MessageRecord(
            conversation_id=conversation_id,
            role=role,
            content=content,
            name=name,
            message_type=message_type,
            token_count=token_count,
            latency_ms=latency_ms,
            created_at=now,
            metadata=metadata or {},
        )

        cursor = self.db.execute(
            """INSERT INTO messages
               (conversation_id, role, content, name, message_type,
                token_count, latency_ms, created_at, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                conversation_id,
                role,
                content,
                name,
                message_type,
                token_count,
                latency_ms,
                now,
                json.dumps(record.metadata),
            ),
        )
        record.id = cursor.lastrowid

        # Update conversation counters
        self.db.execute(
            """UPDATE conversations
               SET message_count = message_count + 1,
                   total_tokens = total_tokens + ?,
                   updated_at = ?
               WHERE conversation_id = ?""",
            (token_count, now, conversation_id),
        )

        return record

    def get_messages(
        self,
        conversation_id: str,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[MessageRecord]:
        """Get messages for a conversation.

        Args:
            conversation_id: Target conversation.
            limit: Maximum messages to return.
            offset: Number of messages to skip.

        Returns:
            List of MessageRecord objects in chronological order.
        """
        sql = "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at ASC"
        params: list = [conversation_id]

        if limit is not None:
            sql += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])

        rows = self.db.fetch_all(sql, tuple(params))
        return [self._row_to_message(row) for row in rows]

    def search_conversations(self, query: str, limit: int = 20) -> List[ConversationRecord]:
        """Search conversations by title or message content.

        Args:
            query: Search query string.
            limit: Maximum results to return.

        Returns:
            List of matching ConversationRecord objects.
        """
        search_pattern = f"%{query}%"

        rows = self.db.fetch_all(
            """SELECT DISTINCT c.* FROM conversations c
               LEFT JOIN messages m ON c.conversation_id = m.conversation_id
               WHERE c.title LIKE ? OR m.content LIKE ?
               ORDER BY c.updated_at DESC LIMIT ?""",
            (search_pattern, search_pattern, limit),
        )

        return [self._row_to_conversation(row) for row in rows]

    def get_conversation_stats(self, conversation_id: str) -> Dict[str, Any]:
        """Get statistics for a specific conversation.

        Args:
            conversation_id: Target conversation.

        Returns:
            Dictionary with conversation statistics.
        """
        msg_stats = self.db.fetch_one(
            """SELECT
                COUNT(*) as total_messages,
                SUM(token_count) as total_tokens,
                AVG(latency_ms) as avg_latency_ms,
                SUM(CASE WHEN role = 'user' THEN 1 ELSE 0 END) as user_messages,
                SUM(CASE WHEN role = 'assistant' THEN 1 ELSE 0 END) as assistant_messages
               FROM messages WHERE conversation_id = ?""",
            (conversation_id,),
        )

        if msg_stats is None:
            return {}

        return {
            "total_messages": msg_stats.get("total_messages", 0),
            "total_tokens": msg_stats.get("total_tokens", 0) or 0,
            "avg_latency_ms": round(msg_stats.get("avg_latency_ms", 0) or 0, 2),
            "user_messages": msg_stats.get("user_messages", 0),
            "assistant_messages": msg_stats.get("assistant_messages", 0),
        }

    def _row_to_conversation(self, row: Dict[str, Any]) -> ConversationRecord:
        """Convert a database row to a ConversationRecord."""
        metadata = row.get("metadata", "{}")
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {}

        return ConversationRecord(
            conversation_id=row["conversation_id"],
            title=row.get("title"),
            model_name=row.get("model_name", ""),
            system_prompt=row.get("system_prompt"),
            message_count=row.get("message_count", 0),
            total_tokens=row.get("total_tokens", 0),
            is_archived=bool(row.get("is_archived", 0)),
            is_pinned=bool(row.get("is_pinned", 0)),
            created_at=row.get("created_at"),
            updated_at=row.get("updated_at"),
            metadata=metadata,
        )

    def _row_to_message(self, row: Dict[str, Any]) -> MessageRecord:
        """Convert a database row to a MessageRecord."""
        metadata = row.get("metadata", "{}")
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {}

        return MessageRecord(
            id=row.get("id"),
            conversation_id=row.get("conversation_id", ""),
            role=row.get("role", "user"),
            content=row.get("content", ""),
            name=row.get("name"),
            message_type=row.get("message_type", "text"),
            token_count=row.get("token_count", 0),
            latency_ms=row.get("latency_ms", 0),
            created_at=row.get("created_at"),
            metadata=metadata,
        )
