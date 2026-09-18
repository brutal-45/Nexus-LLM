"""Conversation store for Nexus-LLM.

Manages persistent storage and retrieval of chat conversations.
"""

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from nexus_llm.storage.backend import StorageBackend

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """A single chat message.

    Attributes:
        role: One of ``"user"``, ``"assistant"``, ``"system"``.
        content: The text content of the message.
        timestamp: ISO-8601 timestamp.
        metadata: Optional metadata (e.g. token count, model name).
    """

    role: str
    content: str
    timestamp: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


@dataclass
class Conversation:
    """A conversation consisting of a sequence of messages.

    Attributes:
        id: Unique conversation identifier.
        title: Short title / summary.
        messages: Ordered list of messages.
        created_at: ISO-8601 creation timestamp.
        updated_at: ISO-8601 last-update timestamp.
        metadata: Arbitrary metadata (model name, settings, etc.).
    """

    id: str = ""
    title: str = ""
    messages: List[Message] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        now = datetime.now(timezone.utc).isoformat()
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now

    def add_message(self, role: str, content: str, **metadata: Any) -> Message:
        """Append a message and update the timestamp.

        Returns:
            The newly created Message.
        """
        msg = Message(role=role, content=content, metadata=metadata)
        self.messages.append(msg)
        self.updated_at = datetime.now(timezone.utc).isoformat()
        return msg

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the conversation to a plain dict."""
        return {
            "id": self.id,
            "title": self.title,
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    "timestamp": m.timestamp,
                    "metadata": m.metadata,
                }
                for m in self.messages
            ],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Conversation":
        """Deserialise a conversation from a dict."""
        messages = [
            Message(
                role=m.get("role", ""),
                content=m.get("content", ""),
                timestamp=m.get("timestamp", ""),
                metadata=m.get("metadata", {}),
            )
            for m in data.get("messages", [])
        ]
        return cls(
            id=data.get("id", ""),
            title=data.get("title", ""),
            messages=messages,
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            metadata=data.get("metadata", {}),
        )


class ConversationStore:
    """Persistent store for :class:`Conversation` objects.

    Delegates actual persistence to a :class:`StorageBackend`
    implementation (e.g. :class:`FileStorage` or :class:`SQLiteStorage`).

    Conversations are stored under keys of the form
    ``conversations/<conv_id>``.

    Example::

        from nexus_llm.storage.file_storage import FileStorage

        backend = FileStorage(base_dir="/tmp/nexus_conv")
        store = ConversationStore(backend)
        conv = Conversation(title="Test chat")
        conv.add_message("user", "Hello!")
        cid = store.save_conversation(conv)
    """

    KEY_PREFIX = "conversations"

    def __init__(self, backend: StorageBackend) -> None:
        self._backend = backend

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def save_conversation(self, conv: Conversation) -> str:
        """Persist a conversation and return its ID.

        If ``conv.id`` is empty a new UUID will be assigned.

        Args:
            conv: The Conversation to save.

        Returns:
            The conversation ID.
        """
        if not conv.id:
            conv.id = str(uuid.uuid4())

        key = self._make_key(conv.id)
        self._backend.save(key, conv.to_dict())
        logger.info("Saved conversation %s", conv.id)
        return conv.id

    def load_conversation(self, conv_id: str) -> Conversation:
        """Load a conversation by ID.

        Args:
            conv_id: The conversation identifier.

        Returns:
            The deserialised Conversation.

        Raises:
            KeyError: If the conversation does not exist.
        """
        key = self._make_key(conv_id)
        data = self._backend.load(key)
        return Conversation.from_dict(data)

    def list_conversations(self) -> List[Dict[str, Any]]:
        """Return a summary list of all stored conversations.

        Each entry contains ``id``, ``title``, ``created_at``,
        ``updated_at``, and ``message_count``.
        """
        summaries: List[Dict[str, Any]] = []
        for key in self._backend.list_keys():
            if not key.startswith(self.KEY_PREFIX):
                continue
            try:
                data = self._backend.load(key)
                summaries.append(
                    {
                        "id": data.get("id", ""),
                        "title": data.get("title", ""),
                        "created_at": data.get("created_at", ""),
                        "updated_at": data.get("updated_at", ""),
                        "message_count": len(data.get("messages", [])),
                    }
                )
            except Exception:
                logger.warning("Failed to load conversation at key %r", key)
        return summaries

    def search_conversations(self, query: str) -> List[Dict[str, Any]]:
        """Search conversations by content.

        Performs a simple case-insensitive substring match against
        message content and conversation title.

        Args:
            query: Search string.

        Returns:
            List of conversation summaries matching the query.
        """
        query_lower = query.lower()
        results: List[Dict[str, Any]] = []

        for key in self._backend.list_keys():
            if not key.startswith(self.KEY_PREFIX):
                continue
            try:
                data = self._backend.load(key)
                # Check title
                if query_lower in data.get("title", "").lower():
                    results.append(self._summarise(data))
                    continue
                # Check message content
                for msg in data.get("messages", []):
                    if query_lower in msg.get("content", "").lower():
                        results.append(self._summarise(data))
                        break
            except Exception:
                logger.warning("Failed to search conversation at key %r", key)

        return results

    def delete_conversation(self, conv_id: str) -> bool:
        """Delete a conversation by ID.

        Args:
            conv_id: The conversation identifier.

        Returns:
            True if the conversation was found and deleted.
        """
        key = self._make_key(conv_id)
        deleted = self._backend.delete(key)
        if deleted:
            logger.info("Deleted conversation %s", conv_id)
        return deleted

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_key(self, conv_id: str) -> str:
        return f"{self.KEY_PREFIX}/{conv_id}"

    @staticmethod
    def _summarise(data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": data.get("id", ""),
            "title": data.get("title", ""),
            "created_at": data.get("created_at", ""),
            "updated_at": data.get("updated_at", ""),
            "message_count": len(data.get("messages", [])),
        }
