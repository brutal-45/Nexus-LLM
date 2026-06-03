"""ConversationMemory — stores and retrieves chat-style message history."""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """A single chat message."""

    role: str
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class ConversationMemory:
    """Buffer-style conversation memory with auto-trimming and context retrieval.

    Parameters
    ----------
    max_messages:
        Maximum number of messages to retain.  When exceeded, the oldest
        messages are dropped automatically (FIFO).
    id:
        Optional explicit identifier; a UUID is generated if not supplied.
    """

    def __init__(
        self,
        max_messages: int = 100,
        id: Optional[str] = None,
    ) -> None:
        if max_messages <= 0:
            raise ValueError("max_messages must be a positive integer")
        self.id: str = id or uuid.uuid4().hex
        self.max_messages = max_messages
        self._messages: List[Message] = []

    # ------------------------------------------------------------------
    # Message management
    # ------------------------------------------------------------------

    def add_message(self, role: str, content: str, **metadata: Any) -> Message:
        """Append a new message and auto-trim if necessary.

        Parameters
        ----------
        role:
            Speaker role (e.g. ``"user"``, ``"assistant"``, ``"system"``).
        content:
            Message text.
        **metadata:
            Arbitrary key-value metadata attached to the message.

        Returns
        -------
        The newly created :class:`Message`.
        """
        msg = Message(role=role, content=content, metadata=metadata)
        self._messages.append(msg)
        self._trim()
        logger.debug(
            "ConversationMemory %s: added %s message (%d total)",
            self.id, role, len(self._messages),
        )
        return msg

    def get_messages(self, n: Optional[int] = None) -> List[Message]:
        """Return the most recent *n* messages (or all if *n* is ``None``)."""
        if n is None:
            return list(self._messages)
        return list(self._messages[-n:])

    def get_all(self) -> List[Message]:
        """Return a copy of all stored messages."""
        return list(self._messages)

    def clear(self) -> None:
        """Remove all messages."""
        self._messages.clear()
        logger.debug("ConversationMemory %s: cleared", self.id)

    # ------------------------------------------------------------------
    # Context retrieval
    # ------------------------------------------------------------------

    def get_context(self, query: str, max_tokens: int = 4096) -> List[Message]:
        """Return messages relevant to *query* within a token budget.

        Relevance is determined by simple keyword overlap.  Messages are
        scored, sorted by relevance (descending), and then returned in their
        original chronological order up to the estimated token budget.

        Parameters
        ----------
        query:
            Search text.
        max_tokens:
            Approximate token budget.  Tokens are estimated as
            ``len(content) / 4``.
        """
        query_terms = set(re.findall(r"\w+", query.lower()))
        if not query_terms:
            return self._fit_budget(self._messages, max_tokens)

        scored: List[tuple[float, Message]] = []
        for msg in self._messages:
            msg_terms = set(re.findall(r"\w+", msg.content.lower()))
            overlap = len(query_terms & msg_terms)
            if overlap > 0:
                scored.append((overlap, msg))

        scored.sort(key=lambda t: t[0], reverse=True)
        selected = [msg for _, msg in scored]
        return self._fit_budget(selected, max_tokens)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _trim(self) -> None:
        """Remove the oldest messages when capacity is exceeded."""
        while len(self._messages) > self.max_messages:
            self._messages.pop(0)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimate (≈ 4 characters per token)."""
        return max(1, len(text) // 4)

    def _fit_budget(self, messages: List[Message], max_tokens: int) -> List[Message]:
        """Return a chronological subset that fits within *max_tokens*."""
        result: List[Message] = []
        used = 0
        for msg in sorted(messages, key=lambda m: m.timestamp):
            cost = self._estimate_tokens(msg.content)
            if used + cost > max_tokens:
                break
            result.append(msg)
            used += cost
        return result

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._messages)

    def __repr__(self) -> str:
        return (
            f"ConversationMemory(id={self.id!r}, "
            f"messages={len(self._messages)}, max={self.max_messages})"
        )
