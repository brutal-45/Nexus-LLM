"""KnowledgeEntry dataclass — a single unit of knowledge."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class KnowledgeEntry:
    """Represents a single piece of knowledge.

    Attributes
    ----------
    id:
        Unique identifier (auto-generated if not supplied).
    title:
        Short descriptive title.
    content:
        Full text body of the entry.
    tags:
        List of categorisation tags.
    source:
        Origin of the entry (URL, document name, etc.).
    created_at:
        Timestamp when the entry was created.
    updated_at:
        Timestamp when the entry was last updated.
    """

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    title: str = ""
    content: str = ""
    tags: List[str] = field(default_factory=list)
    source: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-friendly dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "tags": list(self.tags),
            "source": self.source,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeEntry":
        """Construct a :class:`KnowledgeEntry` from a dictionary.

        ``created_at`` and ``updated_at`` are parsed from ISO-format strings
        when present; missing keys fall back to defaults.
        """
        kwargs: Dict[str, Any] = {k: v for k, v in data.items() if k not in ("created_at", "updated_at")}

        for ts_field in ("created_at", "updated_at"):
            if ts_field in data and isinstance(data[ts_field], str):
                kwargs[ts_field] = datetime.fromisoformat(data[ts_field])

        return cls(**kwargs)

    # ------------------------------------------------------------------
    # Relevance scoring
    # ------------------------------------------------------------------

    def relevance_score(self, query: str) -> float:
        """Compute a simple keyword-matching relevance score against *query*.

        The score is a ``float`` in ``[0.0, 1.0]`` based on the proportion of
        query terms that appear in the entry's content, title, or tags.

        Parameters
        ----------
        query:
            Search query string.
        """
        if not query or not query.strip():
            return 0.0

        query_terms = set(re.findall(r"\w+", query.lower()))
        if not query_terms:
            return 0.0

        # Build a bag-of-words from the searchable fields.
        searchable = " ".join([
            self.content,
            self.title,
            " ".join(self.tags),
            self.source,
        ]).lower()
        doc_terms = set(re.findall(r"\w+", searchable))

        if not doc_terms:
            return 0.0

        overlap = query_terms & doc_terms
        return len(overlap) / len(query_terms)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def touch(self) -> None:
        """Update ``updated_at`` to the current time."""
        self.updated_at = datetime.now(timezone.utc)

    def add_tag(self, tag: str) -> None:
        """Append *tag* if not already present."""
        if tag not in self.tags:
            self.tags.append(tag)
            self.touch()

    def remove_tag(self, tag: str) -> None:
        """Remove *tag* if present."""
        if tag in self.tags:
            self.tags.remove(tag)
            self.touch()

    def __repr__(self) -> str:
        return f"KnowledgeEntry(id={self.id!r}, title={self.title!r})"
