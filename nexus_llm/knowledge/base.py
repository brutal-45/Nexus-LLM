"""KnowledgeBase — CRUD + search over KnowledgeEntry instances."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from nexus_llm.knowledge.entry import KnowledgeEntry

logger = logging.getLogger(__name__)


class EntryNotFoundError(KeyError):
    """Raised when a requested entry does not exist."""


class KnowledgeBase:
    """In-memory knowledge base with keyword search.

    Example
    -------
    >>> kb = KnowledgeBase()
    >>> entry = kb.add_entry(KnowledgeEntry(title="Python", content="Python is a language", tags=["prog"]))
    >>> kb.search("Python", top_k=5)
    [KnowledgeEntry(id='...', title='Python')]
    """

    def __init__(self) -> None:
        self._entries: Dict[str, KnowledgeEntry] = {}

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add_entry(self, entry: KnowledgeEntry) -> KnowledgeEntry:
        """Add an entry to the knowledge base.

        If the entry's ``id`` already exists a :class:`ValueError` is raised.

        Returns
        -------
        The added entry (for chaining convenience).
        """
        if entry.id in self._entries:
            raise ValueError(f"Entry with id {entry.id!r} already exists")
        self._entries[entry.id] = entry
        logger.debug("KnowledgeBase: added entry %r", entry.id)
        return entry

    def get_entry(self, id: str) -> KnowledgeEntry:
        """Return the entry with the given *id*.

        Raises
        ------
        EntryNotFoundError
            If *id* is not found.
        """
        if id not in self._entries:
            raise EntryNotFoundError(id)
        return self._entries[id]

    def update_entry(self, id: str, entry: KnowledgeEntry) -> KnowledgeEntry:
        """Replace the entry at *id* with a new :class:`KnowledgeEntry`.

        The ``updated_at`` timestamp is refreshed automatically.

        Raises
        ------
        EntryNotFoundError
            If *id* is not found.
        """
        if id not in self._entries:
            raise EntryNotFoundError(id)
        entry.touch()
        self._entries[id] = entry
        logger.debug("KnowledgeBase: updated entry %r", id)
        return entry

    def remove_entry(self, id: str) -> None:
        """Remove the entry with the given *id*.

        Raises
        ------
        EntryNotFoundError
            If *id* is not found.
        """
        if id not in self._entries:
            raise EntryNotFoundError(id)
        del self._entries[id]
        logger.debug("KnowledgeBase: removed entry %r", id)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 10) -> List[KnowledgeEntry]:
        """Return entries ranked by relevance to *query*.

        Parameters
        ----------
        query:
            Search text.
        top_k:
            Maximum number of results to return.

        Returns
        -------
        A list of :class:`KnowledgeEntry` sorted by descending relevance.
        """
        if top_k <= 0:
            return []

        scored = [
            (entry.relevance_score(query), entry)
            for entry in self._entries.values()
        ]
        # Filter out zero-relevance entries
        scored = [(score, entry) for score, entry in scored if score > 0]
        scored.sort(key=lambda t: t[0], reverse=True)
        results = [entry for _, entry in scored[:top_k]]
        logger.debug(
            "KnowledgeBase: search %r returned %d result(s)",
            query, len(results),
        )
        return results

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------

    def list_entries(self) -> List[KnowledgeEntry]:
        """Return all entries ordered by creation time."""
        return sorted(self._entries.values(), key=lambda e: e.created_at)

    def count(self) -> int:
        """Return the total number of entries."""
        return len(self._entries)

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Remove all entries."""
        self._entries.clear()
