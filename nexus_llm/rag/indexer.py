"""Indexer for Nexus-LLM RAG.

Builds, saves, and loads inverted indices for fast document retrieval.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from nexus_llm.rag.document_store import Document, DocumentStore
from nexus_llm.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Inverted index
# ---------------------------------------------------------------------------

class InvertedIndex:
    """A simple inverted index mapping terms to document IDs and positions.

    Attributes:
        term_to_docs: Mapping of term -> set of document IDs.
        doc_lengths: Mapping of document ID -> length in tokens.
    """

    def __init__(self) -> None:
        self.term_to_docs: Dict[str, Set[str]] = defaultdict(set)
        self.doc_lengths: Dict[str, int] = {}
        self._doc_count: int = 0

    @property
    def doc_count(self) -> int:
        return self._doc_count

    def add_document(self, doc: Document) -> None:
        """Index a single document."""
        tokens = self._tokenise(doc.content)
        self.doc_lengths[doc.id] = len(tokens)
        for token in tokens:
            self.term_to_docs[token].add(doc.id)
        self._doc_count += 1

    def search(self, term: str) -> Set[str]:
        """Return the set of document IDs that contain *term*."""
        return self.term_to_docs.get(term.lower(), set())

    def search_multi(self, terms: List[str]) -> Set[str]:
        """Return document IDs containing ALL of the given terms."""
        if not terms:
            return set()
        result_sets = [self.search(t) for t in terms]
        return set.intersection(*result_sets) if result_sets else set()

    @staticmethod
    def _tokenise(text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "term_to_docs": {k: list(v) for k, v in self.term_to_docs.items()},
            "doc_lengths": self.doc_lengths,
            "doc_count": self._doc_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InvertedIndex":
        idx = cls()
        idx.term_to_docs = defaultdict(set, {
            k: set(v) for k, v in data.get("term_to_docs", {}).items()
        })
        idx.doc_lengths = data.get("doc_lengths", {})
        idx._doc_count = data.get("doc_count", 0)
        return idx


# ---------------------------------------------------------------------------
# Indexer
# ---------------------------------------------------------------------------

class Indexer:
    """Build and persist inverted indices over document stores.

    Args:
        persist_path: Optional path for saving/loading the index to disk.
    """

    def __init__(self, persist_path: Optional[str] = None) -> None:
        self.persist_path = Path(persist_path) if persist_path else None
        self._index: Optional[InvertedIndex] = None
        logger.info("Indexer initialised (persist=%s)", self.persist_path or "disabled")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index(self, documents: List[Document]) -> InvertedIndex:
        """Build an inverted index from a list of documents.

        Args:
            documents: The documents to index.

        Returns:
            The built ``InvertedIndex``.
        """
        inv = InvertedIndex()
        for doc in documents:
            inv.add_document(doc)
        self._index = inv
        self._maybe_save()
        logger.info(
            "Indexed %d documents (%d unique terms)",
            inv.doc_count, len(inv.term_to_docs),
        )
        return inv

    def build_index(self, store: DocumentStore) -> InvertedIndex:
        """Build an index from all documents in a ``DocumentStore``.

        Args:
            store: The document store to index.

        Returns:
            The built ``InvertedIndex``.
        """
        docs = [store.get(did) for did in store.list_ids()]
        docs = [d for d in docs if d is not None]
        return self.index(docs)

    @property
    def index_obj(self) -> Optional[InvertedIndex]:
        """Return the current index (or ``None`` if not built)."""
        return self._index

    def save_index(self, path: Optional[str] = None) -> None:
        """Save the current index to disk.

        Args:
            path: Override the ``persist_path`` set at init time.
        """
        target = Path(path) if path else self.persist_path
        if target is None:
            raise ValueError("No path specified for saving index")
        if self._index is None:
            raise RuntimeError("No index to save – call index() first")
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(self._index.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info("Index saved to %s", target)

    def load_index(self, path: Optional[str] = None) -> InvertedIndex:
        """Load an index from disk.

        Args:
            path: Override the ``persist_path`` set at init time.

        Returns:
            The loaded ``InvertedIndex``.
        """
        source = Path(path) if path else self.persist_path
        if source is None:
            raise ValueError("No path specified for loading index")
        with open(source, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._index = InvertedIndex.from_dict(data)
        logger.info("Index loaded from %s (%d docs)", source, self._index.doc_count)
        return self._index

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _maybe_save(self) -> None:
        if self.persist_path and self._index is not None:
            self.save_index()
