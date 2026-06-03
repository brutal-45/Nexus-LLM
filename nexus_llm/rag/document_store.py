"""Document store for Nexus-LLM RAG.

In-memory document storage with optional JSON-file persistence.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from nexus_llm.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Document representation
# ---------------------------------------------------------------------------

@dataclass
class Document:
    """A single document in the store.

    Attributes:
        id: Unique identifier (auto-generated if not provided).
        content: The document text.
        metadata: Arbitrary key-value metadata.
    """

    content: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        return cls(**data)


# ---------------------------------------------------------------------------
# Document store
# ---------------------------------------------------------------------------

class DocumentStore:
    """In-memory document store with optional file persistence.

    Args:
        persist_path: If provided, documents are saved to / loaded from
            this JSON file on every write and on initialisation.
    """

    def __init__(self, persist_path: Optional[str] = None) -> None:
        self._docs: Dict[str, Document] = {}
        self.persist_path = Path(persist_path) if persist_path else None
        logger.info(
            "DocumentStore initialised (persist=%s)",
            self.persist_path or "disabled",
        )

        if self.persist_path and self.persist_path.exists():
            self._load_from_disk()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, doc: Document) -> str:
        """Add a document and return its ID.

        If the document already has an ID it will be used; otherwise a
        new one is generated.
        """
        if not doc.id:
            doc.id = uuid.uuid4().hex[:12]
        self._docs[doc.id] = doc
        self._maybe_persist()
        logger.debug("Added document id=%s", doc.id)
        return doc.id

    def get(self, doc_id: str) -> Optional[Document]:
        """Retrieve a document by ID, or ``None`` if not found."""
        return self._docs.get(doc_id)

    def search(self, query: str, top_k: int = 5) -> List[Document]:
        """Simple keyword search over document contents.

        Documents are ranked by the number of query-word occurrences
        (case-insensitive) and the top *k* are returned.
        """
        query_terms = set(query.lower().split())
        scored: List[tuple] = []

        for doc in self._docs.values():
            content_lower = doc.content.lower()
            hits = sum(1 for term in query_terms if term in content_lower)
            if hits > 0:
                scored.append((hits, doc))

        scored.sort(key=lambda t: t[0], reverse=True)
        return [doc for _, doc in scored[:top_k]]

    def delete(self, doc_id: str) -> bool:
        """Delete a document by ID.  Returns ``True`` if found and deleted."""
        if doc_id in self._docs:
            del self._docs[doc_id]
            self._maybe_persist()
            logger.debug("Deleted document id=%s", doc_id)
            return True
        return False

    def count(self) -> int:
        """Return the number of documents in the store."""
        return len(self._docs)

    def list_ids(self) -> List[str]:
        """Return all document IDs."""
        return list(self._docs.keys())

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _maybe_persist(self) -> None:
        if self.persist_path:
            self._save_to_disk()

    def _save_to_disk(self) -> None:
        data = [doc.to_dict() for doc in self._docs.values()]
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)  # type: ignore[union-attr]
        with open(self.persist_path, "w", encoding="utf-8") as f:  # type: ignore[arg-type]
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.debug("Persisted %d documents to %s", len(data), self.persist_path)

    def _load_from_disk(self) -> None:
        with open(self.persist_path, "r", encoding="utf-8") as f:  # type: ignore[arg-type]
            data = json.load(f)
        for entry in data:
            doc = Document.from_dict(entry)
            self._docs[doc.id] = doc
        logger.info("Loaded %d documents from %s", len(self._docs), self.persist_path)
