"""Retrieval engine for Nexus-LLM.

High-level retrieval engine that supports keyword, semantic, and
hybrid retrieval modes.
"""

import logging
from typing import Any, Dict, List, Optional

from nexus_llm.retrieval.vector_index import VectorIndex
from nexus_llm.retrieval.hybrid import HybridRetriever

logger = logging.getLogger(__name__)

# Type aliases
Document = Dict[str, Any]
RetrievalResult = Dict[str, Any]


class RetrievalEngine:
    """High-level document retrieval engine.

    Supports three retrieval modes:
    - ``"keyword"``: BM25-style keyword matching.
    - ``"semantic"``: Vector similarity search.
    - ``"hybrid"``: Combined keyword + vector with RRF fusion.

    Example::

        engine = RetrievalEngine(mode="hybrid")
        engine.index([
            {"id": "1", "text": "Machine learning fundamentals"},
            {"id": "2", "text": "Deep learning with transformers"},
        ])
        results = engine.retrieve("machine learning", top_k=5)
    """

    def __init__(
        self,
        mode: str = "hybrid",
        alpha: float = 0.5,
        beta: float = 0.5,
    ) -> None:
        """Initialise the retrieval engine.

        Args:
            mode: Retrieval mode — ``"keyword"``, ``"semantic"``, or
                  ``"hybrid"``.
            alpha: Keyword weight (for hybrid mode).
            beta: Semantic weight (for hybrid mode).

        Raises:
            ValueError: If *mode* is not recognised.
        """
        valid_modes = {"keyword", "semantic", "hybrid"}
        if mode not in valid_modes:
            raise ValueError(
                f"Invalid retrieval mode {mode!r}. Valid: {sorted(valid_modes)}"
            )

        self._mode = mode
        self._vector_index = VectorIndex()
        self._hybrid_retriever = HybridRetriever(
            alpha=alpha,
            beta=beta,
            vector_index=self._vector_index,
        )

        # Keyword-only index structures (shared with hybrid retriever)
        self._documents: Dict[str, Document] = {}

        logger.info(
            "RetrievalEngine initialised (mode=%s, alpha=%.2f, beta=%.2f)",
            mode,
            alpha,
            beta,
        )

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index(
        self,
        documents: List[Document],
        id_field: str = "id",
        text_field: str = "text",
    ) -> None:
        """Index documents for retrieval.

        Args:
            documents: List of document dicts.
            id_field: Key for the document ID.
            text_field: Key for the document text content.
        """
        for doc in documents:
            doc_id = str(doc.get(id_field, ""))
            if doc_id:
                self._documents[doc_id] = doc

        # Always index in hybrid retriever (it handles both keyword & vector)
        self._hybrid_retriever.index_documents(documents, id_field, text_field)

        logger.info(
            "Indexed %d documents (total: %d)",
            len(documents),
            len(self._documents),
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        mode: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """Retrieve documents matching the query.

        Args:
            query: The search query string.
            top_k: Number of results to return.
            mode: Override the default retrieval mode for this call.

        Returns:
            List of result dicts with ``id``, ``score``, and document
            fields.
        """
        effective_mode = mode or self._mode

        if effective_mode == "keyword":
            raw_results = self._keyword_retrieve(query, top_k)
        elif effective_mode == "semantic":
            raw_results = self._semantic_retrieve(query, top_k)
        elif effective_mode == "hybrid":
            raw_results = self._hybrid_retriever.retrieve(query, top_k)
        else:
            raise ValueError(f"Unknown retrieval mode: {effective_mode!r}")

        # Format results
        results: List[RetrievalResult] = []
        for doc_id, score, metadata in raw_results:
            result = {"id": doc_id, "score": round(score, 6)}
            # Merge document metadata
            doc = self._documents.get(doc_id, {})
            for k, v in doc.items():
                if k not in result:
                    result[k] = v
            results.append(result)

        return results

    # ------------------------------------------------------------------
    # Mode-specific retrieval
    # ------------------------------------------------------------------

    def _keyword_retrieve(
        self,
        query: str,
        top_k: int,
    ) -> List[tuple]:
        """Retrieve using keyword-only search."""
        results = self._hybrid_retriever._keyword_search(query, top_k=top_k)
        return [
            (doc_id, score, self._documents.get(doc_id, {}))
            for doc_id, score in results
        ]

    def _semantic_retrieve(
        self,
        query: str,
        top_k: int,
    ) -> List[tuple]:
        """Retrieve using vector-only search."""
        query_vector = self._hybrid_retriever._text_to_mock_vector(query)
        results = self._vector_index.search_with_metadata(query_vector, top_k=top_k)
        return results

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @property
    def mode(self) -> str:
        """Current default retrieval mode."""
        return self._mode

    @property
    def document_count(self) -> int:
        """Number of indexed documents."""
        return len(self._documents)
