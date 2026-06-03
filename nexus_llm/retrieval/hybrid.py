"""Hybrid retriever for Nexus-LLM.

Combines keyword (BM25-style) and semantic (vector) search using
reciprocal rank fusion for robust retrieval.
"""

import logging
import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from nexus_llm.retrieval.vector_index import VectorIndex

logger = logging.getLogger(__name__)

# RRF constant (prevents division by zero, smooths rankings)
RRF_K = 60

# Type alias
Document = Dict[str, Any]
RetrievalResult = Tuple[str, float, Document]


class HybridRetriever:
    """Hybrid retriever combining keyword and vector search.

    Uses reciprocal rank fusion (RRF) to merge results from both
    retrieval methods, configurable via *alpha* (keyword weight) and
    *beta* (semantic weight).

    Example::

        retriever = HybridRetriever(alpha=0.4, beta=0.6)
        retriever.index_documents(docs)
        results = retriever.retrieve("machine learning", top_k=5)
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        vector_index: Optional[VectorIndex] = None,
    ) -> None:
        """Initialise the hybrid retriever.

        Args:
            alpha: Weight for keyword search results (0-1).
            beta: Weight for semantic search results (0-1).
            vector_index: Optional pre-built VectorIndex.  A new one is
                          created if not provided.

        Raises:
            ValueError: If weights are negative.
        """
        if alpha < 0 or beta < 0:
            raise ValueError("Weights must be non-negative")
        self._alpha = alpha
        self._beta = beta
        self._vector_index = vector_index or VectorIndex()

        # Keyword index: id → term frequency map
        self._doc_texts: Dict[str, str] = {}
        self._doc_metadata: Dict[str, Document] = {}
        self._term_index: Dict[str, Counter] = {}  # term → {doc_id: count}
        self._doc_lengths: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_documents(
        self,
        documents: List[Document],
        id_field: str = "id",
        text_field: str = "text",
    ) -> None:
        """Index a list of documents for keyword and vector search.

        Args:
            documents: List of document dicts.  Each must contain the
                       fields specified by *id_field* and *text_field*.
            id_field: Key for the document ID.
            text_field: Key for the document text content.
        """
        for doc in documents:
            doc_id = str(doc.get(id_field, ""))
            text = doc.get(text_field, "")
            if not doc_id or not text:
                continue

            # Store document
            self._doc_texts[doc_id] = text
            self._doc_metadata[doc_id] = doc

            # Build keyword index
            tokens = self._tokenize(text)
            self._doc_lengths[doc_id] = len(tokens)

            for token, count in Counter(tokens).items():
                if token not in self._term_index:
                    self._term_index[token] = Counter()
                self._term_index[token][doc_id] = count

            # Add to vector index (use a simple TF-based mock vector)
            # In production, this would use a real embedding model
            vector = self._text_to_mock_vector(text)
            self._vector_index.add(doc_id, vector, doc)

        logger.info(
            "Indexed %d documents (%d terms in keyword index)",
            len(self._doc_texts),
            len(self._term_index),
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        query_vector: Optional[List[float]] = None,
    ) -> List[RetrievalResult]:
        """Retrieve documents using hybrid search with RRF.

        Args:
            query: The search query string.
            top_k: Number of results to return.
            query_vector: Optional pre-computed query embedding.
                          If ``None`` a mock vector is generated.

        Returns:
            List of ``(doc_id, score, metadata)`` tuples sorted by
            descending fusion score.
        """
        # --- Keyword search ---
        keyword_results = self._keyword_search(query, top_k=top_k * 2)

        # --- Vector search ---
        if query_vector is None:
            query_vector = self._text_to_mock_vector(query)
        vector_results = self._vector_index.search(query_vector, top_k=top_k * 2)

        # --- Reciprocal Rank Fusion ---
        fused_scores: Dict[str, float] = {}

        for rank, (doc_id, _score) in enumerate(keyword_results):
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + self._alpha / (RRF_K + rank + 1)

        for rank, (doc_id, _score) in enumerate(vector_results):
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + self._beta / (RRF_K + rank + 1)

        # Sort by fused score
        sorted_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

        results: List[RetrievalResult] = []
        for doc_id, score in sorted_results[:top_k]:
            metadata = self._doc_metadata.get(doc_id, {})
            results.append((doc_id, score, metadata))

        return results

    # ------------------------------------------------------------------
    # Keyword search (BM25-like)
    # ------------------------------------------------------------------

    def _keyword_search(
        self,
        query: str,
        top_k: int = 20,
    ) -> List[Tuple[str, float]]:
        """BM25-inspired keyword search.

        Returns:
            List of ``(doc_id, score)`` tuples.
        """
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        N = len(self._doc_texts)
        avg_dl = sum(self._doc_lengths.values()) / N if N > 0 else 1.0

        # BM25 parameters
        k1 = 1.5
        b = 0.75

        scores: Dict[str, float] = {}
        for token in query_tokens:
            if token not in self._term_index:
                continue

            # Document frequency
            df = len(self._term_index[token])
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1)

            for doc_id, tf in self._term_index[token].items():
                dl = self._doc_lengths.get(doc_id, 1)
                tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avg_dl))
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * tf_norm

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores[:top_k]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple whitespace + punctuation tokenization."""
        text = text.lower()
        tokens = re.findall(r"[a-z0-9]+", text)
        return tokens

    @staticmethod
    def _text_to_mock_vector(text: str, dimension: int = 64) -> List[float]:
        """Generate a deterministic mock vector from text.

        Uses character-level hashing for reproducibility.
        """
        import hashlib

        raw = b""
        for i in range(max(1, (dimension * 4 + 31) // 32)):
            h = hashlib.sha256(f"{text}|vec{i}".encode("utf-8")).digest()
            raw += h

        values: List[float] = []
        for i in range(dimension):
            offset = i * 4
            int_val = int.from_bytes(raw[offset : offset + 4], "big")
            values.append(int_val / (2**32 - 1))

        # Centre and normalise
        mean = sum(values) / len(values)
        values = [v - mean for v in values]
        norm = math.sqrt(sum(v * v for v in values))
        if norm > 0:
            values = [v / norm for v in values]

        return values
