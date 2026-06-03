"""Retriever for Nexus-LLM RAG.

Supports keyword (BM25-like), semantic (mock), and hybrid retrieval
strategies.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple

from nexus_llm.rag.document_store import Document
from nexus_llm.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Retrieval strategies
# ---------------------------------------------------------------------------

class RetrievalStrategy(str, Enum):
    """Available retrieval strategies."""
    KEYWORD = "keyword"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


@dataclass
class RetrievalResult:
    """A single retrieval result with a relevance score.

    Attributes:
        document: The matched document.
        score: Relevance score (higher is better).
        strategy: The strategy that produced this result.
    """

    document: Document
    score: float
    strategy: str


# ---------------------------------------------------------------------------
# BM25-like scorer
# ---------------------------------------------------------------------------

class _BM25Scorer:
    """Simplified BM25 scoring for keyword retrieval.

    Args:
        k1: Term frequency saturation parameter.
        b: Length normalisation parameter.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._doc_freqs: Counter = Counter()
        self._doc_lengths: List[int] = []
        self._avgdl: float = 0.0
        self._corpus_size: int = 0

    def fit(self, documents: List[Document]) -> None:
        """Compute corpus-level statistics from *documents*."""
        self._corpus_size = len(documents)
        self._doc_lengths = []
        self._doc_freqs = Counter()

        for doc in documents:
            tokens = self._tokenise(doc.content)
            self._doc_lengths.append(len(tokens))
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self._doc_freqs[token] += 1

        total = sum(self._doc_lengths) or 1
        self._avgdl = total / (self._corpus_size or 1)

    def score(self, query: str, document: Document, doc_index: int) -> float:
        """Score a single document against *query*."""
        query_tokens = self._tokenise(query)
        doc_tokens = self._tokenise(document.content)
        doc_len = self._doc_lengths[doc_index] if doc_index < len(self._doc_lengths) else len(doc_tokens)
        tf_map = Counter(doc_tokens)

        score = 0.0
        for term in query_tokens:
            if term not in self._doc_freqs:
                continue
            df = self._doc_freqs[term]
            idf = math.log(
                (self._corpus_size - df + 0.5) / (df + 0.5) + 1.0
            )
            tf = tf_map.get(term, 0)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (
                1 - self.b + self.b * (doc_len / max(self._avgdl, 1e-8))
            )
            score += idf * (numerator / denominator)

        return max(score, 0.0)

    @staticmethod
    def _tokenise(text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class Retriever:
    """Retrieve documents using keyword, semantic (mock), or hybrid search.

    Args:
        strategy: The retrieval strategy to use.
        top_k: Default number of results to return.
    """

    def __init__(
        self,
        strategy: str = "keyword",
        top_k: int = 5,
    ) -> None:
        self.strategy = RetrievalStrategy(strategy)
        self.top_k = top_k
        self._bm25 = _BM25Scorer()
        self._documents: List[Document] = []
        logger.info("Retriever initialised with strategy=%s", self.strategy.value)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index_documents(self, documents: List[Document]) -> None:
        """Register documents for retrieval.

        Must be called before :meth:`retrieve`.
        """
        self._documents = list(documents)
        if self.strategy in (RetrievalStrategy.KEYWORD, RetrievalStrategy.HYBRID):
            self._bm25.fit(self._documents)
        logger.info("Indexed %d documents for retrieval", len(self._documents))

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[Tuple[Document, float]]:
        """Retrieve the top-k documents for *query*.

        Returns:
            A list of ``(document, score)`` tuples sorted by descending
            score.
        """
        k = top_k or self.top_k
        if self.strategy == RetrievalStrategy.KEYWORD:
            return self._keyword_retrieve(query, k)
        elif self.strategy == RetrievalStrategy.SEMANTIC:
            return self._semantic_retrieve(query, k)
        elif self.strategy == RetrievalStrategy.HYBRID:
            return self._hybrid_retrieve(query, k)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------

    def _keyword_retrieve(self, query: str, top_k: int) -> List[Tuple[Document, float]]:
        results: List[Tuple[Document, float]] = []
        for idx, doc in enumerate(self._documents):
            score = self._bm25.score(query, doc, idx)
            if score > 0:
                results.append((doc, score))
        results.sort(key=lambda t: t[1], reverse=True)
        return results[:top_k]

    def _semantic_retrieve(self, query: str, top_k: int) -> List[Tuple[Document, float]]:
        """Mock semantic retrieval using simple term overlap scoring.

        In production, this would use vector embeddings and cosine
        similarity.  The mock version provides a basic approximation.
        """
        query_terms = set(re.findall(r"\w+", query.lower()))
        results: List[Tuple[Document, float]] = []
        for doc in self._documents:
            doc_terms = set(re.findall(r"\w+", doc.content.lower()))
            if not query_terms or not doc_terms:
                continue
            overlap = len(query_terms & doc_terms)
            score = overlap / len(query_terms | doc_terms)  # Jaccard similarity
            if score > 0:
                results.append((doc, score))
        results.sort(key=lambda t: t[1], reverse=True)
        return results[:top_k]

    def _hybrid_retrieve(self, query: str, top_k: int) -> List[Tuple[Document, float]]:
        """Combine keyword and semantic scores with equal weighting."""
        keyword_results = self._keyword_retrieve(query, top_k * 2)
        semantic_results = self._semantic_retrieve(query, top_k * 2)

        # Merge scores
        scores: dict = {}
        for doc, score in keyword_results:
            scores[doc.id] = scores.get(doc.id, 0.0) + 0.5 * score
        for doc, score in semantic_results:
            scores[doc.id] = scores.get(doc.id, 0.0) + 0.5 * score

        doc_map = {doc.id: doc for doc in self._documents}
        combined = [(doc_map[did], s) for did, s in scores.items() if did in doc_map]
        combined.sort(key=lambda t: t[1], reverse=True)
        return combined[:top_k]
