"""Document retrieval strategies for RAG.

Implements similarity search, BM25 sparse retrieval, and hybrid
retrieval combining dense and sparse methods.
"""

from __future__ import annotations

import math
import logging
import re
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from nexus_llm.rag.vector_store import VectorDocument

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """A single retrieval result with document and relevance information."""

    document: VectorDocument
    score: float
    retrieval_method: str = ""
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        preview = self.document.text[:50].replace("\n", " ")
        return f"RetrievalResult(score={self.score:.4f}, method={self.retrieval_method}, text='{preview}...')"


class Retriever(ABC):
    """Abstract base class for document retrievers."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """Retrieve documents relevant to the query.

        Args:
            query: The search query string.
            top_k: Maximum number of results to return.

        Returns:
            List of RetrievalResult objects sorted by relevance.
        """
        ...

    @abstractmethod
    def index_documents(self, documents: List[VectorDocument]) -> None:
        """Index documents for retrieval.

        Args:
            documents: List of VectorDocument objects to index.
        """
        ...


class SimilarityRetriever(Retriever):
    """Dense similarity-based retrieval using vector store.

    Encodes queries using an embedding model and performs
    similarity search against a vector store.
    """

    def __init__(
        self,
        vector_store=None,
        embedding_model=None,
        default_top_k: int = 10,
    ):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.default_top_k = default_top_k

    def index_documents(self, documents: List[VectorDocument]) -> None:
        """Index documents by generating embeddings and adding to vector store.

        If documents don't have embeddings, generates them using the embedding model.
        """
        if self.embedding_model is None:
            raise ValueError("Embedding model is required for SimilarityRetriever")

        docs_to_embed = [doc for doc in documents if doc.embedding is None]
        if docs_to_embed:
            texts = [doc.text for doc in docs_to_embed]
            embeddings = self.embedding_model.embed(texts)
            for i, doc in enumerate(docs_to_embed):
                doc.embedding = embeddings[i]

        if self.vector_store:
            self.vector_store.add(documents)
        else:
            raise ValueError("Vector store is required for SimilarityRetriever")

    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """Retrieve documents using dense similarity search."""
        if self.embedding_model is None or self.vector_store is None:
            raise ValueError("Both embedding model and vector store are required")

        query_embedding = self.embedding_model.embed_query(query)
        results = self.vector_store.search(query_embedding, top_k=top_k)

        return [
            RetrievalResult(
                document=doc,
                score=score,
                retrieval_method="similarity",
            )
            for doc, score in results
        ]


class BM25Retriever(Retriever):
    """BM25 sparse retrieval based on term frequency and document frequency.

    Implements the Okapi BM25 algorithm for keyword-based document
    retrieval with tunable parameters k1 and b.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon

        self._documents: Dict[str, VectorDocument] = {}
        self._doc_tokens: Dict[str, List[str]] = {}
        self._doc_lengths: Dict[str, int] = {}
        self._avg_doc_length: float = 0.0
        self._doc_freq: Dict[str, int] = defaultdict(int)  # term -> num docs containing term
        self._term_freq: Dict[str, Dict[str, int]] = {}  # doc_id -> {term: count}
        self._num_docs: int = 0
        self._idf_cache: Dict[str, float] = {}

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into lowercase terms."""
        tokens = re.findall(r"\b\w+\b", text.lower())
        return tokens

    def _compute_idf(self, term: str) -> float:
        """Compute inverse document frequency for a term."""
        if term in self._idf_cache:
            return self._idf_cache[term]

        df = self._doc_freq.get(term, 0)
        idf = math.log((self._num_docs - df + 0.5) / (df + 0.5) + 1.0)
        # Ensure non-negative IDF
        idf = max(idf, self.epsilon)
        self._idf_cache[term] = idf
        return idf

    def index_documents(self, documents: List[VectorDocument]) -> None:
        """Index documents for BM25 retrieval."""
        for doc in documents:
            self._documents[doc.doc_id] = doc
            tokens = self._tokenize(doc.text)
            self._doc_tokens[doc.doc_id] = tokens
            self._doc_lengths[doc.doc_id] = len(tokens)

            # Compute term frequencies
            tf = Counter(tokens)
            self._term_freq[doc.doc_id] = dict(tf)

            # Update document frequency
            for term in set(tokens):
                self._doc_freq[term] += 1

            self._num_docs += 1

        # Compute average document length
        if self._num_docs > 0:
            self._avg_doc_length = sum(self._doc_lengths.values()) / self._num_docs

        # Clear IDF cache since corpus changed
        self._idf_cache.clear()

        logger.info("BM25 indexed %d documents. Avg doc length: %.1f", self._num_docs, self._avg_doc_length)

    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """Retrieve documents using BM25 scoring."""
        if self._num_docs == 0:
            return []

        query_tokens = self._tokenize(query)
        scores: Dict[str, float] = {}

        for doc_id, doc in self._documents.items():
            doc_score = 0.0
            doc_len = self._doc_lengths[doc_id]

            for term in query_tokens:
                tf = self._term_freq[doc_id].get(term, 0)
                if tf == 0:
                    continue

                idf = self._compute_idf(term)
                # BM25 formula
                tf_component = (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * doc_len / self._avg_doc_length)
                )
                doc_score += idf * tf_component

            if doc_score > 0:
                scores[doc_id] = doc_score

        # Sort by score descending
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        return [
            RetrievalResult(
                document=self._documents[doc_id],
                score=score,
                retrieval_method="bm25",
            )
            for doc_id, score in sorted_results
        ]

    def remove_documents(self, doc_ids: List[str]) -> int:
        """Remove documents from the BM25 index."""
        removed = 0
        for doc_id in doc_ids:
            if doc_id in self._documents:
                # Decrement document frequencies
                tokens = self._doc_tokens.get(doc_id, [])
                for term in set(tokens):
                    self._doc_freq[term] = max(0, self._doc_freq[term] - 1)

                del self._documents[doc_id]
                self._doc_tokens.pop(doc_id, None)
                self._doc_lengths.pop(doc_id, None)
                self._term_freq.pop(doc_id, None)
                self._num_docs -= 1
                removed += 1

        if self._num_docs > 0:
            self._avg_doc_length = sum(self._doc_lengths.values()) / self._num_docs

        self._idf_cache.clear()
        return removed


class HybridRetriever(Retriever):
    """Hybrid retrieval combining dense similarity and sparse BM25.

    Combines results from multiple retrievers using reciprocal rank
    fusion (RRF) or weighted score combination.
    """

    def __init__(
        self,
        retrievers: Optional[List[Tuple[Retriever, float]]] = None,
        fusion_method: str = "rrf",
        rrf_k: int = 60,
    ):
        """Initialize hybrid retriever.

        Args:
            retrievers: List of (retriever, weight) tuples.
            fusion_method: Fusion method - 'rrf' for reciprocal rank fusion,
                          'weighted' for weighted score combination.
            rrf_k: K parameter for RRF (default 60).
        """
        self.retrievers: List[Tuple[Retriever, float]] = retrievers or []
        self.fusion_method = fusion_method
        self.rrf_k = rrf_k

    def add_retriever(self, retriever: Retriever, weight: float = 1.0) -> None:
        """Add a retriever with a given weight."""
        self.retrievers.append((retriever, weight))

    def index_documents(self, documents: List[VectorDocument]) -> None:
        """Index documents in all sub-retrievers."""
        for retriever, _ in self.retrievers:
            retriever.index_documents(documents)

    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """Retrieve documents using hybrid search with fusion."""
        if not self.retrievers:
            return []

        # Get results from each retriever
        all_results: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
        doc_map: Dict[str, VectorDocument] = {}

        for retriever, weight in self.retrievers:
            # Retrieve more results to ensure good fusion
            results = retriever.retrieve(query, top_k=top_k * 3)
            for rank, result in enumerate(results):
                doc_id = result.document.doc_id
                all_results[doc_id].append((rank, result.score * weight))
                doc_map[doc_id] = result.document

        if not all_results:
            return []

        # Fuse results
        fused_scores: Dict[str, float] = {}

        if self.fusion_method == "rrf":
            # Reciprocal Rank Fusion
            for doc_id, rank_score_list in all_results.items():
                rrf_score = 0.0
                for rank, _ in rank_score_list:
                    rrf_score += 1.0 / (self.rrf_k + rank + 1)
                fused_scores[doc_id] = rrf_score
        elif self.fusion_method == "weighted":
            # Weighted score combination with min-max normalization
            # First, collect scores per retriever for normalization
            all_raw_scores: Dict[str, List[float]] = defaultdict(list)
            for doc_id, rank_score_list in all_results.items():
                for _, score in rank_score_list:
                    all_raw_scores[doc_id].append(score)

            if all_raw_scores:
                flat_scores = [s for scores in all_raw_scores.values() for s in scores]
                min_score = min(flat_scores)
                max_score = max(flat_scores)
                score_range = max_score - min_score if max_score > min_score else 1.0

                for doc_id, rank_score_list in all_results.items():
                    normalized = sum((s - min_score) / score_range for _, s in rank_score_list)
                    fused_scores[doc_id] = normalized
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

        # Sort by fused score
        sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        return [
            RetrievalResult(
                document=doc_map[doc_id],
                score=score,
                retrieval_method="hybrid",
                metadata={"fusion_method": self.fusion_method},
            )
            for doc_id, score in sorted_docs
        ]
