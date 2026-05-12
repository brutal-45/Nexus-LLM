"""
Nexus LLM — Retrieval Module
==============================

Production-grade text retrieval systems covering dense, sparse, hybrid,
ColBERT late-interaction, multi-vector, and cross-encoder reranked
retrieval.  Every retriever produces a list of :class:`RetrievalResult`
objects ranked by relevance score.

Dependencies
------------
* ``numpy`` — array operations and distance computation
* ``torch`` — optional GPU acceleration for dense retrieval
"""

from __future__ import annotations

import abc
import math
import re
import logging
import hashlib
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np

logger: logging.Logger = logging.getLogger("nexus.knowledge.retrieval")


# ═══════════════════════════════════════════════════════════════════════════
#  Data structures
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Document:
    """A single document in the retrieval corpus.

    Attributes
    ----------
    id : str
        Unique identifier (auto-generated if not supplied).
    content : str
        Raw text content.
    title : Optional[str]
        Optional title or headline.
    metadata : Dict[str, Any]
        Arbitrary key-value metadata.
    embedding : Optional[np.ndarray]
        Pre-computed dense embedding.
    """

    id: str = ""
    content: str = ""
    title: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if not self.id:
            raw = f"{self.content}{self.title}"
            self.id = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Document):
            return self.id == other.id
        return NotImplemented

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"id": self.id, "content": self.content}
        if self.title is not None:
            d["title"] = self.title
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Document:
        return cls(
            id=data.get("id", ""),
            content=data.get("content", ""),
            title=data.get("title"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class RetrievalResult:
    """A single retrieval hit.

    Attributes
    ----------
    document : Document
        The retrieved document.
    score : float
        Relevance score (higher = more relevant).
    rank : int
        1-based rank in the result list.
    metadata : Dict[str, Any]
        Extra information from the retriever (e.g. BM25 sub-scores).
    """

    document: Document
    score: float
    rank: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class RetrievalMetric(str, Enum):
    COSINE = "cosine"
    L2 = "l2"
    INNER_PRODUCT = "ip"
    BM25 = "bm25"
    TF_IDF = "tf_idf"
    MAXSIM = "maxsim"


# ═══════════════════════════════════════════════════════════════════════════
#  Abstract base retriever
# ═══════════════════════════════════════════════════════════════════════════

class BaseRetriever(abc.ABC):
    """Interface that all retrievers must implement."""

    @abc.abstractmethod
    def index(self, documents: Sequence[Document]) -> None:
        """Build or update the retrieval index from *documents*."""

    @abc.abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        score_threshold: float = 0.0,
    ) -> List[RetrievalResult]:
        """Retrieve up to *top_k* documents for *query*."""

    def batch_retrieve(
        self,
        queries: Sequence[str],
        top_k: int = 10,
        score_threshold: float = 0.0,
    ) -> List[List[RetrievalResult]]:
        """Retrieve for each query independently (default: sequential)."""
        return [self.retrieve(q, top_k, score_threshold) for q in queries]

    @property
    def document_count(self) -> int:
        return 0

    @property
    def is_indexed(self) -> bool:
        return self.document_count > 0


# ═══════════════════════════════════════════════════════════════════════════
#  Distance helpers
# ═══════════════════════════════════════════════════════════════════════════

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between *a* (N, D) and *b* (M, D)."""
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    a_norm = np.maximum(a_norm, 1e-12)
    b_norm = np.maximum(b_norm, 1e-12)
    a = a / a_norm
    b = b / b_norm
    return a @ b.T


def l2_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise L2 distance between *a* (N, D) and *b* (M, D)."""
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b
    a_sq = np.sum(a ** 2, axis=1, keepdims=True)
    b_sq = np.sum(b ** 2, axis=1, keepdims=True).T
    dist_sq = a_sq + b_sq - 2.0 * (a @ b.T)
    dist_sq = np.maximum(dist_sq, 0.0)
    return np.sqrt(dist_sq)


def inner_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise inner product between *a* (N, D) and *b* (M, D)."""
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)
    return a @ b.T


def compute_scores(
    query_vec: np.ndarray,
    doc_matrix: np.ndarray,
    metric: RetrievalMetric,
) -> np.ndarray:
    """Compute similarity / distance scores and return as-is (higher=more relevant)."""
    if metric == RetrievalMetric.COSINE:
        scores = cosine_similarity(query_vec, doc_matrix)
    elif metric == RetrievalMetric.L2:
        dist = l2_distance(query_vec, doc_matrix)
        scores = -dist  # negate so higher = more relevant
    elif metric == RetrievalMetric.INNER_PRODUCT:
        scores = inner_product(query_vec, doc_matrix)
    else:
        raise ValueError(f"Unsupported metric for dense scoring: {metric}")
    return scores


def top_k_indices(scores: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(indices, scores)`` of the top-*k* highest scores."""
    k = min(k, len(scores))
    if k <= 0:
        return np.array([], dtype=np.intp), np.array([], dtype=np.float64)
    top_idx = np.argpartition(scores, -k)[-k:]
    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
    return top_idx, scores[top_idx]


# ═══════════════════════════════════════════════════════════════════════════
#  Simple embedding encoder (no external dependencies)
# ═══════════════════════════════════════════════════════════════════════════

class SimpleEncoder:
    """Bag-of-words hashing encoder — no model download required.

    Produces dense vectors of configurable dimensionality using character-
    level n-gram hashing with multiple hash functions, followed by L2
    normalisation.  Sufficient for prototyping and testing.
    """

    def __init__(self, dimension: int = 256, ngram_range: Tuple[int, int] = (3, 5)) -> None:
        self._dim = dimension
        self._ngram_lo, self._ngram_hi = ngram_range
        self._seed = 42
        self._rng = np.random.RandomState(self._seed)
        self._projection: Optional[np.ndarray] = None

    def _get_ngrams(self, text: str) -> List[str]:
        text = text.lower().strip()
        ngrams: List[str] = []
        for n in range(self._ngram_lo, self._ngram_hi + 1):
            for i in range(max(0, len(text) - n + 1)):
                ngrams.append(text[i : i + n])
        return ngrams

    def _hash_ngram(self, ngram: str, hash_idx: int) -> int:
        h = hashlib.md5(f"{self._seed}:{hash_idx}:{ngram}".encode()).hexdigest()
        return int(h, 16) % self._dim

    def encode_single(self, text: str) -> np.ndarray:
        ngrams = self._get_ngrams(text)
        if not ngrams:
            vec = np.zeros(self._dim, dtype=np.float64)
        else:
            vec = np.zeros(self._dim, dtype=np.float64)
            counts: Counter = Counter(ngrams)
            for ng, cnt in counts.items():
                for hi in range(4):
                    idx = self._hash_ngram(ng, hi)
                    sign = 1.0 if hi % 2 == 0 else -1.0
                    vec[idx] += sign * cnt
            norm = np.linalg.norm(vec)
            if norm > 1e-12:
                vec /= norm
        return vec

    def encode(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        result = np.zeros((len(texts), self._dim), dtype=np.float64)
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            for i, t in enumerate(batch):
                result[start + i] = self.encode_single(t)
        return result

    @property
    def dimension(self) -> int:
        return self._dim


# ═══════════════════════════════════════════════════════════════════════════
#  DenseRetriever
# ═══════════════════════════════════════════════════════════════════════════

class DenseRetriever(BaseRetriever):
    """Dense passage retrieval using learned embeddings.

    Encodes all documents into a shared embedding space and retrieves via
    nearest-neighbour search over cosine / L2 / inner-product metrics.

    Parameters
    ----------
    dimension : int
        Embedding dimensionality.
    encoder : Optional[Callable[[Sequence[str]], np.ndarray]]
        Custom encoder callable.  Falls back to :class:`SimpleEncoder` when
        ``None``.
    metric : str
        Distance metric (``"cosine"``, ``"l2"``, ``"ip"``).
    device : str
        Compute device string (``"cpu"`` / ``"cuda:0"``).
    batch_size : int
        Batch size for encoding.
    top_k : int
        Default number of results.
    score_threshold : float
        Minimum score to include.
    max_docs : int
        Maximum number of documents to index.
    normalize : bool
        L2-normalise embeddings before search.
    """

    def __init__(
        self,
        dimension: int = 256,
        encoder: Optional[Callable[[Sequence[str]], np.ndarray]] = None,
        metric: str = "cosine",
        device: str = "cpu",
        batch_size: int = 32,
        top_k: int = 10,
        score_threshold: float = 0.0,
        max_docs: int = 100_000,
        normalize: bool = True,
    ) -> None:
        self._dimension = dimension
        self._custom_encoder = encoder
        self._metric = RetrievalMetric(metric)
        self._device = device
        self._batch_size = batch_size
        self._default_top_k = top_k
        self._default_threshold = score_threshold
        self._max_docs = max_docs
        self._normalize = normalize
        self._encoder = encoder or SimpleEncoder(dimension=dimension)
        self._documents: List[Document] = []
        self._embeddings: Optional[np.ndarray] = None
        self._id_to_idx: Dict[str, int] = {}

    # ── Encoding ───────────────────────────────────────────────────────

    def encode_queries(
        self,
        queries: Sequence[str],
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """Encode *queries* to embedding vectors (N, D)."""
        bs = batch_size or self._batch_size
        if callable(self._encoder):
            return self._encoder(queries) if not isinstance(self._encoder, SimpleEncoder) else self._encoder.encode(queries, batch_size=bs)
        return self._encoder.encode(queries, batch_size=bs)

    def encode_documents(
        self,
        documents: Sequence[Union[str, Document]],
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """Encode *documents* to embedding vectors (N, D)."""
        texts = [d.content if isinstance(d, Document) else d for d in documents]
        return self.encode_queries(texts, batch_size)

    def _normalize_embeddings(self, vecs: np.ndarray) -> np.ndarray:
        if not self._normalize:
            return vecs
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return vecs / norms

    # ── Indexing ───────────────────────────────────────────────────────

    def index(self, documents: Sequence[Document]) -> None:
        """Build the index from *documents*.

        Existing index content is replaced.
        """
        docs = list(documents[: self._max_docs])
        if not docs:
            logger.warning("DenseRetriever.index() called with empty document list.")
            return
        self._documents = docs
        self._id_to_idx = {d.id: i for i, d in enumerate(docs)}
        logger.info(
            "Encoding %d documents (dim=%d, metric=%s) …",
            len(docs), self._dimension, self._metric.value,
        )
        t0 = time.perf_counter()
        raw = self.encode_documents(docs)
        self._embeddings = self._normalize_embeddings(raw)
        elapsed = time.perf_counter() - t0
        logger.info(
            "Indexed %d documents in %.2f s  (%.0f docs/s)",
            len(docs), elapsed, len(docs) / max(elapsed, 1e-9),
        )

    def add_documents(self, documents: Sequence[Document]) -> None:
        """Incrementally add documents to the existing index."""
        if not documents:
            return
        start_idx = len(self._documents)
        for i, d in enumerate(documents):
            self._id_to_idx[d.id] = start_idx + i
        self._documents.extend(documents)
        new_embeds = self.encode_documents(documents)
        new_embeds = self._normalize_embeddings(new_embeds)
        if self._embeddings is None:
            self._embeddings = new_embeds
        else:
            self._embeddings = np.vstack([self._embeddings, new_embeds])

    # ── Retrieval ──────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> List[RetrievalResult]:
        """Retrieve top-k documents for a single *query*."""
        if self._embeddings is None or len(self._documents) == 0:
            logger.warning("DenseRetriever.retrieve() called before index().")
            return []
        k = top_k if top_k is not None else self._default_top_k
        threshold = score_threshold if score_threshold is not None else self._default_threshold

        query_vec = self.encode_queries([query])
        query_vec = self._normalize_embeddings(query_vec)

        scores = compute_scores(query_vec[0], self._embeddings, self._metric)

        if threshold > 0.0:
            mask = scores >= threshold
            if not mask.any():
                return []
            valid_idx = np.where(mask)[0]
            valid_scores = scores[mask]
            sorted_order = np.argsort(valid_scores)[::-1][:k]
            ranked_idx = valid_idx[sorted_order]
            ranked_scores = valid_scores[sorted_order]
        else:
            k = min(k, len(scores))
            ranked_idx, ranked_scores = top_k_indices(scores, k)

        results: List[RetrievalResult] = []
        for rank, (idx, score) in enumerate(zip(ranked_idx, ranked_scores), start=1):
            doc = self._documents[int(idx)]
            results.append(RetrievalResult(
                document=doc,
                score=float(score),
                rank=rank,
                metadata={"metric": self._metric.value},
            ))
        return results

    def batch_retrieve(
        self,
        queries: Sequence[str],
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> List[List[RetrievalResult]]:
        """Vectorised batch retrieval for multiple queries."""
        if self._embeddings is None or len(self._documents) == 0:
            return [[] for _ in queries]
        k = top_k if top_k is not None else self._default_top_k
        threshold = score_threshold if score_threshold is not None else self._default_threshold

        query_vecs = self.encode_queries(queries)
        query_vecs = self._normalize_embeddings(query_vecs)

        sim_matrix = compute_scores(query_vecs, self._embeddings, self._metric)

        all_results: List[List[RetrievalResult]] = []
        for qi in range(len(queries)):
            scores = sim_matrix[qi]
            if threshold > 0.0:
                mask = scores >= threshold
                if not mask.any():
                    all_results.append([])
                    continue
                valid_idx = np.where(mask)[0]
                valid_scores = scores[mask]
                sorted_order = np.argsort(valid_scores)[::-1][:k]
                ranked_idx = valid_idx[sorted_order]
                ranked_scores = valid_scores[sorted_order]
            else:
                kk = min(k, len(scores))
                ranked_idx, ranked_scores = top_k_indices(scores, kk)

            results: List[RetrievalResult] = []
            for rank, (idx, score) in enumerate(zip(ranked_idx, ranked_scores), start=1):
                doc = self._documents[int(idx)]
                results.append(RetrievalResult(
                    document=doc,
                    score=float(score),
                    rank=rank,
                    metadata={"metric": self._metric.value},
                ))
            all_results.append(results)
        return all_results

    # ── Properties ─────────────────────────────────────────────────────

    @property
    def document_count(self) -> int:
        return len(self._documents)

    @property
    def is_indexed(self) -> bool:
        return self._embeddings is not None and len(self._documents) > 0

    @property
    def embeddings(self) -> Optional[np.ndarray]:
        return self._embeddings


# ═══════════════════════════════════════════════════════════════════════════
#  SparseRetriever (BM25 + TF-IDF)
# ═══════════════════════════════════════════════════════════════════════════

class SparseRetriever(BaseRetriever):
    """Sparse retrieval using BM25 and TF-IDF scoring.

    Implements a full inverted index with configurable BM25 parameters
    (k1, b) and computes both BM25 and TF-IDF scores per query-document
    pair.

    Parameters
    ----------
    k1 : float
        BM25 term-saturation parameter.
    b : float
        BM25 length-normalisation parameter.
    epsilon : float
        Lower bound on IDF values.
    top_k : int
        Default number of results.
    score_threshold : float
        Minimum score threshold.
    stopwords : Optional[Set[str]]
        Custom stop-word set.  Uses a built-in English list when ``None``.
    min_token_length : int
        Discard tokens shorter than this.
    """

    _DEFAULT_STOPWORDS: Set[str] = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "shall",
        "should", "may", "might", "must", "can", "could", "of", "in", "on",
        "at", "to", "for", "with", "by", "from", "up", "about", "into",
        "through", "during", "before", "after", "above", "below", "between",
        "under", "again", "further", "then", "once", "here", "there", "when",
        "where", "why", "how", "all", "both", "each", "few", "more", "most",
        "other", "some", "such", "no", "nor", "not", "only", "own", "same",
        "so", "than", "too", "very", "just", "because", "as", "until",
        "while", "but", "and", "or", "if", "it", "its", "this", "that",
        "these", "those", "i", "me", "my", "we", "our", "you", "your",
        "he", "him", "his", "she", "her", "they", "them", "their",
        "what", "which", "who", "whom",
    }

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
        top_k: int = 10,
        score_threshold: float = 0.0,
        stopwords: Optional[Set[str]] = None,
        min_token_length: int = 2,
    ) -> None:
        self._k1 = k1
        self._b = b
        self._epsilon = epsilon
        self._default_top_k = top_k
        self._default_threshold = score_threshold
        self._stopwords = stopwords or self._DEFAULT_STOPWORDS
        self._min_token_length = min_token_length

        self._documents: List[Document] = []
        self._doc_tokens: List[List[str]] = []
        self._doc_freqs: Dict[str, int] = defaultdict(int)
        self._inverted_index: Dict[str, List[int]] = defaultdict(list)
        self._term_freqs: List[Dict[str, int]] = []
        self._doc_lengths: List[int] = []
        self._avg_doc_length: float = 0.0
        self._id_to_idx: Dict[str, int] = {}
        self._idf_cache: Dict[str, float] = {}
        self._total_docs: int = 0

    # ── Tokenisation ───────────────────────────────────────────────────

    def tokenize(self, text: str) -> List[str]:
        """Lowercase, strip non-alpha, remove stopwords and short tokens."""
        text = text.lower()
        tokens = re.findall(r"[a-z0-9]+", text)
        tokens = [t for t in tokens if len(t) >= self._min_token_length and t not in self._stopwords]
        return tokens

    def tokenize_batch(self, texts: Sequence[str]) -> List[List[str]]:
        return [self.tokenize(t) for t in texts]

    # ── Index building ─────────────────────────────────────────────────

    def index(self, documents: Sequence[Document]) -> None:
        """Build the inverted index from *documents*."""
        self._documents = list(documents)
        self._total_docs = len(self._documents)
        self._id_to_idx = {d.id: i for i, d in enumerate(self._documents)}

        self._doc_tokens = []
        self._term_freqs = []
        self._doc_lengths = []
        self._doc_freqs = defaultdict(int)
        self._inverted_index = defaultdict(list)
        self._idf_cache = {}

        for i, doc in enumerate(self._documents):
            tokens = self.tokenize(doc.content)
            self._doc_tokens.append(tokens)
            self._doc_lengths.append(len(tokens))

            tf: Counter = Counter(tokens)
            self._term_freqs.append(dict(tf))

            for term in tf:
                if i not in self._inverted_index[term]:
                    self._inverted_index[term].append(i)
                self._doc_freqs[term] += 1

        total_len = sum(self._doc_lengths)
        self._avg_doc_length = total_len / max(self._total_docs, 1)

        self._compute_idf_cache()
        logger.info(
            "SparseRetriever: indexed %d docs, vocab=%d, avg_len=%.1f",
            self._total_docs, len(self._doc_freqs), self._avg_doc_length,
        )

    def add_documents(self, documents: Sequence[Document]) -> None:
        """Incrementally add documents."""
        start_idx = len(self._documents)
        for doc in documents:
            i = len(self._documents)
            self._id_to_idx[doc.id] = i
            self._documents.append(doc)

            tokens = self.tokenize(doc.content)
            self._doc_tokens.append(tokens)
            self._doc_lengths.append(len(tokens))

            tf: Counter = Counter(tokens)
            self._term_freqs.append(dict(tf))

            for term in tf:
                if i not in self._inverted_index[term]:
                    self._inverted_index[term].append(i)
                self._doc_freqs[term] += 1

        self._total_docs = len(self._documents)
        total_len = sum(self._doc_lengths)
        self._avg_doc_length = total_len / max(self._total_docs, 1)
        self._compute_idf_cache()

    # ── IDF ────────────────────────────────────────────────────────────

    def _compute_idf_cache(self) -> None:
        self._idf_cache = {}
        for term, df in self._doc_freqs.items():
            idf = math.log(1.0 + (self._total_docs - df + 0.5) / (df + 0.5) + self._epsilon)
            self._idf_cache[term] = idf

    def _idf(self, term: str) -> float:
        if term in self._idf_cache:
            return self._idf_cache[term]
        return self._epsilon

    # ── BM25 scoring ───────────────────────────────────────────────────

    def compute_bm25_score(self, query: str, doc_id: str) -> float:
        """Compute BM25 score for a single (query, doc_id) pair."""
        idx = self._id_to_idx.get(doc_id)
        if idx is None:
            return 0.0
        query_terms = self.tokenize(query)
        if not query_terms:
            return 0.0

        tf_dict = self._term_freqs[idx]
        doc_len = self._doc_lengths[idx]
        score = 0.0

        for term in query_terms:
            if term not in tf_dict:
                continue
            tf = tf_dict[term]
            idf_val = self._idf(term)
            numerator = tf * (self._k1 + 1.0)
            denominator = tf + self._k1 * (1.0 - self._b + self._b * doc_len / max(self._avg_doc_length, 1e-9))
            score += idf_val * (numerator / denominator)

        return score

    def _compute_bm25_vectorised(self, query_terms: List[str]) -> np.ndarray:
        """Compute BM25 scores for *query_terms* against all documents."""
        scores = np.zeros(self._total_docs, dtype=np.float64)

        for term in query_terms:
            if term not in self._inverted_index:
                continue
            idf_val = self._idf(term)
            postings = self._inverted_index[term]
            for doc_idx in postings:
                tf = self._term_freqs[doc_idx].get(term, 0)
                if tf == 0:
                    continue
                doc_len = self._doc_lengths[doc_idx]
                numerator = tf * (self._k1 + 1.0)
                denominator = tf + self._k1 * (1.0 - self._b + self._b * doc_len / max(self._avg_doc_length, 1e-9))
                scores[doc_idx] += idf_val * (numerator / denominator)

        return scores

    # ── TF-IDF scoring ─────────────────────────────────────────────────

    def compute_tf_idf(self, query: str, doc_id: str) -> float:
        """Compute TF-IDF score for a single (query, doc_id) pair."""
        idx = self._id_to_idx.get(doc_id)
        if idx is None:
            return 0.0
        query_terms = set(self.tokenize(query))
        if not query_terms:
            return 0.0

        tf_dict = self._term_freqs[idx]
        score = 0.0
        for term in query_terms:
            tf = tf_dict.get(term, 0)
            if tf == 0:
                continue
            idf_val = self._idf(term)
            score += tf * idf_val
        return score

    def _compute_tfidf_vectorised(self, query_terms: List[str]) -> np.ndarray:
        """Compute TF-IDF scores for *query_terms* against all documents."""
        scores = np.zeros(self._total_docs, dtype=np.float64)
        query_set = set(query_terms)

        for term in query_set:
            if term not in self._inverted_index:
                continue
            idf_val = self._idf(term)
            postings = self._inverted_index[term]
            for doc_idx in postings:
                tf = self._term_freqs[doc_idx].get(term, 0)
                scores[doc_idx] += tf * idf_val

        return scores

    # ── Retrieval ──────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> List[RetrievalResult]:
        """Retrieve top-k documents using BM25 scoring."""
        if not self._documents:
            return []

        k = top_k if top_k is not None else self._default_top_k
        threshold = score_threshold if score_threshold is not None else self._default_threshold

        query_terms = self.tokenize(query)
        if not query_terms:
            return []

        bm25_scores = self._compute_bm25_vectorised(query_terms)

        if threshold > 0.0:
            mask = bm25_scores >= threshold
            if not mask.any():
                return []
            valid_idx = np.where(mask)[0]
            valid_scores = bm25_scores[mask]
            sorted_order = np.argsort(valid_scores)[::-1][:k]
            ranked_idx = valid_idx[sorted_order]
            ranked_scores = valid_scores[sorted_order]
        else:
            kk = min(k, self._total_docs)
            ranked_idx, ranked_scores = top_k_indices(bm25_scores, kk)

        results: List[RetrievalResult] = []
        for rank, (idx, score) in enumerate(zip(ranked_idx, ranked_scores), start=1):
            doc = self._documents[int(idx)]
            results.append(RetrievalResult(
                document=doc,
                score=float(score),
                rank=rank,
                metadata={"method": "bm25"},
            ))
        return results

    # ── Properties ─────────────────────────────────────────────────────

    @property
    def document_count(self) -> int:
        return self._total_docs

    @property
    def is_indexed(self) -> bool:
        return self._total_docs > 0

    @property
    def vocabulary_size(self) -> int:
        return len(self._doc_freqs)

    def get_term_frequency(self, term: str) -> Dict[str, int]:
        """Return a mapping of doc_id → term-frequency for *term*."""
        postings = self._inverted_index.get(term, [])
        result: Dict[str, int] = {}
        for idx in postings:
            doc_id = self._documents[idx].id
            result[doc_id] = self._term_freqs[idx].get(term, 0)
        return result

    def get_document_frequency(self, term: str) -> int:
        return self._doc_freqs.get(term, 0)


# ═══════════════════════════════════════════════════════════════════════════
#  HybridRetriever
# ═══════════════════════════════════════════════════════════════════════════

class HybridRetriever(BaseRetriever):
    """Combines dense and sparse retrieval with configurable fusion.

    Supports two fusion strategies:
    * **Alpha-weighted** linear combination (default).
    * **Reciprocal Rank Fusion (RRF)** — robust to scale differences.

    Parameters
    ----------
    dense_retriever : DenseRetriever
        The dense retrieval component.
    sparse_retriever : SparseRetriever
        The sparse retrieval component.
    alpha : float
        Weight for dense scores (``1 − alpha`` for sparse).
    fusion : str
        ``"alpha"`` for linear combination, ``"rrf"`` for reciprocal rank
        fusion.
    rrf_k : int
        RRF smoothing constant (only used when ``fusion == "rrf"``).
    normalize_scores : bool
        Min-max normalise scores before alpha fusion.
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseRetriever,
        alpha: float = 0.7,
        fusion: str = "alpha",
        rrf_k: int = 60,
        normalize_scores: bool = True,
    ) -> None:
        self._dense = dense_retriever
        self._sparse = sparse_retriever
        self._alpha = alpha
        self._fusion = fusion
        self._rrf_k = rrf_k
        self._normalize_scores = normalize_scores
        self._documents: List[Document] = []

    def index(self, documents: Sequence[Document]) -> None:
        self._documents = list(documents)
        self._dense.index(documents)
        self._sparse.index(documents)

    def add_documents(self, documents: Sequence[Document]) -> None:
        self._documents.extend(documents)
        self._dense.add_documents(documents)
        self._sparse.add_documents(documents)

    # ── Alpha fusion ───────────────────────────────────────────────────

    def _min_max_normalize(self, scores: Dict[str, float]) -> Dict[str, float]:
        if not scores:
            return scores
        vals = list(scores.values())
        min_val = min(vals)
        max_val = max(vals)
        rng = max_val - min_val
        if rng < 1e-12:
            return {k: 1.0 for k in scores}
        return {k: (v - min_val) / rng for k, v in scores.items()}

    def _alpha_fuse(
        self,
        dense_results: List[RetrievalResult],
        sparse_results: List[RetrievalResult],
    ) -> List[RetrievalResult]:
        dense_scores: Dict[str, float] = {r.document.id: r.score for r in dense_results}
        sparse_scores: Dict[str, float] = {r.document.id: r.score for r in sparse_results}

        if self._normalize_scores:
            dense_scores = self._min_max_normalize(dense_scores)
            sparse_scores = self._min_max_normalize(sparse_scores)

        all_ids = set(dense_scores.keys()) | set(sparse_scores.keys())
        fused: List[Tuple[str, float]] = []
        for doc_id in all_ids:
            d = dense_scores.get(doc_id, 0.0)
            s = sparse_scores.get(doc_id, 0.0)
            fused.append((doc_id, self._alpha * d + (1.0 - self._alpha) * s))

        fused.sort(key=lambda x: x[1], reverse=True)

        results: List[RetrievalResult] = []
        doc_map = {d.id: d for d in self._documents}
        for rank, (doc_id, score) in enumerate(fused, start=1):
            doc = doc_map.get(doc_id)
            if doc is None:
                continue
            results.append(RetrievalResult(
                document=doc,
                score=score,
                rank=rank,
                metadata={
                    "method": "hybrid_alpha",
                    "alpha": self._alpha,
                    "dense_score": dense_scores.get(doc_id, 0.0),
                    "sparse_score": sparse_scores.get(doc_id, 0.0),
                },
            ))
        return results

    # ── RRF ────────────────────────────────────────────────────────────

    def reciprocal_rank_fusion(
        self,
        results_lists: List[List[RetrievalResult]],
        weights: Optional[List[float]] = None,
    ) -> List[RetrievalResult]:
        """Fuse multiple ranked lists using Reciprocal Rank Fusion.

        Parameters
        ----------
        results_lists : List[List[RetrievalResult]]
            Ranked result lists to fuse.
        weights : Optional[List[float]]
            Per-list weights.  Defaults to uniform weighting.

        Returns
        -------
        List[RetrievalResult]
            Fused results sorted by RRF score.
        """
        if weights is None:
            weights = [1.0] * len(results_lists)

        rrf_scores: Dict[str, float] = defaultdict(float)
        doc_map: Dict[str, Document] = {}

        for results, weight in zip(results_lists, weights):
            for rank, result in enumerate(results, start=1):
                doc_id = result.document.id
                doc_map[doc_id] = result.document
                rrf_scores[doc_id] += weight / (self._rrf_k + rank)

        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        fused: List[RetrievalResult] = []
        for rank, (doc_id, score) in enumerate(sorted_items, start=1):
            doc = doc_map[doc_id]
            fused.append(RetrievalResult(
                document=doc,
                score=score,
                rank=rank,
                metadata={"method": "rrf", "rrf_k": self._rrf_k},
            ))
        return fused

    def _rrf_fuse(
        self,
        dense_results: List[RetrievalResult],
        sparse_results: List[RetrievalResult],
    ) -> List[RetrievalResult]:
        return self.reciprocal_rank_fusion([dense_results, sparse_results])

    # ── Retrieval ──────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        score_threshold: float = 0.0,
    ) -> List[RetrievalResult]:
        if not self._documents:
            return []

        # Fetch more than top_k from each to improve fusion quality
        fetch_k = max(top_k * 3, 50)
        dense_results = self._dense.retrieve(query, top_k=fetch_k)
        sparse_results = self._sparse.retrieve(query, top_k=fetch_k)

        if self._fusion == "rrf":
            fused = self._rrf_fuse(dense_results, sparse_results)
        else:
            fused = self._alpha_fuse(dense_results, sparse_results)

        # Apply threshold and top_k
        if score_threshold > 0.0:
            fused = [r for r in fused if r.score >= score_threshold]

        return fused[:top_k]

    def batch_retrieve(
        self,
        queries: Sequence[str],
        top_k: int = 10,
        score_threshold: float = 0.0,
    ) -> List[List[RetrievalResult]]:
        return [self.retrieve(q, top_k, score_threshold) for q in queries]

    @property
    def document_count(self) -> int:
        return len(self._documents)


# ═══════════════════════════════════════════════════════════════════════════
#  ColBERTRetriever (Late Interaction)
# ═══════════════════════════════════════════════════════════════════════════

class ColBERTRetriever(BaseRetriever):
    """ColBERT-style late interaction retrieval.

    Instead of producing a single vector per document, ColBERT produces
    a token-level embedding matrix (L_q × D).  The relevance score between
    a query and a document is the sum of the maximum similarity of each
    query token to any document token (MaxSim).

    This implementation uses a simple hashing-based encoder for
    token-level vectors, producing deterministic embeddings without
    requiring a model download.

    Parameters
    ----------
    dimension : int
        Per-token embedding dimensionality.
    max_tokens : int
        Maximum number of tokens per sequence (truncate/pad).
    score_scale : float
        Temperature scaling for MaxSim scores.
    top_k : int
        Default number of results.
    score_threshold : float
        Minimum score threshold.
    device : str
        Compute device.
    """

    def __init__(
        self,
        dimension: int = 128,
        max_tokens: int = 256,
        score_scale: float = 1.0,
        top_k: int = 10,
        score_threshold: float = 0.0,
        device: str = "cpu",
    ) -> None:
        self._dimension = dimension
        self._max_tokens = max_tokens
        self._score_scale = score_scale
        self._default_top_k = top_k
        self._default_threshold = score_threshold
        self._device = device
        self._encoder = _ColBERTEncoder(dimension)
        self._documents: List[Document] = []
        self._doc_embeddings: List[np.ndarray] = []

    def encode_for_colbert(
        self,
        texts: Sequence[str],
        is_query: bool = False,
    ) -> List[np.ndarray]:
        """Encode texts into token-level embedding matrices (L × D)."""
        prefix = "Q: " if is_query else "D: "
        results: List[np.ndarray] = []
        for text in texts:
            tokens = self._encoder.tokenize(prefix + text)[: self._max_tokens]
            embeds = self._encoder.encode_tokens(tokens)
            results.append(embeds)
        return results

    def max_sim_score(self, query_embeds: np.ndarray, doc_embeds: np.ndarray) -> float:
        """Compute the MaxSim score between query and document embeddings.

        Parameters
        ----------
        query_embeds : np.ndarray
            Query token embeddings (L_q × D).
        doc_embeds : np.ndarray
            Document token embeddings (L_d × D).

        Returns
        -------
        float
            MaxSim score = Σ_q max_d sim(q_i, d_j).
        """
        if query_embeds.shape[0] == 0 or doc_embeds.shape[0] == 0:
            return 0.0

        # Normalise for cosine similarity
        q_norm = np.linalg.norm(query_embeds, axis=1, keepdims=True)
        q_norm = np.maximum(q_norm, 1e-12)
        query_embeds = query_embeds / q_norm

        d_norm = np.linalg.norm(doc_embeds, axis=1, keepdims=True)
        d_norm = np.maximum(d_norm, 1e-12)
        doc_embeds = doc_embeds / d_norm

        # sim matrix: (L_q × L_d)
        sim = query_embeds @ doc_embeds.T

        # MaxSim: for each query token, take max similarity to any doc token
        max_sims = np.max(sim, axis=1)
        score = float(np.sum(max_sims)) * self._score_scale
        return score

    def index(self, documents: Sequence[Document]) -> None:
        self._documents = list(documents)
        self._doc_embeddings = []
        logger.info("ColBERTRetriever: encoding %d documents …", len(self._documents))
        t0 = time.perf_counter()
        for doc in self._documents:
            embeds = self.encode_for_colbert([doc.content], is_query=False)[0]
            self._doc_embeddings.append(embeds)
        elapsed = time.perf_counter() - t0
        logger.info(
            "ColBERTRetriever: indexed %d docs in %.2f s",
            len(self._documents), elapsed,
        )

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> List[RetrievalResult]:
        if not self._documents:
            return []
        k = top_k if top_k is not None else self._default_top_k
        threshold = score_threshold if score_threshold is not None else self._default_threshold

        query_embeds = self.encode_for_colbert([query], is_query=True)[0]

        scores: List[Tuple[int, float]] = []
        for idx, doc_embeds in enumerate(self._doc_embeddings):
            score = self.max_sim_score(query_embeds, doc_embeds)
            scores.append((idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)

        results: List[RetrievalResult] = []
        rank = 0
        for idx, score in scores:
            if score < threshold:
                continue
            rank += 1
            if rank > k:
                break
            results.append(RetrievalResult(
                document=self._documents[idx],
                score=score,
                rank=rank,
                metadata={"method": "colbert_maxsim"},
            ))
        return results

    @property
    def document_count(self) -> int:
        return len(self._documents)


class _ColBERTEncoder:
    """Simple token-level encoder for ColBERT."""

    def __init__(self, dimension: int = 128) -> None:
        self._dim = dimension
        self._rng = np.random.RandomState(42)

    def tokenize(self, text: str) -> List[str]:
        return re.findall(r"\S+", text.lower())

    def encode_tokens(self, tokens: List[str]) -> np.ndarray:
        """Encode a list of tokens to an embedding matrix (L × D)."""
        if not tokens:
            return np.zeros((0, self._dim), dtype=np.float32)
        embeds = np.zeros((len(tokens), self._dim), dtype=np.float32)
        for i, token in enumerate(tokens):
            raw = token.encode("utf-8")
            for j in range(self._dim):
                h = hashlib.sha256(f"colbert:{j}:{token}".encode()).hexdigest()
                val = int(h[:8], 16) / 0xFFFFFFFF - 0.5
                embeds[i, j] = val
            norm = np.linalg.norm(embeds[i])
            if norm > 1e-12:
                embeds[i] /= norm
        return embeds


# ═══════════════════════════════════════════════════════════════════════════
#  MultiVectorRetriever
# ═══════════════════════════════════════════════════════════════════════════

class MultiVectorRetriever(BaseRetriever):
    """Retrieve using multiple vector representations per document.

    Each document can be associated with multiple vectors (e.g. title
    embedding, summary embedding, chunk embeddings).  At query time, the
    best score across all vectors for each document is used for ranking.

    Parameters
    ----------
    dimension : int
        Embedding dimensionality.
    encoder : Optional[Callable]
        Custom encoder.  Falls back to :class:`SimpleEncoder`.
    metric : str
        Distance metric.
    aggregation : str
        How to aggregate per-document multi-vector scores:
        ``"max"``, ``"mean"``, or ``"sum"``.
    """

    def __init__(
        self,
        dimension: int = 256,
        encoder: Optional[Callable[[Sequence[str]], np.ndarray]] = None,
        metric: str = "cosine",
        aggregation: str = "max",
    ) -> None:
        self._dimension = dimension
        self._encoder = encoder or SimpleEncoder(dimension=dimension)
        self._metric = RetrievalMetric(metric)
        self._aggregation = aggregation
        self._documents: List[Document] = []
        self._doc_vectors: Dict[str, np.ndarray] = {}
        self._all_embeddings: Optional[np.ndarray] = None
        self._all_doc_ids: List[str] = []
        self._all_doc_indices: List[int] = []

    def index(self, documents: Sequence[Document]) -> None:
        self._documents = list(documents)
        self._doc_vectors = {}
        self._all_doc_ids = []
        self._all_doc_indices = []

        all_embeds: List[np.ndarray] = []
        for i, doc in enumerate(self._documents):
            vectors = self._compute_doc_vectors(doc)
            self._doc_vectors[doc.id] = vectors
            for vec in vectors:
                all_embeds.append(vec)
                self._all_doc_ids.append(doc.id)
                self._all_doc_indices.append(i)

        if all_embeds:
            self._all_embeddings = np.vstack(all_embeds)
        else:
            self._all_embeddings = None

    def _compute_doc_vectors(self, doc: Document) -> np.ndarray:
        """Compute multiple vectors for a single document."""
        parts: List[str] = []
        if doc.title:
            parts.append(doc.title)
        parts.append(doc.content)

        content = doc.content
        if len(content) > 200:
            parts.append(content[:200])
            parts.append(content[len(content) // 2 - 100 : len(content) // 2 + 100])
            parts.append(content[-200:])

        embeds = self._encoder.encode(parts)
        return embeds

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        score_threshold: float = 0.0,
    ) -> List[RetrievalResult]:
        if self._all_embeddings is None or len(self._documents) == 0:
            return []

        query_vec = self._encoder.encode([query])[0]
        scores = compute_scores(query_vec, self._all_embeddings, self._metric)

        # Aggregate scores per document
        doc_scores: Dict[str, List[float]] = defaultdict(list)
        for i, score in enumerate(scores):
            doc_id = self._all_doc_ids[i]
            doc_scores[doc_id].append(float(score))

        aggregated: Dict[str, float] = {}
        for doc_id, score_list in doc_scores.items():
            if self._aggregation == "max":
                aggregated[doc_id] = max(score_list)
            elif self._aggregation == "mean":
                aggregated[doc_id] = sum(score_list) / len(score_list)
            elif self._aggregation == "sum":
                aggregated[doc_id] = sum(score_list)
            else:
                aggregated[doc_id] = max(score_list)

        sorted_docs = sorted(aggregated.items(), key=lambda x: x[1], reverse=True)

        doc_map = {d.id: d for d in self._documents}
        results: List[RetrievalResult] = []
        rank = 0
        for doc_id, score in sorted_docs:
            if score < score_threshold:
                continue
            rank += 1
            if rank > top_k:
                break
            results.append(RetrievalResult(
                document=doc_map[doc_id],
                score=score,
                rank=rank,
                metadata={"method": "multi_vector", "aggregation": self._aggregation},
            ))
        return results

    @property
    def document_count(self) -> int:
        return len(self._documents)


# ═══════════════════════════════════════════════════════════════════════════
#  CrossEncoderReranker (lightweight — full version lives in reranking.py)
# ═══════════════════════════════════════════════════════════════════════════

class CrossEncoderReranker:
    """Lightweight cross-encoder reranker using character n-gram overlap.

    Computes a relevance score by combining:
    * Character n-gram overlap (Jaccard)
    * Token overlap (Jaccard)
    * Exact match bonus
    * Length normalisation

    For production use, swap this for a transformer-based cross-encoder.

    Parameters
    ----------
    top_k : int
        Number of documents to return after reranking.
    batch_size : int
        Batch size for scoring.
    """

    def __init__(self, top_k: int = 10, batch_size: int = 32) -> None:
        self._top_k = top_k
        self._batch_size = batch_size

    @staticmethod
    def _char_ngrams(text: str, n: int = 3) -> Set[str]:
        text = text.lower().strip()
        return {text[i : i + n] for i in range(max(0, len(text) - n + 1))}

    @staticmethod
    def _tokenize(text: str) -> Set[str]:
        return set(re.findall(r"[a-z0-9]+", text.lower()))

    def score(self, query: str, document: str) -> float:
        """Score a single (query, document) pair."""
        q_ngrams = self._char_ngrams(query, n=3)
        d_ngrams = self._char_ngrams(document, n=3)

        ngram_jaccard = 0.0
        if q_ngrams or d_ngrams:
            ngram_jaccard = len(q_ngrams & d_ngrams) / len(q_ngrams | d_ngrams)

        q_tokens = self._tokenize(query)
        d_tokens = self._tokenize(document)

        token_jaccard = 0.0
        if q_tokens or d_tokens:
            token_jaccard = len(q_tokens & d_tokens) / len(q_tokens | d_tokens)

        exact_match = 0.0
        if any(t in document.lower() for t in q_tokens):
            exact_match = 0.3

        length_norm = min(1.0, len(query) / max(len(document), 1))

        total = 0.4 * ngram_jaccard + 0.3 * token_jaccard + exact_match + 0.1 * length_norm
        return min(total, 1.0)

    def rerank(
        self,
        query: str,
        documents: Sequence[Union[str, Document]],
        top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """Rerank *documents* for *query* and return top-k."""
        k = top_k if top_k is not None else self._top_k
        scored: List[Tuple[int, float]] = []
        for i, doc in enumerate(documents):
            text = doc.content if isinstance(doc, Document) else doc
            s = self.score(query, text)
            scored.append((i, s))

        scored.sort(key=lambda x: x[1], reverse=True)

        results: List[RetrievalResult] = []
        for rank, (idx, s) in enumerate(scored[:k], start=1):
            doc = documents[idx]
            if isinstance(doc, str):
                doc = Document(content=doc)
            results.append(RetrievalResult(
                document=doc,
                score=s,
                rank=rank,
                metadata={"method": "cross_encoder_rerank"},
            ))
        return results

    def batch_rerank(
        self,
        queries: Sequence[str],
        doc_lists: Sequence[Sequence[Union[str, Document]]],
        top_k: Optional[int] = None,
    ) -> List[List[RetrievalResult]]:
        """Rerank for multiple query/document-list pairs."""
        return [self.rerank(q, docs, top_k) for q, docs in zip(queries, doc_lists)]


# ═══════════════════════════════════════════════════════════════════════════
#  RetrievalPipeline
# ═══════════════════════════════════════════════════════════════════════════

class RetrievalPipeline:
    """End-to-end retrieval pipeline: encode → retrieve → rerank.

    Coordinates a retriever and an optional reranker to produce the final
    ranked result list.  Supports pre/post-processing hooks for logging,
    caching, and instrumentation.

    Parameters
    ----------
    retriever : BaseRetriever
        The primary retriever.
    reranker : Optional[CrossEncoderReranker]
        Post-retrieval reranker.
    pre_hooks : Optional[List[Callable]]
        Functions called with ``(query, documents)`` before retrieval.
    post_hooks : Optional[List[Callable]]
        Functions called with ``(query, results)`` after reranking.
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        reranker: Optional[CrossEncoderReranker] = None,
        pre_hooks: Optional[List[Callable]] = None,
        post_hooks: Optional[List[Callable]] = None,
    ) -> None:
        self._retriever = retriever
        self._reranker = reranker
        self._pre_hooks = pre_hooks or []
        self._post_hooks = post_hooks or []
        self._query_count = 0
        self._total_latency = 0.0

    def index(self, documents: Sequence[Document]) -> None:
        """Index documents through the pipeline."""
        self._retriever.index(documents)

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        score_threshold: float = 0.0,
        rerank_top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """Full pipeline: retrieve → rerank → return."""
        t0 = time.perf_counter()
        self._query_count += 1

        # Pre-hooks
        for hook in self._pre_hooks:
            hook(query, self._retriever._documents if hasattr(self._retriever, "_documents") else [])

        # Retrieve more documents than final top_k for reranking
        fetch_k = top_k * 3 if self._reranker is not None else top_k
        results = self._retriever.retrieve(query, top_k=fetch_k, score_threshold=score_threshold)

        # Rerank
        if self._reranker is not None and results:
            docs = [r.document for r in results]
            rk = rerank_top_k if rerank_top_k is not None else top_k
            results = self._reranker.rerank(query, docs, top_k=rk)

        # Post-hooks
        for hook in self._post_hooks:
            hook(query, results)

        elapsed = time.perf_counter() - t0
        self._total_latency += elapsed

        return results

    def batch_retrieve(
        self,
        queries: Sequence[str],
        top_k: int = 10,
        score_threshold: float = 0.0,
    ) -> List[List[RetrievalResult]]:
        return [self.retrieve(q, top_k, score_threshold) for q in queries]

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "query_count": self._query_count,
            "total_latency_seconds": round(self._total_latency, 3),
            "avg_latency_seconds": round(self._total_latency / max(self._query_count, 1), 4),
            "document_count": self._retriever.document_count,
            "reranker_enabled": self._reranker is not None,
        }


# ═══════════════════════════════════════════════════════════════════════════
#  Factory helpers
# ═══════════════════════════════════════════════════════════════════════════

def create_retriever(
    method: str = "dense",
    dimension: int = 256,
    **kwargs: Any,
) -> BaseRetriever:
    """Factory function to create a retriever by name.

    Parameters
    ----------
    method : str
        ``"dense"``, ``"sparse"``, ``"hybrid"``, ``"colbert"``, or ``"multi_vector"``.
    dimension : int
        Embedding dimensionality (for dense-based retrievers).
    **kwargs
        Forwarded to the retriever constructor.

    Returns
    -------
    BaseRetriever
    """
    method = method.lower().strip()
    if method == "dense":
        return DenseRetriever(dimension=dimension, **kwargs)
    elif method == "sparse":
        return SparseRetriever(**kwargs)
    elif method == "hybrid":
        dense_kwargs = {k: v for k, v in kwargs.items() if k in ("dimension", "encoder", "metric", "device", "batch_size", "top_k", "score_threshold", "max_docs", "normalize")}
        sparse_kwargs = {k: v for k, v in kwargs.items() if k in ("k1", "b", "epsilon", "top_k", "score_threshold", "stopwords", "min_token_length")}
        dense = DenseRetriever(dimension=dimension, **dense_kwargs)
        sparse = SparseRetriever(**sparse_kwargs)
        alpha = kwargs.get("alpha", 0.7)
        return HybridRetriever(dense, sparse, alpha=alpha)
    elif method == "colbert":
        return ColBERTRetriever(dimension=kwargs.get("colbert_dim", dimension), **{k: v for k, v in kwargs.items() if k != "colbert_dim"})
    elif method == "multi_vector":
        return MultiVectorRetriever(dimension=dimension, **kwargs)
    else:
        raise ValueError(f"Unknown retrieval method: {method!r}. "
                         f"Choose from: dense, sparse, hybrid, colbert, multi_vector")
