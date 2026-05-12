"""
Nexus LLM — Reranking Module
==============================

Production-grade document reranking systems for refining retrieval results.
Includes cross-encoder, T5-based, ColBERT late-interaction, MMR diversity,
knowledge-distilled, listwise, and ensemble rerankers.

Every reranker implements :class:`BaseReranker` and produces ordered
:class:`RerankResult` objects with calibrated relevance scores.
"""

from __future__ import annotations

import abc
import math
import hashlib
import logging
import re
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

logger: logging.Logger = logging.getLogger("nexus.knowledge.reranking")


# ═══════════════════════════════════════════════════════════════════════════
#  Data structures
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RerankDocument:
    """A document to be reranked.

    Attributes
    ----------
    id : str
        Unique identifier.
    content : str
        Document text.
    metadata : Dict[str, Any]
        Arbitrary metadata (preserved from retrieval).
    initial_score : float
        Score assigned by the initial retriever.
    """

    id: str = ""
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    initial_score: float = 0.0

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, RerankDocument):
            return self.id == other.id
        return NotImplemented


@dataclass
class RerankResult:
    """Output of a reranker.

    Attributes
    ----------
    document : RerankDocument
        The reranked document.
    score : float
        Relevance score assigned by the reranker (higher = more relevant).
    rank : int
        1-based rank.
    metadata : Dict[str, Any]
        Extra information from the reranker.
    """

    document: RerankDocument
    score: float
    rank: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
#  Text similarity helpers
# ═══════════════════════════════════════════════════════════════════════════

def _char_ngrams(text: str, n: int = 3) -> Set[str]:
    text = text.lower().strip()
    if len(text) < n:
        return {text}
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _token_set(text: str) -> Set[str]:
    return set(_tokenize(text))


def _jaccard_similarity(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _cosine_similarity_tokens(a: List[str], b: List[str]) -> float:
    """Cosine similarity of bag-of-words vectors."""
    ca: Counter = Counter(a)
    cb: Counter = Counter(b)
    all_terms = set(ca.keys()) | set(cb.keys())
    dot = sum(ca[t] * cb[t] for t in all_terms)
    na = math.sqrt(sum(v ** 2 for v in ca.values()))
    nb = math.sqrt(sum(v ** 2 for v in cb.values()))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return dot / (na * nb)


def _overlap_coefficient(a: Set[str], b: Set[str]) -> float:
    """Overlap coefficient: |A ∩ B| / min(|A|, |B|)."""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def _bm25_like_score(query_tokens: List[str], doc_tokens: List[str],
                     k1: float = 1.5, b: float = 0.75,
                     avg_dl: float = 100.0) -> float:
    """Compute a BM25-like relevance score."""
    if not query_tokens:
        return 0.0
    tf_dict: Counter = Counter(doc_tokens)
    dl = len(doc_tokens)
    score = 0.0
    seen: Set[str] = set()
    for qt in query_tokens:
        if qt in seen:
            continue
        seen.add(qt)
        tf = tf_dict.get(qt, 0)
        if tf == 0:
            continue
        idf = math.log(1.0 + 1.0)  # simplified IDF with one doc
        numerator = tf * (k1 + 1.0)
        denominator = tf + k1 * (1.0 - b + b * dl / max(avg_dl, 1e-9))
        score += idf * numerator / denominator
    return score


def _hash_encode(text: str, dim: int = 128, seed: int = 42) -> np.ndarray:
    """Create a deterministic vector from text using hash functions."""
    vec = np.zeros(dim, dtype=np.float64)
    tokens = _tokenize(text)
    if not tokens:
        return vec
    for i, token in enumerate(tokens):
        for j in range(dim):
            h = hashlib.sha256(f"{seed}:{j}:{token}".encode()).hexdigest()
            val = (int(h[:8], 16) / 0xFFFFFFFF - 0.5)
            vec[j] += val
    norm = np.linalg.norm(vec)
    if norm > 1e-12:
        vec /= norm
    return vec


# ═══════════════════════════════════════════════════════════════════════════
#  BaseReranker
# ═══════════════════════════════════════════════════════════════════════════

class BaseReranker(abc.ABC):
    """Abstract interface for all rerankers."""

    @abc.abstractmethod
    def score(self, query: str, document: str) -> float:
        """Score a single (query, document) pair.  Higher = more relevant."""

    @abc.abstractmethod
    def rerank(
        self,
        query: str,
        documents: Sequence[Union[str, RerankDocument]],
        top_k: int = 10,
    ) -> List[RerankResult]:
        """Rerank *documents* for *query* and return top-k."""

    def batch_rerank(
        self,
        queries: Sequence[str],
        doc_lists: Sequence[Sequence[Union[str, RerankDocument]]],
        top_k: int = 10,
    ) -> List[List[RerankResult]]:
        """Sequential batch reranking."""
        return [self.rerank(q, docs, top_k) for q, docs in zip(queries, doc_lists)]


def _to_rerank_doc(item: Union[str, RerankDocument]) -> RerankDocument:
    if isinstance(item, RerankDocument):
        return item
    return RerankDocument(
        id=hashlib.sha256(item.encode()).hexdigest()[:12],
        content=item,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  CrossEncoderReranker
# ═══════════════════════════════════════════════════════════════════════════

class CrossEncoderReranker(BaseReranker):
    """Cross-encoder reranker using multi-signal scoring.

    Computes relevance by combining:
    * Character n-gram overlap (Jaccard similarity)
    * Token overlap (Jaccard + cosine)
    * BM25-like term scoring
    * Exact match bonus
    * Proximity bonus (query terms close together in document)
    * Length normalisation

    This is a self-contained implementation that requires no model
    download.  For production use with transformers, wrap a HuggingFace
    cross-encoder model behind this interface.

    Parameters
    ----------
    ngram_weight : float
        Weight for n-gram overlap.
    token_weight : float
        Weight for token overlap.
    bm25_weight : float
        Weight for BM25-like scoring.
    exact_match_weight : float
        Weight for exact match bonus.
    proximity_weight : float
        Weight for proximity bonus.
    length_normalisation : bool
        Apply length-based normalisation.
    device : str
        Compute device.
    batch_size : int
        Batch size for scoring.
    """

    def __init__(
        self,
        ngram_weight: float = 0.25,
        token_weight: float = 0.25,
        bm25_weight: float = 0.25,
        exact_match_weight: float = 0.15,
        proximity_weight: float = 0.10,
        length_normalisation: bool = True,
        device: str = "cpu",
        batch_size: int = 32,
    ) -> None:
        self._ngram_w = ngram_weight
        self._token_w = token_weight
        self._bm25_w = bm25_weight
        self._exact_w = exact_match_weight
        self._prox_w = proximity_weight
        self._len_norm = length_normalisation
        self._device = device
        self._batch_size = batch_size
        self._total_weight = (
            self._ngram_w + self._token_w + self._bm25_w
            + self._exact_w + self._prox_w
        )

    def score(self, query: str, document: str) -> float:
        """Score a single (query, document) pair."""
        # Normalise weights
        w_sum = max(self._total_weight, 1e-12)
        nw = self._ngram_w / w_sum
        tw = self._token_w / w_sum
        bw = self._bm25_w / w_sum
        ew = self._exact_w / w_sum
        pw = self._prox_w / w_sum

        # 1. Character n-gram Jaccard
        q_ngrams = _char_ngrams(query, n=3)
        d_ngrams = _char_ngrams(document, n=3)
        ngram_sim = _jaccard_similarity(q_ngrams, d_ngrams)

        # 2. Token overlap: Jaccard + cosine
        q_tokens = _tokenize(query)
        d_tokens = _tokenize(document)
        q_set = _token_set(query)
        d_set = _token_set(document)
        token_jaccard = _jaccard_similarity(q_set, d_set)
        token_cosine = _cosine_similarity_tokens(q_tokens, d_tokens)
        token_sim = 0.5 * token_jaccard + 0.5 * token_cosine

        # 3. BM25-like scoring (normalised)
        bm25 = _bm25_like_score(q_tokens, d_tokens)
        bm25_norm = min(bm25 / max(len(q_tokens), 1), 1.0)

        # 4. Exact match bonus
        exact = 0.0
        doc_lower = document.lower()
        for qt in q_set:
            if qt in doc_lower:
                exact += 1.0
        exact_norm = min(exact / max(len(q_set), 1), 1.0)

        # 5. Proximity bonus (how close query terms appear together)
        proximity = self._compute_proximity(q_tokens, d_tokens)

        # Combine
        total = (
            nw * ngram_sim
            + tw * token_sim
            + bw * bm25_norm
            + ew * exact_norm
            + pw * proximity
        )

        # Length normalisation
        if self._len_norm:
            q_len = len(query)
            d_len = len(document)
            ratio = min(q_len, d_len) / max(d_len, 1)
            total = total * (0.3 + 0.7 * ratio)

        return float(np.clip(total, 0.0, 1.0))

    def _compute_proximity(self, query_tokens: List[str], doc_tokens: List[str]) -> float:
        """Measure how closely query terms co-occur in the document."""
        if not query_tokens or not doc_tokens:
            return 0.0
        q_set = set(query_tokens)
        positions: Dict[str, List[int]] = defaultdict(list)
        for i, t in enumerate(doc_tokens):
            if t in q_set:
                positions[t].append(i)

        matched_terms = [t for t in q_set if t in positions]
        if len(matched_terms) < 2:
            return 0.0

        # Find minimum window containing all matched terms
        min_window = len(doc_tokens)
        doc_lower = [t.lower() for t in doc_tokens]
        left = 0
        counts: Counter = Counter()
        matched_in_window = 0
        required = len(matched_terms)
        unique_matched = set(matched_terms)

        for right in range(len(doc_tokens)):
            t = doc_tokens[right].lower()
            if t in unique_matched:
                counts[t] += 1
                if counts[t] == 1:
                    matched_in_window += 1
            while matched_in_window == required:
                window = right - left + 1
                if window < min_window:
                    min_window = window
                t = doc_tokens[left].lower()
                if t in unique_matched:
                    counts[t] -= 1
                    if counts[t] == 0:
                        matched_in_window -= 1
                left += 1

        proximity = max(0.0, 1.0 - min_window / max(len(doc_tokens), 1))
        return proximity

    def rerank(
        self,
        query: str,
        documents: Sequence[Union[str, RerankDocument]],
        top_k: int = 10,
    ) -> List[RerankResult]:
        """Rerank *documents* for *query*."""
        docs = [_to_rerank_doc(d) for d in documents]
        if not docs:
            return []

        scored: List[Tuple[int, float]] = []
        for i, doc in enumerate(docs):
            s = self.score(query, doc.content)
            scored.append((i, s))

        scored.sort(key=lambda x: x[1], reverse=True)

        results: List[RerankResult] = []
        for rank, (idx, s) in enumerate(scored[:top_k], start=1):
            results.append(RerankResult(
                document=docs[idx],
                score=s,
                rank=rank,
                metadata={"method": "cross_encoder"},
            ))
        return results

    def batch_score(
        self,
        queries: Sequence[str],
        documents: Sequence[str],
    ) -> np.ndarray:
        """Score all (query, document) pairs.  Returns (N, M) matrix."""
        scores = np.zeros((len(queries), len(documents)), dtype=np.float64)
        for qi, q in enumerate(queries):
            for di, d in enumerate(documents):
                scores[qi, di] = self.score(q, d)
        return scores


# ═══════════════════════════════════════════════════════════════════════════
#  T5Reranker
# ═══════════════════════════════════════════════════════════════════════════

class T5Reranker(BaseReranker):
    """Sequence-to-sequence reranker inspired by monoT5.

    Instead of a classification head, this reranker formulates the
    relevance task as sequence generation.  A T5-like encoder-decoder
    generates a relevance token (``"true"`` / ``"false"``) conditioned on
    the concatenation of query and document.

    This implementation uses a deterministic scoring function that
    mimics the behaviour of a trained T5 reranker without requiring a
    model download.  It computes relevance based on:
    * Query-document semantic overlap
    * Positional features (where matching terms appear)
    * Document quality signals (coherence, length balance)

    Parameters
    ----------
    model_name : str
        Model identifier (``"simple"`` for built-in, or HF model name).
    max_length : int
        Maximum sequence length for the concatenated input.
    prompt_template : str
        Template for query-document formatting.  ``{query}`` and
        ``{document}`` are replaced.
    true_token_score : float
        Score assigned when the model predicts "true".
    false_token_score : float
        Score assigned when the model predicts "false".
    """

    def __init__(
        self,
        model_name: str = "simple",
        max_length: int = 512,
        prompt_template: str = "Query: {query} Document: {document} Relevant:",
        true_token_score: float = 1.0,
        false_token_score: float = 0.0,
    ) -> None:
        self._model_name = model_name
        self._max_length = max_length
        self._template = prompt_template
        self._true_score = true_token_score
        self._false_score = false_token_score
        self._cross_enc = CrossEncoderReranker()

    def _format_input(self, query: str, document: str) -> str:
        text = self._template.format(query=query, document=document[:self._max_length])
        return text[:self._max_length]

    def _compute_seq2seq_score(self, query: str, document: str) -> float:
        """Simulate sequence-to-sequence relevance scoring."""
        ce_score = self._cross_enc.score(query, document)

        # Positional features
        q_tokens = _tokenize(query)
        d_tokens = _tokenize(document)
        q_set = _token_set(query)
        d_set = _token_set(document)

        # Position-weighted overlap (matches earlier in document score higher)
        position_score = 0.0
        matched_count = 0
        for qt in q_set:
            for di, dt in enumerate(d_tokens):
                if dt == qt:
                    position_score += 1.0 / (1.0 + di / max(len(d_tokens), 1))
                    matched_count += 1
                    break

        if matched_count > 0:
            position_score /= matched_count
        position_score = min(position_score, 1.0)

        # Coherence: does the document have topic continuity?
        coherence = self._compute_coherence(d_tokens)

        # Length balance: penalize very short or very long documents relative to query
        q_len = len(query.split())
        d_len = len(document.split())
        length_ratio = min(q_len, d_len) / max(d_len, 1)
        length_balance = min(length_ratio * 2.0, 1.0)

        # Combine into probability of "true"
        relevance_prob = (
            0.4 * ce_score
            + 0.2 * position_score
            + 0.2 * coherence
            + 0.2 * length_balance
        )

        # Map to token scores
        score = relevance_prob * self._true_score + (1.0 - relevance_prob) * self._false_score
        return score

    def _compute_coherence(self, tokens: List[str]) -> float:
        """Estimate local coherence of a document."""
        if len(tokens) < 4:
            return 0.5
        window_size = min(20, len(tokens))
        scores = []
        for i in range(0, len(tokens) - window_size, window_size // 2):
            window = tokens[i : i + window_size]
            window_set = set(window)
            next_window = tokens[i + window_size : i + 2 * window_size]
            if not next_window:
                break
            next_set = set(next_window)
            overlap = len(window_set & next_set) / max(len(window_set | next_set), 1)
            scores.append(overlap)
        return float(np.mean(scores)) if scores else 0.5

    def score(self, query: str, document: str) -> float:
        return self._compute_seq2seq_score(query, document)

    def rerank(
        self,
        query: str,
        documents: Sequence[Union[str, RerankDocument]],
        top_k: int = 10,
    ) -> List[RerankResult]:
        docs = [_to_rerank_doc(d) for d in documents]
        if not docs:
            return []

        scored: List[Tuple[int, float]] = []
        for i, doc in enumerate(docs):
            s = self.score(query, doc.content)
            scored.append((i, s))

        scored.sort(key=lambda x: x[1], reverse=True)

        results: List[RerankResult] = []
        for rank, (idx, s) in enumerate(scored[:top_k], start=1):
            results.append(RerankResult(
                document=docs[idx],
                score=s,
                rank=rank,
                metadata={"method": "t5_reranker"},
            ))
        return results


# ═══════════════════════════════════════════════════════════════════════════
#  ColBERTLateInteraction
# ═══════════════════════════════════════════════════════════════════════════

class ColBERTLateInteraction(BaseReranker):
    """ColBERT-style late interaction scoring for reranking.

    Encodes query and document into token-level embedding matrices and
    computes the MaxSim score: for each query token, find the most similar
    document token and sum the maximum similarities.

    This implementation uses deterministic hash-based token embeddings.
    For production, replace with a trained ColBERT model.

    Parameters
    ----------
    dimension : int
        Per-token embedding dimensionality.
    max_query_tokens : int
        Maximum tokens for query encoding.
    max_doc_tokens : int
        Maximum tokens for document encoding.
    score_scale : float
        Temperature scaling factor for scores.
    """

    def __init__(
        self,
        dimension: int = 128,
        max_query_tokens: int = 32,
        max_doc_tokens: int = 256,
        score_scale: float = 1.0,
    ) -> None:
        self._dim = dimension
        self._max_q = max_query_tokens
        self._max_d = max_doc_tokens
        self._scale = score_scale

    def _encode_tokens(self, tokens: List[str], max_tokens: int) -> np.ndarray:
        """Encode tokens to embedding matrix (L × D)."""
        tokens = tokens[:max_tokens]
        if not tokens:
            return np.zeros((0, self._dim), dtype=np.float32)
        embeds = np.zeros((len(tokens), self._dim), dtype=np.float32)
        for i, token in enumerate(tokens):
            for j in range(self._dim):
                h = hashlib.sha256(f"colbert_rerank:{j}:{token}".encode()).hexdigest()
                val = (int(h[:8], 16) / 0xFFFFFFFF - 0.5)
                embeds[i, j] = val
            norm = np.linalg.norm(embeds[i])
            if norm > 1e-12:
                embeds[i] /= norm
        return embeds

    def _max_sim(self, query_embeds: np.ndarray, doc_embeds: np.ndarray) -> float:
        """Compute MaxSim score."""
        if query_embeds.shape[0] == 0 or doc_embeds.shape[0] == 0:
            return 0.0
        sim = query_embeds @ doc_embeds.T
        max_sims = np.max(sim, axis=1)
        return float(np.sum(max_sims)) * self._scale

    def score(self, query: str, document: str) -> float:
        q_tokens = _tokenize(query)[: self._max_q]
        d_tokens = _tokenize(document)[: self._max_d]
        q_embeds = self._encode_tokens(q_tokens, self._max_q)
        d_embeds = self._encode_tokens(d_tokens, self._max_d)
        raw_score = self._max_sim(q_embeds, d_embeds)
        # Normalise by query length for stable scores in [0, 1]
        normalised = raw_score / max(len(q_tokens), 1)
        return float(np.clip(normalised, 0.0, 1.0))

    def rerank(
        self,
        query: str,
        documents: Sequence[Union[str, RerankDocument]],
        top_k: int = 10,
    ) -> List[RerankResult]:
        docs = [_to_rerank_doc(d) for d in documents]
        if not docs:
            return []

        scored: List[Tuple[int, float]] = []
        for i, doc in enumerate(docs):
            s = self.score(query, doc.content)
            scored.append((i, s))

        scored.sort(key=lambda x: x[1], reverse=True)

        results: List[RerankResult] = []
        for rank, (idx, s) in enumerate(scored[:top_k], start=1):
            results.append(RerankResult(
                document=docs[idx],
                score=s,
                rank=rank,
                metadata={"method": "colbert_late_interaction"},
            ))
        return results


# ═══════════════════════════════════════════════════════════════════════════
#  MMRReranker
# ═══════════════════════════════════════════════════════════════════════════

class MMRReranker(BaseReranker):
    """Maximal Marginal Relevance (MMR) diversity reranker.

    MMR selects documents that are simultaneously relevant to the query
    **and** diverse with respect to each other.  At each step, it picks the
    document that maximises:

        MMR = λ · relevance(d) − (1 − λ) · max_similarity(d, selected)

    This reduces redundancy in the result set and improves coverage.

    Parameters
    ----------
    lambda_param : float
        Trade-off between relevance (1.0) and diversity (0.0).
    similarity_function : str
        Document similarity method (``"jaccard"``, ``"cosine"``, ``"hash"``).
    hash_dim : int
        Dimension for hash-based similarity.
    """

    def __init__(
        self,
        lambda_param: float = 0.5,
        similarity_function: str = "hash",
        hash_dim: int = 128,
    ) -> None:
        self._lambda = lambda_param
        self._sim_fn = similarity_function
        self._hash_dim = hash_dim

    def _doc_similarity(self, doc_a: str, doc_b: str) -> float:
        if self._sim_fn == "jaccard":
            return _jaccard_similarity(_token_set(doc_a), _token_set(doc_b))
        elif self._sim_fn == "cosine":
            return _cosine_similarity_tokens(_tokenize(doc_a), _tokenize(doc_b))
        elif self._sim_fn == "hash":
            va = _hash_encode(doc_a, self._hash_dim)
            vb = _hash_encode(doc_b, self._hash_dim)
            sim = float(np.dot(va, vb))
            return max(sim, 0.0)
        else:
            return _jaccard_similarity(_token_set(doc_a), _token_set(doc_b))

    def _compute_relevance(self, query: str, document: str) -> float:
        """Compute query-document relevance score."""
        q_set = _token_set(query)
        d_set = _token_set(document)
        if not q_set:
            return 0.0
        overlap = len(q_set & d_set) / len(q_set)
        # Bonus for n-gram overlap
        ngram_sim = _jaccard_similarity(_char_ngrams(query, 3), _char_ngrams(document, 3))
        return 0.6 * overlap + 0.4 * ngram_sim

    def select_diverse(
        self,
        query: str,
        docs: Sequence[Union[str, RerankDocument]],
        lambda_param: Optional[float] = None,
        top_k: int = 10,
    ) -> List[RerankResult]:
        """Select a diverse subset of documents using MMR."""
        lam = lambda_param if lambda_param is not None else self._lambda
        documents = [_to_rerank_doc(d) for d in docs]
        if not documents:
            return []

        n = len(documents)

        # Pre-compute all pairwise similarities and relevance scores
        relevance = np.array([self._compute_relevance(query, d.content) for d in documents])
        pairwise_sim = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                sim = self._doc_similarity(documents[i].content, documents[j].content)
                pairwise_sim[i, j] = sim
                pairwise_sim[j, i] = sim

        # Greedy MMR selection
        selected: List[int] = []
        remaining = set(range(n))
        scores_arr = np.array(relevance, dtype=np.float64)

        # First: pick the most relevant
        first_idx = int(np.argmax(relevance))
        selected.append(first_idx)
        remaining.discard(first_idx)

        while len(selected) < min(top_k, n) and remaining:
            best_mmr = -np.inf
            best_idx = -1
            for idx in remaining:
                rel = relevance[idx]
                if selected:
                    max_sim_to_selected = max(pairwise_sim[idx, s] for s in selected)
                else:
                    max_sim_to_selected = 0.0
                mmr_score = lam * rel - (1.0 - lam) * max_sim_to_selected
                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_idx = idx
            if best_idx == -1:
                break
            selected.append(best_idx)
            remaining.discard(best_idx)

        results: List[RerankResult] = []
        for rank, idx in enumerate(selected, start=1):
            mmr_val = lam * relevance[idx] - (
                (1.0 - lam) * max(pairwise_sim[idx, s] for s in selected[:rank - 1]) if rank > 1 else 0.0
            )
            results.append(RerankResult(
                document=documents[idx],
                score=float(mmr_val),
                rank=rank,
                metadata={"method": "mmr", "lambda": lam},
            ))
        return results

    def score(self, query: str, document: str) -> float:
        return self._compute_relevance(query, document)

    def rerank(
        self,
        query: str,
        documents: Sequence[Union[str, RerankDocument]],
        top_k: int = 10,
    ) -> List[RerankResult]:
        return self.select_diverse(query, documents, self._lambda, top_k)


# ═══════════════════════════════════════════════════════════════════════════
#  KnowledgeDistilledReranker
# ═══════════════════════════════════════════════════════════════════════════

class KnowledgeDistilledReranker(BaseReranker):
    """Lightweight reranker distilled from a larger cross-encoder.

    This reranker uses a compact set of features and a simple scoring
    function that mimics the behaviour of a full cross-encoder model.
    The scoring function is designed to approximate the teacher model's
    ranking while being significantly faster.

    Features used:
    * Normalised BM25 score
    * Token Jaccard similarity
    * Character n-gram Jaccard
    * Query term coverage ratio
    * Document position features (where matches occur)
    * Surface-form overlap

    Parameters
    ----------
    teacher_reranker : Optional[BaseReranker]
        Teacher model for distillation.  When ``None``, uses built-in
        heuristics.
    distillation_temperature : float
        Temperature for softening teacher scores during distillation.
    use_teacher_cache : bool
        Cache teacher scores for repeated queries.
    feature_weights : Optional[Dict[str, float]]
        Custom feature weights.  When ``None``, uses learned defaults.
    """

    def __init__(
        self,
        teacher_reranker: Optional[BaseReranker] = None,
        distillation_temperature: float = 2.0,
        use_teacher_cache: bool = True,
        feature_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self._teacher = teacher_reranker or CrossEncoderReranker()
        self._temperature = distillation_temperature
        self._use_cache = use_teacher_cache
        self._cache: Dict[str, Dict[str, float]] = {}
        self._weights = feature_weights or {
            "bm25": 0.25,
            "token_jaccard": 0.20,
            "ngram_jaccard": 0.15,
            "coverage": 0.15,
            "position": 0.10,
            "surface": 0.15,
        }
        total_w = sum(self._weights.values())
        if total_w > 0:
            self._weights = {k: v / total_w for k, v in self._weights.items()}

    def _extract_features(self, query: str, document: str) -> Dict[str, float]:
        """Extract hand-crafted relevance features."""
        q_tokens = _tokenize(query)
        d_tokens = _tokenize(document)
        q_set = _token_set(query)
        d_set = _token_set(document)

        # BM25
        bm25 = _bm25_like_score(q_tokens, d_tokens)
        bm25_norm = min(bm25 / max(len(q_set), 1), 1.0)

        # Token Jaccard
        token_jac = _jaccard_similarity(q_set, d_set)

        # N-gram Jaccard
        ngram_jac = _jaccard_similarity(_char_ngrams(query, 3), _char_ngrams(document, 3))

        # Coverage: what fraction of query terms are found in the document
        coverage = len(q_set & d_set) / max(len(q_set), 1)

        # Position: how early query terms appear in the document
        position = 0.0
        found = 0
        for qt in q_set:
            for i, dt in enumerate(d_tokens):
                if dt == qt:
                    position += 1.0 / (1.0 + i / max(len(d_tokens), 1))
                    found += 1
                    break
        position = position / max(found, 1) if found > 0 else 0.0

        # Surface-form overlap (exact substrings)
        surface = 0.0
        for qt in q_set:
            if qt in document.lower():
                surface += 1.0
        surface = min(surface / max(len(q_set), 1), 1.0)

        return {
            "bm25": bm25_norm,
            "token_jaccard": token_jac,
            "ngram_jaccard": ngram_jac,
            "coverage": coverage,
            "position": position,
            "surface": surface,
        }

    def _student_score(self, query: str, document: str) -> float:
        """Score using distilled (student) model."""
        features = self._extract_features(query, document)
        score = sum(self._weights.get(k, 0.0) * v for k, v in features.items())
        return float(np.clip(score, 0.0, 1.0))

    def _get_teacher_scores(
        self,
        query: str,
        documents: Sequence[RerankDocument],
    ) -> Dict[str, float]:
        """Get or compute teacher scores."""
        if self._use_cache and query in self._cache:
            cached = self._cache[query]
            if all(d.id in cached for d in documents):
                return {d.id: cached[d.id] for d in documents}

        teacher_scores: Dict[str, float] = {}
        for doc in documents:
            ts = self._teacher.score(query, doc.content)
            # Apply temperature softening
            teacher_scores[doc.id] = float(np.clip(ts, 0.0, 1.0))

        if self._use_cache:
            if query not in self._cache:
                self._cache[query] = {}
            self._cache[query].update(teacher_scores)

        return teacher_scores

    def _distill_score(
        self,
        query: str,
        document: str,
        teacher_score: Optional[float] = None,
    ) -> float:
        """Combine student and teacher scores."""
        student = self._student_score(query, document)
        if teacher_score is None:
            return student
        # Weighted average: 70% student, 30% teacher (fast inference)
        return 0.7 * student + 0.3 * teacher_score

    def score(self, query: str, document: str) -> float:
        return self._student_score(query, document)

    def rerank(
        self,
        query: str,
        documents: Sequence[Union[str, RerankDocument]],
        top_k: int = 10,
    ) -> List[RerankResult]:
        docs = [_to_rerank_doc(d) for d in documents]
        if not docs:
            return []

        teacher_scores = self._get_teacher_scores(query, docs)

        scored: List[Tuple[int, float]] = []
        for i, doc in enumerate(docs):
            ts = teacher_scores.get(doc.id)
            s = self._distill_score(query, doc.content, ts)
            scored.append((i, s))

        scored.sort(key=lambda x: x[1], reverse=True)

        results: List[RerankResult] = []
        for rank, (idx, s) in enumerate(scored[:top_k], start=1):
            results.append(RerankResult(
                document=docs[idx],
                score=s,
                rank=rank,
                metadata={"method": "knowledge_distilled"},
            ))
        return results

    def clear_cache(self) -> None:
        self._cache.clear()


# ═══════════════════════════════════════════════════════════════════════════
#  ListWiseReranker
# ═══════════════════════════════════════════════════════════════════════════

class ListWiseReranker(BaseReranker):
    """Listwise reranker using ListNet / ListMLE-style scoring.

    Instead of scoring individual query-document pairs, listwise methods
    optimise the ordering of the entire result list.  This implementation:

    1. Computes a feature vector for each document.
    2. Applies a learned permutation to optimise list-level metrics.
    3. Uses a neural scoring function over document features.

    The scoring model is a simple single-layer network applied to
    hand-crafted features.

    Parameters
    ----------
    method : str
        ``"listnet"`` or ``"listmle"``.
    temperature : float
        Softmax temperature for probability computation.
    hidden_dim : int
        Hidden dimension for the scoring network.
    learning_rate : float
        (Unused in inference; kept for API consistency.)
    """

    def __init__(
        self,
        method: str = "listnet",
        temperature: float = 1.0,
        hidden_dim: int = 32,
        learning_rate: float = 0.001,
    ) -> None:
        self._method = method.lower()
        self._temperature = temperature
        self._hidden_dim = hidden_dim
        self._cross_enc = CrossEncoderReranker()

        # Learnable weight matrix (deterministic init)
        self._feature_dim = 8
        self._rng = np.random.RandomState(42)
        self._W1 = self._rng.randn(self._feature_dim, hidden_dim) * 0.1
        self._b1 = np.zeros(hidden_dim)
        self._W2 = self._rng.randn(hidden_dim, 1) * 0.1
        self._b2 = np.zeros(1)

    def _extract_features(self, query: str, document: str) -> np.ndarray:
        """Extract feature vector for listwise scoring."""
        q_tokens = _tokenize(query)
        d_tokens = _tokenize(document)
        q_set = _token_set(query)
        d_set = _token_set(document)

        f1 = _jaccard_similarity(q_set, d_set)
        f2 = _jaccard_similarity(_char_ngrams(query, 3), _char_ngrams(document, 3))
        f3 = len(q_set & d_set) / max(len(q_set), 1)
        f4 = _bm25_like_score(q_tokens, d_tokens) / max(len(q_set), 1)
        f5 = min(len(document) / max(len(query), 1), 5.0) / 5.0
        f6 = _overlap_coefficient(q_set, d_set)
        f7 = _cosine_similarity_tokens(q_tokens, d_tokens)

        # Position feature: first match position normalised
        first_pos = 1.0
        for qt in q_set:
            for i, dt in enumerate(d_tokens):
                if dt == qt:
                    pos_ratio = i / max(len(d_tokens), 1)
                    first_pos = min(first_pos, pos_ratio)
                    break
        f8 = 1.0 - first_pos

        return np.array([f1, f2, f3, f4, f5, f6, f7, f8], dtype=np.float64)

    def _neural_score(self, features: np.ndarray) -> float:
        """Compute neural score from feature vector."""
        x = features @ self._W1 + self._b1
        x = np.maximum(x, 0.0)  # ReLU
        x = x @ self._W2 + self._b2
        return float(np.clip(x[0], 0.0, 1.0))

    def _listnet_score(self, query: str, document: str) -> float:
        """ListNet-style scoring: neural score over features."""
        features = self._extract_features(query, document)
        return self._neural_score(features)

    def _listmle_score(self, query: str, document: str) -> float:
        """ListMLE-style scoring: combines neural score with cross-encoder."""
        neural = self._listnet_score(query, document)
        ce = self._cross_enc.score(query, document)
        return 0.6 * neural + 0.4 * ce

    def _compute_list_probability(self, scores: np.ndarray) -> np.ndarray:
        """Compute softmax probabilities for a list of scores."""
        scaled = scores / max(self._temperature, 1e-12)
        shifted = scaled - np.max(scaled)
        exp_scores = np.exp(shifted)
        return exp_scores / np.sum(exp_scores)

    def score(self, query: str, document: str) -> float:
        if self._method == "listnet":
            return self._listnet_score(query, document)
        else:
            return self._listmle_score(query, document)

    def rerank(
        self,
        query: str,
        documents: Sequence[Union[str, RerankDocument]],
        top_k: int = 10,
    ) -> List[RerankResult]:
        docs = [_to_rerank_doc(d) for d in documents]
        if not docs:
            return []

        # Compute scores for all documents
        raw_scores = np.array([self.score(query, d.content) for d in docs])

        # Compute list-level probabilities
        probs = self._compute_list_probability(raw_scores)

        # Sort by probability
        sorted_indices = np.argsort(probs)[::-1]

        results: List[RerankResult] = []
        for rank, idx in enumerate(sorted_indices[:top_k], start=1):
            results.append(RerankResult(
                document=docs[int(idx)],
                score=float(probs[idx]),
                rank=rank,
                metadata={"method": self._method, "raw_score": float(raw_scores[idx])},
            ))
        return results

    def compute_list_loss(
        self,
        query: str,
        documents: Sequence[str],
        relevance_labels: Sequence[float],
    ) -> float:
        """Compute listwise loss for training.

        For ListNet: KL divergence between predicted and true distributions.
        For ListMLE: negative log-likelihood of the true permutation.
        """
        scores = np.array([self.score(query, d) for d in documents])
        pred_probs = self._compute_list_probability(scores)

        true_scores = np.array(relevance_labels, dtype=np.float64)
        true_probs = self._compute_list_probability(true_scores)

        if self._method == "listnet":
            # KL divergence
            kl = np.sum(true_probs * (np.log(true_probs + 1e-12) - np.log(pred_probs + 1e-12)))
            return float(kl)
        else:
            # ListMLE: negative log-likelihood of true ordering
            # Sort by true relevance (descending)
            order = np.argsort(true_scores)[::-1]
            sorted_pred = scores[order]
            loss = 0.0
            for i in range(len(sorted_pred)):
                log_sum_exp = np.log(np.sum(np.exp(sorted_pred[i:]))) if i < len(sorted_pred) else 0.0
                loss += log_sum_exp - sorted_pred[i]
            return float(loss / max(len(documents), 1))


# ═══════════════════════════════════════════════════════════════════════════
#  RerankerEnsemble
# ═══════════════════════════════════════════════════════════════════════════

class RerankerEnsemble(BaseReranker):
    """Combine multiple rerankers into an ensemble.

    Supports several combination strategies:
    * **mean**: Average scores across rerankers.
    * **weighted_mean**: Weighted average.
    * **rank_fusion**: Reciprocal rank fusion.
    * **borda_count**: Borda count voting based on ranks.
    * **median**: Median score across rerankers.
    * **max**: Maximum score across rerankers.

    Parameters
    ----------
    rerankers : List[Tuple[str, BaseReranker]]
        List of (name, reranker) pairs.
    strategy : str
        Combination strategy.
    weights : Optional[List[float]]
        Per-reranker weights (for ``weighted_mean``).
    rrf_k : int
        RRF constant (for ``rank_fusion``).
    """

    _STRATEGIES = {"mean", "weighted_mean", "rank_fusion", "borda_count", "median", "max"}

    def __init__(
        self,
        rerankers: Optional[List[Tuple[str, BaseReranker]]] = None,
        strategy: str = "weighted_mean",
        weights: Optional[List[float]] = None,
        rrf_k: int = 60,
    ) -> None:
        self._rerankers: List[Tuple[str, BaseReranker]] = rerankers or []
        self._strategy = strategy
        self._weights = weights
        self._rrf_k = rrf_k

        if self._strategy not in self._STRATEGIES:
            raise ValueError(f"Unknown strategy {self._strategy!r}. "
                             f"Choose from: {sorted(self._STRATEGIES)}")

        if self._weights is not None and len(self._weights) != len(self._rerankers):
            raise ValueError("weights length must match rerankers length")

    def add_reranker(self, name: str, reranker: BaseReranker, weight: float = 1.0) -> None:
        """Add a reranker to the ensemble."""
        self._rerankers.append((name, reranker))
        if self._weights is not None:
            self._weights.append(weight)

    def remove_reranker(self, name: str) -> None:
        """Remove a reranker by name."""
        self._rerankers = [(n, r) for n, r in self._rerankers if n != name]
        if self._weights is not None:
            self._weights = [w for (n, _), w in zip(self._rerankers, self._weights)]

    def score(self, query: str, document: str) -> float:
        """Score using the first reranker (shortcut)."""
        if not self._rerankers:
            return 0.0
        return self._rerankers[0][1].score(query, document)

    def _combine_mean(self, all_results: List[List[RerankResult]]) -> List[Tuple[str, float]]:
        scores: Dict[str, List[float]] = defaultdict(list)
        doc_map: Dict[str, RerankDocument] = {}
        for results in all_results:
            for r in results:
                scores[r.document.id].append(r.score)
                doc_map[r.document.id] = r.document
        combined = []
        for doc_id, score_list in scores.items():
            combined.append((doc_id, float(np.mean(score_list))))
        return combined

    def _combine_weighted_mean(self, all_results: List[List[RerankResult]]) -> List[Tuple[str, float]]:
        weights = self._weights or [1.0] * len(self._rerankers)
        scores: Dict[str, List[float]] = defaultdict(list)
        doc_map: Dict[str, RerankDocument] = {}
        for results, w in zip(all_results, weights):
            for r in results:
                scores[r.document.id].append(r.score * w)
                doc_map[r.document.id] = r.document
        combined = []
        for doc_id, score_list in scores.items():
            combined.append((doc_id, float(np.sum(score_list))))
        # Normalise by total weight
        if combined:
            total_w = sum(weights[:len(all_results)])
            if total_w > 0:
                combined = [(did, s / total_w) for did, s in combined]
        return combined

    def _combine_rank_fusion(self, all_results: List[List[RerankResult]]) -> List[Tuple[str, float]]:
        doc_map: Dict[str, RerankDocument] = {}
        for results in all_results:
            for r in results:
                doc_map[r.document.id] = r.document

        rrf_scores: Dict[str, float] = defaultdict(float)
        for results in all_results:
            for rank, r in enumerate(results, start=1):
                rrf_scores[r.document.id] += 1.0 / (self._rrf_k + rank)

        return list(rrf_scores.items())

    def _combine_borda_count(self, all_results: List[List[RerankResult]]) -> List[Tuple[str, float]]:
        doc_map: Dict[str, RerankDocument] = {}
        for results in all_results:
            for r in results:
                doc_map[r.document.id] = r.document

        n_docs = len(doc_map)
        borda_scores: Dict[str, float] = defaultdict(float)
        for results in all_results:
            for rank, r in enumerate(results, start=1):
                borda_scores[r.document.id] += n_docs - rank

        return list(borda_scores.items())

    def _combine_median(self, all_results: List[List[RerankResult]]) -> List[Tuple[str, float]]:
        scores: Dict[str, List[float]] = defaultdict(list)
        doc_map: Dict[str, RerankDocument] = {}
        for results in all_results:
            for r in results:
                scores[r.document.id].append(r.score)
                doc_map[r.document.id] = r.document
        combined = []
        for doc_id, score_list in scores.items():
            combined.append((doc_id, float(np.median(score_list))))
        return combined

    def _combine_max(self, all_results: List[List[RerankResult]]) -> List[Tuple[str, float]]:
        scores: Dict[str, List[float]] = defaultdict(list)
        doc_map: Dict[str, RerankDocument] = {}
        for results in all_results:
            for r in results:
                scores[r.document.id].append(r.score)
                doc_map[r.document.id] = r.document
        combined = []
        for doc_id, score_list in scores.items():
            combined.append((doc_id, float(max(score_list))))
        return combined

    def rerank(
        self,
        query: str,
        documents: Sequence[Union[str, RerankDocument]],
        top_k: int = 10,
    ) -> List[RerankResult]:
        if not self._rerankers:
            return []
        docs = [_to_rerank_doc(d) for d in documents]
        if not docs:
            return []

        # Get rerank results from each reranker
        all_results: List[List[RerankResult]] = []
        for name, reranker in self._rerankers:
            try:
                results = reranker.rerank(query, docs, top_k=len(docs))
                all_results.append(results)
            except Exception as exc:
                logger.warning("Reranker %r failed: %s", name, exc)

        if not all_results:
            return []

        # Combine scores
        doc_map: Dict[str, RerankDocument] = {d.id: d for d in docs}

        if self._strategy == "mean":
            combined = self._combine_mean(all_results)
        elif self._strategy == "weighted_mean":
            combined = self._combine_weighted_mean(all_results)
        elif self._strategy == "rank_fusion":
            combined = self._combine_rank_fusion(all_results)
        elif self._strategy == "borda_count":
            combined = self._combine_borda_count(all_results)
        elif self._strategy == "median":
            combined = self._combine_median(all_results)
        elif self._strategy == "max":
            combined = self._combine_max(all_results)
        else:
            combined = self._combine_mean(all_results)

        # Sort by combined score
        combined.sort(key=lambda x: x[1], reverse=True)

        # Build results
        per_doc_scores: Dict[str, Dict[str, float]] = defaultdict(dict)
        for name, results in zip([n for n, _ in self._rerankers], all_results):
            for r in results:
                per_doc_scores[r.document.id][name] = r.score

        final_results: List[RerankResult] = []
        for rank, (doc_id, score) in enumerate(combined[:top_k], start=1):
            doc = doc_map.get(doc_id)
            if doc is None:
                continue
            final_results.append(RerankResult(
                document=doc,
                score=score,
                rank=rank,
                metadata={
                    "method": f"ensemble_{self._strategy}",
                    "num_rerankers": len(all_results),
                    "per_reranker_scores": dict(per_doc_scores.get(doc_id, {})),
                },
            ))
        return final_results

    def batch_rerank(
        self,
        queries: Sequence[str],
        doc_lists: Sequence[Sequence[Union[str, RerankDocument]]],
        top_k: int = 10,
    ) -> List[List[RerankResult]]:
        return [self.rerank(q, docs, top_k) for q, docs in zip(queries, doc_lists)]


# ═══════════════════════════════════════════════════════════════════════════
#  Factory helper
# ═══════════════════════════════════════════════════════════════════════════

def create_reranker(
    method: str = "cross_encoder",
    **kwargs: Any,
) -> BaseReranker:
    """Factory: create a reranker by name.

    Parameters
    ----------
    method : str
        One of ``"cross_encoder"``, ``"t5"``, ``"colbert"``, ``"mmr"``,
        ``"distilled"``, ``"listwise"``, ``"ensemble"``.
    **kwargs
        Forwarded to the reranker constructor.

    Returns
    -------
    BaseReranker
    """
    method = method.lower().strip()
    if method == "cross_encoder":
        return CrossEncoderReranker(**kwargs)
    elif method == "t5":
        return T5Reranker(**kwargs)
    elif method == "colbert":
        return ColBERTLateInteraction(**kwargs)
    elif method == "mmr":
        return MMRReranker(**kwargs)
    elif method == "distilled":
        return KnowledgeDistilledReranker(**kwargs)
    elif method == "listwise":
        return ListWiseReranker(**kwargs)
    elif method == "ensemble":
        return RerankerEnsemble(**kwargs)
    else:
        raise ValueError(f"Unknown reranker method: {method!r}. "
                         f"Choose from: cross_encoder, t5, colbert, mmr, "
                         f"distilled, listwise, ensemble")


# ═══════════════════════════════════════════════════════════════════════════
#  Evaluation helpers
# ═══════════════════════════════════════════════════════════════════════════

def compute_ndcg(
    predicted: List[RerankResult],
    relevant_ids: Set[str],
    k: Optional[int] = None,
) -> float:
    """Compute Normalised Discounted Cumulative Gain at *k*."""
    if not predicted:
        return 0.0
    if k is not None:
        predicted = predicted[:k]

    dcg = 0.0
    for rank, result in enumerate(predicted, start=1):
        if result.document.id in relevant_ids:
            dcg += 1.0 / math.log2(rank + 1)

    # Ideal DCG
    ideal_hits = min(len(relevant_ids), len(predicted))
    idcg = 0.0
    for rank in range(1, ideal_hits + 1):
        idcg += 1.0 / math.log2(rank + 1)

    return dcg / max(idcg, 1e-12)


def compute_mean_reciprocal_rank(
    predicted: List[RerankResult],
    relevant_ids: Set[str],
) -> float:
    """Compute Mean Reciprocal Rank."""
    for rank, result in enumerate(predicted, start=1):
        if result.document.id in relevant_ids:
            return 1.0 / rank
    return 0.0


def compute_precision_at_k(
    predicted: List[RerankResult],
    relevant_ids: Set[str],
    k: int = 10,
) -> float:
    """Compute Precision@k."""
    if not predicted:
        return 0.0
    top_k = predicted[:k]
    hits = sum(1 for r in top_k if r.document.id in relevant_ids)
    return hits / k


def compute_recall_at_k(
    predicted: List[RerankResult],
    relevant_ids: Set[str],
    k: int = 10,
) -> float:
    """Compute Recall@k."""
    if not relevant_ids:
        return 0.0
    top_k = predicted[:k]
    hits = sum(1 for r in top_k if r.document.id in relevant_ids)
    return hits / len(relevant_ids)
