"""Result reranking strategies for improving retrieval quality.

Provides cross-encoder reranking, diversity-based reranking,
and relevance scoring with configurable methods.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Set

import numpy as np

from nexus_llm.rag.retriever import RetrievalResult

logger = logging.getLogger(__name__)


class Reranker(ABC):
    """Abstract base class for result rerankers."""

    @abstractmethod
    def rerank(self, query: str, results: List[RetrievalResult], top_k: int = 10) -> List[RetrievalResult]:
        """Rerank retrieval results.

        Args:
            query: The original query string.
            results: Initial retrieval results to rerank.
            top_k: Maximum number of results to return.

        Returns:
            Reranked list of RetrievalResult objects.
        """
        ...


class CrossEncoderReranker(Reranker):
    """Reranks results using a cross-encoder model.

    Uses a cross-encoder to compute relevance scores by jointly
    encoding the query and each document, producing more accurate
    relevance estimates than bi-encoder approaches.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size: int = 16,
        max_length: int = 512,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self._model = None

    def _load_model(self):
        """Lazily load the cross-encoder model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name, max_length=self.max_length)
                logger.info("Loaded cross-encoder model: %s", self.model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for CrossEncoderReranker. "
                    "Install with: pip install sentence-transformers"
                )

    def rerank(self, query: str, results: List[RetrievalResult], top_k: int = 10) -> List[RetrievalResult]:
        """Rerank using cross-encoder relevance scores."""
        if not results:
            return []

        self._load_model()

        # Create query-document pairs
        pairs = [(query, result.document.text) for result in results]

        # Score pairs in batches
        all_scores = []
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i : i + self.batch_size]
            scores = self._model.predict(batch, show_progress_bar=False)
            all_scores.extend(scores.tolist() if hasattr(scores, "tolist") else list(scores))

        # Create new results with cross-encoder scores
        reranked = []
        for i, result in enumerate(results):
            reranked.append(
                RetrievalResult(
                    document=result.document,
                    score=float(all_scores[i]),
                    retrieval_method=f"{result.retrieval_method}_cross_encoder",
                    metadata={
                        **result.metadata,
                        "original_score": result.score,
                        "cross_encoder_score": float(all_scores[i]),
                    },
                )
            )

        # Sort by cross-encoder score descending
        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked[:top_k]


class DiversityReranker(Reranker):
    """Reranks results to maximize diversity while maintaining relevance.

    Uses a maximal marginal relevance (MMR) approach to balance
    relevance to the query with diversity among selected results,
    reducing redundancy in the returned documents.
    """

    def __init__(
        self,
        lambda_param: float = 0.7,
        similarity_fn: Optional[callable] = None,
    ):
        """Initialize diversity reranker.

        Args:
            lambda_param: Trade-off between relevance and diversity (0-1).
                         1.0 = pure relevance, 0.0 = pure diversity.
            similarity_fn: Optional function to compute similarity between
                          two texts. If None, uses token-based Jaccard.
        """
        self.lambda_param = lambda_param
        self.similarity_fn = similarity_fn or self._jaccard_similarity

    @staticmethod
    def _jaccard_similarity(text_a: str, text_b: str) -> float:
        """Compute Jaccard similarity between two texts based on token overlap."""
        tokens_a = set(text_a.lower().split())
        tokens_b = set(text_b.lower().split())
        if not tokens_a or not tokens_b:
            return 0.0
        intersection = tokens_a & tokens_b
        union = tokens_a | tokens_b
        return len(intersection) / len(union)

    def _compute_mmr(
        self,
        query: str,
        candidate: RetrievalResult,
        selected: List[RetrievalResult],
    ) -> float:
        """Compute Maximal Marginal Relevance score for a candidate."""
        # Relevance component
        relevance = candidate.score

        # Diversity component: max similarity to already selected
        if not selected:
            return self.lambda_param * relevance

        max_sim = max(
            self.similarity_fn(candidate.document.text, sel.document.text)
            for sel in selected
        )

        # MMR formula
        mmr = self.lambda_param * relevance - (1 - self.lambda_param) * max_sim
        return mmr

    def rerank(self, query: str, results: List[RetrievalResult], top_k: int = 10) -> List[RetrievalResult]:
        """Rerank using MMR for diversity."""
        if not results:
            return []

        top_k = min(top_k, len(results))
        selected: List[RetrievalResult] = []
        remaining = list(results)

        # Normalize scores to [0, 1] for fair MMR computation
        scores = [r.score for r in remaining]
        min_score = min(scores)
        max_score = max(scores)
        score_range = max_score - min_score if max_score > min_score else 1.0

        for r in remaining:
            r.score = (r.score - min_score) / score_range

        for _ in range(top_k):
            if not remaining:
                break

            # Select candidate with highest MMR
            best_idx = 0
            best_mmr = float("-inf")

            for i, candidate in enumerate(remaining):
                mmr = self._compute_mmr(query, candidate, selected)
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = i

            selected_result = remaining.pop(best_idx)
            selected.append(
                RetrievalResult(
                    document=selected_result.document,
                    score=selected_result.score,
                    retrieval_method=f"{selected_result.retrieval_method}_diversity",
                    metadata={
                        **selected_result.metadata,
                        "mmr_score": best_mmr,
                    },
                )
            )

        return selected


class RelevanceReranker(Reranker):
    """Reranks results based on multi-signal relevance scoring.

    Combines multiple relevance signals including original score,
    query term coverage, text length normalization, and position
    bias to produce a composite relevance score.
    """

    def __init__(
        self,
        term_coverage_weight: float = 0.3,
        length_penalty_weight: float = 0.1,
        position_bias_weight: float = 0.1,
        original_score_weight: float = 0.5,
        ideal_length: int = 200,
    ):
        """Initialize relevance reranker.

        Args:
            term_coverage_weight: Weight for query term coverage signal.
            length_penalty_weight: Weight for length normalization signal.
            position_bias_weight: Weight for original position bias.
            original_score_weight: Weight for the original retrieval score.
            ideal_length: Ideal chunk length for length penalty.
        """
        self.term_coverage_weight = term_coverage_weight
        self.length_penalty_weight = length_penalty_weight
        self.position_bias_weight = position_bias_weight
        self.original_score_weight = original_score_weight
        self.ideal_length = ideal_length

        # Normalize weights
        total = (term_coverage_weight + length_penalty_weight +
                 position_bias_weight + original_score_weight)
        self.term_coverage_weight /= total
        self.length_penalty_weight /= total
        self.position_bias_weight /= total
        self.original_score_weight /= total

    def _compute_term_coverage(self, query: str, text: str) -> float:
        """Compute the fraction of query terms present in the text."""
        query_terms = set(query.lower().split())
        text_lower = text.lower()
        if not query_terms:
            return 0.0

        covered = sum(1 for term in query_terms if term in text_lower)
        return covered / len(query_terms)

    def _compute_length_score(self, text: str) -> float:
        """Compute length-based score. Penalizes very short or very long texts."""
        length = len(text.split())
        if length == 0:
            return 0.0

        # Gaussian-like penalty around ideal length
        sigma = self.ideal_length * 0.5
        score = math.exp(-0.5 * ((length - self.ideal_length) / sigma) ** 2)
        return score

    def _compute_position_score(self, original_rank: int, total_results: int) -> float:
        """Compute position-based score. Higher ranks get higher scores."""
        if total_results <= 1:
            return 1.0
        return 1.0 - (original_rank / total_results)

    def rerank(self, query: str, results: List[RetrievalResult], top_k: int = 10) -> List[RetrievalResult]:
        """Rerank using multi-signal relevance scoring."""
        if not results:
            return []

        total_results = len(results)

        # Normalize original scores to [0, 1]
        scores = [r.score for r in results]
        min_score = min(scores)
        max_score = max(scores)
        score_range = max_score - min_score if max_score > min_score else 1.0

        reranked = []
        for rank, result in enumerate(results):
            # Compute individual signals
            normalized_score = (result.score - min_score) / score_range
            term_coverage = self._compute_term_coverage(query, result.document.text)
            length_score = self._compute_length_score(result.document.text)
            position_score = self._compute_position_score(rank, total_results)

            # Combine signals
            composite_score = (
                self.original_score_weight * normalized_score
                + self.term_coverage_weight * term_coverage
                + self.length_penalty_weight * length_score
                + self.position_bias_weight * position_score
            )

            reranked.append(
                RetrievalResult(
                    document=result.document,
                    score=composite_score,
                    retrieval_method=f"{result.retrieval_method}_relevance",
                    metadata={
                        **result.metadata,
                        "original_score": result.score,
                        "term_coverage": term_coverage,
                        "length_score": length_score,
                        "position_score": position_score,
                        "composite_score": composite_score,
                    },
                )
            )

        # Sort by composite score descending
        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked[:top_k]
