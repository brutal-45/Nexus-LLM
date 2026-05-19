"""Test result reranking for Nexus-LLM."""
import pytest
from dataclasses import dataclass
from typing import List, Dict, Any, Callable, Optional


@dataclass
class RankedResult:
    id: str
    text: str
    original_score: float
    reranked_score: float
    rank: int = 0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class Reranker:
    def __init__(self, method: str = "score"):
        self._method = method

    def rerank_by_score(self, results: List[Dict], score_key: str = "score") -> List[RankedResult]:
        sorted_results = sorted(results, key=lambda x: x.get(score_key, 0), reverse=True)
        ranked = []
        for i, r in enumerate(sorted_results):
            ranked.append(RankedResult(
                id=r.get("id", str(i)),
                text=r.get("text", ""),
                original_score=r.get(score_key, 0),
                reranked_score=r.get(score_key, 0),
                rank=i + 1,
                metadata=r.get("metadata", {}),
            ))
        return ranked

    def rerank_with_freshness(self, results: List[Dict], score_key: str = "score",
                               timestamp_key: str = "timestamp", freshness_weight: float = 0.3) -> List[RankedResult]:
        if not results:
            return []
        max_score = max(r.get(score_key, 0) for r in results)
        timestamps = [r.get(timestamp_key, 0) for r in results]
        max_ts = max(timestamps) if timestamps else 1
        min_ts = min(timestamps) if timestamps else 0
        ts_range = max_ts - min_ts if max_ts != min_ts else 1

        scored = []
        for r in results:
            relevance = r.get(score_key, 0) / max_score if max_score > 0 else 0
            freshness = (r.get(timestamp_key, 0) - min_ts) / ts_range
            combined = (1 - freshness_weight) * relevance + freshness_weight * freshness
            scored.append((r, combined))

        scored.sort(key=lambda x: x[1], reverse=True)
        ranked = []
        for i, (r, score) in enumerate(scored):
            ranked.append(RankedResult(
                id=r.get("id", str(i)),
                text=r.get("text", ""),
                original_score=r.get(score_key, 0),
                reranked_score=score,
                rank=i + 1,
            ))
        return ranked

    def rerank_with_diversity(self, results: List[Dict], score_key: str = "score",
                               diversity_key: str = "category", diversity_weight: float = 0.2) -> List[RankedResult]:
        sorted_results = sorted(results, key=lambda x: x.get(score_key, 0), reverse=True)
        selected = []
        used_categories = set()

        for r in sorted_results:
            category = r.get(diversity_key, "default")
            if category not in used_categories or len(selected) < len(results) // 2:
                selected.append(r)
                used_categories.add(category)

        for r in sorted_results:
            if r not in selected:
                selected.append(r)

        ranked = []
        for i, r in enumerate(selected):
            ranked.append(RankedResult(
                id=r.get("id", str(i)),
                text=r.get("text", ""),
                original_score=r.get(score_key, 0),
                reranked_score=r.get(score_key, 0) * (1 - diversity_weight if r.get(diversity_key) in used_categories else 1),
                rank=i + 1,
            ))
        return ranked


class TestRankedResult:
    def test_creation(self):
        r = RankedResult(id="1", text="test", original_score=0.9, reranked_score=0.95, rank=1)
        assert r.rank == 1

    def test_default_metadata(self):
        r = RankedResult(id="1", text="test", original_score=0.9, reranked_score=0.9)
        assert r.metadata == {}


class TestReranker:
    def test_rerank_by_score(self):
        reranker = Reranker()
        results = [
            {"id": "1", "text": "low", "score": 0.3},
            {"id": "2", "text": "high", "score": 0.9},
            {"id": "3", "text": "mid", "score": 0.6},
        ]
        ranked = reranker.rerank_by_score(results)
        assert ranked[0].id == "2"
        assert ranked[0].rank == 1
        assert ranked[1].id == "3"

    def test_rerank_empty(self):
        reranker = Reranker()
        assert reranker.rerank_by_score([]) == []

    def test_rerank_preserves_original_score(self):
        reranker = Reranker()
        results = [{"id": "1", "text": "test", "score": 0.7}]
        ranked = reranker.rerank_by_score(results)
        assert ranked[0].original_score == 0.7

    def test_rerank_with_freshness(self):
        reranker = Reranker()
        results = [
            {"id": "1", "text": "old but relevant", "score": 0.9, "timestamp": 100},
            {"id": "2", "text": "new less relevant", "score": 0.5, "timestamp": 200},
        ]
        ranked = reranker.rerank_with_freshness(results, freshness_weight=0.5)
        assert len(ranked) == 2
        assert all(r.reranked_score != r.original_score for r in ranked)

    def test_rerank_with_diversity(self):
        reranker = Reranker()
        results = [
            {"id": "1", "text": "ml", "score": 0.9, "category": "ml"},
            {"id": "2", "text": "dl", "score": 0.8, "category": "ml"},
            {"id": "3", "text": "nlp", "score": 0.7, "category": "nlp"},
        ]
        ranked = reranker.rerank_with_diversity(results)
        assert len(ranked) == 3
        # Diverse categories should be promoted
        categories = [r.metadata.get("category") or results[int(r.id)-1].get("category") for r in ranked]
