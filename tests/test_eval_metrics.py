"""Test evaluation metrics for Nexus-LLM."""
import math
import pytest
from typing import List, Dict, Any


def accuracy(predictions: List[Any], references: List[Any]) -> float:
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have same length")
    if not predictions:
        return 0.0
    correct = sum(1 for p, r in zip(predictions, references) if p == r)
    return correct / len(predictions)


def precision(true_positives: int, false_positives: int) -> float:
    total = true_positives + false_positives
    if total == 0:
        return 0.0
    return true_positives / total


def recall(true_positives: int, false_negatives: int) -> float:
    total = true_positives + false_negatives
    if total == 0:
        return 0.0
    return true_positives / total


def f1_score(prec: float, rec: float) -> float:
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def bleu_score(hypothesis: List[str], reference: List[str], max_n: int = 4) -> float:
    if not hypothesis or not reference:
        return 0.0
    scores = []
    for n in range(1, max_n + 1):
        hyp_ngrams = [tuple(hypothesis[i:i+n]) for i in range(len(hypothesis) - n + 1)]
        ref_ngrams = [tuple(reference[i:i+n]) for i in range(len(reference) - n + 1)]
        ref_counts = {}
        for ng in ref_ngrams:
            ref_counts[ng] = ref_counts.get(ng, 0) + 1
        matches = 0
        hyp_counts = {}
        for ng in hyp_ngrams:
            hyp_counts[ng] = hyp_counts.get(ng, 0) + 1
        for ng, count in hyp_counts.items():
            matches += min(count, ref_counts.get(ng, 0))
        total = len(hyp_ngrams)
        if total > 0:
            scores.append(matches / total)
        else:
            scores.append(0)
    if not scores or all(s == 0 for s in scores):
        return 0.0
    avg = sum(scores) / len(scores)
    bp = min(1.0, math.exp(1 - len(reference) / len(hypothesis))) if len(hypothesis) > 0 else 0
    return bp * avg


def rouge_l_score(hypothesis: List[str], reference: List[str]) -> float:
    if not hypothesis or not reference:
        return 0.0
    m = len(hypothesis)
    n = len(reference)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if hypothesis[i-1] == reference[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    if lcs == 0:
        return 0.0
    prec = lcs / m
    rec = lcs / n
    return f1_score(prec, rec)


def exact_match(predictions: List[str], references: List[str]) -> float:
    if len(predictions) != len(references):
        raise ValueError("Length mismatch")
    if not predictions:
        return 0.0
    matches = sum(1 for p, r in zip(predictions, references) if p.strip() == r.strip())
    return matches / len(predictions)


class TestAccuracy:
    def test_perfect(self):
        assert accuracy([1, 2, 3], [1, 2, 3]) == 1.0

    def test_partial(self):
        assert accuracy([1, 2, 3], [1, 3, 3]) == pytest.approx(2/3)

    def test_zero(self):
        assert accuracy([1, 2], [3, 4]) == 0.0

    def test_empty(self):
        assert accuracy([], []) == 0.0

    def test_length_mismatch(self):
        with pytest.raises(ValueError):
            accuracy([1], [1, 2])


class TestPrecisionRecall:
    def test_precision(self):
        assert precision(8, 2) == 0.8

    def test_precision_zero(self):
        assert precision(0, 0) == 0.0

    def test_recall(self):
        assert recall(8, 2) == 0.8

    def test_recall_zero(self):
        assert recall(0, 0) == 0.0


class TestF1Score:
    def test_balanced(self):
        assert f1_score(0.8, 0.8) == pytest.approx(0.8)

    def test_zero(self):
        assert f1_score(0.0, 0.0) == 0.0

    def test_one_zero(self):
        assert f1_score(1.0, 0.0) == 0.0


class TestBLEU:
    def test_identical(self):
        hyp = ["the", "cat", "sat"]
        ref = ["the", "cat", "sat"]
        assert bleu_score(hyp, ref) > 0.5

    def test_different(self):
        hyp = ["the", "dog", "ran"]
        ref = ["the", "cat", "sat"]
        score = bleu_score(hyp, ref)
        assert 0 <= score <= 1

    def test_empty(self):
        assert bleu_score([], ["test"]) == 0.0

    def test_score_range(self):
        hyp = ["a", "b", "c"]
        ref = ["a", "b", "d"]
        score = bleu_score(hyp, ref)
        assert 0 <= score <= 1


class TestROUGE:
    def test_identical(self):
        hyp = ["the", "cat", "sat"]
        ref = ["the", "cat", "sat"]
        assert rouge_l_score(hyp, ref) == 1.0

    def test_partial(self):
        hyp = ["the", "cat"]
        ref = ["the", "cat", "sat"]
        score = rouge_l_score(hyp, ref)
        assert 0 < score < 1

    def test_empty(self):
        assert rouge_l_score([], ["test"]) == 0.0


class TestExactMatch:
    def test_perfect(self):
        assert exact_match(["hello", "world"], ["hello", "world"]) == 1.0

    def test_partial(self):
        assert exact_match(["hello", "world"], ["hello", "earth"]) == 0.5

    def test_whitespace(self):
        assert exact_match(["hello "], ["hello"]) == 1.0

    def test_empty(self):
        assert exact_match([], []) == 0.0
