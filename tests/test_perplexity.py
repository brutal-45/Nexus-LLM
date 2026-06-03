"""Test perplexity calculation for Nexus-LLM."""
import math
import pytest
from typing import List, Optional


def perplexity_from_loss(avg_loss: float) -> float:
    if avg_loss < 0:
        raise ValueError("Loss must be non-negative")
    return math.exp(avg_loss)


def perplexity_from_probs(log_probs: List[float]) -> float:
    if not log_probs:
        raise ValueError("Log probabilities list cannot be empty")
    avg_neg_log_prob = -sum(log_probs) / len(log_probs)
    return math.exp(avg_neg_log_prob)


def bits_per_character(log_probs: List[float], num_chars: int) -> float:
    if num_chars <= 0:
        raise ValueError("Number of characters must be positive")
    total_neg_log_prob = -sum(log_probs)
    return total_neg_log_prob / num_chars / math.log(2)


def cross_entropy_loss(log_probs: List[float]) -> float:
    if not log_probs:
        return 0.0
    return -sum(log_probs) / len(log_probs)


def normalize_log_probs(log_probs: List[float]) -> List[float]:
    if not log_probs:
        return []
    max_lp = max(log_probs)
    shifted = [lp - max_lp for lp in log_probs]
    total = sum(math.exp(lp) for lp in shifted)
    log_total = math.log(total)
    return [lp - log_total for lp in shifted]


class PerplexityCalculator:
    def __init__(self):
        self._losses: List[float] = []

    def add_loss(self, loss: float):
        if loss < 0:
            raise ValueError("Loss must be non-negative")
        self._losses.append(loss)

    def compute(self) -> float:
        if not self._losses:
            raise ValueError("No losses recorded")
        avg_loss = sum(self._losses) / len(self._losses)
        return perplexity_from_loss(avg_loss)

    def compute_from_tokens(self, token_log_probs: List[List[float]]) -> float:
        if not token_log_probs:
            raise ValueError("No token probabilities")
        all_log_probs = [lp for seq in token_log_probs for lp in seq]
        return perplexity_from_probs(all_log_probs)

    def reset(self):
        self._losses.clear()

    @property
    def num_samples(self):
        return len(self._losses)


class TestPerplexityFromLoss:
    def test_zero_loss(self):
        assert perplexity_from_loss(0.0) == 1.0

    def test_known_value(self):
        loss = 2.0
        ppl = perplexity_from_loss(loss)
        assert abs(ppl - math.exp(2.0)) < 0.001

    def test_negative_loss(self):
        with pytest.raises(ValueError, match="non-negative"):
            perplexity_from_loss(-1.0)

    def test_high_loss(self):
        ppl = perplexity_from_loss(10.0)
        assert ppl > 1000


class TestPerplexityFromProbs:
    def test_uniform(self):
        log_probs = [math.log(1/10)] * 10
        ppl = perplexity_from_probs(log_probs)
        assert abs(ppl - 10.0) < 0.1

    def test_perfect(self):
        log_probs = [0.0]
        ppl = perplexity_from_probs(log_probs)
        assert abs(ppl - 1.0) < 0.001

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            perplexity_from_probs([])


class TestBitsPerCharacter:
    def test_calculation(self):
        log_probs = [math.log(0.5), math.log(0.5)]
        bpc = bits_per_character(log_probs, 10)
        assert bpc > 0

    def test_zero_chars(self):
        with pytest.raises(ValueError, match="positive"):
            bits_per_character([0.1], 0)


class TestCrossEntropyLoss:
    def test_uniform(self):
        log_probs = [math.log(0.5)] * 2
        loss = cross_entropy_loss(log_probs)
        assert abs(loss - math.log(2)) < 0.001

    def test_empty(self):
        assert cross_entropy_loss([]) == 0.0


class TestNormalizeLogProbs:
    def test_normalization(self):
        log_probs = [0.0, 0.0]
        normalized = normalize_log_probs(log_probs)
        total = sum(math.exp(lp) for lp in normalized)
        assert abs(total - 1.0) < 0.001

    def test_empty(self):
        assert normalize_log_probs([]) == []


class TestPerplexityCalculator:
    def test_compute(self):
        calc = PerplexityCalculator()
        calc.add_loss(2.0)
        calc.add_loss(3.0)
        ppl = calc.compute()
        assert ppl > 1.0

    def test_compute_empty(self):
        calc = PerplexityCalculator()
        with pytest.raises(ValueError, match="No losses"):
            calc.compute()

    def test_negative_loss(self):
        calc = PerplexityCalculator()
        with pytest.raises(ValueError, match="non-negative"):
            calc.add_loss(-1.0)

    def test_compute_from_tokens(self):
        calc = PerplexityCalculator()
        token_log_probs = [[math.log(0.1)] * 10, [math.log(0.1)] * 10]
        ppl = calc.compute_from_tokens(token_log_probs)
        assert ppl > 1.0

    def test_reset(self):
        calc = PerplexityCalculator()
        calc.add_loss(1.0)
        calc.reset()
        assert calc.num_samples == 0

    def test_num_samples(self):
        calc = PerplexityCalculator()
        calc.add_loss(1.0)
        calc.add_loss(2.0)
        assert calc.num_samples == 2
