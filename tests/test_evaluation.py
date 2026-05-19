"""Tests for training evaluation."""
import pytest
import math


class EvaluationMetrics:
    """Simple evaluation metrics for testing."""
    def __init__(self):
        self.predictions = []
        self.references = []

    def add(self, prediction, reference):
        self.predictions.append(prediction)
        self.references.append(reference)

    def exact_match(self):
        if not self.predictions:
            return 0.0
        matches = sum(1 for p, r in zip(self.predictions, self.references) if p == r)
        return matches / len(self.predictions)

    def character_f1(self):
        if not self.predictions:
            return 0.0
        f1_scores = []
        for pred, ref in zip(self.predictions, self.references):
            pred_set = set(pred)
            ref_set = set(ref)
            if not pred_set and not ref_set:
                f1_scores.append(1.0)
                continue
            precision = len(pred_set & ref_set) / max(len(pred_set), 1)
            recall = len(pred_set & ref_set) / max(len(ref_set), 1)
            if precision + recall == 0:
                f1_scores.append(0.0)
            else:
                f1_scores.append(2 * precision * recall / (precision + recall))
        return sum(f1_scores) / len(f1_scores)


@pytest.fixture
def eval_metrics():
    return EvaluationMetrics()


def test_exact_match(eval_metrics):
    """Test exact match calculation."""
    eval_metrics.add("Hello", "Hello")
    eval_metrics.add("World", "World")
    eval_metrics.add("Test", "Fail")
    assert eval_metrics.exact_match() == pytest.approx(2/3, abs=0.01)


def test_exact_match_all_correct(eval_metrics):
    """Test 100% exact match."""
    eval_metrics.add("a", "a")
    eval_metrics.add("b", "b")
    assert eval_metrics.exact_match() == 1.0


def test_exact_match_none_correct(eval_metrics):
    """Test 0% exact match."""
    eval_metrics.add("a", "b")
    eval_metrics.add("c", "d")
    assert eval_metrics.exact_match() == 0.0


def test_character_f1(eval_metrics):
    """Test character F1 calculation."""
    eval_metrics.add("hello", "hello")
    assert eval_metrics.character_f1() == 1.0


def test_evaluation_empty(eval_metrics):
    """Test evaluation with no data."""
    assert eval_metrics.exact_match() == 0.0
