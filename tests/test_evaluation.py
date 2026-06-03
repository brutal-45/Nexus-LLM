"""Tests for the evaluation module.

Covers Evaluator, BenchmarkRunner, MetricsCalculator, EvaluationReport,
and ComparisonEngine.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from nexus_llm.evaluation.metrics import MetricsCalculator
from nexus_llm.evaluation.evaluator import Evaluator
from nexus_llm.evaluation.report import EvaluationReport
from nexus_llm.evaluation.benchmark import BenchmarkRunner
from nexus_llm.evaluation.comparison import ComparisonEngine, ComparisonResult


# ---------------------------------------------------------------------------
# MetricsCalculator
# ---------------------------------------------------------------------------

class TestMetricsCalculator:
    """Tests for MetricsCalculator."""

    def test_perplexity(self):
        calc = MetricsCalculator()
        import math
        # Simple case: 2 tokens, vocab_size 3
        logits = [[1.0, 2.0, 0.5], [0.5, 1.0, 2.0]]
        labels = [1, 2]
        ppl = calc.perplexity(logits, labels)
        assert ppl > 0
        assert isinstance(ppl, float)

    def test_perplexity_mismatched_lengths(self):
        calc = MetricsCalculator()
        with pytest.raises(ValueError):
            calc.perplexity([[1.0, 2.0]], [1, 2])

    def test_perplexity_empty(self):
        calc = MetricsCalculator()
        ppl = calc.perplexity([], [])
        assert ppl == float("inf")

    def test_bleu_identical(self):
        calc = MetricsCalculator()
        score = calc.bleu_score("the cat sat on the mat", "the cat sat on the mat")
        assert score == 1.0

    def test_bleu_no_overlap(self):
        calc = MetricsCalculator()
        score = calc.bleu_score("aaa bbb ccc", "ddd eee fff")
        assert score == 0.0

    def test_bleu_empty_hypothesis(self):
        calc = MetricsCalculator()
        score = calc.bleu_score("reference text", "")
        assert score == 0.0

    def test_rouge_identical(self):
        calc = MetricsCalculator()
        scores = calc.rouge_score("the cat sat on the mat", "the cat sat on the mat")
        assert scores["rouge1"] == 1.0
        assert scores["rouge2"] == 1.0
        assert scores["rougeL"] == 1.0

    def test_rouge_no_overlap(self):
        calc = MetricsCalculator()
        scores = calc.rouge_score("aaa bbb", "ccc ddd")
        assert scores["rouge1"] == 0.0

    def test_rouge_partial_overlap(self):
        calc = MetricsCalculator()
        scores = calc.rouge_score("the cat sat", "the cat walked")
        assert 0 < scores["rouge1"] < 1.0

    def test_distinct_n(self):
        calc = MetricsCalculator()
        texts = ["the cat sat on the mat", "the dog sat on the rug"]
        d2 = calc.distinct_n(texts, n=2)
        assert 0 < d2 <= 1.0

    def test_distinct_n_empty(self):
        calc = MetricsCalculator()
        assert calc.distinct_n([], n=2) == 0.0

    def test_average_length(self):
        calc = MetricsCalculator()
        avg = calc.average_length(["hello world", "foo bar baz"])
        assert avg == 2.5

    def test_average_length_empty(self):
        calc = MetricsCalculator()
        assert calc.average_length([]) == 0.0


# ---------------------------------------------------------------------------
# EvaluationReport
# ---------------------------------------------------------------------------

class TestEvaluationReport:
    """Tests for EvaluationReport."""

    def test_create_report(self):
        report = EvaluationReport(model_name="test-model", dataset_name="test")
        assert report.model_name == "test-model"
        assert report.dataset_name == "test"

    def test_add_metric(self):
        report = EvaluationReport(model_name="test", dataset_name="test")
        report.add_metric("bleu", 0.85)
        assert "bleu" in report.scores
        assert report.scores["bleu"] == 0.85

    def test_summary(self):
        report = EvaluationReport(model_name="test", dataset_name="test")
        report.add_metric("bleu", 0.85)
        s = report.summary()
        assert isinstance(s, str)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class TestEvaluator:
    """Tests for Evaluator."""

    def test_evaluate_with_mock_model(self):
        model = MagicMock()
        model.generate = MagicMock(return_value="Paris is the capital of France.")
        evaluator = Evaluator()
        dataset = [
            {"prompt": "What is the capital of France?", "reference": "Paris"},
        ]
        report = evaluator.evaluate(model, dataset=dataset, model_name="mock")
        assert isinstance(report, EvaluationReport)
        assert report.model_name == "mock"

    def test_evaluate_generation(self):
        model = MagicMock()
        model.generate = MagicMock(return_value="Generated text output")
        evaluator = Evaluator()
        scores = evaluator.evaluate_generation(
            model,
            prompts=["Hello"],
            references=["Hello"],
        )
        assert isinstance(scores, dict)
        assert "distinct_2" in scores
        assert "avg_length" in scores

    def test_evaluate_chat(self):
        model = MagicMock()
        model.chat = MagicMock(return_value="Chat response")
        evaluator = Evaluator()
        conversations = [[{"role": "user", "content": "Hi"}]]
        scores = evaluator.evaluate_chat(
            model,
            conversations=conversations,
            references=["Hello"],
        )
        assert isinstance(scores, dict)

    def test_evaluate_default_dataset(self):
        model = MagicMock()
        model.generate = MagicMock(return_value="Test output")
        evaluator = Evaluator()
        report = evaluator.evaluate(model, model_name="test")
        assert isinstance(report, EvaluationReport)


# ---------------------------------------------------------------------------
# BenchmarkRunner
# ---------------------------------------------------------------------------

class TestBenchmarkRunner:
    """Tests for BenchmarkRunner."""

    def test_init(self):
        runner = BenchmarkRunner()
        assert runner is not None

    def test_run_benchmark(self):
        runner = BenchmarkRunner()
        model = MagicMock()
        model.generate = MagicMock(return_value="output")
        result = runner.run(model, benchmark_name="speed")
        assert isinstance(result, EvaluationReport)


# ---------------------------------------------------------------------------
# ComparisonEngine
# ---------------------------------------------------------------------------

class TestComparisonEngine:
    """Tests for ComparisonEngine."""

    def test_compare(self):
        engine = ComparisonEngine()
        report_a = EvaluationReport(model_name="model_a", dataset_name="test")
        report_a.add_metric("bleu", 0.8)
        report_b = EvaluationReport(model_name="model_b", dataset_name="test")
        report_b.add_metric("bleu", 0.9)
        result = engine.compare(report_a, report_b)
        assert isinstance(result, ComparisonResult)
        assert "bleu" in result.metrics
