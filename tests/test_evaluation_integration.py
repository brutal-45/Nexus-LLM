"""Integration tests for the evaluation package's public surface."""

from __future__ import annotations

import pytest

import nexus_llm.evaluation as evaluation
from nexus_llm.evaluation import (
    BenchmarkRunner,
    ComparisonEngine,
    Evaluator,
    GenerationEvaluator,
    MetricsCalculator,
    PerplexityCalculator,
    tokenize,
)


class TestPublicSurface:
    @pytest.mark.parametrize(
        "name",
        [
            "BenchmarkRunner",
            "ComparisonEngine",
            "ComparisonResult",
            "EvaluationReport",
            "Evaluator",
            "GenerationEvaluator",
            "MetricsCalculator",
            "PerplexityCalculator",
            "tokenize",
        ],
    )
    def test_package_exports(self, name):
        assert getattr(evaluation, name, None) is not None

    def test_all_names_resolve(self):
        for name in evaluation.__all__:
            assert hasattr(evaluation, name), f"__all__ lists missing name {name}"


class TestTokenize:
    def test_lowercases_and_drops_punctuation(self):
        assert tokenize("Hello, World!") == ["hello", "world"]

    def test_empty_input(self):
        assert tokenize("") == []

    def test_keeps_inner_apostrophes(self):
        assert tokenize("don't stop") == ["don't", "stop"]


class TestMetricsCalculator:
    @pytest.fixture
    def calc(self):
        return MetricsCalculator()

    def test_bleu_is_one_for_identical_text(self, calc):
        assert calc.bleu_score("the cat sat on the mat", "the cat sat on the mat") == pytest.approx(1.0)

    def test_bleu_is_zero_without_overlap(self, calc):
        assert calc.bleu_score("aaa bbb ccc", "ddd eee fff") == 0.0

    def test_rouge_within_unit_interval(self, calc):
        score = calc.rouge_score("the quick brown fox", "the quick fox")
        assert 0.0 <= score <= 1.0

    def test_distinct_n_penalises_repetition(self, calc):
        repetitive = calc.distinct_n(["yes yes yes yes"], n=1)
        varied = calc.distinct_n(["one two three four"], n=1)
        assert repetitive < varied

    def test_perplexity_rejects_mismatched_shapes(self, calc):
        with pytest.raises(ValueError):
            calc.perplexity([[1.0, 2.0]], [0])

    def test_average_length(self, calc):
        assert calc.average_length(["a b c", "a b c"]) == 3.0


class TestEvaluatorSurface:
    """Constructors take optional config, so they must work with no arguments."""

    @pytest.mark.parametrize("factory", [Evaluator, GenerationEvaluator, ComparisonEngine, PerplexityCalculator])
    def test_constructible(self, factory):
        assert factory() is not None

    def test_benchmark_runner_is_a_class(self):
        assert isinstance(BenchmarkRunner, type)

    def test_result_types_are_dataclasses(self):
        import dataclasses

        from nexus_llm.evaluation.perplexity import PerplexityResult

        assert dataclasses.is_dataclass(PerplexityResult)
