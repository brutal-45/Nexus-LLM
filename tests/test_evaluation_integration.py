"""Tests for evaluation integration."""
import pytest

from nexus_llm.evaluation import (
    Evaluator,
    EvaluationResult,
    ModelComparison,
    BenchmarkRunner,
    BenchmarkConfig,
    BenchmarkResult,
    MetricRegistry,
    PerplexityCalculator,
    PerplexityResult,
    GenerationEvaluator,
    GenerationQualityResult,
    ReportGenerator,
    ReportFormat,
)


class TestEvaluationModuleImports:
    """Test that all evaluation module components can be imported."""

    def test_evaluator_imports(self):
        assert Evaluator is not None
        assert EvaluationResult is not None
        assert ModelComparison is not None

    def test_benchmark_imports(self):
        assert BenchmarkRunner is not None
        assert BenchmarkConfig is not None
        assert BenchmarkResult is not None

    def test_metrics_import(self):
        assert MetricRegistry is not None

    def test_perplexity_imports(self):
        assert PerplexityCalculator is not None
        assert PerplexityResult is not None

    def test_generation_eval_imports(self):
        assert GenerationEvaluator is not None
        assert GenerationQualityResult is not None

    def test_report_imports(self):
        assert ReportGenerator is not None
        assert ReportFormat is not None


class TestEvaluatorIntegration:
    """Test Evaluator creation."""

    def test_create_evaluator(self):
        evaluator = Evaluator()
        assert evaluator is not None


class TestMetricRegistryIntegration:
    """Test MetricRegistry."""

    def test_create_registry(self):
        registry = MetricRegistry()
        assert registry is not None
