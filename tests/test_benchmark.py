"""Tests for the benchmark module.

Covers BenchmarkSuite, SpeedBenchmark, QualityBenchmark, and BenchmarkReport.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from nexus_llm.benchmark.suite import BenchmarkSuite
from nexus_llm.benchmark.speed import SpeedBenchmark
from nexus_llm.benchmark.quality import QualityBenchmark
from nexus_llm.benchmark.report import BenchmarkReport


# ---------------------------------------------------------------------------
# SpeedBenchmark
# ---------------------------------------------------------------------------

class TestSpeedBenchmark:
    """Tests for SpeedBenchmark."""

    def test_create_benchmark(self):
        bench = SpeedBenchmark()
        assert bench is not None

    def test_run_benchmark(self):
        bench = SpeedBenchmark()
        model = MagicMock()
        model.generate = MagicMock(return_value="Generated output")
        result = bench.run(model, num_requests=5)
        assert result is not None

    def test_get_metrics(self):
        bench = SpeedBenchmark()
        metrics = bench.get_metrics()
        assert isinstance(metrics, dict)


# ---------------------------------------------------------------------------
# QualityBenchmark
# ---------------------------------------------------------------------------

class TestQualityBenchmark:
    """Tests for QualityBenchmark."""

    def test_create_benchmark(self):
        bench = QualityBenchmark()
        assert bench is not None

    def test_run_benchmark(self):
        bench = QualityBenchmark()
        model = MagicMock()
        model.generate = MagicMock(return_value="The capital of France is Paris.")
        result = bench.run(model)
        assert result is not None

    def test_get_metrics(self):
        bench = QualityBenchmark()
        metrics = bench.get_metrics()
        assert isinstance(metrics, dict)


# ---------------------------------------------------------------------------
# BenchmarkReport
# ---------------------------------------------------------------------------

class TestBenchmarkReport:
    """Tests for BenchmarkReport."""

    def test_create_report(self):
        report = BenchmarkReport(name="test-report")
        assert report.name == "test-report"

    def test_add_metric(self):
        report = BenchmarkReport(name="test")
        report.add_metric("latency_ms", 42.5)
        metrics = report.get_metrics()
        assert "latency_ms" in metrics

    def test_summary(self):
        report = BenchmarkReport(name="test")
        report.add_metric("throughput", 100.0)
        s = report.summary()
        assert isinstance(s, str)

    def test_to_dict(self):
        report = BenchmarkReport(name="test")
        report.add_metric("score", 0.95)
        d = report.to_dict()
        assert isinstance(d, dict)
        assert d["name"] == "test"


# ---------------------------------------------------------------------------
# BenchmarkSuite
# ---------------------------------------------------------------------------

class TestBenchmarkSuite:
    """Tests for BenchmarkSuite."""

    def test_create_suite(self):
        suite = BenchmarkSuite()
        assert suite is not None

    def test_add_benchmark(self):
        suite = BenchmarkSuite()
        bench = SpeedBenchmark()
        suite.add_benchmark("speed", bench)
        assert suite.has_benchmark("speed")

    def test_run_all(self):
        suite = BenchmarkSuite()
        model = MagicMock()
        model.generate = MagicMock(return_value="output")
        speed_bench = SpeedBenchmark()
        suite.add_benchmark("speed", speed_bench)
        results = suite.run_all(model)
        assert isinstance(results, dict)
        assert "speed" in results

    def test_run_single(self):
        suite = BenchmarkSuite()
        bench = SpeedBenchmark()
        suite.add_benchmark("speed", bench)
        model = MagicMock()
        model.generate = MagicMock(return_value="output")
        result = suite.run("speed", model)
        assert result is not None

    def test_list_benchmarks(self):
        suite = BenchmarkSuite()
        suite.add_benchmark("speed", SpeedBenchmark())
        suite.add_benchmark("quality", QualityBenchmark())
        names = suite.list_benchmarks()
        assert "speed" in names
        assert "quality" in names

    def test_remove_benchmark(self):
        suite = BenchmarkSuite()
        suite.add_benchmark("speed", SpeedBenchmark())
        suite.remove_benchmark("speed")
        assert not suite.has_benchmark("speed")
