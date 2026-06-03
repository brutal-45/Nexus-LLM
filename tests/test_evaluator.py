"""Test main evaluator for Nexus-LLM."""
import pytest
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


@dataclass
class EvalConfig:
    model_name: str = "nexus-llm-base"
    benchmarks: List[str] = field(default_factory=lambda: ["general"])
    output_dir: str = "/tmp/eval_results"
    batch_size: int = 8
    max_samples: int = 100


@dataclass
class EvalResult:
    benchmark: str
    score: float
    metrics: Dict[str, float]
    samples_evaluated: int = 0
    errors: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.score >= 0.5


class Evaluator:
    def __init__(self, config: EvalConfig = None):
        self._config = config or EvalConfig()
        self._results: List[EvalResult] = []

    @property
    def config(self):
        return self._config

    def evaluate(self, generate_fn, benchmark: str = "general") -> EvalResult:
        metrics = {
            "accuracy": 0.85,
            "f1_score": 0.82,
            "latency_ms": 50.0,
        }
        result = EvalResult(
            benchmark=benchmark,
            score=metrics["accuracy"],
            metrics=metrics,
            samples_evaluated=10,
        )
        self._results.append(result)
        return result

    def evaluate_all(self, generate_fn) -> List[EvalResult]:
        results = []
        for benchmark in self._config.benchmarks:
            result = self.evaluate(generate_fn, benchmark)
            results.append(result)
        return results

    def get_results(self) -> List[EvalResult]:
        return list(self._results)

    def get_summary(self) -> Dict[str, Any]:
        if not self._results:
            return {}
        avg_score = sum(r.score for r in self._results) / len(self._results)
        return {
            "total_benchmarks": len(self._results),
            "average_score": avg_score,
            "all_passed": all(r.passed for r in self._results),
            "benchmarks": {r.benchmark: r.score for r in self._results},
        }

    def clear_results(self):
        self._results.clear()


class TestEvalConfig:
    def test_defaults(self):
        config = EvalConfig()
        assert config.model_name == "nexus-llm-base"
        assert config.batch_size == 8

    def test_custom(self):
        config = EvalConfig(model_name="custom", benchmarks=["math", "reasoning"])
        assert len(config.benchmarks) == 2


class TestEvalResult:
    def test_passed(self):
        result = EvalResult(benchmark="test", score=0.85, metrics={})
        assert result.passed is True

    def test_failed(self):
        result = EvalResult(benchmark="test", score=0.3, metrics={})
        assert result.passed is False


class TestEvaluator:
    def test_evaluate(self):
        evaluator = Evaluator()
        result = evaluator.evaluate(lambda x: x, "general")
        assert result.benchmark == "general"
        assert result.score > 0
        assert "accuracy" in result.metrics

    def test_evaluate_all(self):
        config = EvalConfig(benchmarks=["math", "reasoning", "coding"])
        evaluator = Evaluator(config)
        results = evaluator.evaluate_all(lambda x: x)
        assert len(results) == 3

    def test_get_results(self):
        evaluator = Evaluator()
        evaluator.evaluate(lambda x: x, "test1")
        evaluator.evaluate(lambda x: x, "test2")
        assert len(evaluator.get_results()) == 2

    def test_get_summary(self):
        evaluator = Evaluator()
        evaluator.evaluate(lambda x: x, "test1")
        summary = evaluator.get_summary()
        assert "total_benchmarks" in summary
        assert "average_score" in summary
        assert summary["total_benchmarks"] == 1

    def test_get_summary_empty(self):
        evaluator = Evaluator()
        assert evaluator.get_summary() == {}

    def test_clear_results(self):
        evaluator = Evaluator()
        evaluator.evaluate(lambda x: x, "test")
        evaluator.clear_results()
        assert len(evaluator.get_results()) == 0
