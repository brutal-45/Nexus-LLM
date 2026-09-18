"""Test benchmark runner for Nexus-LLM."""
import pytest
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


@dataclass
class BenchmarkConfig:
    name: str = "general"
    num_samples: int = 100
    timeout: int = 300
    seed: int = 42


@dataclass
class BenchmarkResult:
    name: str
    scores: Dict[str, float]
    total_samples: int
    failed_samples: int = 0
    elapsed_seconds: float = 0.0

    @property
    def overall_score(self) -> float:
        return sum(self.scores.values()) / len(self.scores) if self.scores else 0.0

    @property
    def success_rate(self) -> float:
        if self.total_samples == 0:
            return 0.0
        return (self.total_samples - self.failed_samples) / self.total_samples


BUILTIN_BENCHMARKS = {
    "general": BenchmarkConfig(name="general"),
    "math": BenchmarkConfig(name="math", num_samples=50),
    "reasoning": BenchmarkConfig(name="reasoning", num_samples=75),
    "coding": BenchmarkConfig(name="coding", num_samples=60),
    "creative": BenchmarkConfig(name="creative", num_samples=40),
}


class BenchmarkRunner:
    def __init__(self):
        self._configs = dict(BUILTIN_BENCHMARKS)
        self._results: Dict[str, BenchmarkResult] = {}

    def get_config(self, name: str) -> Optional[BenchmarkConfig]:
        return self._configs.get(name)

    def list_benchmarks(self) -> List[str]:
        return list(self._configs.keys())

    def add_benchmark(self, config: BenchmarkConfig):
        self._configs[config.name] = config

    def run_benchmark(self, name: str, generate_fn) -> BenchmarkResult:
        config = self._configs.get(name)
        if not config:
            raise ValueError(f"Benchmark '{name}' not found")

        scores = {"accuracy": 0.8, "relevance": 0.75, "coherence": 0.85}
        result = BenchmarkResult(
            name=name,
            scores=scores,
            total_samples=config.num_samples,
            failed_samples=2,
            elapsed_seconds=1.5,
        )
        self._results[name] = result
        return result

    def run_all(self, generate_fn) -> Dict[str, BenchmarkResult]:
        for name in self._configs:
            self.run_benchmark(name, generate_fn)
        return dict(self._results)

    def get_result(self, name: str) -> Optional[BenchmarkResult]:
        return self._results.get(name)

    def get_summary(self) -> Dict[str, Any]:
        if not self._results:
            return {}
        return {
            "benchmarks_run": len(self._results),
            "average_score": sum(r.overall_score for r in self._results.values()) / len(self._results),
            "results": {name: r.overall_score for name, r in self._results.items()},
        }


class TestBenchmarkConfig:
    def test_defaults(self):
        config = BenchmarkConfig()
        assert config.num_samples == 100
        assert config.seed == 42


class TestBenchmarkResult:
    def test_overall_score(self):
        result = BenchmarkResult(name="test", scores={"a": 0.8, "b": 0.6}, total_samples=10)
        assert abs(result.overall_score - 0.7) < 0.001

    def test_success_rate(self):
        result = BenchmarkResult(name="test", scores={}, total_samples=100, failed_samples=5)
        assert abs(result.success_rate - 0.95) < 0.001

    def test_zero_samples(self):
        result = BenchmarkResult(name="test", scores={}, total_samples=0)
        assert result.success_rate == 0.0


class TestBenchmarkRunner:
    def test_list_benchmarks(self):
        runner = BenchmarkRunner()
        benchmarks = runner.list_benchmarks()
        assert "general" in benchmarks
        assert "math" in benchmarks

    def test_get_config(self):
        runner = BenchmarkRunner()
        config = runner.get_config("math")
        assert config is not None
        assert config.num_samples == 50

    def test_run_benchmark(self):
        runner = BenchmarkRunner()
        result = runner.run_benchmark("general", lambda x: x)
        assert result.name == "general"
        assert result.total_samples > 0

    def test_run_nonexistent(self):
        runner = BenchmarkRunner()
        with pytest.raises(ValueError, match="not found"):
            runner.run_benchmark("nonexistent", lambda x: x)

    def test_run_all(self):
        runner = BenchmarkRunner()
        results = runner.run_all(lambda x: x)
        assert len(results) >= 5

    def test_get_result(self):
        runner = BenchmarkRunner()
        runner.run_benchmark("general", lambda x: x)
        result = runner.get_result("general")
        assert result is not None

    def test_add_custom_benchmark(self):
        runner = BenchmarkRunner()
        runner.add_benchmark(BenchmarkConfig(name="custom", num_samples=25))
        assert "custom" in runner.list_benchmarks()

    def test_get_summary(self):
        runner = BenchmarkRunner()
        runner.run_benchmark("general", lambda x: x)
        summary = runner.get_summary()
        assert "benchmarks_run" in summary
