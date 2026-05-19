"""Test model benchmarking for Nexus-LLM."""
import time
import pytest
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from statistics import mean, stdev


@dataclass
class BenchmarkResult:
    model_name: str
    total_time: float
    num_requests: int
    latencies: List[float]
    tokens_generated: int = 0

    @property
    def avg_latency(self) -> float:
        return mean(self.latencies) if self.latencies else 0.0

    @property
    def p50_latency(self) -> float:
        if not self.latencies:
            return 0.0
        sorted_lat = sorted(self.latencies)
        idx = len(sorted_lat) // 2
        return sorted_lat[idx]

    @property
    def p95_latency(self) -> float:
        if not self.latencies:
            return 0.0
        sorted_lat = sorted(self.latencies)
        idx = int(len(sorted_lat) * 0.95)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]

    @property
    def p99_latency(self) -> float:
        if not self.latencies:
            return 0.0
        sorted_lat = sorted(self.latencies)
        idx = int(len(sorted_lat) * 0.99)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]

    @property
    def throughput(self) -> float:
        if self.total_time == 0:
            return 0.0
        return self.num_requests / self.total_time

    @property
    def tokens_per_second(self) -> float:
        if self.total_time == 0:
            return 0.0
        return self.tokens_generated / self.total_time


@dataclass
class BenchmarkConfig:
    num_warmup: int = 2
    num_iterations: int = 10
    prompt_length: int = 128
    max_new_tokens: int = 64
    batch_sizes: List[int] = field(default_factory=lambda: [1])


class ModelBenchmark:
    def __init__(self, config: BenchmarkConfig = None):
        self._config = config or BenchmarkConfig()

    def run_benchmark(self, generate_fn, model_name: str = "model") -> BenchmarkResult:
        latencies = []
        total_tokens = 0

        for i in range(self._config.num_warmup + self._config.num_iterations):
            start = time.perf_counter()
            result = generate_fn("benchmark prompt")
            elapsed = time.perf_counter() - start
            if i >= self._config.num_warmup:
                latencies.append(elapsed)
                total_tokens += self._config.max_new_tokens

        return BenchmarkResult(
            model_name=model_name,
            total_time=sum(latencies),
            num_requests=len(latencies),
            latencies=latencies,
            tokens_generated=total_tokens,
        )

    def compare_benchmarks(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        if not results:
            return {}
        comparison = {}
        for r in results:
            comparison[r.model_name] = {
                "avg_latency": r.avg_latency,
                "throughput": r.throughput,
                "tokens_per_second": r.tokens_per_second,
            }
        fastest = min(results, key=lambda r: r.avg_latency)
        comparison["fastest"] = fastest.model_name
        return comparison


class TestBenchmarkResult:
    def test_avg_latency(self):
        result = BenchmarkResult(model_name="test", total_time=1.0, num_requests=4, latencies=[0.1, 0.2, 0.3, 0.4])
        assert abs(result.avg_latency - 0.25) < 0.001

    def test_p50_latency(self):
        result = BenchmarkResult(model_name="test", total_time=1.0, num_requests=4, latencies=[0.1, 0.2, 0.3, 0.4])
        assert result.p50_latency == 0.3

    def test_p95_latency(self):
        latencies = list(range(1, 101))
        result = BenchmarkResult(model_name="test", total_time=1.0, num_requests=100, latencies=latencies)
        assert result.p95_latency >= 90

    def test_throughput(self):
        result = BenchmarkResult(model_name="test", total_time=2.0, num_requests=10, latencies=[0.2]*10)
        assert abs(result.throughput - 5.0) < 0.001

    def test_tokens_per_second(self):
        result = BenchmarkResult(model_name="test", total_time=2.0, num_requests=10, latencies=[0.2]*10, tokens_generated=640)
        assert abs(result.tokens_per_second - 320.0) < 0.001

    def test_empty_latencies(self):
        result = BenchmarkResult(model_name="test", total_time=0, num_requests=0, latencies=[])
        assert result.avg_latency == 0.0
        assert result.throughput == 0.0


class TestBenchmarkConfig:
    def test_defaults(self):
        config = BenchmarkConfig()
        assert config.num_warmup == 2
        assert config.num_iterations == 10

    def test_custom(self):
        config = BenchmarkConfig(num_iterations=50, batch_sizes=[1, 4, 8])
        assert config.num_iterations == 50


class TestModelBenchmark:
    def test_run_benchmark(self):
        bench = ModelBenchmark(BenchmarkConfig(num_warmup=1, num_iterations=3))
        def fake_generate(prompt):
            time.sleep(0.001)
            return "output"
        result = bench.run_benchmark(fake_generate, "test-model")
        assert result.model_name == "test-model"
        assert result.num_requests == 3
        assert result.avg_latency > 0

    def test_compare_benchmarks(self):
        bench = ModelBenchmark()
        r1 = BenchmarkResult(model_name="model_a", total_time=1.0, num_requests=10, latencies=[0.1]*10, tokens_generated=100)
        r2 = BenchmarkResult(model_name="model_b", total_time=2.0, num_requests=10, latencies=[0.2]*10, tokens_generated=100)
        comparison = bench.compare_benchmarks([r1, r2])
        assert comparison["fastest"] == "model_a"
        assert "model_a" in comparison
        assert "model_b" in comparison

    def test_compare_empty(self):
        bench = ModelBenchmark()
        assert bench.compare_benchmarks([]) == {}
