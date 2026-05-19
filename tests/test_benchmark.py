"""Tests for benchmarking."""
import pytest
import time
import torch


class SimpleBenchmark:
    """Simple benchmarking utility for testing."""
    def __init__(self):
        self.results = {}

    def measure(self, name, fn, iterations=10):
        start = time.perf_counter()
        for _ in range(iterations):
            fn()
        elapsed = time.perf_counter() - start
        self.results[name] = {
            "total_time": elapsed,
            "avg_time": elapsed / iterations,
            "iterations": iterations,
        }
        return elapsed

    def get_result(self, name):
        return self.results.get(name)


@pytest.fixture
def benchmark():
    return SimpleBenchmark()


def test_benchmark_measure(benchmark):
    """Test measuring a function."""
    elapsed = benchmark.measure("simple", lambda: sum(range(100)), iterations=100)
    assert elapsed > 0
    assert "simple" in benchmark.results


def test_benchmark_avg_time(benchmark):
    """Test average time calculation."""
    benchmark.measure("test", lambda: None, iterations=10)
    result = benchmark.get_result("test")
    assert result["avg_time"] > 0
    assert result["iterations"] == 10


def test_benchmark_tensor_op(benchmark):
    """Test benchmarking a tensor operation."""
    a = torch.randn(100, 100)
    b = torch.randn(100, 100)
    benchmark.measure("matmul", lambda: torch.mm(a, b), iterations=10)
    result = benchmark.get_result("matmul")
    assert result["total_time"] > 0


def test_benchmark_multiple_measures(benchmark):
    """Test measuring multiple operations."""
    benchmark.measure("op1", lambda: 1 + 1, iterations=100)
    benchmark.measure("op2", lambda: 2 * 2, iterations=100)
    assert len(benchmark.results) == 2
