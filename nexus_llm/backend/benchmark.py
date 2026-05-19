"""Backend benchmarking for Nexus-LLM.

Provides inference speed, token throughput, memory usage, and latency
measurement tools for benchmarking model performance.
"""

import torch
import time
import statistics
from typing import Optional, Dict, Any, List, Callable, Tuple
from dataclasses import dataclass, field
import logging
import json

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""
    name: str
    num_iterations: int = 0
    total_time_seconds: float = 0.0
    mean_time_seconds: float = 0.0
    median_time_seconds: float = 0.0
    p95_time_seconds: float = 0.0
    p99_time_seconds: float = 0.0
    min_time_seconds: float = 0.0
    max_time_seconds: float = 0.0
    std_dev_seconds: float = 0.0
    tokens_per_second: float = 0.0
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    memory_before_mb: float = 0.0
    memory_after_mb: float = 0.0
    memory_peak_mb: float = 0.0
    gpu_utilization_pct: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "num_iterations": self.num_iterations,
            "total_time_seconds": round(self.total_time_seconds, 6),
            "mean_time_seconds": round(self.mean_time_seconds, 6),
            "median_time_seconds": round(self.median_time_seconds, 6),
            "p95_time_seconds": round(self.p95_time_seconds, 6),
            "p99_time_seconds": round(self.p99_time_seconds, 6),
            "min_time_seconds": round(self.min_time_seconds, 6),
            "max_time_seconds": round(self.max_time_seconds, 6),
            "std_dev_seconds": round(self.std_dev_seconds, 6),
            "tokens_per_second": round(self.tokens_per_second, 2),
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "memory_before_mb": round(self.memory_before_mb, 2),
            "memory_after_mb": round(self.memory_after_mb, 2),
            "memory_peak_mb": round(self.memory_peak_mb, 2),
            "gpu_utilization_pct": round(self.gpu_utilization_pct, 2),
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def summary(self) -> str:
        lines = [
            f"=== Benchmark: {self.name} ===",
            f"Iterations: {self.num_iterations}",
            f"Total Time: {self.total_time_seconds:.4f}s",
            f"Mean Latency: {self.mean_time_seconds * 1000:.2f}ms",
            f"Median Latency: {self.median_time_seconds * 1000:.2f}ms",
            f"P95 Latency: {self.p95_time_seconds * 1000:.2f}ms",
            f"P99 Latency: {self.p99_time_seconds * 1000:.2f}ms",
            f"Tokens/sec: {self.tokens_per_second:.2f}",
            f"Total Tokens: {self.total_tokens}",
            f"Peak Memory: {self.memory_peak_mb:.2f} MB",
        ]
        return "\n".join(lines)


class LatencyMeasurer:
    """Measures latency of callable operations with warmup."""

    def __init__(self, warmup_iterations: int = 3, use_cuda_events: bool = True):
        self._warmup_iterations = warmup_iterations
        self._use_cuda_events = use_cuda_events and torch.cuda.is_available()

    def measure(
        self,
        fn: Callable,
        iterations: int = 10,
        args: Tuple = (),
        kwargs: Optional[Dict] = None,
    ) -> BenchmarkResult:
        """Measure the latency of a function over multiple iterations."""
        kwargs = kwargs or {}

        for _ in range(self._warmup_iterations):
            fn(*args, **kwargs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        latencies = []
        total_tokens = 0
        result_data = None

        for i in range(iterations):
            if self._use_cuda_events:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()

            start_time = time.perf_counter()
            result_data = fn(*args, **kwargs)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            end_time = time.perf_counter()

            if self._use_cuda_events:
                end_event.record()
                torch.cuda.synchronize()
                elapsed = start_event.elapsed_time(end_event) / 1000.0
            else:
                elapsed = end_time - start_time

            latencies.append(elapsed)

            if isinstance(result_data, dict):
                total_tokens += result_data.get("tokens", 0)
            elif isinstance(result_data, (list, tuple)) and len(result_data) > 0:
                if hasattr(result_data[0], "num_tokens"):
                    total_tokens += result_data[0].num_tokens

        return self._compute_result("latency_benchmark", latencies, iterations, total_tokens)

    def _compute_result(
        self,
        name: str,
        latencies: List[float],
        iterations: int,
        total_tokens: int,
    ) -> BenchmarkResult:
        """Compute benchmark statistics from latency measurements."""
        sorted_lat = sorted(latencies)
        total_time = sum(latencies)

        result = BenchmarkResult(
            name=name,
            num_iterations=iterations,
            total_time_seconds=total_time,
            mean_time_seconds=statistics.mean(latencies),
            median_time_seconds=statistics.median(latencies),
            p95_time_seconds=self._percentile(sorted_lat, 95),
            p99_time_seconds=self._percentile(sorted_lat, 99),
            min_time_seconds=min(latencies),
            max_time_seconds=max(latencies),
            std_dev_seconds=statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
            total_tokens=total_tokens,
            tokens_per_second=total_tokens / total_time if total_time > 0 else 0.0,
        )
        return result

    @staticmethod
    def _percentile(sorted_data: List[float], pct: float) -> float:
        """Compute a percentile from sorted data."""
        if not sorted_data:
            return 0.0
        idx = (pct / 100.0) * (len(sorted_data) - 1)
        lower = int(idx)
        upper = min(lower + 1, len(sorted_data) - 1)
        frac = idx - lower
        return sorted_data[lower] * (1 - frac) + sorted_data[upper] * frac


class ThroughputBenchmarker:
    """Benchmarks token generation throughput."""

    def __init__(self, model_manager=None):
        self._model_manager = model_manager

    def benchmark_generation(
        self,
        prompt: str = "The quick brown fox jumps over the lazy dog.",
        max_new_tokens: int = 128,
        num_iterations: int = 5,
        batch_sizes: Optional[List[int]] = None,
        model_id: Optional[str] = None,
    ) -> Dict[int, BenchmarkResult]:
        """Benchmark generation throughput at different batch sizes."""
        if batch_sizes is None:
            batch_sizes = [1]

        results = {}
        for batch_size in batch_sizes:
            result = self._benchmark_batch(
                prompt, max_new_tokens, batch_size, num_iterations, model_id
            )
            results[batch_size] = result

        return results

    def _benchmark_batch(
        self,
        prompt: str,
        max_new_tokens: int,
        batch_size: int,
        num_iterations: int,
        model_id: Optional[str],
    ) -> BenchmarkResult:
        """Benchmark a specific batch size."""
        if self._model_manager is None:
            raise RuntimeError("Model manager required for throughput benchmarking")

        model = self._model_manager.get_model(model_id)
        tokenizer = self._model_manager.get_tokenizer(model_id)
        device = next(model.parameters()).device

        prompts = [prompt] * batch_size
        input_ids = tokenizer(prompts, return_tensors="pt", padding=True).input_ids.to(device)
        prompt_tokens = input_ids.shape[-1]

        latencies = []
        total_completion_tokens = 0

        for _ in range(3):
            with torch.no_grad():
                _ = model.generate(input_ids, max_new_tokens=max_new_tokens)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        for _ in range(num_iterations):
            start = time.perf_counter()
            with torch.no_grad():
                output = model.generate(input_ids, max_new_tokens=max_new_tokens)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            latencies.append(elapsed)
            total_completion_tokens += (output.shape[-1] - prompt_tokens) * batch_size

        sorted_lat = sorted(latencies)
        total_time = sum(latencies)

        return BenchmarkResult(
            name=f"throughput_batch_{batch_size}",
            num_iterations=num_iterations,
            total_time_seconds=total_time,
            mean_time_seconds=statistics.mean(latencies),
            median_time_seconds=statistics.median(latencies),
            p95_time_seconds=LatencyMeasurer._percentile(sorted_lat, 95),
            p99_time_seconds=LatencyMeasurer._percentile(sorted_lat, 99),
            min_time_seconds=min(latencies),
            max_time_seconds=max(latencies),
            std_dev_seconds=statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
            tokens_per_second=total_completion_tokens / total_time if total_time > 0 else 0.0,
            total_tokens=total_completion_tokens,
            prompt_tokens=prompt_tokens * batch_size * num_iterations,
            completion_tokens=total_completion_tokens,
            metadata={"batch_size": batch_size, "max_new_tokens": max_new_tokens},
        )


class MemoryBenchmarker:
    """Benchmarks memory usage during model inference."""

    @staticmethod
    def benchmark_memory(
        model: Any,
        tokenizer: Any,
        prompt: str = "Hello, world!",
        max_new_tokens_list: Optional[List[int]] = None,
        batch_sizes: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Benchmark memory usage at different generation lengths and batch sizes."""
        if max_new_tokens_list is None:
            max_new_tokens_list = [32, 64, 128, 256, 512]
        if batch_sizes is None:
            batch_sizes = [1]

        results = {}
        device = next(model.parameters()).device

        for batch_size in batch_sizes:
            for max_tokens in max_new_tokens_list:
                key = f"batch_{batch_size}_tokens_{max_tokens}"

                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.empty_cache()
                    mem_before = torch.cuda.memory_allocated() / (1024 * 1024)

                prompts = [prompt] * batch_size
                input_ids = tokenizer(prompts, return_tensors="pt", padding=True).input_ids.to(device)

                with torch.no_grad():
                    _ = model.generate(input_ids, max_new_tokens=max_tokens)

                if torch.cuda.is_available():
                    mem_after = torch.cuda.memory_allocated() / (1024 * 1024)
                    mem_peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
                else:
                    mem_after = 0.0
                    mem_peak = 0.0
                    mem_before = 0.0

                results[key] = {
                    "batch_size": batch_size,
                    "max_new_tokens": max_tokens,
                    "memory_before_mb": round(mem_before, 2),
                    "memory_after_mb": round(mem_after, 2),
                    "memory_peak_mb": round(mem_peak, 2),
                    "memory_delta_mb": round(mem_after - mem_before, 2),
                }

        return results


def run_full_benchmark(
    model_manager,
    model_id: Optional[str] = None,
    prompt: str = "Write a detailed essay about artificial intelligence and its impact on society.",
    max_new_tokens: int = 256,
    num_iterations: int = 5,
    batch_sizes: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Run a comprehensive benchmark suite."""
    batch_sizes = batch_sizes or [1]

    throughput_benchmarker = ThroughputBenchmarker(model_manager)
    throughput_results = throughput_benchmarker.benchmark_generation(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        num_iterations=num_iterations,
        batch_sizes=batch_sizes,
        model_id=model_id,
    )

    model = model_manager.get_model(model_id)
    tokenizer = model_manager.get_tokenizer(model_id)
    memory_results = MemoryBenchmarker.benchmark_memory(model, tokenizer, prompt)

    combined = {
        "throughput": {str(k): v.to_dict() for k, v in throughput_results.items()},
        "memory": memory_results,
        "model_info": {
            "model_id": model_id or model_manager.get_active_model_id(),
            "num_parameters": sum(p.numel() for p in model.parameters()),
        },
    }
    return combined
