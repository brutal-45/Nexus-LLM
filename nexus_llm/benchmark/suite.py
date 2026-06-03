"""Benchmark suite for Nexus-LLM.

Orchestrates multiple benchmarks, collects results into a
BenchmarkReport, and provides a simple API for running individual
or all registered benchmarks.
"""

import logging
import time
from typing import Any, Callable, Dict, List, Optional

from nexus_llm.benchmark.report import BenchmarkReport

logger = logging.getLogger(__name__)


class BenchmarkSuite:
    """Orchestrator that manages and runs named benchmarks.

    Example::

        suite = BenchmarkSuite()
        suite.add_benchmark("speed", my_speed_fn)
        suite.add_benchmark("quality", my_quality_fn)

        report = suite.run_all()
        print(report.summary())
    """

    def __init__(self, title: str = "Benchmark Suite") -> None:
        self._title = title
        self._benchmarks: Dict[str, Callable[[], Dict[str, Any]]] = {}

    # ------------------------------------------------------------------
    # Benchmark management
    # ------------------------------------------------------------------

    def add_benchmark(
        self,
        name: str,
        benchmark_fn: Callable[[], Dict[str, Any]],
    ) -> None:
        """Register a named benchmark function.

        Args:
            name: Unique name for the benchmark.
            benchmark_fn: A callable that returns a dict of metrics.

        Raises:
            ValueError: If *benchmark_fn* is not callable.
            ValueError: If *name* is already registered.
        """
        if not callable(benchmark_fn):
            raise ValueError(f"Benchmark function for {name!r} must be callable")
        if name in self._benchmarks:
            raise ValueError(
                f"Benchmark {name!r} is already registered. "
                f"Remove it first or use a different name."
            )
        self._benchmarks[name] = benchmark_fn
        logger.info("Registered benchmark %r", name)

    def list_benchmarks(self) -> List[str]:
        """Return the names of all registered benchmarks, sorted alphabetically."""
        return sorted(self._benchmarks.keys())

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(self, name: str) -> Dict[str, Any]:
        """Run a single benchmark by name.

        Args:
            name: The registered benchmark name.

        Returns:
            A dict of metric results from the benchmark function.

        Raises:
            ValueError: If *name* is not a registered benchmark.
        """
        if name not in self._benchmarks:
            raise ValueError(
                f"Unknown benchmark {name!r}. "
                f"Available: {self.list_benchmarks()}"
            )

        fn = self._benchmarks[name]
        logger.info("Running benchmark %r …", name)
        start = time.perf_counter()

        try:
            result = fn()
        except Exception as exc:
            logger.exception("Benchmark %r failed", name)
            result = {"error": str(exc), "success": False}

        elapsed = time.perf_counter() - start
        result["_elapsed_sec"] = round(elapsed, 6)
        result["_success"] = result.get("success", True)

        logger.info(
            "Benchmark %r completed in %.4fs — %d metric(s)",
            name,
            elapsed,
            len(result),
        )
        return result

    def run_all(self) -> BenchmarkReport:
        """Run every registered benchmark and return a BenchmarkReport.

        Benchmarks are executed sequentially.  If a benchmark raises
        an exception its result is recorded with an ``"error"`` key.

        Returns:
            A BenchmarkReport containing all results.
        """
        report = BenchmarkReport(title=self._title)
        logger.info("Running all benchmarks (%d registered) …", len(self._benchmarks))

        for name in self.list_benchmarks():
            try:
                result = self.run(name)
                report.add_result(name, result)
            except Exception as exc:
                logger.exception("Benchmark %r raised unexpectedly", name)
                report.add_result(name, {"error": str(exc), "_success": False})

        logger.info("All benchmarks complete — %d result(s)", len(report.get_results()))
        return report
