"""Quantization benchmarking for Nexus-LLM.

Provides QuantizationBenchmark which runs a model through multiple
quantization configurations, collects accuracy / size / speed metrics,
and renders a human-readable comparison table.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from nexus_llm.quantization.config import QuantConfig, VALID_METHODS
from nexus_llm.quantization.quantizer import Quantizer

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Container for a single quantization benchmark run.

    Attributes:
        config: The QuantConfig that was benchmarked.
        accuracy_metrics: Output of ``Quantizer.measure_accuracy``.
        size_metrics: Output of ``Quantizer.compare_sizes``.
        quantization_time_ms: Wall-clock time for quantization in ms.
    """

    config: QuantConfig
    accuracy_metrics: Dict[str, Any] = field(default_factory=dict)
    size_metrics: Dict[str, Any] = field(default_factory=dict)
    quantization_time_ms: float = 0.0

    def summary(self) -> Dict[str, Any]:
        """Return a flat summary dict for tabulation."""
        return {
            "method": self.config.method,
            "group_size": self.config.group_size,
            "sym": self.config.sym,
            "avg_output_diff": self.accuracy_metrics.get("avg_output_diff", float("nan")),
            "speedup": self.accuracy_metrics.get("speedup", float("nan")),
            "compression_ratio": self.size_metrics.get("compression_ratio", float("nan")),
            "reduction_pct": self.size_metrics.get("reduction_pct", float("nan")),
            "quantization_time_ms": round(self.quantization_time_ms, 2),
        }


class QuantizationBenchmark:
    """Benchmark multiple quantization configurations against a model.

    Usage::

        bench = QuantizationBenchmark()
        configs = [
            QuantConfig(method="int8"),
            QuantConfig(method="int4", group_size=128),
            QuantConfig(method="fp16"),
        ]
        results = bench.benchmark(model, configs, test_data=samples)
        print(bench.plot_comparison(results))
    """

    def __init__(self, quantizer: Optional[Quantizer] = None) -> None:
        """Initialise the benchmark runner.

        Args:
            quantizer: A Quantizer instance.  If *None*, a default one
                is created.
        """
        self._quantizer = quantizer or Quantizer()

    # ------------------------------------------------------------------
    # Main benchmark loop
    # ------------------------------------------------------------------

    def benchmark(
        self,
        model: Any,
        configs: List[QuantConfig],
        test_data: Optional[List[Dict[str, Any]]] = None,
    ) -> List[BenchmarkResult]:
        """Run benchmarking across multiple quantization configurations.

        For each configuration the method:

        1. Quantizes the model and measures quantization time.
        2. Compares sizes of the original and quantized models.
        3. Optionally measures accuracy if *test_data* is provided.

        Args:
            model: The original (un-quantized) model.
            configs: List of QuantConfig instances to benchmark.
            test_data: Optional test data for accuracy measurement.
                If *None*, accuracy metrics are omitted.

        Returns:
            List of BenchmarkResult objects, one per configuration.
        """
        results: List[BenchmarkResult] = []

        logger.info(
            "Starting quantization benchmark with %d configuration(s)",
            len(configs),
        )

        for i, config in enumerate(configs, 1):
            logger.info(
                "[%d/%d] Benchmarking method=%s group_size=%d sym=%s",
                i,
                len(configs),
                config.method,
                config.group_size,
                config.sym,
            )

            result = BenchmarkResult(config=config)

            # --- Quantize & time ---
            t0 = time.perf_counter()
            try:
                quantized = self._quantizer.quantize(model, config)
            except Exception as exc:
                logger.error("Quantization failed for %s: %s", config.method, exc)
                result.accuracy_metrics = {"error": str(exc)}
                results.append(result)
                continue
            result.quantization_time_ms = (time.perf_counter() - t0) * 1000

            # --- Size comparison ---
            result.size_metrics = self._quantizer.compare_sizes(model, quantized)

            # --- Accuracy measurement ---
            if test_data:
                try:
                    result.accuracy_metrics = self._quantizer.measure_accuracy(
                        model, quantized, test_data
                    )
                except Exception as exc:
                    logger.warning(
                        "Accuracy measurement failed for %s: %s",
                        config.method,
                        exc,
                    )
                    result.accuracy_metrics = {"error": str(exc)}
            else:
                result.accuracy_metrics = {
                    "num_samples": 0,
                    "avg_output_diff": 0.0,
                    "max_output_diff": 0.0,
                    "speedup": 1.0,
                }

            results.append(result)

        logger.info("Benchmark complete — %d results collected", len(results))
        return results

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    @staticmethod
    def plot_comparison(results: List[BenchmarkResult]) -> str:
        """Render a text-table comparison of benchmark results.

        Args:
            results: List of BenchmarkResult objects.

        Returns:
            A formatted multi-line string containing the comparison table.
        """
        if not results:
            return "(no results to display)"

        # Build rows
        rows: List[Dict[str, Any]] = [r.summary() for r in results]

        # Column definitions: (header, key, width, align_right)
        columns = [
            ("Method", "method", 8, False),
            ("GroupSz", "group_size", 8, True),
            ("Sym", "sym", 5, False),
            ("AvgDiff", "avg_output_diff", 10, True),
            ("Speedup", "speedup", 8, True),
            ("Compress", "compression_ratio", 9, True),
            ("Reduc%", "reduction_pct", 8, True),
            ("QuantMs", "quantization_time_ms", 9, True),
        ]

        # Header line
        header = " | ".join(h.ljust(w) if not ar else h.rjust(w) for h, _, w, ar in columns)
        separator = "-+-".join("-" * w for _, _, w, _ in columns)

        lines: List[str] = [
            "Quantization Benchmark Comparison",
            "=" * len(header),
            header,
            separator,
        ]

        for row in rows:
            cells = []
            for header_name, key, width, align_right in columns:
                value = row.get(key, "N/A")
                if isinstance(value, float):
                    try:
                        text = f"{value:.4f}" if abs(value) < 100 else f"{value:.1f}"
                    except (ValueError, TypeError):
                        text = str(value)
                else:
                    text = str(value)

                if align_right:
                    cells.append(text.rjust(width))
                else:
                    cells.append(text.ljust(width))
            lines.append(" | ".join(cells))

        lines.append(separator)
        lines.append(f"Total configurations: {len(results)}")

        table = "\n".join(lines)
        logger.debug("Comparison table:\n%s", table)
        return table
