"""Benchmark report for Nexus-LLM.

Collects, summarises, and serialises benchmark results.  Supports
comparison between two reports and export to JSON/CSV.
"""

import csv
import io
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class BenchmarkReport:
    """Aggregates benchmark results into a structured report.

    Example::

        report = BenchmarkReport(title="My Benchmark Run")
        report.add_result("inference_speed", {"avg_time": 0.12, "tokens_per_sec": 850})
        report.add_result("coherence", {"avg_score": 0.87})
        print(report.summary())
    """

    def __init__(self, title: str = "Benchmark Report") -> None:
        self.title = title
        self.created_at = datetime.now(timezone.utc).isoformat()
        self._results: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Mutating operations
    # ------------------------------------------------------------------

    def add_result(self, name: str, result_dict: Dict[str, Any]) -> None:
        """Add a named benchmark result to the report.

        Args:
            name: Identifier for the benchmark (e.g. ``"inference_speed"``).
            result_dict: Mapping of metric names to values.

        Raises:
            ValueError: If *name* is already present in the report.
        """
        if name in self._results:
            logger.warning("Overwriting existing result for %r", name)
        self._results[name] = dict(result_dict)
        logger.debug("Added result %r with %d metric(s)", name, len(result_dict))

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get_results(self) -> Dict[str, Dict[str, Any]]:
        """Return a deep copy of all results."""
        return {k: dict(v) for k, v in self._results.items()}

    def summary(self) -> str:
        """Return a human-readable summary string.

        The summary includes the report title, creation timestamp,
        and a table of all benchmark metrics.
        """
        lines: List[str] = [
            f"{'=' * 60}",
            f"  {self.title}",
            f"  Created: {self.created_at}",
            f"{'=' * 60}",
        ]

        if not self._results:
            lines.append("  (no results)")
        else:
            for name, metrics in self._results.items():
                lines.append(f"\n  [{name}]")
                for metric, value in metrics.items():
                    if isinstance(value, float):
                        lines.append(f"    {metric}: {value:.4f}")
                    else:
                        lines.append(f"    {metric}: {value}")

        lines.append(f"\n{'=' * 60}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_json(self, indent: int = 2) -> str:
        """Serialise the report to a JSON string.

        Args:
            indent: Number of spaces for JSON indentation.

        Returns:
            A JSON-formatted string of the full report.
        """
        payload = {
            "title": self.title,
            "created_at": self.created_at,
            "results": self._results,
        }
        return json.dumps(payload, indent=indent, default=str)

    def to_csv(self) -> str:
        """Serialise the report to CSV format.

        Each row contains: benchmark_name, metric_name, metric_value.

        Returns:
            A CSV-formatted string.
        """
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["benchmark", "metric", "value"])
        for name, metrics in self._results.items():
            for metric, value in metrics.items():
                writer.writerow([name, metric, value])
        return output.getvalue()

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def compare(self, other: "BenchmarkReport") -> Dict[str, Any]:
        """Compare this report with another and return a diff.

        For every metric that exists in both reports a ``delta`` (other -
        self) and ``pct_change`` is computed.

        Args:
            other: Another BenchmarkReport to compare against.

        Returns:
            A dict with ``"shared"``, ``"only_self"``, ``"only_other"``,
            and ``"differences"`` keys.
        """
        self_keys = set(self._results.keys())
        other_keys = set(other._results.keys())

        shared = sorted(self_keys & other_keys)
        only_self = sorted(self_keys - other_keys)
        only_other = sorted(other_keys - self_keys)

        differences: Dict[str, Dict[str, Any]] = {}
        for name in shared:
            self_metrics = self._results[name]
            other_metrics = other._results[name]
            all_metric_keys = sorted(set(self_metrics) | set(other_metrics))
            diffs: Dict[str, Any] = {}

            for mk in all_metric_keys:
                sv = self_metrics.get(mk)
                ov = other_metrics.get(mk)
                if sv is None or ov is None:
                    diffs[mk] = {"self": sv, "other": ov, "delta": None}
                elif isinstance(sv, (int, float)) and isinstance(ov, (int, float)):
                    delta = ov - sv
                    pct = (delta / abs(sv) * 100) if sv != 0 else None
                    diffs[mk] = {
                        "self": sv,
                        "other": ov,
                        "delta": round(delta, 6),
                        "pct_change": round(pct, 2) if pct is not None else None,
                    }
                else:
                    diffs[mk] = {"self": sv, "other": ov, "delta": None}

            if diffs:
                differences[name] = diffs

        return {
            "shared": shared,
            "only_self": only_self,
            "only_other": only_other,
            "differences": differences,
        }
