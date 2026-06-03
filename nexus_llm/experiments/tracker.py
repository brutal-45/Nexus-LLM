"""Experiment tracker for Nexus-LLM.

Provides metric tracking, history queries, best-value lookups,
and export to JSON, CSV, and Markdown.
"""

import csv
import io
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Track and query experiment metrics over time.

    The tracker stores metric records per experiment and supports
    exporting the history in multiple formats.

    Example::

        tracker = ExperimentTracker()
        tracker.track("exp-001", {"loss": 0.42, "accuracy": 0.88})
        best_step, best_val = tracker.get_best("exp-001", "loss")
        md = tracker.export("exp-001", format="markdown")
    """

    def __init__(self) -> None:
        # experiment_id -> list of metric records
        self._records: Dict[str, List[Dict[str, Any]]] = {}
        self._global_step: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def track(self, experiment_id: str, metrics_dict: Dict[str, float]) -> None:
        """Record a snapshot of metrics for an experiment.

        Each call increments the internal step counter for the experiment.

        Args:
            experiment_id: Unique experiment identifier.
            metrics_dict: Mapping of metric name to numeric value.
        """
        if experiment_id not in self._records:
            self._records[experiment_id] = []
            self._global_step[experiment_id] = 0

        step = self._global_step[experiment_id]
        record: Dict[str, Any] = {"step": step, "metrics": dict(metrics_dict)}
        self._records[experiment_id].append(record)
        self._global_step[experiment_id] = step + 1
        logger.debug(
            "Tracked metrics for %s at step %d: %s",
            experiment_id, step, metrics_dict,
        )

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def get_history(self, experiment_id: str) -> List[Dict[str, Any]]:
        """Return the full metric history for an experiment.

        Args:
            experiment_id: Experiment identifier.

        Returns:
            List of records, each with ``"step"`` and ``"metrics"`` keys.

        Raises:
            KeyError: If the experiment has no tracked records.
        """
        if experiment_id not in self._records:
            raise KeyError(f"No tracked records for experiment {experiment_id!r}")
        return list(self._records[experiment_id])

    def get_best(
        self,
        experiment_id: str,
        metric: str,
        mode: str = "min",
    ) -> Tuple[int, float]:
        """Find the step with the best (min or max) value for a metric.

        Args:
            experiment_id: Experiment identifier.
            metric: Name of the metric to evaluate.
            mode: ``"min"`` to find the lowest value, ``"max"`` for highest.

        Returns:
            Tuple of ``(step, value)`` at the best point.

        Raises:
            KeyError: If the experiment or metric is not found.
            ValueError: If no records exist.
        """
        history = self.get_history(experiment_id)
        if not history:
            raise ValueError(f"No records for experiment {experiment_id!r}")

        candidates: List[Tuple[int, float]] = []
        for record in history:
            if metric not in record["metrics"]:
                continue
            candidates.append((record["step"], record["metrics"][metric]))

        if not candidates:
            raise KeyError(
                f"Metric {metric!r} not found in records for experiment "
                f"{experiment_id!r}"
            )

        if mode == "min":
            return min(candidates, key=lambda x: x[1])
        return max(candidates, key=lambda x: x[1])

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export(self, experiment_id: str, format: str = "json") -> str:
        """Export metric history in the requested format.

        Args:
            experiment_id: Experiment identifier.
            format: One of ``"json"``, ``"csv"``, or ``"markdown"``.

        Returns:
            String representation of the exported data.

        Raises:
            KeyError: If the experiment has no tracked records.
            ValueError: If the format is unsupported.
        """
        history = self.get_history(experiment_id)
        if format == "json":
            return self._export_json(history, experiment_id)
        elif format == "csv":
            return self._export_csv(history)
        elif format == "markdown":
            return self._export_markdown(history, experiment_id)
        else:
            raise ValueError(
                f"Unsupported export format {format!r}; "
                f"expected 'json', 'csv', or 'markdown'."
            )

    # ------------------------------------------------------------------
    # Private export helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _export_json(history: List[Dict[str, Any]], experiment_id: str) -> str:
        payload = {"experiment_id": experiment_id, "records": history}
        return json.dumps(payload, indent=2)

    @staticmethod
    def _export_csv(history: List[Dict[str, Any]]) -> str:
        if not history:
            return ""
        # Collect all metric names across all records
        metric_names: List[str] = sorted(
            {name for rec in history for name in rec["metrics"]}
        )
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["step"] + metric_names)
        for rec in history:
            row = [rec["step"]] + [
                rec["metrics"].get(name, "") for name in metric_names
            ]
            writer.writerow(row)
        return output.getvalue()

    @staticmethod
    def _export_markdown(
        history: List[Dict[str, Any]], experiment_id: str
    ) -> str:
        if not history:
            return f"## Experiment: {experiment_id}\n\n_No records._\n"
        metric_names: List[str] = sorted(
            {name for rec in history for name in rec["metrics"]}
        )
        lines = [f"## Experiment: {experiment_id}", ""]
        header = "| Step | " + " | ".join(metric_names) + " |"
        separator = "|------|" + "|".join(["------" for _ in metric_names]) + "|"
        lines.append(header)
        lines.append(separator)
        for rec in history:
            values = " | ".join(
                str(rec["metrics"].get(name, "—")) for name in metric_names
            )
            lines.append(f"| {rec['step']} | {values} |")
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"<ExperimentTracker experiments={len(self._records)} "
            f"total_records={sum(len(r) for r in self._records.values())}>"
        )
