"""Evaluation report for Nexus-LLM.

Collects metric scores, provides serialisation helpers (dict, JSON, CSV),
and generates a human-readable summary.
"""

import csv
import io
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class EvaluationReport:
    """Container for evaluation results.

    Attributes:
        model_name: Identifier of the evaluated model.
        dataset_name: Name of the dataset or benchmark used.
        scores: Mapping of metric names to numeric scores.
        metadata: Arbitrary metadata (timestamp, config, etc.).
    """

    model_name: str = ""
    dataset_name: str = ""
    scores: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if "timestamp" not in self.metadata:
            self.metadata["timestamp"] = datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def add_metric(self, name: str, value: float) -> None:
        """Record a metric score.

        Args:
            name: Metric name (e.g. ``"bleu"``).
            value: Numeric score.
        """
        self.scores[name] = value

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain dict representation of the report."""
        return {
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "scores": dict(self.scores),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationReport":
        """Construct an EvaluationReport from a dict.

        Args:
            data: Dict with keys matching the dataclass fields.

        Returns:
            A populated EvaluationReport.
        """
        return cls(
            model_name=data.get("model_name", ""),
            dataset_name=data.get("dataset_name", ""),
            scores=dict(data.get("scores", {})),
            metadata=dict(data.get("metadata", {})),
        )

    def to_json(self, indent: int = 2) -> str:
        """Return a JSON string of the report.

        Args:
            indent: JSON indentation level.

        Returns:
            JSON-encoded string.
        """
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_json(cls, json_str: str) -> "EvaluationReport":
        """Construct an EvaluationReport from a JSON string."""
        return cls.from_dict(json.loads(json_str))

    def to_csv(self) -> str:
        """Return a CSV string with one row per metric.

        Columns: ``metric_name``, ``score``, ``model_name``, ``dataset_name``.
        """
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["metric_name", "score", "model_name", "dataset_name"])
        for name, value in sorted(self.scores.items()):
            writer.writerow([name, value, self.model_name, self.dataset_name])
        return output.getvalue()

    # ------------------------------------------------------------------
    # Human-readable summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a formatted multi-line summary of the report.

        Example output::

            Evaluation Report
            =================
            Model:    gpt2-medium
            Dataset:  perplexity_benchmark
            Date:     2025-01-15T10:30:00+00:00

            Metrics
            -------
            perplexity        : 23.4567
            bleu              : 0.3421
            rouge1            : 0.5123
        """
        lines: List[str] = [
            "Evaluation Report",
            "=" * 40,
            f"Model:    {self.model_name}",
            f"Dataset:  {self.dataset_name}",
            f"Date:     {self.metadata.get('timestamp', 'N/A')}",
            "",
            "Metrics",
            "-" * 30,
        ]
        for name, value in sorted(self.scores.items()):
            lines.append(f"{name:<20s}: {value:.4f}")
        if not self.scores:
            lines.append("(no metrics recorded)")
        return "\n".join(lines)
