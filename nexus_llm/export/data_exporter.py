"""Data exporter for Nexus-LLM.

Exports datasets, evaluation metrics, and full evaluation reports
to JSON, CSV, or Markdown.
"""

import csv
import io
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class DataExporter:
    """Export datasets, metrics, and evaluation reports.

    Example::

        de = DataExporter()
        de.export_dataset(dataset, "json", "output.json")
        de.export_metrics(metrics, "csv", "metrics.csv")
    """

    # ------------------------------------------------------------------
    # Dataset export
    # ------------------------------------------------------------------

    def export_dataset(
        self,
        dataset: Union[List[Dict[str, Any]], Dict[str, List[Any]]],
        format: str = "json",
        path: Optional[Union[str, Path]] = None,
    ) -> str:
        """Export a dataset in the specified format.

        Args:
            dataset: A list of records (list-of-dicts) or a columnar
                     dataset (dict-of-lists).
            format: ``"json"``, ``"csv"``, or ``"markdown"``.
            path: Optional destination file path.

        Returns:
            The serialised string, or *path* if written to disk.

        Raises:
            ValueError: If the format is unsupported.
        """
        # Normalise columnar to row-wise
        records = self._normalise_dataset(dataset)
        format = format.lower()

        if format == "json":
            result = json.dumps(records, indent=2, default=str, ensure_ascii=False)
        elif format == "csv":
            result = self._records_to_csv(records)
        elif format == "markdown":
            result = self._records_to_markdown(records)
        else:
            raise ValueError(
                f"Unsupported dataset export format {format!r}. "
                f"Supported: json, csv, markdown"
            )

        if path is not None:
            path = str(path)
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(result)
            logger.info("Exported dataset to %s (format=%s)", path, format)
            return path

        return result

    # ------------------------------------------------------------------
    # Metrics export
    # ------------------------------------------------------------------

    def export_metrics(
        self,
        metrics: Dict[str, Any],
        format: str = "json",
        path: Optional[Union[str, Path]] = None,
    ) -> str:
        """Export a metrics dictionary.

        Args:
            metrics: Mapping of metric names to values.
            format: ``"json"``, ``"csv"``, or ``"markdown"``.
            path: Optional destination file path.

        Returns:
            The serialised string, or *path* if written to disk.
        """
        format = format.lower()

        if format == "json":
            result = json.dumps(metrics, indent=2, default=str, ensure_ascii=False)
        elif format == "csv":
            # Flatten to rows of (metric, value)
            rows = [{"metric": k, "value": v} for k, v in metrics.items()]
            result = self._records_to_csv(rows)
        elif format == "markdown":
            rows = [{"metric": k, "value": v} for k, v in metrics.items()]
            result = self._records_to_markdown(rows)
        else:
            raise ValueError(
                f"Unsupported metrics export format {format!r}. "
                f"Supported: json, csv, markdown"
            )

        if path is not None:
            path = str(path)
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(result)
            logger.info("Exported metrics to %s (format=%s)", path, format)
            return path

        return result

    # ------------------------------------------------------------------
    # Evaluation report export
    # ------------------------------------------------------------------

    def export_evaluation_report(
        self,
        report: Dict[str, Any],
        format: str = "json",
        path: Optional[Union[str, Path]] = None,
    ) -> str:
        """Export a full evaluation report.

        The report dict should contain ``"model_name"``, ``"scores"``,
        and optional ``"metadata"``.

        Args:
            report: Evaluation report dictionary.
            format: ``"json"``, ``"csv"``, or ``"markdown"``.
            path: Optional destination file path.

        Returns:
            The serialised string, or *path* if written to disk.
        """
        format = format.lower()

        if format == "json":
            result = json.dumps(report, indent=2, default=str, ensure_ascii=False)
        elif format == "csv":
            scores = report.get("scores", {})
            rows = [
                {"model": report.get("model_name", "unknown"), "metric": k, "value": v}
                for k, v in scores.items()
            ]
            result = self._records_to_csv(rows)
        elif format == "markdown":
            result = self._report_to_markdown(report)
        else:
            raise ValueError(
                f"Unsupported report export format {format!r}. "
                f"Supported: json, csv, markdown"
            )

        if path is not None:
            path = str(path)
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(result)
            logger.info("Exported evaluation report to %s (format=%s)", path, format)
            return path

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_dataset(
        dataset: Union[List[Dict[str, Any]], Dict[str, List[Any]]],
    ) -> List[Dict[str, Any]]:
        """Convert columnar dataset to row-wise records."""
        if isinstance(dataset, list):
            return dataset
        if isinstance(dataset, dict):
            # Dict of lists → list of dicts
            keys = list(dataset.keys())
            if not keys:
                return []
            length = len(dataset[keys[0]])
            records: List[Dict[str, Any]] = []
            for i in range(length):
                record = {k: dataset[k][i] for k in keys if i < len(dataset[k])}
                records.append(record)
            return records
        raise TypeError(f"Expected list or dict, got {type(dataset).__name__}")

    @staticmethod
    def _records_to_csv(records: List[Dict[str, Any]]) -> str:
        """Serialise records to CSV."""
        if not records:
            return ""
        fieldnames = sorted({k for r in records for k in r.keys()})
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow(record)
        return output.getvalue()

    @staticmethod
    def _records_to_markdown(records: List[Dict[str, Any]]) -> str:
        """Serialise records as a Markdown table."""
        if not records:
            return ""
        headers = sorted(records[0].keys())
        lines: List[str] = []
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join("---" for _ in headers) + " |")
        for record in records:
            cells = [str(record.get(h, "")) for h in headers]
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _report_to_markdown(report: Dict[str, Any]) -> str:
        """Render an evaluation report as Markdown."""
        lines: List[str] = [
            f"# Evaluation Report: {report.get('model_name', 'Unknown')}",
            "",
        ]

        scores = report.get("scores", {})
        if scores:
            lines.append("## Scores")
            lines.append("")
            rows = [{"metric": k, "value": v} for k, v in scores.items()]
            headers = ["metric", "value"]
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("| " + " | ".join("---" for _ in headers) + " |")
            for k, v in scores.items():
                lines.append(f"| {k} | {v} |")
            lines.append("")

        metadata = report.get("metadata", {})
        if metadata:
            lines.append("## Metadata")
            lines.append("")
            for k, v in metadata.items():
                lines.append(f"- **{k}**: {v}")
            lines.append("")

        return "\n".join(lines)
