"""
Evaluation Report Generation Module

Generates evaluation reports in multiple formats:
- JSON: structured data for programmatic consumption
- HTML: styled, self-contained HTML report with tables and charts
- Text: plain-text summary for terminal output

Supports visualization stubs, comparison tables, and benchmark summaries.
"""

import json
import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

logger = logging.getLogger(__name__)


class ReportFormat(str, Enum):
    """Supported output formats."""
    JSON = "json"
    HTML = "html"
    TEXT = "text"


@dataclass
class ReportSection:
    """A section within a report."""
    title: str
    content: str = ""
    tables: List[Dict[str, Any]] = field(default_factory=list)
    subsections: List["ReportSection"] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "content": self.content,
            "tables": self.tables,
            "subsections": [s.to_dict() for s in self.subsections],
        }


class ReportGenerator:
    """
    Generate evaluation reports from results.

    Takes evaluation results (dicts or structured objects) and produces
    formatted reports in JSON, HTML, or plain text.
    """

    def __init__(
        self,
        title: str = "Nexus-LLM Evaluation Report",
        model_name: Optional[str] = None,
        include_timestamp: bool = True,
    ):
        self.title = title
        self.model_name = model_name
        self.include_timestamp = include_timestamp

    # ------------------------------------------------------------------
    # JSON Report
    # ------------------------------------------------------------------

    def generate_json(
        self,
        results: Union[Dict[str, Any], List[Dict[str, Any]]],
        output_path: Optional[Union[str, Path]] = None,
    ) -> str:
        """
        Generate a JSON report.

        Args:
            results: Evaluation results (single dict or list of dicts).
            output_path: Optional file path to write the report.

        Returns:
            JSON string of the report.
        """
        report = self._build_report_data(results)
        report_json = json.dumps(report, indent=2, ensure_ascii=False, default=str)

        if output_path:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(report_json)
            logger.info("JSON report saved to %s", path)

        return report_json

    # ------------------------------------------------------------------
    # HTML Report
    # ------------------------------------------------------------------

    def generate_html(
        self,
        results: Union[Dict[str, Any], List[Dict[str, Any]]],
        output_path: Optional[Union[str, Path]] = None,
    ) -> str:
        """
        Generate a self-contained HTML report with styling and tables.

        Args:
            results: Evaluation results.
            output_path: Optional file path to write the report.

        Returns:
            HTML string of the report.
        """
        report_data = self._build_report_data(results)
        html = self._render_html(report_data)

        if output_path:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(html)
            logger.info("HTML report saved to %s", path)

        return html

    # ------------------------------------------------------------------
    # Text Report
    # ------------------------------------------------------------------

    def generate_text(
        self,
        results: Union[Dict[str, Any], List[Dict[str, Any]]],
        output_path: Optional[Union[str, Path]] = None,
    ) -> str:
        """
        Generate a plain-text summary report.

        Args:
            results: Evaluation results.
            output_path: Optional file path to write the report.

        Returns:
            Text string of the report.
        """
        report_data = self._build_report_data(results)
        lines = self._render_text(report_data)
        text = "\n".join(lines)

        if output_path:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            logger.info("Text report saved to %s", path)

        return text

    # ------------------------------------------------------------------
    # Convenience: generate in all formats
    # ------------------------------------------------------------------

    def generate_all(
        self,
        results: Union[Dict[str, Any], List[Dict[str, Any]]],
        output_dir: Union[str, Path],
        base_name: str = "eval_report",
    ) -> Dict[ReportFormat, str]:
        """
        Generate reports in all formats.

        Returns:
            Dict mapping format to output file path.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            ReportFormat.JSON: str(output_dir / f"{base_name}.json"),
            ReportFormat.HTML: str(output_dir / f"{base_name}.html"),
            ReportFormat.TEXT: str(output_dir / f"{base_name}.txt"),
        }
        self.generate_json(results, paths[ReportFormat.JSON])
        self.generate_html(results, paths[ReportFormat.HTML])
        self.generate_text(results, paths[ReportFormat.TEXT])
        return paths

    # ------------------------------------------------------------------
    # Comparison Table
    # ------------------------------------------------------------------

    @staticmethod
    def comparison_table(
        model_results: Dict[str, Dict[str, float]],
        metrics: Optional[List[str]] = None,
    ) -> str:
        """
        Generate a Markdown comparison table for multiple models.

        Args:
            model_results: Mapping of model name → metric name → score.
            metrics: Optional ordered list of metric names.

        Returns:
            Markdown table string.
        """
        if not model_results:
            return "| No data |"

        models = list(model_results.keys())
        if metrics is None:
            metric_set: set = set()
            for res in model_results.values():
                metric_set.update(res.keys())
            metrics = sorted(metric_set)

        header = "| Model | " + " | ".join(metrics) + " |"
        separator = "|-------|" + "|".join(["-------" for _ in metrics]) + "|"
        rows = []
        for model in models:
            values = []
            for m in metrics:
                val = model_results[model].get(m, float("nan"))
                if isinstance(val, float) and math.isnan(val):
                    values.append("N/A")
                elif isinstance(val, float):
                    values.append(f"{val:.4f}")
                else:
                    values.append(str(val))
            rows.append(f"| {model} | " + " | ".join(values) + " |")

        return "\n".join([header, separator] + rows)

    # ==================================================================
    # Internal helpers
    # ==================================================================

    def _build_report_data(
        self,
        results: Union[Dict[str, Any], List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """Build structured report data from raw results."""
        if isinstance(results, dict):
            results_list = [results]
        else:
            results_list = list(results)

        report: Dict[str, Any] = {
            "title": self.title,
            "model_name": self.model_name,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "num_evaluations": len(results_list),
            "evaluations": results_list,
        }

        # Aggregate metrics across evaluations
        all_metrics: Dict[str, List[float]] = {}
        for res in results_list:
            metrics = res.get("metrics", {})
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    all_metrics.setdefault(k, []).append(v)

        aggregated: Dict[str, Dict[str, float]] = {}
        for metric_name, values in all_metrics.items():
            aggregated[metric_name] = {
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "count": len(values),
            }
        report["aggregated_metrics"] = aggregated

        return report

    def _render_html(self, report: Dict[str, Any]) -> str:
        """Render report data as styled HTML."""
        title = report.get("title", "Evaluation Report")
        generated_at = report.get("generated_at", "")
        model_name = report.get("model_name", "N/A")
        agg = report.get("aggregated_metrics", {})
        evals = report.get("evaluations", [])

        # Build metrics table rows
        metric_rows = ""
        for metric_name, stats in agg.items():
            metric_rows += (
                f"<tr>"
                f"<td>{metric_name}</td>"
                f"<td>{stats['mean']:.4f}</td>"
                f"<td>{stats['min']:.4f}</td>"
                f"<td>{stats['max']:.4f}</td>"
                f"<td>{int(stats['count'])}</td>"
                f"</tr>"
            )

        # Build evaluation detail rows
        eval_rows = ""
        for i, ev in enumerate(evals):
            ev_model = ev.get("model_name", "N/A")
            ev_task = ev.get("task_name", "N/A")
            ev_status = ev.get("status", "N/A")
            metrics_str = ", ".join(
                f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                for k, v in ev.get("metrics", {}).items()
            )
            eval_rows += (
                f"<tr>"
                f"<td>{i + 1}</td>"
                f"<td>{ev_model}</td>"
                f"<td>{ev_task}</td>"
                f"<td>{ev_status}</td>"
                f"<td>{metrics_str}</td>"
                f"</tr>"
            )

        # Build bar chart using pure CSS
        chart_bars = ""
        colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
                   "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac"]
        for idx, (metric_name, stats) in enumerate(agg.items()):
            color = colors[idx % len(colors)]
            width = min(stats["mean"] * 100, 100)
            chart_bars += (
                f'<div class="chart-row">'
                f'<span class="chart-label">{metric_name}</span>'
                f'<div class="chart-bar-bg">'
                f'<div class="chart-bar" style="width:{width:.1f}%;background:{color}"></div>'
                f'</div>'
                f'<span class="chart-value">{stats["mean"]:.4f}</span>'
                f'</div>'
            )

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         max-width: 960px; margin: 0 auto; padding: 20px; background: #f8f9fa; color: #333; }}
  h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
  h2 {{ color: #34495e; margin-top: 30px; }}
  .meta {{ color: #7f8c8d; font-size: 0.9em; margin-bottom: 20px; }}
  table {{ width: 100%; border-collapse: collapse; margin: 15px 0; background: #fff;
           box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  th, td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid #ecf0f1; }}
  th {{ background: #2c3e50; color: #fff; font-weight: 600; }}
  tr:hover {{ background: #f1f2f6; }}
  .chart-row {{ display: flex; align-items: center; margin: 6px 0; }}
  .chart-label {{ width: 140px; font-size: 0.9em; text-align: right; padding-right: 10px; }}
  .chart-bar-bg {{ flex: 1; background: #ecf0f1; border-radius: 4px; height: 24px; overflow: hidden; }}
  .chart-bar {{ height: 100%; border-radius: 4px; transition: width 0.3s; }}
  .chart-value {{ width: 80px; font-size: 0.9em; padding-left: 10px; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 0.8em;
            font-weight: 600; }}
  .badge-completed {{ background: #d5f5e3; color: #27ae60; }}
  .badge-failed {{ background: #fadbd8; color: #e74c3c; }}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="meta">
  <strong>Model:</strong> {model_name} &nbsp;|&nbsp;
  <strong>Generated:</strong> {generated_at} &nbsp;|&nbsp;
  <strong>Evaluations:</strong> {len(evals)}
</div>

<h2>Metrics Overview</h2>
<table>
  <thead>
    <tr><th>Metric</th><th>Mean</th><th>Min</th><th>Max</th><th>Count</th></tr>
  </thead>
  <tbody>{metric_rows}</tbody>
</table>

<h2>Score Distribution</h2>
{chart_bars}

<h2>Evaluation Details</h2>
<table>
  <thead>
    <tr><th>#</th><th>Model</th><th>Task</th><th>Status</th><th>Metrics</th></tr>
  </thead>
  <tbody>{eval_rows}</tbody>
</table>

</body>
</html>"""
        return html

    def _render_text(self, report: Dict[str, Any]) -> List[str]:
        """Render report data as plain text lines."""
        lines: List[str] = []
        title = report.get("title", "Evaluation Report")
        lines.append("=" * 60)
        lines.append(title)
        lines.append("=" * 60)
        lines.append(f"Model: {report.get('model_name', 'N/A')}")
        lines.append(f"Generated: {report.get('generated_at', '')}")
        lines.append(f"Total evaluations: {report.get('num_evaluations', 0)}")
        lines.append("")

        agg = report.get("aggregated_metrics", {})
        if agg:
            lines.append("-" * 60)
            lines.append("AGGREGATED METRICS")
            lines.append("-" * 60)
            for metric_name, stats in agg.items():
                lines.append(
                    f"  {metric_name:30s}  "
                    f"mean={stats['mean']:.4f}  "
                    f"min={stats['min']:.4f}  "
                    f"max={stats['max']:.4f}  "
                    f"n={int(stats['count'])}"
                )
            lines.append("")

        evals = report.get("evaluations", [])
        for i, ev in enumerate(evals):
            lines.append("-" * 60)
            lines.append(f"EVALUATION #{i + 1}")
            lines.append("-" * 60)
            lines.append(f"  Model:  {ev.get('model_name', 'N/A')}")
            lines.append(f"  Task:   {ev.get('task_name', 'N/A')}")
            lines.append(f"  Status: {ev.get('status', 'N/A')}")
            metrics = ev.get("metrics", {})
            if metrics:
                lines.append("  Metrics:")
                for k, v in metrics.items():
                    lines.append(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")
            lines.append("")

        lines.append("=" * 60)
        return lines
