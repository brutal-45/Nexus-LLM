"""Experiment Comparison and Visualization for Nexus-LLM.

Provides tools for comparing experiments side-by-side, computing
metric differences, ranking experiments, and generating comparison
reports suitable for analysis and decision-making.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from nexus_llm.experiments.experiment import Experiment, ExperimentStatus, MetricRecord


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MetricComparison:
    """Comparison of a single metric across experiments."""
    metric_name: str
    values: Dict[str, Optional[float]]  # experiment_id -> value
    best_experiment_id: Optional[str] = None
    best_value: Optional[float] = None
    worst_experiment_id: Optional[str] = None
    worst_value: Optional[float] = None
    difference: Optional[float] = None  # best - worst

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "values": self.values,
            "best_experiment_id": self.best_experiment_id,
            "best_value": self.best_value,
            "worst_experiment_id": self.worst_experiment_id,
            "worst_value": self.worst_value,
            "difference": round(self.difference, 6) if self.difference is not None else None,
        }


@dataclass
class ParamDifference:
    """Difference in a parameter between experiments."""
    param_name: str
    values: Dict[str, Any]  # experiment_id -> value
    is_same: bool = True
    unique_values: List[Any] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "param_name": self.param_name,
            "values": self.values,
            "is_same": self.is_same,
            "unique_values": self.unique_values,
        }


@dataclass
class ComparisonResult:
    """Result of comparing multiple experiments."""
    experiment_ids: List[str]
    experiment_names: Dict[str, str] = field(default_factory=dict)
    metric_comparisons: List[MetricComparison] = field(default_factory=list)
    param_differences: List[ParamDifference] = field(default_factory=list)
    common_metrics: List[str] = field(default_factory=list)
    common_params: List[str] = field(default_factory=list)
    rankings: Dict[str, int] = field(default_factory=dict)  # exp_id -> rank

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_ids": self.experiment_ids,
            "experiment_names": self.experiment_names,
            "metric_comparisons": [mc.to_dict() for mc in self.metric_comparisons],
            "param_differences": [pd.to_dict() for pd in self.param_differences],
            "common_metrics": self.common_metrics,
            "common_params": self.common_params,
            "rankings": self.rankings,
        }


# ---------------------------------------------------------------------------
# Experiment Comparator
# ---------------------------------------------------------------------------

class ExperimentComparator:
    """Compare and rank multiple experiments.

    Provides side-by-side comparison of metrics and parameters,
    ranking based on specified criteria, and report generation.

    Example::

        comparator = ExperimentComparator()
        result = comparator.compare([exp1, exp2, exp3])

        # Rank by a specific metric
        ranked = comparator.rank_by(result, "val_loss", mode="min")

        # Find differing parameters
        diffs = comparator.find_param_differences(exp1, exp2)

        # Generate a report
        report = comparator.generate_report(result)
    """

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def compare(
        self,
        experiments: List[Experiment],
        metric_mode: str = "min",
        rank_metric: Optional[str] = None,
    ) -> ComparisonResult:
        """Compare multiple experiments.

        Args:
            experiments: List of Experiment objects to compare.
            metric_mode: Default mode for best/worst ('min' or 'max').
            rank_metric: Metric to use for ranking. If None, uses first common metric.

        Returns:
            ComparisonResult with detailed comparison data.
        """
        if len(experiments) < 2:
            raise ValueError("At least 2 experiments are required for comparison")

        exp_map = {exp.id: exp for exp in experiments}
        exp_ids = [exp.id for exp in experiments]
        exp_names = {exp.id: exp.name for exp in experiments}

        # Find common and unique metrics
        all_metric_names: Dict[str, Set[str]] = {}
        for exp in experiments:
            for name in exp.get_metric_names():
                if name not in all_metric_names:
                    all_metric_names[name] = set()
                all_metric_names[name].add(exp.id)

        common_metrics = sorted([
            name for name, ids in all_metric_names.items()
            if len(ids) == len(experiments)
        ])

        # Compare metrics
        metric_comparisons = []
        for metric_name in common_metrics:
            comparison = self._compare_metric(experiments, metric_name, metric_mode)
            metric_comparisons.append(comparison)

        # Find parameter differences
        all_param_names: Set[str] = set()
        for exp in experiments:
            all_param_names.update(exp.params.keys())

        common_params = sorted([
            name for name in all_param_names
            if all(name in exp.params for exp in experiments)
        ])

        param_diffs = []
        for param_name in sorted(all_param_names):
            values = {exp.id: exp.params.get(param_name) for exp in experiments}
            unique = list(set(v for v in values.values() if v is not None))
            param_diffs.append(ParamDifference(
                param_name=param_name,
                values=values,
                is_same=len(unique) <= 1,
                unique_values=unique,
            ))

        # Rank experiments
        rankings = {}
        rank_by = rank_metric or (common_metrics[0] if common_metrics else None)
        if rank_by:
            rankings = self._rank_experiments(experiments, rank_by, metric_mode)

        return ComparisonResult(
            experiment_ids=exp_ids,
            experiment_names=exp_names,
            metric_comparisons=metric_comparisons,
            param_differences=param_diffs,
            common_metrics=common_metrics,
            common_params=common_params,
            rankings=rankings,
        )

    def _compare_metric(
        self,
        experiments: List[Experiment],
        metric_name: str,
        mode: str,
    ) -> MetricComparison:
        """Compare a single metric across experiments."""
        values: Dict[str, Optional[float]] = {}
        for exp in experiments:
            best = exp.get_best_metric(metric_name, mode=mode)
            values[exp.id] = best.value if best else None

        valid = {eid: v for eid, v in values.items() if v is not None}
        best_eid = None
        best_val = None
        worst_eid = None
        worst_val = None

        if valid:
            if mode == "min":
                best_eid = min(valid, key=valid.get)  # type: ignore[arg-type]
                worst_eid = max(valid, key=valid.get)  # type: ignore[arg-type]
            else:
                best_eid = max(valid, key=valid.get)  # type: ignore[arg-type]
                worst_eid = min(valid, key=valid.get)  # type: ignore[arg-type]

            best_val = valid[best_eid]
            worst_val = valid[worst_eid]

        diff = None
        if best_val is not None and worst_val is not None:
            diff = best_val - worst_val

        return MetricComparison(
            metric_name=metric_name,
            values=values,
            best_experiment_id=best_eid,
            best_value=best_val,
            worst_experiment_id=worst_eid,
            worst_value=worst_val,
            difference=diff,
        )

    def _rank_experiments(
        self,
        experiments: List[Experiment],
        metric_name: str,
        mode: str,
    ) -> Dict[str, int]:
        """Rank experiments by a metric."""
        scored = []
        for exp in experiments:
            best = exp.get_best_metric(metric_name, mode=mode)
            score = best.value if best else None
            scored.append((exp.id, score))

        # Sort by score (handle None values)
        if mode == "min":
            scored.sort(key=lambda x: x[1] if x[1] is not None else float("inf"))
        else:
            scored.sort(key=lambda x: x[1] if x[1] is not None else float("-inf"), reverse=True)

        return {eid: rank + 1 for rank, (eid, _) in enumerate(scored)}

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    def rank_by(
        self,
        comparison: ComparisonResult,
        metric_name: str,
        mode: str = "min",
    ) -> Dict[str, int]:
        """Re-rank experiments by a specific metric from comparison data.

        Args:
            comparison: Existing ComparisonResult.
            metric_name: Metric to rank by.
            mode: 'min' for ascending, 'max' for descending.

        Returns:
            Dictionary of experiment_id -> rank.
        """
        for mc in comparison.metric_comparisons:
            if mc.metric_name == metric_name:
                valid = {eid: v for eid, v in mc.values.items() if v is not None}
                if mode == "min":
                    sorted_ids = sorted(valid, key=valid.get)  # type: ignore[arg-type]
                else:
                    sorted_ids = sorted(valid, key=valid.get, reverse=True)  # type: ignore[arg-type]
                return {eid: rank + 1 for rank, eid in enumerate(sorted_ids)}
        return {}

    # ------------------------------------------------------------------
    # Pairwise difference
    # ------------------------------------------------------------------

    def find_param_differences(
        self,
        exp1: Experiment,
        exp2: Experiment,
    ) -> List[ParamDifference]:
        """Find parameters that differ between two experiments.

        Args:
            exp1: First experiment.
            exp2: Second experiment.

        Returns:
            List of ParamDifference for differing parameters.
        """
        all_params = set(exp1.params.keys()) | set(exp2.params.keys())
        diffs = []
        for name in sorted(all_params):
            v1 = exp1.params.get(name)
            v2 = exp2.params.get(name)
            if v1 != v2:
                diffs.append(ParamDifference(
                    param_name=name,
                    values={exp1.id: v1, exp2.id: v2},
                    is_same=False,
                    unique_values=list({v1, v2} - {None}),
                ))
        return diffs

    def find_metric_differences(
        self,
        exp1: Experiment,
        exp2: Experiment,
        mode: str = "min",
    ) -> List[MetricComparison]:
        """Compare metrics between two experiments.

        Args:
            exp1: First experiment.
            exp2: Second experiment.
            mode: Best metric mode.

        Returns:
            List of MetricComparison for common metrics.
        """
        common = set(exp1.get_metric_names()) & set(exp2.get_metric_names())
        comparisons = []
        for name in sorted(common):
            comparisons.append(self._compare_metric([exp1, exp2], name, mode))
        return comparisons

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_report(
        self,
        comparison: ComparisonResult,
        format: str = "text",
    ) -> str:
        """Generate a comparison report.

        Args:
            comparison: ComparisonResult to report on.
            format: 'text', 'json', or 'markdown'.

        Returns:
            Formatted report string.
        """
        if format == "json":
            return json.dumps(comparison.to_dict(), indent=2, default=str)

        if format == "markdown":
            return self._generate_markdown_report(comparison)

        # Default: text format
        lines: List[str] = []
        lines.append("=" * 60)
        lines.append("EXPERIMENT COMPARISON REPORT")
        lines.append("=" * 60)

        # Experiments overview
        lines.append("\nExperiments:")
        for eid in comparison.experiment_ids:
            name = comparison.experiment_names.get(eid, eid)
            rank = comparison.rankings.get(eid, "-")
            lines.append(f"  [{rank}] {name} (ID: {eid})")

        # Metric comparisons
        if comparison.metric_comparisons:
            lines.append("\nMetric Comparisons:")
            for mc in comparison.metric_comparisons:
                lines.append(f"\n  {mc.metric_name}:")
                for eid, val in mc.values.items():
                    marker = " <-- BEST" if eid == mc.best_experiment_id else ""
                    marker += " WORST" if eid == mc.worst_experiment_id and eid != mc.best_experiment_id else ""
                    name = comparison.experiment_names.get(eid, eid)
                    lines.append(f"    {name}: {val}{marker}")
                if mc.difference is not None:
                    lines.append(f"    Difference: {mc.difference:.6f}")

        # Parameter differences
        differing = [pd for pd in comparison.param_differences if not pd.is_same]
        if differing:
            lines.append(f"\nDiffering Parameters ({len(differing)}):")
            for pd in differing:
                vals = ", ".join(
                    f"{comparison.experiment_names.get(eid, eid)}={val}"
                    for eid, val in pd.values.items()
                )
                lines.append(f"  {pd.param_name}: {vals}")

        same = [pd for pd in comparison.param_differences if pd.is_same]
        if same:
            lines.append(f"\nCommon Parameters ({len(same)}):")
            for pd in same[:10]:
                val = list(pd.values.values())[0]
                lines.append(f"  {pd.param_name} = {val}")
            if len(same) > 10:
                lines.append(f"  ... and {len(same) - 10} more")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)

    def _generate_markdown_report(self, comparison: ComparisonResult) -> str:
        """Generate a Markdown comparison report."""
        lines: List[str] = []
        lines.append("# Experiment Comparison Report\n")

        # Overview table
        lines.append("## Experiments\n")
        lines.append("| Rank | Name | ID |")
        lines.append("|------|------|----|")
        for eid in comparison.experiment_ids:
            name = comparison.experiment_names.get(eid, eid)
            rank = comparison.rankings.get(eid, "-")
            lines.append(f"| {rank} | {name} | `{eid}` |")

        # Metrics table
        if comparison.metric_comparisons:
            lines.append("\n## Metric Comparisons\n")
            header = "| Metric | " + " | ".join(
                comparison.experiment_names.get(eid, eid) for eid in comparison.experiment_ids
            ) + " | Best | Difference |"
            sep = "|--------" + "|--------" * len(comparison.experiment_ids) + "|------|------------|"
            lines.append(header)
            lines.append(sep)
            for mc in comparison.metric_comparisons:
                vals = " | ".join(
                    f"{mc.values.get(eid, 'N/A')}" if mc.values.get(eid) is not None else "N/A"
                    for eid in comparison.experiment_ids
                )
                best_name = comparison.experiment_names.get(mc.best_experiment_id, mc.best_experiment_id or "")
                diff = f"{mc.difference:.6f}" if mc.difference is not None else "N/A"
                lines.append(f"| {mc.metric_name} | {vals} | {best_name} | {diff} |")

        # Differing params
        differing = [pd for pd in comparison.param_differences if not pd.is_same]
        if differing:
            lines.append("\n## Differing Parameters\n")
            header = "| Parameter | " + " | ".join(
                comparison.experiment_names.get(eid, eid) for eid in comparison.experiment_ids
            ) + " |"
            sep = "|-----------" + "|----------" * len(comparison.experiment_ids) + "|"
            lines.append(header)
            lines.append(sep)
            for pd in differing:
                vals = " | ".join(str(pd.values.get(eid, "N/A")) for eid in comparison.experiment_ids)
                lines.append(f"| {pd.param_name} | {vals} |")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_comparison(
        self,
        comparison: ComparisonResult,
        path: str,
        format: str = "json",
    ) -> str:
        """Export comparison results to a file.

        Args:
            comparison: ComparisonResult to export.
            path: Output file path.
            format: 'json', 'text', or 'markdown'.

        Returns:
            Path to the written file.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        content = self.generate_report(comparison, format=format)

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

        return path


# Required import for type hint
from typing import Set  # noqa: E402 - used in method bodies
