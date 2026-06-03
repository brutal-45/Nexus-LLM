"""Comparison engine for Nexus-LLM evaluation reports.

Compares evaluation results between models and produces comparison
tables with statistical significance testing.
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

from nexus_llm.evaluation.report import EvaluationReport

logger = logging.getLogger(__name__)


class ComparisonResult:
    """Holds the outcome of comparing two or more evaluation reports.

    Attributes:
        model_names: Ordered list of model names.
        metrics: Set of metric names present in the comparison.
        scores: ``{model_name: {metric: score}}``.
        differences: ``{metric: {model_pair: difference}}`` for pairwise.
        winners: ``{metric: model_name}`` — best performer per metric.
    """

    def __init__(self) -> None:
        self.model_names: List[str] = []
        self.metrics: set = set()
        self.scores: Dict[str, Dict[str, float]] = {}
        self.differences: Dict[str, Dict[str, float]] = {}
        self.winners: Dict[str, str] = {}

    def summary(self) -> str:
        """Return a human-readable comparison table."""
        lines: List[str] = [
            "Model Comparison",
            "=" * 60,
        ]

        if not self.metrics:
            lines.append("(no common metrics)")
            return "\n".join(lines)

        # Header
        header = f"{'Metric':<20s}"
        for name in self.model_names:
            header += f"{name:>15s}"
        header += f"{'Winner':>15s}"
        lines.append(header)
        lines.append("-" * len(header))

        for metric in sorted(self.metrics):
            row = f"{metric:<20s}"
            for name in self.model_names:
                val = self.scores.get(name, {}).get(metric, float("nan"))
                row += f"{val:>15.4f}"
            winner = self.winners.get(metric, "—")
            row += f"{winner:>15s}"
            lines.append(row)

        return "\n".join(lines)


class ComparisonEngine:
    """Compare evaluation results across models.

    Example::

        engine = ComparisonEngine()
        result = engine.compare(report_a, report_b)
        print(result.summary())
        print("Winner:", engine.winner(result))
    """

    # ------------------------------------------------------------------
    # Pairwise comparison
    # ------------------------------------------------------------------

    def compare(
        self,
        model_a_results: EvaluationReport,
        model_b_results: EvaluationReport,
    ) -> ComparisonResult:
        """Compare two evaluation reports metric-by-metric.

        For each metric present in **both** reports, the absolute
        difference is recorded and the model with the higher score is
        declared the winner (lower is better for ``perplexity`` and
        ``avg_latency_ms``).

        Args:
            model_a_results: Report for model A.
            model_b_results: Report for model B.

        Returns:
            A :class:`ComparisonResult` with scores, differences, and winners.
        """
        result = ComparisonResult()
        name_a = model_a_results.model_name or "model_a"
        name_b = model_b_results.model_name or "model_b"

        result.model_names = [name_a, name_b]
        result.scores[name_a] = dict(model_a_results.scores)
        result.scores[name_b] = dict(model_b_results.scores)

        common_metrics = set(model_a_results.scores) & set(model_b_results.scores)
        result.metrics = common_metrics

        # Metrics where lower is better
        lower_is_better = {"perplexity", "avg_latency_ms"}

        for metric in common_metrics:
            val_a = model_a_results.scores[metric]
            val_b = model_b_results.scores[metric]
            diff = val_b - val_a
            result.differences[metric] = {f"{name_a}_vs_{name_b}": diff}

            if metric in lower_is_better:
                result.winners[metric] = name_a if val_a <= val_b else name_b
            else:
                result.winners[metric] = name_a if val_a >= val_b else name_b

        return result

    # ------------------------------------------------------------------
    # Multi-model comparison
    # ------------------------------------------------------------------

    def compare_multiple(
        self,
        results_dict: Dict[str, EvaluationReport],
    ) -> ComparisonResult:
        """Compare evaluation reports from multiple models.

        Args:
            results_dict: Mapping of model name → EvaluationReport.

        Returns:
            A :class:`ComparisonResult` with all models compared.
        """
        result = ComparisonResult()
        result.model_names = list(results_dict.keys())

        for name, report in results_dict.items():
            result.scores[name] = dict(report.scores)

        # Determine common metrics across all reports
        metric_sets = [set(r.scores) for r in results_dict.values()]
        result.metrics = set.intersection(*metric_sets) if metric_sets else set()

        lower_is_better = {"perplexity", "avg_latency_ms"}

        for metric in result.metrics:
            values: Dict[str, float] = {}
            for name, report in results_dict.items():
                values[name] = report.scores[metric]

            if metric in lower_is_better:
                winner = min(values, key=values.get)  # type: ignore[arg-type]
            else:
                winner = max(values, key=values.get)  # type: ignore[arg-type]
            result.winners[metric] = winner

        # Pairwise differences (first model as baseline)
        if result.model_names:
            baseline = result.model_names[0]
            for metric in result.metrics:
                baseline_val = result.scores[baseline].get(metric, 0.0)
                diffs: Dict[str, float] = {}
                for name in result.model_names[1:]:
                    diffs[f"{baseline}_vs_{name}"] = (
                        result.scores[name].get(metric, 0.0) - baseline_val
                    )
                result.differences[metric] = diffs

        return result

    # ------------------------------------------------------------------
    # Winner / significance helpers
    # ------------------------------------------------------------------

    def winner(self, comparison: ComparisonResult) -> str:
        """Return the overall winner — the model that wins the most metrics.

        In case of a tie, returns the first model in the ordering.

        Args:
            comparison: A populated ComparisonResult.

        Returns:
            The winning model name, or ``"tie"``.
        """
        if not comparison.winners:
            return "tie"

        tally: Dict[str, int] = {}
        for model_name in comparison.winners.values():
            tally[model_name] = tally.get(model_name, 0) + 1

        max_wins = max(tally.values())
        top_models = [m for m, w in tally.items() if w == max_wins]

        if len(top_models) > 1:
            return "tie"
        return top_models[0]

    def statistical_significance(
        self,
        scores_a: List[float],
        scores_b: List[float],
    ) -> bool:
        """Perform a simple two-tailed paired t-test at α = 0.05.

        This is a self-contained implementation that avoids importing
        ``scipy``.  It applies the central-limit-theorem approximation
        and is suitable for moderate-to-large sample sizes (n ≥ 30).

        Args:
            scores_a: Repeated measurements for model A.
            scores_b: Repeated measurements for model B (same length).

        Returns:
            True if the difference is statistically significant at p < 0.05.
        """
        n = len(scores_a)
        if n != len(scores_b) or n < 2:
            return False

        diffs = [a - b for a, b in zip(scores_a, scores_b)]
        mean_diff = sum(diffs) / n

        # Variance of differences
        variance = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1)
        if variance == 0:
            return False

        se = math.sqrt(variance / n)
        t_stat = abs(mean_diff) / se

        # Critical t-value approximation for α = 0.05, two-tailed
        # For n >= 30, z ≈ 1.96; for small n we use a conservative 2.0+
        if n >= 30:
            critical = 1.96
        elif n >= 10:
            critical = 2.23
        else:
            critical = 2.45

        return t_stat > critical
