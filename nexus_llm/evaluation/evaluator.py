"""
Main Evaluator Module

Orchestrates evaluation runs, aggregates results, and supports model comparison.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from nexus_llm.evaluation.metrics import (
    Accuracy,
    BLEUScore,
    BERTScore,
    ExactMatch,
    F1Score,
    MetricRegistry,
    ROUGEScore,
)
from nexus_llm.evaluation.perplexity import PerplexityCalculator, PerplexityResult
from nexus_llm.evaluation.generation_eval import GenerationEvaluator, GenerationQualityResult

logger = logging.getLogger(__name__)


class EvaluationStatus(str, Enum):
    """Status of an evaluation run."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class EvaluationResult:
    """Container for a single evaluation result."""
    model_name: str
    task_name: str
    status: EvaluationStatus = EvaluationStatus.PENDING
    metrics: Dict[str, float] = field(default_factory=dict)
    per_example_metrics: List[Dict[str, float]] = field(default_factory=list)
    num_examples: int = 0
    elapsed_seconds: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "model_name": self.model_name,
            "task_name": self.task_name,
            "status": self.status.value,
            "metrics": self.metrics,
            "per_example_metrics": self.per_example_metrics,
            "num_examples": self.num_examples,
            "elapsed_seconds": self.elapsed_seconds,
            "error_message": self.error_message,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationResult":
        """Deserialize from dictionary."""
        data["status"] = EvaluationStatus(data["status"])
        return cls(**data)

    def save(self, path: Union[str, Path]) -> None:
        """Save result to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "EvaluationResult":
        """Load result from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


@dataclass
class ModelComparison:
    """Comparison results between multiple models."""
    models: List[str]
    task_name: str
    metrics: List[str]
    results: Dict[str, Dict[str, float]] = field(default_factory=dict)
    rankings: Dict[str, List[Tuple[str, float]]] = field(default_factory=dict)
    best_model: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def compute_rankings(self) -> None:
        """Compute model rankings for each metric (higher is better)."""
        self.rankings = {}
        for metric in self.metrics:
            scores = []
            for model in self.models:
                if model in self.results and metric in self.results[model]:
                    scores.append((model, self.results[model][metric]))
            scores.sort(key=lambda x: x[1], reverse=True)
            self.rankings[metric] = scores
        if self.rankings:
            first_metric = self.metrics[0]
            if first_metric in self.rankings and self.rankings[first_metric]:
                self.best_model = self.rankings[first_metric][0][0]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "models": self.models,
            "task_name": self.task_name,
            "metrics": self.metrics,
            "results": self.results,
            "rankings": {k: v for k, v in self.rankings.items()},
            "best_model": self.best_model,
            "timestamp": self.timestamp,
        }

    def to_markdown_table(self) -> str:
        """Render comparison as a Markdown table."""
        header = "| Model | " + " | ".join(self.metrics) + " |"
        separator = "|-------|" + "|".join(["-------" for _ in self.metrics]) + "|"
        rows = []
        for model in self.models:
            values = []
            for metric in self.metrics:
                val = self.results.get(model, {}).get(metric, float("nan"))
                values.append(f"{val:.4f}" if isinstance(val, float) else str(val))
            rows.append(f"| {model} | " + " | ".join(values) + " |")
        return "\n".join([header, separator] + rows)


class Evaluator:
    """
    Main evaluator that orchestrates evaluation runs.

    Supports:
    - Running evaluation on datasets with configurable metrics
    - Aggregating results across examples
    - Comparing multiple models side-by-side
    - Computing perplexity and generation quality
    """

    def __init__(
        self,
        model_name: str = "default",
        metrics: Optional[List[str]] = None,
        batch_size: int = 32,
        device: str = "cpu",
        output_dir: Optional[Union[str, Path]] = None,
        verbose: bool = True,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.output_dir = Path(output_dir) if output_dir else None
        self.verbose = verbose
        self._results: List[EvaluationResult] = []

        # Initialize metric objects
        if metrics is None:
            metrics = ["accuracy", "exact_match", "bleu", "rouge", "f1"]
        self.metric_names = metrics
        self.metric_objects = self._build_metrics(metrics)

        # Optional sub-evaluators
        self.perplexity_calculator: Optional[PerplexityCalculator] = None
        self.generation_evaluator: Optional[GenerationEvaluator] = None

    @staticmethod
    def _extract_scalar(score: Any) -> float:
        """
        Recursively extract a scalar float from a possibly-nested dict.

        ROUGE returns {"rouge1": {"precision": ..., "recall": ..., "fmeasure": ...}, ...}
        BERTScore returns {"precision": ..., "recall": ..., "f1": ...}
        This method prefers 'fmeasure' or 'f1' keys and falls back to the
        first numeric value found.
        """
        if isinstance(score, (int, float)):
            return float(score)
        if not isinstance(score, dict):
            return float("nan")

        # Direct preference keys
        for key in ("fmeasure", "f1"):
            if key in score and isinstance(score[key], (int, float)):
                return float(score[key])

        # Check nested dicts — prefer "rougeL" for ROUGE
        for preferred_key in ("rougeL", "rouge2", "rouge1"):
            if preferred_key in score and isinstance(score[preferred_key], dict):
                inner = score[preferred_key]
                for k in ("fmeasure", "f1"):
                    if k in inner and isinstance(inner[k], (int, float)):
                        return float(inner[k])

        # Recurse into first dict value
        for v in score.values():
            if isinstance(v, (int, float)):
                return float(v)
            if isinstance(v, dict):
                result = Evaluator._extract_scalar(v)
                if not math.isnan(result):
                    return result

        return float("nan")

    @staticmethod
    def _build_metrics(metric_names: List[str]) -> Dict[str, Any]:
        """Instantiate metric objects from names."""
        registry = MetricRegistry()
        return {name: registry.get(name) for name in metric_names}

    def evaluate(
        self,
        predictions: Sequence[str],
        references: Sequence[str],
        task_name: str = "default",
        compute_perplexity: bool = False,
        compute_generation_quality: bool = False,
        model_log_probs: Optional[List[float]] = None,
    ) -> EvaluationResult:
        """
        Run evaluation on predictions vs. references.

        Args:
            predictions: Model-generated texts.
            references: Ground-truth texts.
            task_name: Name of the evaluation task.
            compute_perplexity: Whether to compute perplexity.
            compute_generation_quality: Whether to compute generation quality scores.
            model_log_probs: Optional pre-computed log probabilities for perplexity.

        Returns:
            EvaluationResult with aggregated and per-example metrics.
        """
        start_time = time.time()
        result = EvaluationResult(
            model_name=self.model_name,
            task_name=task_name,
            status=EvaluationStatus.RUNNING,
            num_examples=len(predictions),
        )

        try:
            # Compute per-example metrics
            per_example: List[Dict[str, float]] = []
            for pred, ref in zip(predictions, references):
                example_metrics: Dict[str, float] = {}
                for name, metric_obj in self.metric_objects.items():
                    try:
                        score = metric_obj.compute(pred, ref)
                        # Some metrics return dicts (e.g. ROUGE, BERTScore);
                        # extract a scalar summary for aggregation.
                        if isinstance(score, dict):
                            score = self._extract_scalar(score)
                        example_metrics[name] = float(score) if isinstance(score, (int, float)) else float("nan")
                    except Exception as exc:
                        logger.warning("Metric %s failed for example: %s", name, exc)
                        example_metrics[name] = float("nan")
                per_example.append(example_metrics)

            result.per_example_metrics = per_example

            # Aggregate: mean across examples
            for name in self.metric_objects:
                values = [ex[name] for ex in per_example if name in ex and isinstance(ex[name], (int, float))]
                if values:
                    result.metrics[name] = sum(values) / len(values)

            # Perplexity
            if compute_perplexity and model_log_probs is not None:
                self.perplexity_calculator = PerplexityCalculator()
                ppl_result = self.perplexity_calculator.compute(model_log_probs)
                result.metrics["perplexity"] = ppl_result.perplexity
                result.metadata["perplexity_details"] = {
                    "total_log_prob": ppl_result.total_log_prob,
                    "num_tokens": ppl_result.num_tokens,
                    "per_token_perplexity": ppl_result.per_token_perplexity,
                }

            # Generation quality
            if compute_generation_quality:
                self.generation_evaluator = GenerationEvaluator()
                gen_result = self.generation_evaluator.evaluate(predictions, references)
                result.metrics["diversity"] = gen_result.diversity
                result.metrics["coherence"] = gen_result.coherence
                result.metrics["relevance"] = gen_result.relevance
                result.metrics["fluency"] = gen_result.fluency
                result.metrics["overall_quality"] = gen_result.overall_quality

            result.status = EvaluationStatus.COMPLETED

        except Exception as exc:
            result.status = EvaluationStatus.FAILED
            result.error_message = str(exc)
            logger.error("Evaluation failed: %s", exc, exc_info=True)

        result.elapsed_seconds = time.time() - start_time
        self._results.append(result)

        if self.output_dir:
            out_path = self.output_dir / f"{task_name}_{self.model_name}_{int(result.timestamp)}.json"
            result.save(out_path)
            if self.verbose:
                logger.info("Saved evaluation result to %s", out_path)

        if self.verbose:
            logger.info(
                "Evaluation '%s' for model '%s': %s (%.2fs)",
                task_name,
                self.model_name,
                result.status.value,
                result.elapsed_seconds,
            )

        return result

    def compare_models(
        self,
        model_results: Dict[str, EvaluationResult],
        task_name: str = "comparison",
    ) -> ModelComparison:
        """
        Compare evaluation results across models.

        Args:
            model_results: Mapping from model name to EvaluationResult.
            task_name: Name of the comparison task.

        Returns:
            ModelComparison with rankings and best model.
        """
        models = list(model_results.keys())
        all_metrics: List[str] = []
        for res in model_results.values():
            for m in res.metrics:
                if m not in all_metrics:
                    all_metrics.append(m)

        results_dict: Dict[str, Dict[str, float]] = {}
        for model, res in model_results.items():
            results_dict[model] = {}
            for metric in all_metrics:
                results_dict[model][metric] = res.metrics.get(metric, float("nan"))

        comparison = ModelComparison(
            models=models,
            task_name=task_name,
            metrics=all_metrics,
            results=results_dict,
        )
        comparison.compute_rankings()
        return comparison

    @property
    def results(self) -> List[EvaluationResult]:
        """Return all accumulated results."""
        return list(self._results)

    def clear_results(self) -> None:
        """Clear accumulated results."""
        self._results.clear()

    def summary(self) -> Dict[str, Any]:
        """Return summary of all evaluation results."""
        return {
            "model_name": self.model_name,
            "total_evaluations": len(self._results),
            "metrics_used": self.metric_names,
            "results": [r.to_dict() for r in self._results],
        }
