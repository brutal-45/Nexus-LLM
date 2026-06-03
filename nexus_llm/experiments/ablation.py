"""Ablation Study Support for Nexus-LLM.

Provides structured ablation study management, enabling systematic
removal or modification of model components to measure their individual
contributions.  Supports component-level, parameter-level, and
layer-level ablations with automatic result aggregation.
"""

from __future__ import annotations

import copy
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class AblationType(str, Enum):
    """Type of ablation."""
    REMOVE = "remove"           # Remove component entirely
    REPLACE = "replace"         # Replace with default/zero
    DISABLE = "disable"         # Disable (e.g., set dropout to 0)
    SCALE = "scale"             # Scale a parameter value
    ZERO_INIT = "zero_init"     # Zero-initialize weights


class AblationStatus(str, Enum):
    """Status of an ablation run."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AblationComponent:
    """Defines a component that can be ablated."""
    name: str
    description: str = ""
    component_type: str = "module"  # module, layer, parameter, feature
    default_value: Any = None
    ablation_types: List[AblationType] = field(default_factory=lambda: [AblationType.REMOVE])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "component_type": self.component_type,
            "default_value": self.default_value,
            "ablation_types": [t.value for t in self.ablation_types],
        }


@dataclass
class AblationVariant:
    """A single ablation variant (one component modified)."""
    variant_id: str = ""
    component_name: str = ""
    ablation_type: AblationType = AblationType.REMOVE
    original_value: Any = None
    ablated_value: Any = None
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.variant_id:
            self.variant_id = f"ablate-{str(uuid.uuid4())[:8]}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "component_name": self.component_name,
            "ablation_type": self.ablation_type.value,
            "original_value": self.original_value,
            "ablated_value": self.ablated_value,
            "params": self.params,
        }


@dataclass
class AblationResult:
    """Result from a single ablation run."""
    variant: AblationVariant
    metrics: Dict[str, float] = field(default_factory=dict)
    baseline_metrics: Dict[str, float] = field(default_factory=dict)
    metric_deltas: Dict[str, float] = field(default_factory=dict)
    status: AblationStatus = AblationStatus.PENDING
    error: Optional[str] = None
    duration_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def primary_delta(self) -> Optional[float]:
        """Delta of the primary metric (first metric)."""
        if self.metric_deltas:
            return next(iter(self.metric_deltas.values()))
        return None

    @property
    def is_degradation(self) -> bool:
        """Whether the ablation caused performance degradation."""
        if self.metric_deltas:
            return any(v < 0 for v in self.metric_deltas.values())
        return False

    def compute_deltas(self) -> None:
        """Compute metric deltas relative to baseline."""
        self.metric_deltas = {}
        for name, value in self.metrics.items():
            baseline = self.baseline_metrics.get(name, 0.0)
            self.metric_deltas[name] = value - baseline

    def to_dict(self) -> Dict[str, Any]:
        return {
            "variant": self.variant.to_dict(),
            "metrics": self.metrics,
            "baseline_metrics": self.baseline_metrics,
            "metric_deltas": self.metric_deltas,
            "status": self.status.value,
            "error": self.error,
            "duration_seconds": round(self.duration_seconds, 4),
            "is_degradation": self.is_degradation,
            "metadata": self.metadata,
        }


@dataclass
class AblationStudyConfig:
    """Configuration for an ablation study."""
    name: str = "ablation_study"
    description: str = ""
    include_baseline: bool = True
    ablation_type: AblationType = AblationType.REMOVE
    skip_failed: bool = True
    parallel: bool = False
    max_workers: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "include_baseline": self.include_baseline,
            "ablation_type": self.ablation_type.value,
            "skip_failed": self.skip_failed,
            "parallel": self.parallel,
            "max_workers": self.max_workers,
        }


# ---------------------------------------------------------------------------
# Ablation Study
# ---------------------------------------------------------------------------

class AblationStudy:
    """Structured ablation study for measuring component contributions.

    Provides a systematic framework for ablating model components,
    running evaluations, and analyzing the impact of each component
    on overall performance.

    Example::

        study = AblationStudy(
            config=AblationStudyConfig(name="transformer_ablation"),
        )

        # Register components
        study.register_component(AblationComponent(
            name="attention", component_type="module",
            ablation_types=[AblationType.REMOVE],
        ))
        study.register_component(AblationComponent(
            name="layer_norm", component_type="module",
            ablation_types=[AblationType.REPLACE],
            default_value="identity",
        ))

        # Define the evaluation function
        def evaluate_fn(params):
            model = build_model(params)
            return {"accuracy": evaluate(model), "loss": compute_loss(model)}

        # Run the study
        results = study.run(evaluate_fn)

        # Get ranked impact
        impact = study.rank_by_impact("accuracy")
    """

    def __init__(self, config: Optional[AblationStudyConfig] = None) -> None:
        self._config = config or AblationStudyConfig()
        self._components: Dict[str, AblationComponent] = {}
        self._results: List[AblationResult] = []
        self._baseline_metrics: Dict[str, float] = {}
        self._baseline_params: Dict[str, Any] = {}
        self._running = False

    @property
    def config(self) -> AblationStudyConfig:
        return self._config

    @property
    def components(self) -> Dict[str, AblationComponent]:
        return dict(self._components)

    @property
    def results(self) -> List[AblationResult]:
        return list(self._results)

    @property
    def baseline_metrics(self) -> Dict[str, float]:
        return dict(self._baseline_metrics)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_component(self, component: AblationComponent) -> None:
        """Register a component for ablation.

        Args:
            component: AblationComponent to register.
        """
        self._components[component.name] = component

    def register_components(self, components: List[AblationComponent]) -> None:
        """Register multiple components.

        Args:
            components: List of AblationComponent objects.
        """
        for comp in components:
            self.register_component(comp)

    def set_baseline(self, metrics: Dict[str, float], params: Optional[Dict[str, Any]] = None) -> None:
        """Set the baseline (full model) metrics.

        Args:
            metrics: Baseline metric values.
            params: Baseline model parameters.
        """
        self._baseline_metrics = metrics
        self._baseline_params = params or {}

    # ------------------------------------------------------------------
    # Generate variants
    # ------------------------------------------------------------------

    def generate_variants(self) -> List[AblationVariant]:
        """Generate all ablation variants from registered components.

        Returns:
            List of AblationVariant objects.
        """
        variants: List[AblationVariant] = []

        for comp in self._components.values():
            for ablation_type in comp.ablation_types:
                ablated_value = self._compute_ablated_value(comp, ablation_type)
                variant = AblationVariant(
                    component_name=comp.name,
                    ablation_type=ablation_type,
                    original_value=comp.default_value,
                    ablated_value=ablated_value,
                )
                variants.append(variant)

        return variants

    @staticmethod
    def _compute_ablated_value(component: AblationComponent, ablation_type: AblationType) -> Any:
        """Compute the value after ablation."""
        if ablation_type == AblationType.REMOVE:
            return None
        elif ablation_type == AblationType.ZERO_INIT:
            return 0
        elif ablation_type == AblationType.DISABLE:
            return False
        elif ablation_type == AblationType.REPLACE:
            return component.default_value
        elif ablation_type == AblationType.SCALE:
            return 0.5  # Default: scale by 50%
        return None

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(
        self,
        evaluate_fn: Callable[[Dict[str, Any]], Dict[str, float]],
        baseline_params: Optional[Dict[str, Any]] = None,
    ) -> List[AblationResult]:
        """Execute the ablation study.

        Args:
            evaluate_fn: Function that takes a params dict (with one
                        component ablated) and returns metrics dict.
            baseline_params: Full model parameters as starting point.

        Returns:
            List of AblationResult for each variant.
        """
        self._running = True
        base_params = baseline_params or dict(self._baseline_params)

        # Run baseline first if not already set
        if self._config.include_baseline and not self._baseline_metrics:
            try:
                self._baseline_metrics = evaluate_fn(base_params)
            except Exception as e:
                self._running = False
                raise RuntimeError(f"Baseline evaluation failed: {e}") from e

        # Generate and run variants
        variants = self.generate_variants()
        self._results = []

        for variant in variants:
            if not self._running:
                break

            # Create ablated params
            ablated_params = copy.deepcopy(base_params)
            ablated_params[variant.component_name] = variant.ablated_value

            start_time = time.time()
            result = AblationResult(
                variant=variant,
                baseline_metrics=dict(self._baseline_metrics),
                status=AblationStatus.RUNNING,
            )

            try:
                metrics = evaluate_fn(ablated_params)
                result.metrics = metrics
                result.status = AblationStatus.COMPLETED
            except Exception as e:
                result.status = AblationStatus.FAILED
                result.error = str(e)
                if not self._config.skip_failed:
                    self._running = False
                    self._results.append(result)
                    break

            result.duration_seconds = time.time() - start_time
            result.compute_deltas()
            self._results.append(result)

        self._running = False
        return self._results

    def stop(self) -> None:
        """Stop the running ablation study."""
        self._running = False

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def rank_by_impact(self, metric_name: str) -> List[Tuple[str, float]]:
        """Rank components by their impact on a metric.

        Args:
            metric_name: The metric to rank by.

        Returns:
            List of (component_name, absolute_delta) sorted by impact (descending).
        """
        impacts: Dict[str, float] = {}
        for result in self._results:
            if result.status != AblationStatus.COMPLETED:
                continue
            delta = result.metric_deltas.get(metric_name, 0.0)
            comp_name = result.variant.component_name
            # Take the maximum absolute impact for each component
            abs_delta = abs(delta)
            if comp_name not in impacts or abs_delta > abs(impacts[comp_name]):
                impacts[comp_name] = delta

        # Sort by absolute impact
        ranked = sorted(impacts.items(), key=lambda x: abs(x[1]), reverse=True)
        return ranked

    def get_critical_components(self, metric_name: str, threshold: float = 0.05) -> List[str]:
        """Identify components whose ablation causes significant degradation.

        Args:
            metric_name: The metric to analyze.
            threshold: Minimum absolute delta to consider significant.

        Returns:
            List of component names with significant impact.
        """
        ranked = self.rank_by_impact(metric_name)
        return [name for name, delta in ranked if abs(delta) >= threshold]

    def get_result_for_component(self, component_name: str) -> List[AblationResult]:
        """Get all results for a specific component.

        Args:
            component_name: The component name.

        Returns:
            List of AblationResult for that component.
        """
        return [r for r in self._results if r.variant.component_name == component_name]

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def generate_report(self, format: str = "text") -> str:
        """Generate an ablation study report.

        Args:
            format: 'text', 'json', or 'markdown'.

        Returns:
            Formatted report string.
        """
        if format == "json":
            return json.dumps({
                "config": self._config.to_dict(),
                "baseline_metrics": self._baseline_metrics,
                "results": [r.to_dict() for r in self._results],
            }, indent=2, default=str)

        if format == "markdown":
            return self._generate_markdown_report()

        # Text format
        lines: List[str] = []
        lines.append("=" * 60)
        lines.append(f"ABLATION STUDY: {self._config.name}")
        lines.append("=" * 60)

        lines.append(f"\nBaseline Metrics:")
        for name, val in self._baseline_metrics.items():
            lines.append(f"  {name}: {val:.4f}")

        completed = [r for r in self._results if r.status == AblationStatus.COMPLETED]
        lines.append(f"\nCompleted Ablations: {len(completed)}/{len(self._results)}")

        for result in self._results:
            variant = result.variant
            status_marker = "OK" if result.status == AblationStatus.COMPLETED else "FAIL"
            lines.append(f"\n  [{status_marker}] {variant.component_name} ({variant.ablation_type.value})")
            if result.metrics:
                for name, val in result.metrics.items():
                    delta = result.metric_deltas.get(name, 0.0)
                    arrow = "UP" if delta > 0 else "DOWN" if delta < 0 else "="
                    lines.append(f"    {name}: {val:.4f} (delta: {delta:+.4f} [{arrow}])")

        # Impact ranking
        if self._baseline_metrics:
            metric_name = next(iter(self._baseline_metrics), "")
            if metric_name:
                ranked = self.rank_by_impact(metric_name)
                lines.append(f"\nImpact Ranking (by {metric_name}):")
                for rank, (name, delta) in enumerate(ranked, 1):
                    lines.append(f"  {rank}. {name}: delta = {delta:+.4f}")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)

    def _generate_markdown_report(self) -> str:
        """Generate a Markdown ablation report."""
        lines: List[str] = []
        lines.append(f"# Ablation Study: {self._config.name}\n")

        if self._config.description:
            lines.append(f"{self._config.description}\n")

        lines.append("## Baseline Metrics\n")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        for name, val in self._baseline_metrics.items():
            lines.append(f"| {name} | {val:.4f} |")

        lines.append("\n## Ablation Results\n")
        if self._baseline_metrics:
            metric_names = list(self._baseline_metrics.keys())
            header = "| Component | Ablation Type | " + " | ".join(metric_names) + " |"
            sep = "|-----------|--------------|" + "|----------" * len(metric_names) + "|"
            lines.append(header)
            lines.append(sep)

            for result in self._results:
                if result.status != AblationStatus.COMPLETED:
                    continue
                v = result.variant
                vals = " | ".join(
                    f"{result.metrics.get(n, 'N/A'):.4f} ({result.metric_deltas.get(n, 0):+.4f})"
                    if n in result.metrics else "N/A"
                    for n in metric_names
                )
                lines.append(f"| {v.component_name} | {v.ablation_type.value} | {vals} |")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> str:
        """Save the ablation study to a JSON file.

        Args:
            path: Output file path.

        Returns:
            Path written.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {
            "config": self._config.to_dict(),
            "components": {name: c.to_dict() for name, c in self._components.items()},
            "baseline_metrics": self._baseline_metrics,
            "baseline_params": self._baseline_params,
            "results": [r.to_dict() for r in self._results],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        return path

    @classmethod
    def load(cls, path: str) -> "AblationStudy":
        """Load an ablation study from a JSON file.

        Args:
            path: Path to the saved study.

        Returns:
            Reconstructed AblationStudy.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        config = AblationStudyConfig(
            name=data.get("config", {}).get("name", "ablation_study"),
            description=data.get("config", {}).get("description", ""),
            include_baseline=data.get("config", {}).get("include_baseline", True),
            ablation_type=AblationType(data.get("config", {}).get("ablation_type", "remove")),
        )

        study = cls(config=config)

        for name, comp_data in data.get("components", {}).items():
            study._components[name] = AblationComponent(
                name=comp_data["name"],
                description=comp_data.get("description", ""),
                component_type=comp_data.get("component_type", "module"),
                default_value=comp_data.get("default_value"),
                ablation_types=[AblationType(t) for t in comp_data.get("ablation_types", ["remove"])],
            )

        study._baseline_metrics = data.get("baseline_metrics", {})
        study._baseline_params = data.get("baseline_params", {})

        for r_data in data.get("results", []):
            variant_data = r_data.get("variant", {})
            variant = AblationVariant(
                variant_id=variant_data.get("variant_id", ""),
                component_name=variant_data.get("component_name", ""),
                ablation_type=AblationType(variant_data.get("ablation_type", "remove")),
                original_value=variant_data.get("original_value"),
                ablated_value=variant_data.get("ablated_value"),
                params=variant_data.get("params", {}),
            )
            result = AblationResult(
                variant=variant,
                metrics=r_data.get("metrics", {}),
                baseline_metrics=r_data.get("baseline_metrics", {}),
                metric_deltas=r_data.get("metric_deltas", {}),
                status=AblationStatus(r_data.get("status", "pending")),
                error=r_data.get("error"),
                duration_seconds=r_data.get("duration_seconds", 0),
            )
            study._results.append(result)

        return study
