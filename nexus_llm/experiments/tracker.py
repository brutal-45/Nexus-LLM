"""ML Experiment Tracker for Nexus-LLM.

Provides an MLflow-like experiment tracking system for managing
multiple experiments, comparing results, and persisting data.
Supports experiment creation, metric aggregation, and experiment
search/filtering.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from nexus_llm.experiments.experiment import (
    Artifact,
    Experiment,
    ExperimentConfig,
    ExperimentStatus,
    MetricRecord,
    MetricType,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TrackerConfig:
    """Configuration for the experiment tracker."""
    storage_dir: str = "./experiments"
    auto_save: bool = True
    auto_start: bool = True
    max_experiments: int = 1000
    purge_after_days: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "storage_dir": self.storage_dir,
            "auto_save": self.auto_save,
            "auto_start": self.auto_start,
            "max_experiments": self.max_experiments,
            "purge_after_days": self.purge_after_days,
        }


@dataclass
class ExperimentSummary:
    """Summary of an experiment for listing and comparison."""
    id: str
    name: str
    status: ExperimentStatus
    tags: List[str]
    params: Dict[str, Any]
    best_metrics: Dict[str, float]
    metric_count: int
    artifact_count: int
    created_at: float
    duration: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status.value,
            "tags": self.tags,
            "params": self.params,
            "best_metrics": self.best_metrics,
            "metric_count": self.metric_count,
            "artifact_count": self.artifact_count,
            "created_at": self.created_at,
            "duration": round(self.duration, 4) if self.duration else None,
        }


# ---------------------------------------------------------------------------
# Experiment Tracker
# ---------------------------------------------------------------------------

class ExperimentTracker:
    """ML experiment tracker for managing multiple experiments.

    Provides a centralized system for creating, tracking, and
    querying experiments.  Inspired by MLflow's tracking API.

    Example::

        tracker = ExperimentTracker(TrackerConfig(storage_dir="./ml_runs"))

        # Create and run an experiment
        exp = tracker.create_experiment(
            "baseline_run",
            tags=["baseline", "v1"],
            params={"lr": 2e-5, "epochs": 3},
        )

        tracker.log_metric(exp.id, "loss", 0.45, step=1)
        tracker.log_metric(exp.id, "loss", 0.32, step=2)
        tracker.log_metric(exp.id, "accuracy", 0.89, step=2)

        # Search experiments
        results = tracker.search_experiments(tags=["baseline"])

        # Get best metric
        best = tracker.get_best_metric(exp.id, "loss", mode="min")
    """

    def __init__(self, config: Optional[TrackerConfig] = None) -> None:
        self._config = config or TrackerConfig()
        self._experiments: Dict[str, Experiment] = {}
        self._active_experiment_id: Optional[str] = None

        # Ensure storage directory exists
        os.makedirs(self._config.storage_dir, exist_ok=True)

        # Load existing experiments from disk
        self._load_from_disk()

    @property
    def active_experiment(self) -> Optional[Experiment]:
        """Currently active experiment, if any."""
        if self._active_experiment_id:
            return self._experiments.get(self._active_experiment_id)
        return None

    @property
    def experiment_count(self) -> int:
        """Number of tracked experiments."""
        return len(self._experiments)

    # ------------------------------------------------------------------
    # Experiment creation
    # ------------------------------------------------------------------

    def create_experiment(
        self,
        name: str,
        description: str = "",
        tags: Optional[List[str]] = None,
        params: Optional[Dict[str, Any]] = None,
        set_active: bool = True,
    ) -> Experiment:
        """Create a new experiment.

        Args:
            name: Experiment name.
            description: Optional description.
            tags: Optional tags for categorization.
            params: Optional initial parameters.
            set_active: Whether to set this as the active experiment.

        Returns:
            The created Experiment.
        """
        exp_config = ExperimentConfig(
            storage_dir=os.path.join(self._config.storage_dir, name),
        )
        exp = Experiment(
            name=name,
            description=description,
            tags=tags or [],
            config=exp_config,
        )

        if params:
            exp.log_params(params)

        if self._config.auto_start:
            exp.start()

        self._experiments[exp.id] = exp

        if set_active:
            self._active_experiment_id = exp.id

        if self._config.auto_save:
            self._save_experiment(exp)

        return exp

    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Get an experiment by ID.

        Args:
            experiment_id: The experiment ID.

        Returns:
            Experiment or None.
        """
        return self._experiments.get(experiment_id)

    def set_active(self, experiment_id: str) -> None:
        """Set the active experiment.

        Args:
            experiment_id: Experiment ID to make active.
        """
        if experiment_id not in self._experiments:
            raise KeyError(f"Experiment '{experiment_id}' not found")
        self._active_experiment_id = experiment_id

    # ------------------------------------------------------------------
    # Logging shortcuts (active experiment)
    # ------------------------------------------------------------------

    def log_param(self, key: str, value: Any, experiment_id: Optional[str] = None) -> None:
        """Log a parameter to an experiment.

        Args:
            key: Parameter name.
            value: Parameter value.
            experiment_id: Target experiment (uses active if None).
        """
        exp = self._resolve_experiment(experiment_id)
        exp.log_param(key, value)

    def log_params(self, params: Dict[str, Any], experiment_id: Optional[str] = None) -> None:
        """Log multiple parameters.

        Args:
            params: Parameter dictionary.
            experiment_id: Target experiment.
        """
        exp = self._resolve_experiment(experiment_id)
        exp.log_params(params)

    def log_metric(
        self,
        name: str,
        value: float,
        step: int = 0,
        experiment_id: Optional[str] = None,
        metric_type: MetricType = MetricType.SCALAR,
    ) -> None:
        """Record a metric measurement.

        Args:
            name: Metric name.
            value: Metric value.
            step: Step number.
            experiment_id: Target experiment.
            metric_type: Type of metric.
        """
        exp = self._resolve_experiment(experiment_id)
        exp.log_metric(name, value, step=step, metric_type=metric_type)

    def log_metrics(self, metrics: Dict[str, float], step: int = 0, experiment_id: Optional[str] = None) -> None:
        """Record multiple metrics.

        Args:
            metrics: Metric dictionary.
            step: Step number.
            experiment_id: Target experiment.
        """
        exp = self._resolve_experiment(experiment_id)
        exp.log_metrics(metrics, step=step)

    def log_artifact(
        self,
        name: str,
        path: str,
        artifact_type: str = "file",
        experiment_id: Optional[str] = None,
    ) -> Artifact:
        """Log an artifact to an experiment.

        Args:
            name: Artifact name.
            path: File path.
            artifact_type: Type of artifact.
            experiment_id: Target experiment.

        Returns:
            The created Artifact.
        """
        exp = self._resolve_experiment(experiment_id)
        return exp.log_artifact(name, path, artifact_type=artifact_type)

    # ------------------------------------------------------------------
    # Metric queries
    # ------------------------------------------------------------------

    def get_best_metric(
        self,
        experiment_id: str,
        metric_name: str,
        mode: str = "min",
    ) -> Optional[MetricRecord]:
        """Get the best value for a metric across an experiment.

        Args:
            experiment_id: Experiment ID.
            metric_name: Metric to query.
            mode: 'min' for lowest, 'max' for highest.

        Returns:
            Best MetricRecord or None.
        """
        exp = self._experiments.get(experiment_id)
        if exp is None:
            return None
        return exp.get_best_metric(metric_name, mode=mode)

    def get_metric_history(
        self,
        experiment_id: str,
        metric_name: str,
    ) -> List[MetricRecord]:
        """Get the full history of a metric.

        Args:
            experiment_id: Experiment ID.
            metric_name: Metric name.

        Returns:
            List of MetricRecord objects.
        """
        exp = self._experiments.get(experiment_id)
        if exp is None:
            return []
        return exp.get_metric_history(metric_name)

    # ------------------------------------------------------------------
    # Search / Filter
    # ------------------------------------------------------------------

    def search_experiments(
        self,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        status: Optional[ExperimentStatus] = None,
        param_filter: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> List[Experiment]:
        """Search experiments by criteria.

        Args:
            name: Filter by name (substring match).
            tags: Filter by tags (all must match).
            status: Filter by status.
            param_filter: Callable that takes params dict and returns True.

        Returns:
            List of matching Experiment objects.
        """
        results = []
        for exp in self._experiments.values():
            if name and name.lower() not in exp.name.lower():
                continue
            if tags and not all(t in exp.tags for t in tags):
                continue
            if status and exp.status != status:
                continue
            if param_filter and not param_filter(exp.params):
                continue
            results.append(exp)
        return results

    def list_experiments(self) -> List[ExperimentSummary]:
        """List all experiments with summary information.

        Returns:
            List of ExperimentSummary objects.
        """
        summaries = []
        for exp in self._experiments.values():
            best_metrics: Dict[str, float] = {}
            for name in exp.get_metric_names():
                best = exp.get_best_metric(name, mode="min")
                if best:
                    best_metrics[name] = best.value

            summaries.append(ExperimentSummary(
                id=exp.id,
                name=exp.name,
                status=exp.status,
                tags=exp.tags,
                params=exp.params,
                best_metrics=best_metrics,
                metric_count=sum(len(v) for v in exp._metrics.values()),
                artifact_count=len(exp._artifacts),
                created_at=exp.created_at,
                duration=exp.duration,
            ))
        return summaries

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def complete_experiment(self, experiment_id: Optional[str] = None) -> None:
        """Mark an experiment as completed.

        Args:
            experiment_id: Experiment to complete (active if None).
        """
        exp = self._resolve_experiment(experiment_id)
        exp.complete()
        if self._config.auto_save:
            self._save_experiment(exp)

    def fail_experiment(self, experiment_id: Optional[str] = None, message: str = "") -> None:
        """Mark an experiment as failed.

        Args:
            experiment_id: Experiment to fail.
            message: Failure message.
        """
        exp = self._resolve_experiment(experiment_id)
        exp.fail(message)
        if self._config.auto_save:
            self._save_experiment(exp)

    def delete_experiment(self, experiment_id: str) -> bool:
        """Delete an experiment.

        Args:
            experiment_id: Experiment to delete.

        Returns:
            True if deleted.
        """
        if experiment_id in self._experiments:
            exp = self._experiments.pop(experiment_id)
            # Clean up storage
            exp_dir = os.path.join(self._config.storage_dir, exp.name, exp.id)
            if os.path.isdir(exp_dir):
                import shutil
                shutil.rmtree(exp_dir, ignore_errors=True)
            if self._active_experiment_id == experiment_id:
                self._active_experiment_id = None
            return True
        return False

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_all(self) -> None:
        """Save all experiments to disk."""
        for exp in self._experiments.values():
            self._save_experiment(exp)

    def _save_experiment(self, exp: Experiment) -> None:
        """Save a single experiment."""
        try:
            exp_dir = os.path.join(self._config.storage_dir, exp.id)
            os.makedirs(exp_dir, exist_ok=True)
            exp.save(os.path.join(exp_dir, "experiment.json"))
        except Exception:
            pass

    def _load_from_disk(self) -> None:
        """Load experiments from the storage directory."""
        if not os.path.isdir(self._config.storage_dir):
            return

        for entry in os.listdir(self._config.storage_dir):
            exp_file = os.path.join(self._config.storage_dir, entry, "experiment.json")
            if os.path.isfile(exp_file):
                try:
                    exp = Experiment.load(exp_file)
                    self._experiments[exp.id] = exp
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Get tracker statistics.

        Returns:
            Dictionary with experiment counts and aggregate metrics.
        """
        status_counts: Dict[str, int] = {}
        for exp in self._experiments.values():
            key = exp.status.value
            status_counts[key] = status_counts.get(key, 0) + 1

        return {
            "total_experiments": len(self._experiments),
            "status_counts": status_counts,
            "storage_dir": self._config.storage_dir,
            "active_experiment": self._active_experiment_id,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _resolve_experiment(self, experiment_id: Optional[str] = None) -> Experiment:
        """Resolve an experiment ID, falling back to active."""
        eid = experiment_id or self._active_experiment_id
        if eid is None:
            raise ValueError("No active experiment. Create one first or specify experiment_id.")
        exp = self._experiments.get(eid)
        if exp is None:
            raise KeyError(f"Experiment '{eid}' not found")
        return exp
