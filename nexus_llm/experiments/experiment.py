"""Experiment Tracking for Nexus-LLM.

Provides experiment tracking with parameter logging, metric recording,
artifact management, and full experiment lifecycle support.  Experiments
can be created, logged to, and compared with one another.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ExperimentStatus(str, Enum):
    """Status of an experiment."""
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MetricType(str, Enum):
    """Type of metric."""
    SCALAR = "scalar"
    TIME_SERIES = "time_series"
    HISTOGRAM = "histogram"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MetricRecord:
    """A single metric measurement."""
    name: str
    value: float
    step: int = 0
    timestamp: float = field(default_factory=time.time)
    metric_type: MetricType = MetricType.SCALAR
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "step": self.step,
            "timestamp": self.timestamp,
            "metric_type": self.metric_type.value,
            "metadata": self.metadata,
        }


@dataclass
class Artifact:
    """An artifact associated with an experiment."""
    name: str
    path: str
    artifact_type: str = "file"  # file, model, image, plot, log
    description: str = ""
    file_size: int = 0
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "artifact_type": self.artifact_type,
            "description": self.description,
            "file_size": self.file_size,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }


@dataclass
class ExperimentConfig:
    """Configuration for experiment tracking."""
    storage_dir: str = "./experiments"
    auto_save: bool = True
    save_interval: float = 60.0  # Auto-save every N seconds
    max_artifacts: int = 100
    keep_history: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "storage_dir": self.storage_dir,
            "auto_save": self.auto_save,
            "save_interval": self.save_interval,
            "max_artifacts": self.max_artifacts,
            "keep_history": self.keep_history,
        }


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

class Experiment:
    """A single tracked experiment with parameters, metrics, and artifacts.

    Provides methods for logging parameters, recording metrics,
    managing artifacts, and persisting experiment data to disk.

    Example::

        exp = Experiment("my_experiment", tags=["baseline", "v1"])
        exp.log_param("learning_rate", 2e-5)
        exp.log_param("batch_size", 32)

        exp.start()
        for step in range(100):
            loss = train_step()
            exp.log_metric("loss", loss, step=step)
        exp.complete()
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        tags: Optional[List[str]] = None,
        config: Optional[ExperimentConfig] = None,
        experiment_id: Optional[str] = None,
    ) -> None:
        self.id = experiment_id or str(uuid.uuid4())[:12]
        self.name = name
        self.description = description
        self.tags = tags or []
        self._config = config or ExperimentConfig()
        self._status = ExperimentStatus.CREATED

        self._params: Dict[str, Any] = {}
        self._metrics: Dict[str, List[MetricRecord]] = {}
        self._artifacts: List[Artifact] = []
        self._notes: List[Dict[str, Any]] = []

        self._created_at = time.time()
        self._started_at: Optional[float] = None
        self._completed_at: Optional[float] = None
        self._duration: Optional[float] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def status(self) -> ExperimentStatus:
        return self._status

    @property
    def params(self) -> Dict[str, Any]:
        return dict(self._params)

    @property
    def created_at(self) -> float:
        return self._created_at

    @property
    def started_at(self) -> Optional[float]:
        return self._started_at

    @property
    def completed_at(self) -> Optional[float]:
        return self._completed_at

    @property
    def duration(self) -> Optional[float]:
        if self._completed_at and self._started_at:
            return self._completed_at - self._started_at
        if self._started_at:
            return time.time() - self._started_at
        return None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Mark the experiment as running."""
        if self._status == ExperimentStatus.CREATED:
            self._status = ExperimentStatus.RUNNING
            self._started_at = time.time()

    def complete(self) -> None:
        """Mark the experiment as completed."""
        if self._status == ExperimentStatus.RUNNING:
            self._status = ExperimentStatus.COMPLETED
            self._completed_at = time.time()
            if self._started_at:
                self._duration = self._completed_at - self._started_at

    def fail(self, message: str = "") -> None:
        """Mark the experiment as failed.

        Args:
            message: Optional failure description.
        """
        self._status = ExperimentStatus.FAILED
        self._completed_at = time.time()
        if message:
            self.add_note(f"FAILED: {message}")

    def cancel(self) -> None:
        """Mark the experiment as cancelled."""
        self._status = ExperimentStatus.CANCELLED
        self._completed_at = time.time()

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    def log_param(self, key: str, value: Any) -> None:
        """Log a single parameter.

        Args:
            key: Parameter name.
            value: Parameter value (must be JSON-serializable).
        """
        self._params[key] = value

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log multiple parameters.

        Args:
            params: Dictionary of parameter key-value pairs.
        """
        self._params.update(params)

    def get_param(self, key: str, default: Any = None) -> Any:
        """Get a parameter value.

        Args:
            key: Parameter name.
            default: Default value if key not found.

        Returns:
            Parameter value or default.
        """
        return self._params.get(key, default)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def log_metric(
        self,
        name: str,
        value: float,
        step: int = 0,
        metric_type: MetricType = MetricType.SCALAR,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a metric measurement.

        Args:
            name: Metric name.
            value: Metric value.
            step: Training step or iteration.
            metric_type: Type of metric.
            metadata: Optional metadata.
        """
        record = MetricRecord(
            name=name,
            value=value,
            step=step,
            metric_type=metric_type,
            metadata=metadata or {},
        )
        if name not in self._metrics:
            self._metrics[name] = []
        self._metrics[name].append(record)

    def log_metrics(self, metrics: Dict[str, float], step: int = 0) -> None:
        """Record multiple metrics at the same step.

        Args:
            metrics: Dictionary of metric name-value pairs.
            step: Training step.
        """
        for name, value in metrics.items():
            self.log_metric(name, value, step=step)

    def get_metric_history(self, name: str) -> List[MetricRecord]:
        """Get all recorded values for a metric.

        Args:
            name: Metric name.

        Returns:
            List of MetricRecord objects.
        """
        return list(self._metrics.get(name, []))

    def get_metric_names(self) -> List[str]:
        """Get all logged metric names."""
        return list(self._metrics.keys())

    def get_latest_metric(self, name: str) -> Optional[MetricRecord]:
        """Get the most recent value for a metric.

        Args:
            name: Metric name.

        Returns:
            Most recent MetricRecord, or None.
        """
        records = self._metrics.get(name, [])
        return records[-1] if records else None

    def get_best_metric(
        self, name: str, mode: str = "min"
    ) -> Optional[MetricRecord]:
        """Get the best (min or max) value for a metric.

        Args:
            name: Metric name.
            mode: 'min' for lowest, 'max' for highest.

        Returns:
            Best MetricRecord, or None.
        """
        records = self._metrics.get(name, [])
        if not records:
            return None
        if mode == "min":
            return min(records, key=lambda r: r.value)
        return max(records, key=lambda r: r.value)

    # ------------------------------------------------------------------
    # Artifacts
    # ------------------------------------------------------------------

    def log_artifact(
        self,
        name: str,
        path: str,
        artifact_type: str = "file",
        description: str = "",
        copy: bool = True,
    ) -> Artifact:
        """Log an artifact associated with the experiment.

        Args:
            name: Artifact name.
            path: File path of the artifact.
            artifact_type: Type (file, model, image, plot, log).
            description: Optional description.
            copy: Whether to copy the file into the experiment directory.

        Returns:
            The created Artifact.
        """
        file_size = 0
        actual_path = path

        if os.path.isfile(path):
            file_size = os.path.getsize(path)
            if copy and self._config.storage_dir:
                dest_dir = os.path.join(self._config.storage_dir, self.id, "artifacts")
                os.makedirs(dest_dir, exist_ok=True)
                dest_path = os.path.join(dest_dir, os.path.basename(path))
                shutil.copy2(path, dest_path)
                actual_path = dest_path

        artifact = Artifact(
            name=name,
            path=actual_path,
            artifact_type=artifact_type,
            description=description,
            file_size=file_size,
        )
        self._artifacts.append(artifact)
        return artifact

    def get_artifacts(self, artifact_type: Optional[str] = None) -> List[Artifact]:
        """Get artifacts, optionally filtered by type.

        Args:
            artifact_type: Optional type filter.

        Returns:
            List of Artifact objects.
        """
        if artifact_type:
            return [a for a in self._artifacts if a.artifact_type == artifact_type]
        return list(self._artifacts)

    # ------------------------------------------------------------------
    # Notes
    # ------------------------------------------------------------------

    def add_note(self, text: str) -> None:
        """Add a timestamped note to the experiment.

        Args:
            text: Note content.
        """
        self._notes.append({
            "text": text,
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
        })

    def get_notes(self) -> List[Dict[str, Any]]:
        """Get all notes."""
        return list(self._notes)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the experiment to a dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "tags": self.tags,
            "status": self._status.value,
            "params": self._params,
            "metrics": {
                name: [r.to_dict() for r in records]
                for name, records in self._metrics.items()
            },
            "artifacts": [a.to_dict() for a in self._artifacts],
            "notes": self._notes,
            "created_at": self._created_at,
            "started_at": self._started_at,
            "completed_at": self._completed_at,
            "duration": round(self.duration, 4) if self.duration else None,
        }

    def save(self, path: Optional[str] = None) -> str:
        """Save the experiment to a JSON file.

        Args:
            path: Custom save path. Defaults to storage_dir/id/experiment.json.

        Returns:
            Path to the saved file.
        """
        save_path = path
        if save_path is None:
            save_path = os.path.join(
                self._config.storage_dir, self.id, "experiment.json"
            )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        return save_path

    @classmethod
    def load(cls, path: str) -> "Experiment":
        """Load an experiment from a JSON file.

        Args:
            path: Path to the experiment.json file.

        Returns:
            Reconstructed Experiment object.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        exp = cls(
            name=data["name"],
            description=data.get("description", ""),
            tags=data.get("tags", []),
            experiment_id=data.get("id"),
        )
        exp._status = ExperimentStatus(data.get("status", "created"))
        exp._params = data.get("params", {})
        exp._created_at = data.get("created_at", time.time())
        exp._started_at = data.get("started_at")
        exp._completed_at = data.get("completed_at")
        exp._notes = data.get("notes", [])

        # Reconstruct metrics
        for name, records in data.get("metrics", {}).items():
            exp._metrics[name] = [
                MetricRecord(
                    name=r["name"],
                    value=r["value"],
                    step=r.get("step", 0),
                    timestamp=r.get("timestamp", 0),
                    metric_type=MetricType(r.get("metric_type", "scalar")),
                    metadata=r.get("metadata", {}),
                )
                for r in records
            ]

        # Reconstruct artifacts
        exp._artifacts = [
            Artifact(
                name=a["name"],
                path=a["path"],
                artifact_type=a.get("artifact_type", "file"),
                description=a.get("description", ""),
                file_size=a.get("file_size", 0),
                created_at=a.get("created_at", 0),
                metadata=a.get("metadata", {}),
            )
            for a in data.get("artifacts", [])
        ]

        return exp
