"""Experiment class for Nexus-LLM.

Represents a single experiment with lifecycle management (start, stop,
pause, resume) and logging of metrics, parameters, and artifacts.
"""

import enum
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ExperimentState(enum.Enum):
    """Possible states of an experiment."""

    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class Experiment:
    """A single experiment with lifecycle and logging support.

    An experiment tracks metrics, parameters, and artifacts throughout
    its lifecycle.  State transitions follow::

        CREATED -> RUNNING -> COMPLETED
                       |         ^
                       v         |
                      PAUSED ----+

    Any state can transition to FAILED on error.

    Example::

        exp = Experiment("my-exp", config={"lr": 0.01})
        exp.start()
        exp.log_metric("loss", 0.5, step=1)
        exp.log_parameter("optimizer", "adam")
        exp.stop()
    """

    def __init__(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
        experiment_id: Optional[str] = None,
    ) -> None:
        self.id: str = experiment_id or uuid.uuid4().hex[:12]
        self.name: str = name
        self.config: Dict[str, Any] = config or {}
        self.state: ExperimentState = ExperimentState.CREATED

        self._metrics: List[Dict[str, Any]] = []
        self._parameters: Dict[str, Any] = {}
        self._artifacts: List[str] = []
        self._created_at: float = time.time()
        self._started_at: Optional[float] = None
        self._stopped_at: Optional[float] = None
        self._error_message: Optional[str] = None

    # ------------------------------------------------------------------
    # Lifecycle methods
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Transition the experiment to RUNNING.

        Raises:
            RuntimeError: If the experiment is not in CREATED or PAUSED state.
        """
        if self.state not in (ExperimentState.CREATED, ExperimentState.PAUSED):
            raise RuntimeError(
                f"Cannot start experiment in {self.state.value} state; "
                f"expected CREATED or PAUSED."
            )
        self.state = ExperimentState.RUNNING
        self._started_at = self._started_at or time.time()
        logger.info("Experiment %s (%s) started.", self.name, self.id)

    def stop(self) -> None:
        """Transition the experiment to COMPLETED.

        Raises:
            RuntimeError: If the experiment is not in RUNNING state.
        """
        if self.state != ExperimentState.RUNNING:
            raise RuntimeError(
                f"Cannot stop experiment in {self.state.value} state; "
                f"expected RUNNING."
            )
        self.state = ExperimentState.COMPLETED
        self._stopped_at = time.time()
        logger.info("Experiment %s (%s) completed.", self.name, self.id)

    def pause(self) -> None:
        """Transition the experiment to PAUSED.

        Raises:
            RuntimeError: If the experiment is not in RUNNING state.
        """
        if self.state != ExperimentState.RUNNING:
            raise RuntimeError(
                f"Cannot pause experiment in {self.state.value} state; "
                f"expected RUNNING."
            )
        self.state = ExperimentState.PAUSED
        logger.info("Experiment %s (%s) paused.", self.name, self.id)

    def resume(self) -> None:
        """Transition the experiment back to RUNNING from PAUSED.

        Raises:
            RuntimeError: If the experiment is not in PAUSED state.
        """
        if self.state != ExperimentState.PAUSED:
            raise RuntimeError(
                f"Cannot resume experiment in {self.state.value} state; "
                f"expected PAUSED."
            )
        self.state = ExperimentState.RUNNING
        logger.info("Experiment %s (%s) resumed.", self.name, self.id)

    def fail(self, message: str = "") -> None:
        """Transition the experiment to FAILED.

        This can be called from any non-terminal state.

        Args:
            message: Optional error description.
        """
        self.state = ExperimentState.FAILED
        self._stopped_at = time.time()
        self._error_message = message
        logger.error("Experiment %s (%s) failed: %s", self.name, self.id, message)

    # ------------------------------------------------------------------
    # Logging methods
    # ------------------------------------------------------------------

    def log_metric(self, name: str, value: float, step: int) -> None:
        """Log a metric value at a given step.

        Args:
            name: Metric name (e.g. ``"loss"``, ``"accuracy"``).
            value: Numeric metric value.
            step: Training or evaluation step number.

        Raises:
            RuntimeError: If the experiment is not in RUNNING state.
        """
        if self.state != ExperimentState.RUNNING:
            raise RuntimeError(
                f"Cannot log metrics when experiment is {self.state.value}; "
                f"must be RUNNING."
            )
        record = {
            "name": name,
            "value": value,
            "step": step,
            "timestamp": time.time(),
        }
        self._metrics.append(record)
        logger.debug(
            "Experiment %s: metric %s=%s at step %d",
            self.id, name, value, step,
        )

    def log_parameter(self, name: str, value: Any) -> None:
        """Log a hyperparameter value.

        Parameters can be logged in any experiment state.

        Args:
            name: Parameter name.
            value: Parameter value (any JSON-serialisable type).
        """
        self._parameters[name] = value
        logger.debug("Experiment %s: param %s=%s", self.id, name, value)

    def log_artifact(self, path: str) -> None:
        """Record an artifact file path associated with this experiment.

        Args:
            path: Filesystem path to the artifact.
        """
        self._artifacts.append(path)
        logger.debug("Experiment %s: artifact %s", self.id, path)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Return a status dictionary for this experiment.

        Returns:
            Dict with keys: id, name, state, config, parameters,
            num_metrics, num_artifacts, created_at, started_at,
            stopped_at, duration, error_message.
        """
        duration: Optional[float] = None
        if self._started_at is not None and self._stopped_at is not None:
            duration = self._stopped_at - self._started_at
        elif self._started_at is not None and self.state == ExperimentState.RUNNING:
            duration = time.time() - self._started_at

        return {
            "id": self.id,
            "name": self.name,
            "state": self.state.value,
            "config": self.config,
            "parameters": dict(self._parameters),
            "num_metrics": len(self._metrics),
            "num_artifacts": len(self._artifacts),
            "created_at": self._created_at,
            "started_at": self._started_at,
            "stopped_at": self._stopped_at,
            "duration": duration,
            "error_message": self._error_message,
        }

    @property
    def metrics(self) -> List[Dict[str, Any]]:
        """Return a copy of the metric records list."""
        return list(self._metrics)

    @property
    def parameters(self) -> Dict[str, Any]:
        """Return a copy of the parameters dict."""
        return dict(self._parameters)

    @property
    def artifacts(self) -> List[str]:
        """Return a copy of the artifact paths list."""
        return list(self._artifacts)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"<Experiment id={self.id!r} name={self.name!r} "
            f"state={self.state.value}>"
        )
