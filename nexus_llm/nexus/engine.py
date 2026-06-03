"""Nexus-LLM Engine Orchestration.

Provides the Engine class that manages the lifecycle of model execution,
including initialization, warm-up, execution of inference or training jobs,
and state transitions.
"""

import enum
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from nexus_llm.exceptions import InferenceError, NexusLLMError

logger = logging.getLogger(__name__)


class EngineState(enum.Enum):
    """Possible states of the engine."""

    IDLE = "idle"
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    STOPPED = "stopped"


@dataclass
class EngineConfig:
    """Configuration for the engine.

    Attributes:
        max_concurrent_jobs: Maximum number of concurrent jobs.
        warmup_iterations: Number of warm-up iterations on start.
        default_timeout: Default timeout in seconds for jobs.
        auto_restart: Whether to auto-restart on failure.
        retry_count: Number of retries for failed jobs.
    """

    max_concurrent_jobs: int = 4
    warmup_iterations: int = 3
    default_timeout: float = 300.0
    auto_restart: bool = False
    retry_count: int = 2


@dataclass
class JobResult:
    """Result from an engine job execution.

    Attributes:
        job_id: Unique identifier for the job.
        success: Whether the job completed successfully.
        output: The output data from the job.
        duration_ms: Execution duration in milliseconds.
        error: Error message if the job failed.
        metadata: Additional metadata about the execution.
    """

    job_id: str
    success: bool
    output: Any = None
    duration_ms: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class Engine:
    """Orchestration engine for managing model execution jobs.

    The Engine handles the lifecycle of inference and training jobs,
    managing state transitions, concurrency limits, and result collection.

    Attributes:
        state: Current state of the engine.
        config: Active engine configuration.
    """

    def __init__(self, config: Optional[EngineConfig] = None) -> None:
        self._config = config or EngineConfig()
        self._state = EngineState.IDLE
        self._active_jobs: Dict[str, Dict[str, Any]] = {}
        self._job_counter = 0
        self._callbacks: Dict[str, List[Callable]] = {}
        logger.info("Engine created with max_concurrent_jobs=%d", self._config.max_concurrent_jobs)

    @property
    def state(self) -> EngineState:
        """Current state of the engine."""
        return self._state

    @property
    def config(self) -> EngineConfig:
        """Active engine configuration."""
        return self._config

    @property
    def active_job_count(self) -> int:
        """Number of currently active jobs."""
        return len(self._active_jobs)

    def start(self) -> None:
        """Start the engine, performing initialization and warm-up.

        Raises:
            NexusLLMError: If the engine is already running or in an error state.
        """
        if self._state in (EngineState.RUNNING, EngineState.READY):
            raise NexusLLMError("Engine is already running", error_code="ENGINE_ALREADY_RUNNING")
        if self._state == EngineState.ERROR:
            raise NexusLLMError("Engine is in error state; call reset() first", error_code="ENGINE_ERROR_STATE")

        self._transition(EngineState.INITIALIZING)
        try:
            self._warmup()
            self._transition(EngineState.READY)
            logger.info("Engine started and ready")
        except Exception as exc:
            self._transition(EngineState.ERROR)
            raise InferenceError(message=f"Engine startup failed: {exc}", error_code="ENGINE_STARTUP_FAILED") from exc

    def stop(self) -> None:
        """Stop the engine gracefully, cancelling active jobs."""
        if self._state == EngineState.STOPPED:
            return
        logger.info("Stopping engine with %d active jobs", len(self._active_jobs))
        self._active_jobs.clear()
        self._transition(EngineState.STOPPED)

    def pause(self) -> None:
        """Pause the engine, preventing new job submissions."""
        if self._state != EngineState.RUNNING:
            raise NexusLLMError("Can only pause a running engine", error_code="ENGINE_NOT_RUNNING")
        self._transition(EngineState.PAUSED)
        logger.info("Engine paused")

    def resume(self) -> None:
        """Resume a paused engine."""
        if self._state != EngineState.PAUSED:
            raise NexusLLMError("Can only resume a paused engine", error_code="ENGINE_NOT_PAUSED")
        self._transition(EngineState.RUNNING)
        logger.info("Engine resumed")

    def reset(self) -> None:
        """Reset the engine from an error state to idle."""
        self._active_jobs.clear()
        self._transition(EngineState.IDLE)
        logger.info("Engine reset to idle")

    def submit(self, job_fn: Callable, job_id: Optional[str] = None, **kwargs: Any) -> JobResult:
        """Submit a job for execution.

        Args:
            job_fn: Callable to execute.
            job_id: Optional unique identifier for the job.
            **kwargs: Additional keyword arguments passed to job_fn.

        Returns:
            A JobResult with execution details.

        Raises:
            InferenceError: If the engine is not ready or capacity is exceeded.
        """
        if self._state not in (EngineState.READY, EngineState.RUNNING):
            raise InferenceError(message="Engine is not ready for jobs", error_code="ENGINE_NOT_READY")

        if len(self._active_jobs) >= self._config.max_concurrent_jobs:
            raise InferenceError(
                message="Maximum concurrent jobs reached",
                error_code="ENGINE_CAPACITY_EXCEEDED",
            )

        if job_id is None:
            self._job_counter += 1
            job_id = f"job_{self._job_counter}"

        self._transition(EngineState.RUNNING)
        self._active_jobs[job_id] = {"fn": job_fn, "kwargs": kwargs, "start_time": time.time()}

        start = time.perf_counter()
        try:
            output = job_fn(**kwargs)
            duration_ms = (time.perf_counter() - start) * 1000
            result = JobResult(job_id=job_id, success=True, output=output, duration_ms=duration_ms)
        except Exception as exc:
            duration_ms = (time.perf_counter() - start) * 1000
            result = JobResult(job_id=job_id, success=False, error=str(exc), duration_ms=duration_ms)
            logger.error("Job %s failed: %s", job_id, exc)
        finally:
            self._active_jobs.pop(job_id, None)
            if not self._active_jobs:
                self._transition(EngineState.READY)

        self._emit("job_complete", result)
        return result

    def on(self, event: str, callback: Callable) -> None:
        """Register a callback for an engine event.

        Args:
            event: Event name (e.g., 'job_complete', 'state_change').
            callback: Callable to invoke when the event fires.
        """
        self._callbacks.setdefault(event, []).append(callback)

    def health_check(self) -> Dict[str, Any]:
        """Return health status of the engine.

        Returns:
            Dictionary with state, active_jobs, and config summary.
        """
        return {
            "status": "healthy" if self._state in (EngineState.READY, EngineState.RUNNING) else str(self._state.value),
            "state": self._state.value,
            "active_jobs": len(self._active_jobs),
            "max_concurrent_jobs": self._config.max_concurrent_jobs,
        }

    def _transition(self, new_state: EngineState) -> None:
        """Transition to a new state, emitting a state_change event."""
        old_state = self._state
        self._state = new_state
        logger.debug("Engine state: %s -> %s", old_state.value, new_state.value)
        self._emit("state_change", {"old": old_state.value, "new": new_state.value})

    def _warmup(self) -> None:
        """Perform warm-up iterations."""
        for i in range(self._config.warmup_iterations):
            logger.debug("Warmup iteration %d/%d", i + 1, self._config.warmup_iterations)

    def _emit(self, event: str, data: Any) -> None:
        """Emit an event to registered callbacks."""
        for cb in self._callbacks.get(event, []):
            try:
                cb(data)
            except Exception as exc:
                logger.warning("Callback error for event '%s': %s", event, exc)
