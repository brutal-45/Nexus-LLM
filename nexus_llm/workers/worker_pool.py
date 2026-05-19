"""
Worker Pool for Nexus-LLM

Manages a pool of worker processes for parallel task execution.
Supports both inference and training workers with automatic
scaling, load balancing, and fault recovery.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

from nexus_llm.workers.inference_worker import InferenceWorker, InferenceTask, InferenceResult
from nexus_llm.workers.training_worker import TrainingWorker, TrainingTask, TrainingResult
from nexus_llm.workers.task_queue import TaskQueue, TaskPriority


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class WorkerPoolError(Exception):
    """Base exception for worker pool errors."""


class WorkerPoolFullError(WorkerPoolError):
    """Raised when the pool has no available workers."""

    def __init__(self, pool_size: int) -> None:
        self.pool_size = pool_size
        super().__init__(f"Worker pool is full ({pool_size} workers)")


class NoAvailableWorkerError(WorkerPoolError):
    """Raised when no worker is available to handle a task."""


# ---------------------------------------------------------------------------
# Worker info
# ---------------------------------------------------------------------------

@dataclass
class WorkerInfo:
    """Tracks the state of a single worker in the pool."""

    worker_id: str
    worker_type: str  # "inference" or "training"
    is_busy: bool = False
    current_task_id: Optional[str] = None
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_processing_time: float = 0.0
    started_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "worker_id": self.worker_id,
            "worker_type": self.worker_type,
            "is_busy": self.is_busy,
            "current_task_id": self.current_task_id,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "total_processing_time": round(self.total_processing_time, 2),
            "started_at": self.started_at,
            "last_activity": self.last_activity,
        }


# ---------------------------------------------------------------------------
# Worker Pool
# ---------------------------------------------------------------------------

class WorkerPool:
    """Manages a pool of worker processes for parallel execution.

    Supports:
    - Dynamic scaling of inference and training workers
    - Round-robin and least-busy load balancing
    - Automatic worker restart on failure
    - Pool-wide statistics and monitoring

    Example::

        pool = WorkerPool(num_inference_workers=2, num_training_workers=1)
        pool.start()

        # Submit inference
        task = InferenceTask(prompt="Hello!")
        result = pool.submit_inference(task)

        # Submit training
        train_task = TrainingTask(model_name="llama-7b")
        pool.submit_training(train_task)

        pool.stop()
    """

    def __init__(
        self,
        num_inference_workers: int = 2,
        num_training_workers: int = 1,
        auto_restart: bool = True,
        health_check_interval: float = 30.0,
        pool_id: Optional[str] = None,
    ) -> None:
        """Initialize the WorkerPool.

        Args:
            num_inference_workers: Number of inference workers to create.
            num_training_workers: Number of training workers to create.
            auto_restart: Whether to automatically restart failed workers.
            health_check_interval: Seconds between health checks.
            pool_id: Optional pool identifier.
        """
        self.pool_id = pool_id or str(uuid.uuid4())[:8]
        self.auto_restart = auto_restart
        self.health_check_interval = health_check_interval

        self._inference_workers: List[InferenceWorker] = []
        self._training_workers: List[TrainingWorker] = []
        self._worker_info: Dict[str, WorkerInfo] = {}
        self._inference_index = 0  # Round-robin counter
        self._training_index = 0

        self._lock = threading.Lock()
        self._running = False
        self._health_thread: Optional[threading.Thread] = None

        # Create inference workers
        for i in range(num_inference_workers):
            worker = InferenceWorker(worker_id=f"inf-{i}")
            self._inference_workers.append(worker)
            self._worker_info[worker.worker_id] = WorkerInfo(
                worker_id=worker.worker_id,
                worker_type="inference",
            )

        # Create training workers
        for i in range(num_training_workers):
            worker = TrainingWorker(worker_id=f"train-{i}")
            self._training_workers.append(worker)
            self._worker_info[worker.worker_id] = WorkerInfo(
                worker_id=worker.worker_id,
                worker_type="training",
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start all workers in the pool."""
        if self._running:
            return

        for worker in self._inference_workers + self._training_workers:
            worker.start()

        self._running = True

        # Start health check thread
        if self.auto_restart:
            self._health_thread = threading.Thread(
                target=self._health_check_loop,
                daemon=True,
                name=f"pool-health-{self.pool_id}",
            )
            self._health_thread.start()

    def stop(self, timeout: float = 30.0) -> None:
        """Stop all workers in the pool.

        Args:
            timeout: Seconds to wait per worker for graceful shutdown.
        """
        self._running = False

        for worker in self._inference_workers:
            worker.stop(timeout=timeout)

        for worker in self._training_workers:
            worker.stop(timeout=timeout)

        if self._health_thread and self._health_thread.is_alive():
            self._health_thread.join(timeout=5.0)

    # ------------------------------------------------------------------
    # Inference task submission
    # ------------------------------------------------------------------

    def submit_inference(
        self,
        task: InferenceTask,
        strategy: str = "round_robin",
    ) -> InferenceResult:
        """Submit an inference task to the pool.

        Args:
            task: The inference task.
            strategy: Load balancing strategy ('round_robin' or 'least_busy').

        Returns:
            InferenceResult from the assigned worker.

        Raises:
            NoAvailableWorkerError: If no inference workers are available.
        """
        worker = self._select_inference_worker(strategy)
        if worker is None:
            raise NoAvailableWorkerError("No inference workers available")

        info = self._worker_info[worker.worker_id]
        info.is_busy = True
        info.current_task_id = task.task_id

        try:
            result = worker.submit(task)
            info.tasks_completed += 1
            return result
        except Exception:
            info.tasks_failed += 1
            raise
        finally:
            info.is_busy = False
            info.current_task_id = None
            info.last_activity = time.time()

    def submit_inference_async(
        self,
        task: InferenceTask,
        strategy: str = "round_robin",
    ) -> str:
        """Submit an inference task without waiting.

        Args:
            task: The inference task.
            strategy: Load balancing strategy.

        Returns:
            The task ID.

        Raises:
            NoAvailableWorkerError: If no inference workers are available.
        """
        worker = self._select_inference_worker(strategy)
        if worker is None:
            raise NoAvailableWorkerError("No inference workers available")

        info = self._worker_info[worker.worker_id]
        info.is_busy = True
        info.current_task_id = task.task_id

        try:
            return worker.submit_async(task)
        except Exception:
            info.is_busy = False
            info.current_task_id = None
            raise

    # ------------------------------------------------------------------
    # Training task submission
    # ------------------------------------------------------------------

    def submit_training(self, task: TrainingTask) -> TrainingResult:
        """Submit a training task to the pool.

        Args:
            task: The training task configuration.

        Returns:
            TrainingResult from the assigned worker.

        Raises:
            NoAvailableWorkerError: If no training workers are available.
        """
        worker = self._select_training_worker()
        if worker is None:
            raise NoAvailableWorkerError("No training workers available")

        info = self._worker_info[worker.worker_id]
        info.is_busy = True
        info.current_task_id = task.task_id

        try:
            result = worker.submit(task)
            info.tasks_completed += 1
            return result
        except Exception:
            info.tasks_failed += 1
            raise
        finally:
            info.is_busy = False
            info.current_task_id = None
            info.last_activity = time.time()

    def submit_training_async(self, task: TrainingTask) -> str:
        """Submit a training task without waiting.

        Args:
            task: The training task configuration.

        Returns:
            The task ID.

        Raises:
            NoAvailableWorkerError: If no training workers are available.
        """
        worker = self._select_training_worker()
        if worker is None:
            raise NoAvailableWorkerError("No training workers available")

        info = self._worker_info[worker.worker_id]
        info.is_busy = True
        info.current_task_id = task.task_id

        try:
            return worker.submit_async(task)
        except Exception:
            info.is_busy = False
            info.current_task_id = None
            raise

    # ------------------------------------------------------------------
    # Pool statistics
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Get pool status and statistics.

        Returns:
            Dictionary with pool metrics and per-worker info.
        """
        inference_info = [
            self._worker_info[w.worker_id].to_dict()
            for w in self._inference_workers
        ]
        training_info = [
            self._worker_info[w.worker_id].to_dict()
            for w in self._training_workers
        ]

        return {
            "pool_id": self.pool_id,
            "running": self._running,
            "inference_workers": {
                "total": len(self._inference_workers),
                "active": sum(1 for w in self._inference_workers if w.is_alive()),
                "busy": sum(
                    1 for w in self._inference_workers
                    if self._worker_info[w.worker_id].is_busy
                ),
                "workers": inference_info,
            },
            "training_workers": {
                "total": len(self._training_workers),
                "active": sum(1 for w in self._training_workers if w.is_alive()),
                "busy": sum(
                    1 for w in self._training_workers
                    if self._worker_info[w.worker_id].is_busy
                ),
                "workers": training_info,
            },
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _select_inference_worker(
        self, strategy: str = "round_robin"
    ) -> Optional[InferenceWorker]:
        """Select an inference worker using the specified strategy."""
        if not self._inference_workers:
            return None

        alive = [w for w in self._inference_workers if w.is_alive()]
        if not alive:
            return None

        if strategy == "least_busy":
            # Pick the worker with fewest completed tasks (approximation of load)
            return min(alive, key=lambda w: self._worker_info[w.worker_id].tasks_completed)
        else:
            # Round-robin
            with self._lock:
                worker = alive[self._inference_index % len(alive)]
                self._inference_index += 1
                return worker

    def _select_training_worker(self) -> Optional[TrainingWorker]:
        """Select the first available training worker."""
        if not self._training_workers:
            return None

        alive = [w for w in self._training_workers if w.is_alive()]
        if not alive:
            return None

        # Prefer least busy
        not_busy = [
            w for w in alive
            if not self._worker_info[w.worker_id].is_busy
        ]
        if not_busy:
            return not_busy[0]

        return alive[0]

    def _health_check_loop(self) -> None:
        """Periodically check worker health and restart failed workers."""
        while self._running:
            try:
                time.sleep(self.health_check_interval)

                for worker in self._inference_workers + self._training_workers:
                    if not worker.is_alive() and self._running:
                        # Restart dead worker
                        try:
                            worker.stop(timeout=5.0)
                            worker.start()
                        except Exception:
                            pass

            except Exception:
                pass
