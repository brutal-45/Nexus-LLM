"""Nexus-LLM Task Coordinator.

Provides the Coordinator class that manages concurrent tasks, tracks
dependencies between tasks, and coordinates execution order and
resource allocation.
"""

import enum
import heapq
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

from nexus_llm.exceptions import NexusLLMError

logger = logging.getLogger(__name__)


class TaskPriority(enum.IntEnum):
    """Priority levels for coordinated tasks."""

    LOW = 0
    NORMAL = 5
    HIGH = 10
    CRITICAL = 15


class TaskState(enum.Enum):
    """States a task can be in."""

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(order=True)
class Task:
    """A coordinated task with priority and dependencies.

    Attributes:
        task_id: Unique identifier.
        fn: Callable to execute.
        priority: Task priority (higher runs first).
        dependencies: Set of task IDs that must complete first.
        state: Current task state.
        result: Output from execution.
        error: Error message if failed.
        created_at: Timestamp when the task was created.
    """

    # sort_key is used for heap ordering: (-priority, created_at)
    sort_key: Any = field(init=False)
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    fn: Callable = field(default=None, repr=False)
    priority: TaskPriority = TaskPriority.NORMAL
    dependencies: Set[str] = field(default_factory=set)
    state: TaskState = field(default=TaskState.PENDING, repr=False)
    result: Any = field(default=None, repr=False)
    error: Optional[str] = field(default=None, repr=False)
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        self.sort_key = (-self.priority, self.created_at)


class Coordinator:
    """Task coordinator that manages dependencies and execution order.

    The Coordinator maintains a task queue with priority ordering and
    dependency tracking. Tasks are only dispatched when all their
    dependencies have completed successfully.

    Attributes:
        pending_count: Number of tasks still pending.
    """

    def __init__(self, max_workers: int = 4) -> None:
        self._max_workers = max_workers
        self._tasks: Dict[str, Task] = {}
        self._queue: List[Task] = []
        self._running: Set[str] = set()
        self._lock = threading.RLock()
        self._completed_event = threading.Event()
        logger.info("Coordinator initialized with max_workers=%d", max_workers)

    @property
    def pending_count(self) -> int:
        """Number of tasks not yet completed."""
        with self._lock:
            return sum(
                1 for t in self._tasks.values()
                if t.state in (TaskState.PENDING, TaskState.QUEUED, TaskState.RUNNING)
            )

    def submit(
        self,
        fn: Callable,
        priority: TaskPriority = TaskPriority.NORMAL,
        dependencies: Optional[Set[str]] = None,
        task_id: Optional[str] = None,
    ) -> str:
        """Submit a task for coordinated execution.

        Args:
            fn: Callable to execute.
            priority: Task priority.
            dependencies: Set of task IDs that must complete first.
            task_id: Optional explicit task ID.

        Returns:
            The task ID.
        """
        task = Task(
            task_id=task_id or str(uuid.uuid4()),
            fn=fn,
            priority=priority,
            dependencies=dependencies or set(),
        )
        with self._lock:
            self._tasks[task.task_id] = task
            # Check if dependencies are already satisfied
            if self._dependencies_met(task):
                task.state = TaskState.QUEUED
                heapq.heappush(self._queue, task)
            logger.debug("Submitted task %s with priority %s", task.task_id, priority.name)
        return task.task_id

    def cancel(self, task_id: str) -> bool:
        """Cancel a pending or queued task.

        Args:
            task_id: ID of the task to cancel.

        Returns:
            True if the task was cancelled, False if not possible.
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return False
            if task.state in (TaskState.PENDING, TaskState.QUEUED):
                task.state = TaskState.CANCELLED
                self._queue = [t for t in self._queue if t.task_id != task_id]
                heapq.heapify(self._queue)
                logger.debug("Cancelled task %s", task_id)
                return True
            return False

    def process(self) -> int:
        """Process all ready tasks synchronously.

        Runs tasks whose dependencies are met until the queue is empty
        or no more tasks can proceed.

        Returns:
            Number of tasks processed in this round.
        """
        processed = 0
        while True:
            task = self._next_ready_task()
            if task is None:
                break
            self._execute(task)
            processed += 1
        return processed

    def process_all(self) -> int:
        """Process all tasks until none remain.

        Continues processing rounds until all tasks are completed,
        failed, or cancelled.

        Returns:
            Total number of tasks processed.
        """
        total = 0
        while self.pending_count > 0:
            count = self.process()
            total += count
            if count == 0:
                # Deadlock or unresolvable dependencies
                logger.warning("No progress made; possible deadlock or unresolved dependencies")
                break
        return total

    def get_task(self, task_id: str) -> Optional[Task]:
        """Retrieve a task by ID.

        Args:
            task_id: Task identifier.

        Returns:
            The Task, or None if not found.
        """
        return self._tasks.get(task_id)

    def get_result(self, task_id: str) -> Any:
        """Get the result of a completed task.

        Args:
            task_id: Task identifier.

        Returns:
            The task result.

        Raises:
            NexusLLMError: If the task is not completed or not found.
        """
        task = self._tasks.get(task_id)
        if task is None:
            raise NexusLLMError(f"Task not found: {task_id}", error_code="TASK_NOT_FOUND")
        if task.state != TaskState.COMPLETED:
            raise NexusLLMError(
                f"Task {task_id} is {task.state.value}, not completed",
                error_code="TASK_NOT_COMPLETED",
            )
        return task.result

    def wait_for(self, task_id: str, timeout: Optional[float] = None) -> Task:
        """Wait for a task to complete.

        Args:
            task_id: Task identifier.
            timeout: Maximum seconds to wait.

        Returns:
            The completed Task.

        Raises:
            NexusLLMError: If the task is not found.
            TimeoutError: If the timeout expires.
        """
        deadline = time.time() + timeout if timeout else float("inf")
        while time.time() < deadline:
            task = self._tasks.get(task_id)
            if task is None:
                raise NexusLLMError(f"Task not found: {task_id}", error_code="TASK_NOT_FOUND")
            if task.state == TaskState.COMPLETED:
                return task
            if task.state == TaskState.FAILED:
                raise NexusLLMError(f"Task {task_id} failed: {task.error}", error_code="TASK_FAILED")
            time.sleep(0.05)
        raise TimeoutError(f"Timeout waiting for task {task_id}")

    def stats(self) -> Dict[str, int]:
        """Return coordination statistics.

        Returns:
            Dictionary with task counts by state.
        """
        counts: Dict[str, int] = {s.value: 0 for s in TaskState}
        with self._lock:
            for task in self._tasks.values():
                counts[task.state.value] += 1
        return counts

    def health_check(self) -> Dict[str, Any]:
        """Return health status of the coordinator."""
        return {
            "status": "healthy",
            "total_tasks": len(self._tasks),
            "pending_count": self.pending_count,
            "max_workers": self._max_workers,
            "stats": self.stats(),
        }

    def _dependencies_met(self, task: Task) -> bool:
        """Check whether all dependencies of a task are completed."""
        for dep_id in task.dependencies:
            dep = self._tasks.get(dep_id)
            if dep is None or dep.state != TaskState.COMPLETED:
                return False
        return True

    def _next_ready_task(self) -> Optional[Task]:
        """Pop the next task from the queue whose dependencies are met."""
        with self._lock:
            # Re-queue tasks whose dependencies are now met
            for task in list(self._tasks.values()):
                if task.state == TaskState.PENDING and self._dependencies_met(task):
                    task.state = TaskState.QUEUED
                    heapq.heappush(self._queue, task)

            while self._queue:
                task = heapq.heappop(self._queue)
                if task.state == TaskState.CANCELLED:
                    continue
                if self._dependencies_met(task):
                    return task
                # Put back if dependencies not met
                task.state = TaskState.PENDING
            return None

    def _execute(self, task: Task) -> None:
        """Execute a single task."""
        task.state = TaskState.RUNNING
        with self._lock:
            self._running.add(task.task_id)
        try:
            task.result = task.fn()
            task.state = TaskState.COMPLETED
            logger.debug("Task %s completed", task.task_id)
        except Exception as exc:
            task.error = str(exc)
            task.state = TaskState.FAILED
            logger.error("Task %s failed: %s", task.task_id, exc)
        finally:
            with self._lock:
                self._running.discard(task.task_id)
