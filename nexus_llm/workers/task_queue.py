"""
Task Queue for Nexus-LLM

Priority-based task queue implementation for managing pending
inference and training tasks. Supports task prioritization,
cancellation, and status tracking.
"""

from __future__ import annotations

import heapq
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TaskPriority(IntEnum):
    """Task priority levels. Lower values = higher priority."""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


class TaskStatus(str):
    """Status of a queued task."""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    RETRYING = "retrying"


# ---------------------------------------------------------------------------
# Task wrapper
# ---------------------------------------------------------------------------

@dataclass(order=False)
class _PrioritizedTask:
    """Internal wrapper for priority queue ordering."""

    priority: int
    created_at: float
    sequence: int  # Tiebreaker for FIFO within same priority
    task_id: str
    payload: Any

    def __lt__(self, other: "_PrioritizedTask") -> bool:
        if self.priority != other.priority:
            return self.priority < other.priority
        if self.created_at != other.created_at:
            return self.created_at < other.created_at
        return self.sequence < other.sequence


@dataclass
class TaskRecord:
    """Tracks the state and history of a task."""

    task_id: str
    status: str = TaskStatus.QUEUED
    priority: int = TaskPriority.NORMAL
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    attempts: int = 0
    max_attempts: int = 1
    error: Optional[str] = None
    result: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def wait_time(self) -> Optional[float]:
        """Time spent waiting in queue (seconds)."""
        if self.started_at is not None:
            return self.started_at - self.created_at
        return time.time() - self.created_at

    @property
    def processing_time(self) -> Optional[float]:
        """Time spent processing (seconds)."""
        if self.started_at is not None and self.completed_at is not None:
            return self.completed_at - self.started_at
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "task_id": self.task_id,
            "status": self.status,
            "priority": self.priority,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "error": self.error,
            "wait_time": round(self.wait_time, 4) if self.wait_time else None,
            "processing_time": round(self.processing_time, 4) if self.processing_time else None,
        }


# ---------------------------------------------------------------------------
# Task Queue
# ---------------------------------------------------------------------------

class TaskQueue:
    """Thread-safe priority task queue for Nexus-LLM.

    Features:
    - Priority-based ordering (lower number = higher priority)
    - FIFO ordering within the same priority level
    - Task cancellation and status tracking
    - Maximum queue size with backpressure
    - Task retry support
    - Statistics and monitoring

    Example::

        queue = TaskQueue(max_size=1000)

        # Enqueue tasks
        task_id = queue.enqueue("process_data", priority=TaskPriority.HIGH)

        # Dequeue the next task
        task_id, payload = queue.dequeue()

        # Mark task complete
        queue.complete(task_id, result={"status": "ok"})

        # Check stats
        stats = queue.get_stats()
    """

    def __init__(
        self,
        max_size: int = 10000,
        default_priority: int = TaskPriority.NORMAL,
        default_max_attempts: int = 1,
    ) -> None:
        """Initialize the TaskQueue.

        Args:
            max_size: Maximum number of pending tasks.
            default_priority: Default priority for new tasks.
            default_max_attempts: Default max retry attempts.
        """
        self._max_size = max_size
        self._default_priority = default_priority
        self._default_max_attempts = default_max_attempts

        self._heap: List[_PrioritizedTask] = []
        self._records: Dict[str, TaskRecord] = {}
        self._counter = 0  # Sequence counter for FIFO tiebreaking

        self._lock = threading.RLock()
        self._not_empty = threading.Condition(self._lock)

    # ------------------------------------------------------------------
    # Enqueue / Dequeue
    # ------------------------------------------------------------------

    def enqueue(
        self,
        payload: Any,
        *,
        task_id: Optional[str] = None,
        priority: Optional[int] = None,
        max_attempts: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add a task to the queue.

        Args:
            payload: The task data (any serializable object).
            task_id: Optional custom task ID. Auto-generated if not provided.
            priority: Task priority. Defaults to the queue's default.
            max_attempts: Maximum retry attempts on failure.
            metadata: Optional metadata dictionary.

        Returns:
            The task ID.

        Raises:
            OverflowError: If the queue is at maximum capacity.
        """
        with self._not_empty:
            if len(self._heap) >= self._max_size:
                raise OverflowError(
                    f"Task queue is full ({self._max_size} tasks)"
                )

            tid = task_id or str(uuid.uuid4())
            prio = priority if priority is not None else self._default_priority
            attempts = max_attempts if max_attempts is not None else self._default_max_attempts

            self._counter += 1
            task = _PrioritizedTask(
                priority=prio,
                created_at=time.time(),
                sequence=self._counter,
                task_id=tid,
                payload=payload,
            )
            heapq.heappush(self._heap, task)

            self._records[tid] = TaskRecord(
                task_id=tid,
                status=TaskStatus.QUEUED,
                priority=prio,
                max_attempts=attempts,
                metadata=metadata or {},
            )

            self._not_empty.notify()
            return tid

    def dequeue(self, timeout: Optional[float] = None) -> tuple:
        """Remove and return the highest-priority task.

        Args:
            timeout: Maximum wait time in seconds. None means wait forever.

        Returns:
            Tuple of (task_id, payload).

        Raises:
            TimeoutError: If no task is available within the timeout.
        """
        with self._not_empty:
            end_time = None if timeout is None else time.time() + timeout

            while not self._heap:
                if end_time is not None:
                    remaining = end_time - time.time()
                    if remaining <= 0:
                        raise TimeoutError("No task available within timeout")
                    self._not_empty.wait(timeout=remaining)
                else:
                    self._not_empty.wait()

            task = heapq.heappop(self._heap)
            record = self._records.get(task.task_id)
            if record:
                record.status = TaskStatus.RUNNING
                record.started_at = time.time()
                record.attempts += 1

            return (task.task_id, task.payload)

    def peek(self) -> Optional[tuple]:
        """View the highest-priority task without removing it.

        Returns:
            Tuple of (task_id, payload) or None if empty.
        """
        with self._lock:
            if not self._heap:
                return None
            task = self._heap[0]
            return (task.task_id, task.payload)

    # ------------------------------------------------------------------
    # Task lifecycle
    # ------------------------------------------------------------------

    def complete(self, task_id: str, result: Any = None) -> None:
        """Mark a task as completed.

        Args:
            task_id: The task ID.
            result: Optional result data.

        Raises:
            KeyError: If the task ID is not found.
        """
        with self._lock:
            record = self._records.get(task_id)
            if record is None:
                raise KeyError(f"Task '{task_id}' not found")
            record.status = TaskStatus.COMPLETED
            record.completed_at = time.time()
            record.result = result

    def fail(self, task_id: str, error: str = "", retry: bool = False) -> None:
        """Mark a task as failed.

        Args:
            task_id: The task ID.
            error: Error description.
            retry: Whether to re-enqueue the task for retry.

        Raises:
            KeyError: If the task ID is not found.
        """
        with self._lock:
            record = self._records.get(task_id)
            if record is None:
                raise KeyError(f"Task '{task_id}' not found")

            record.error = error

            if retry and record.attempts < record.max_attempts:
                # Re-enqueue for retry
                record.status = TaskStatus.RETRYING
                self._counter += 1
                prioritized = _PrioritizedTask(
                    priority=record.priority,
                    created_at=time.time(),
                    sequence=self._counter,
                    task_id=task_id,
                    payload=None,  # Payload already consumed
                )
                heapq.heappush(self._heap, prioritized)
            else:
                record.status = TaskStatus.FAILED
                record.completed_at = time.time()

    def cancel(self, task_id: str) -> bool:
        """Cancel a queued task.

        Args:
            task_id: The task ID.

        Returns:
            True if the task was cancelled, False if not found or already running.
        """
        with self._lock:
            record = self._records.get(task_id)
            if record is None:
                return False

            if record.status != TaskStatus.QUEUED:
                return False

            record.status = TaskStatus.CANCELLED
            record.completed_at = time.time()

            # Remove from heap (lazy deletion - will be skipped on dequeue)
            return True

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def size(self) -> int:
        """Return the number of pending tasks in the queue."""
        with self._lock:
            return len(self._heap)

    def is_empty(self) -> bool:
        """Check if the queue is empty."""
        with self._lock:
            return len(self._heap) == 0

    def is_full(self) -> bool:
        """Check if the queue is at capacity."""
        with self._lock:
            return len(self._heap) >= self._max_size

    def get_task(self, task_id: str) -> Optional[TaskRecord]:
        """Retrieve a task's record by ID.

        Args:
            task_id: The task ID.

        Returns:
            TaskRecord or None if not found.
        """
        with self._lock:
            return self._records.get(task_id)

    def get_stats(self) -> Dict[str, Any]:
        """Return queue statistics.

        Returns:
            Dictionary with queue metrics.
        """
        with self._lock:
            status_counts: Dict[str, int] = {}
            for record in self._records.values():
                status_counts[record.status] = status_counts.get(record.status, 0) + 1

            return {
                "pending": len(self._heap),
                "total_processed": sum(
                    1 for r in self._records.values()
                    if r.status in (TaskStatus.COMPLETED, TaskStatus.FAILED)
                ),
                "completed": status_counts.get(TaskStatus.COMPLETED, 0),
                "failed": status_counts.get(TaskStatus.FAILED, 0),
                "cancelled": status_counts.get(TaskStatus.CANCELLED, 0),
                "running": status_counts.get(TaskStatus.RUNNING, 0),
                "queued": status_counts.get(TaskStatus.QUEUED, 0),
                "retrying": status_counts.get(TaskStatus.RETRYING, 0),
                "max_size": self._max_size,
            }

    def clear(self) -> int:
        """Remove all pending tasks from the queue.

        Returns:
            Number of tasks cleared.
        """
        with self._lock:
            count = len(self._heap)
            self._heap.clear()
            return count

    def purge_completed(self, max_age: float = 3600.0) -> int:
        """Remove completed/failed task records older than max_age.

        Args:
            max_age: Maximum age in seconds for keeping records.

        Returns:
            Number of records purged.
        """
        with self._lock:
            cutoff = time.time() - max_age
            to_remove = [
                tid
                for tid, record in self._records.items()
                if record.completed_at is not None
                and record.completed_at < cutoff
                and record.status in (
                    TaskStatus.COMPLETED,
                    TaskStatus.FAILED,
                    TaskStatus.CANCELLED,
                )
            ]
            for tid in to_remove:
                del self._records[tid]
            return len(to_remove)
