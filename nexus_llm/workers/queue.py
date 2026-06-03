"""Priority-based task queue."""

from __future__ import annotations

import heapq
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(order=True)
class _QueueItem:
    """Internal heap item ordered by (priority, sequence_number).

    Lower priority values are dequeued first.
    """

    sort_key: tuple = field(compare=True)
    task_id: str = field(compare=False)
    task: Any = field(compare=False)
    priority: int = field(compare=False)
    enqueued_at: float = field(compare=False, default_factory=time.monotonic)


class TaskQueue:
    """Thread-safe priority queue for tasks.

    Tasks are dequeued in priority order (lower value = higher
    priority).  Within the same priority level, tasks are processed
    in FIFO order.

    Example::

        q = TaskQueue()
        q.put(lambda: "low", priority=10)
        q.put(lambda: "high", priority=1)
        task = q.get()   # the high-priority task
    """

    def __init__(self, max_size: int = 0) -> None:
        """Initialise the task queue.

        Args:
            max_size: Maximum queue size.  ``0`` means unbounded.
        """
        self._max_size = max_size
        self._heap: List[_QueueItem] = []
        self._counter = 0
        self._lock = threading.RLock()
        self._not_empty = threading.Condition(self._lock)
        self._not_full = threading.Condition(self._lock)
        self._completed: Dict[str, bool] = {}

    # ------------------------------------------------------------------
    # Enqueue / dequeue
    # ------------------------------------------------------------------

    def put(
        self,
        task: Any,
        priority: int = 5,
        block: bool = True,
        timeout: Optional[float] = None,
    ) -> str:
        """Add a task to the queue.

        Args:
            task: The task payload (typically a callable).
            priority: Priority value — lower is higher priority.
            block: If ``True`` and the queue is full, block until space
                is available.
            timeout: Maximum seconds to wait if blocking.

        Returns:
            A unique task ID.

        Raises:
            queue.Full: If the queue is full and *block* is ``False``.
        """
        with self._not_full:
            if self._max_size > 0:
                while len(self._heap) >= self._max_size:
                    if not block:
                        from queue import Full
                        raise Full("TaskQueue is full.")
                    if not self._not_full.wait(timeout):
                        from queue import Full
                        raise Full("Timed out waiting for queue space.")

            task_id = uuid.uuid4().hex[:12]
            self._counter += 1
            item = _QueueItem(
                sort_key=(priority, self._counter),
                task_id=task_id,
                task=task,
                priority=priority,
            )
            heapq.heappush(self._heap, item)
            self._not_empty.notify()
            return task_id

    def get(
        self,
        block: bool = True,
        timeout: Optional[float] = None,
    ) -> Any:
        """Remove and return the highest-priority task.

        Args:
            block: If ``True`` and the queue is empty, block until a
                task is available.
            timeout: Maximum seconds to wait if blocking.

        Returns:
            The task payload.

        Raises:
            queue.Empty: If the queue is empty and *block* is ``False``.
        """
        with self._not_empty:
            while not self._heap:
                if not block:
                    from queue import Empty
                    raise Empty("TaskQueue is empty.")
                if not self._not_empty.wait(timeout):
                    from queue import Empty
                    raise Empty("Timed out waiting for task.")

            item = heapq.heappop(self._heap)
            self._not_full.notify()
            return item.task

    # ------------------------------------------------------------------
    # Completion tracking
    # ------------------------------------------------------------------

    def task_done(self, task_id: str) -> None:
        """Mark a task as completed.

        Args:
            task_id: The ID returned by :meth:`put`.
        """
        with self._lock:
            self._completed[task_id] = True

    def is_done(self, task_id: str) -> bool:
        """Check whether a task has been marked as done."""
        with self._lock:
            return self._completed.get(task_id, False)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def size(self) -> int:
        """Return the number of tasks currently in the queue."""
        with self._lock:
            return len(self._heap)

    def empty(self) -> bool:
        """Return ``True`` if the queue is empty."""
        with self._lock:
            return len(self._heap) == 0

    def full(self) -> bool:
        """Return ``True`` if the queue is full."""
        if self._max_size <= 0:
            return False
        with self._lock:
            return len(self._heap) >= self._max_size

    def clear(self) -> int:
        """Remove all tasks from the queue.

        Returns:
            The number of tasks removed.
        """
        with self._lock:
            count = len(self._heap)
            self._heap.clear()
            self._not_full.notify_all()
            return count

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:  # noqa: D105
        return self.size()

    def __repr__(self) -> str:  # noqa: D105
        return f"<TaskQueue size={self.size()} max_size={self._max_size}>"
