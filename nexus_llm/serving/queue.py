"""Request queue with priority support for Nexus-LLM serving."""

import enum
import heapq
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class Priority(enum.IntEnum):
    """Request priority levels (lower value = higher priority)."""
    URGENT = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


@dataclass(order=False)
class QueuedRequest:
    """A request waiting in the queue."""
    request_id: str
    data: Any
    priority: Priority
    enqueue_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other: "QueuedRequest") -> bool:
        """Compare by priority (lower int = higher priority), then FIFO."""
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.enqueue_time < other.enqueue_time


class QueueFullError(Exception):
    """Raised when the queue has reached its maximum size."""


class RequestQueue:
    """Thread-safe priority queue for incoming requests.

    Supports four priority levels and configurable maximum size with
    backpressure via ``QueueFullError``.
    """

    def __init__(self, max_size: int = 1000) -> None:
        self._max_size = max_size
        self._heap: List[QueuedRequest] = []
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._total_enqueued: int = 0
        self._total_dequeued: int = 0
        self._dropped: int = 0

    # -- Core operations ------------------------------------------------------

    def enqueue(
        self,
        request: Any,
        priority: Priority = Priority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add a request to the queue.

        Args:
            request: The request payload.
            priority: Request priority (default NORMAL).
            metadata: Optional metadata dict.

        Returns:
            Generated request ID.

        Raises:
            QueueFullError: If the queue is at maximum capacity.
        """
        with self._lock:
            if len(self._heap) >= self._max_size:
                self._dropped += 1
                raise QueueFullError(
                    f"Queue is full (max_size={self._max_size}). "
                    f"Request rejected."
                )

            request_id = str(uuid.uuid4())
            queued = QueuedRequest(
                request_id=request_id,
                data=request,
                priority=priority,
                enqueue_time=time.time(),
                metadata=metadata or {},
            )
            heapq.heappush(self._heap, queued)
            self._total_enqueued += 1
            self._not_empty.notify()

        return request_id

    def dequeue(self, timeout: Optional[float] = None) -> Optional[QueuedRequest]:
        """Remove and return the highest-priority request.

        Blocks if the queue is empty until a request arrives or the
        timeout expires.

        Args:
            timeout: Maximum seconds to wait (``None`` = wait forever).

        Returns:
            ``QueuedRequest`` or ``None`` if the timeout expired.
        """
        with self._not_empty:
            end_time = None
            if timeout is not None:
                end_time = time.monotonic() + timeout

            while not self._heap:
                if end_time is not None:
                    remaining = end_time - time.monotonic()
                    if remaining <= 0:
                        return None
                    self._not_empty.wait(timeout=remaining)
                else:
                    self._not_empty.wait()

            item = heapq.heappop(self._heap)
            self._total_dequeued += 1
            return item

    def peek(self) -> Optional[QueuedRequest]:
        """Return the highest-priority request without removing it.

        Returns:
            ``QueuedRequest`` at the front, or ``None`` if empty.
        """
        with self._lock:
            if not self._heap:
                return None
            # Peek at the smallest item
            return self._heap[0]

    def size(self) -> int:
        """Return the current number of requests in the queue."""
        with self._lock:
            return len(self._heap)

    # -- Stats ----------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return queue statistics.

        Returns:
            Dict with ``current_size``, ``max_size``, ``total_enqueued``,
            ``total_dequeued``, ``dropped``, ``utilization_percent``.
        """
        with self._lock:
            current = len(self._heap)
        return {
            "current_size": current,
            "max_size": self._max_size,
            "total_enqueued": self._total_enqueued,
            "total_dequeued": self._total_dequeued,
            "dropped": self._dropped,
            "utilization_percent": round(
                (current / self._max_size) * 100, 2
            ) if self._max_size > 0 else 0.0,
        }

    def clear(self) -> int:
        """Remove all requests from the queue.

        Returns:
            Number of requests that were cleared.
        """
        with self._lock:
            count = len(self._heap)
            self._heap.clear()
            return count
