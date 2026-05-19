"""Queue Manager for Nexus-LLM.

Provides a priority-based request queue for managing incoming
inference requests with support for priority levels, request
grouping, timeout handling, and queue statistics.
"""

from __future__ import annotations

import heapq
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class RequestPriority(IntEnum):
    """Priority levels for queued requests. Lower = higher priority."""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


class RequestStatus(str):
    """Status of a queued request."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(order=False)
class _PrioritizedEntry:
    """Internal heap entry for priority ordering."""
    priority: int
    created_at: float
    sequence: int
    request_id: str

    def __lt__(self, other: "_PrioritizedEntry") -> bool:
        if self.priority != other.priority:
            return self.priority < other.priority
        if self.created_at != other.created_at:
            return self.created_at < other.created_at
        return self.sequence < other.sequence


@dataclass
class QueuedRequest:
    """A request in the serving queue."""
    request_id: str = ""
    payload: Any = None
    priority: int = RequestPriority.NORMAL
    status: str = RequestStatus.PENDING
    group: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    timeout_seconds: Optional[float] = None
    result: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.request_id:
            self.request_id = str(uuid.uuid4())[:12]

    @property
    def wait_time(self) -> Optional[float]:
        """Time spent waiting in queue."""
        if self.started_at is not None:
            return self.started_at - self.created_at
        return time.time() - self.created_at

    @property
    def processing_time(self) -> Optional[float]:
        """Time spent processing."""
        if self.started_at is not None and self.completed_at is not None:
            return self.completed_at - self.started_at
        return None

    @property
    def is_expired(self) -> bool:
        """Whether the request has exceeded its timeout."""
        if self.timeout_seconds is None:
            return False
        return time.time() - self.created_at > self.timeout_seconds

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "priority": self.priority,
            "status": self.status,
            "group": self.group,
            "created_at": self.created_at,
            "wait_time": round(self.wait_time, 4) if self.wait_time else None,
            "processing_time": round(self.processing_time, 4) if self.processing_time else None,
            "timeout_seconds": self.timeout_seconds,
            "error": self.error,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Queue Manager
# ---------------------------------------------------------------------------

class QueueManager:
    """Priority request queue for model serving.

    Manages incoming inference requests with priority ordering,
    request grouping, timeout enforcement, and comprehensive
    statistics tracking.

    Example::

        qm = QueueManager(max_size=5000, default_timeout=60.0)

        req_id = qm.enqueue(
            payload={"prompt": "Hello"},
            priority=RequestPriority.HIGH,
            group="chat",
        )

        req_id, payload = qm.dequeue()
        qm.complete(req_id, result={"text": "Hi there!"})

        stats = qm.get_stats()
    """

    def __init__(
        self,
        max_size: int = 10000,
        default_priority: int = RequestPriority.NORMAL,
        default_timeout: Optional[float] = None,
        enable_timeout_check: bool = True,
        timeout_check_interval: float = 10.0,
    ) -> None:
        self._max_size = max_size
        self._default_priority = default_priority
        self._default_timeout = default_timeout

        self._heap: List[_PrioritizedEntry] = []
        self._requests: Dict[str, QueuedRequest] = {}
        self._counter = 0

        self._lock = threading.RLock()
        self._not_empty = threading.Condition(self._lock)

        self._running = True
        self._timeout_thread: Optional[threading.Thread] = None

        if enable_timeout_check and default_timeout is not None:
            self._timeout_thread = threading.Thread(
                target=self._timeout_check_loop,
                daemon=True,
                name="queue-timeout",
            )
            self._timeout_thread.start()

    # ------------------------------------------------------------------
    # Enqueue / Dequeue
    # ------------------------------------------------------------------

    def enqueue(
        self,
        payload: Any,
        *,
        request_id: Optional[str] = None,
        priority: Optional[int] = None,
        group: Optional[str] = None,
        timeout: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add a request to the queue.

        Args:
            payload: Request payload data.
            request_id: Optional custom request ID.
            priority: Request priority level.
            group: Optional group label for batch operations.
            timeout: Timeout in seconds (None = no timeout).
            metadata: Optional metadata dictionary.

        Returns:
            The request ID.

        Raises:
            OverflowError: If the queue is full.
        """
        with self._not_empty:
            if len(self._heap) >= self._max_size:
                raise OverflowError(f"Queue is full ({self._max_size} requests)")

            rid = request_id or str(uuid.uuid4())[:12]
            prio = priority if priority is not None else self._default_priority
            timeout_val = timeout if timeout is not None else self._default_timeout

            self._counter += 1
            entry = _PrioritizedEntry(
                priority=prio,
                created_at=time.time(),
                sequence=self._counter,
                request_id=rid,
            )
            heapq.heappush(self._heap, entry)

            self._requests[rid] = QueuedRequest(
                request_id=rid,
                payload=payload,
                priority=prio,
                group=group,
                timeout_seconds=timeout_val,
                metadata=metadata or {},
            )

            self._not_empty.notify()
            return rid

    def dequeue(self, timeout: Optional[float] = None) -> tuple:
        """Remove and return the highest-priority request.

        Args:
            timeout: Maximum seconds to wait. None waits forever.

        Returns:
            Tuple of (request_id, payload).

        Raises:
            TimeoutError: If no request is available in time.
        """
        with self._not_empty:
            end_time = None if timeout is None else time.time() + timeout

            while self._running:
                # Skip expired or cancelled entries
                while self._heap:
                    entry = self._heap[0]
                    req = self._requests.get(entry.request_id)

                    if req is None or req.status == RequestStatus.CANCELLED:
                        heapq.heappop(self._heap)
                        continue

                    if req.is_expired:
                        req.status = RequestStatus.TIMEOUT
                        heapq.heappop(self._heap)
                        continue

                    break
                else:
                    if end_time is not None:
                        remaining = end_time - time.time()
                        if remaining <= 0:
                            raise TimeoutError("No request available within timeout")
                        self._not_empty.wait(timeout=remaining)
                    else:
                        self._not_empty.wait()
                    continue

                if not self._heap:
                    continue

                entry = heapq.heappop(self._heap)
                req = self._requests.get(entry.request_id)
                if req:
                    req.status = RequestStatus.PROCESSING
                    req.started_at = time.time()
                    return (req.request_id, req.payload)

            raise TimeoutError("Queue manager is stopped")

    def peek(self) -> Optional[tuple]:
        """View the highest-priority request without removing it.

        Returns:
            Tuple of (request_id, payload) or None.
        """
        with self._lock:
            for entry in self._heap:
                req = self._requests.get(entry.request_id)
                if req and req.status == RequestStatus.PENDING:
                    return (req.request_id, req.payload)
            return None

    # ------------------------------------------------------------------
    # Request lifecycle
    # ------------------------------------------------------------------

    def complete(self, request_id: str, result: Any = None) -> None:
        """Mark a request as completed.

        Args:
            request_id: The request ID.
            result: Optional result data.

        Raises:
            KeyError: If the request is not found.
        """
        with self._lock:
            req = self._requests.get(request_id)
            if req is None:
                raise KeyError(f"Request '{request_id}' not found")
            req.status = RequestStatus.COMPLETED
            req.completed_at = time.time()
            req.result = result

    def fail(self, request_id: str, error: str = "") -> None:
        """Mark a request as failed.

        Args:
            request_id: The request ID.
            error: Error description.
        """
        with self._lock:
            req = self._requests.get(request_id)
            if req is None:
                raise KeyError(f"Request '{request_id}' not found")
            req.status = RequestStatus.FAILED
            req.completed_at = time.time()
            req.error = error

    def cancel(self, request_id: str) -> bool:
        """Cancel a pending request.

        Args:
            request_id: The request ID.

        Returns:
            True if cancelled, False if not found or already processing.
        """
        with self._lock:
            req = self._requests.get(request_id)
            if req is None or req.status != RequestStatus.PENDING:
                return False
            req.status = RequestStatus.CANCELLED
            req.completed_at = time.time()
            return True

    def cancel_group(self, group: str) -> int:
        """Cancel all pending requests in a group.

        Args:
            group: The group label.

        Returns:
            Number of requests cancelled.
        """
        with self._lock:
            count = 0
            for req in self._requests.values():
                if req.group == group and req.status == RequestStatus.PENDING:
                    req.status = RequestStatus.CANCELLED
                    req.completed_at = time.time()
                    count += 1
            return count

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def size(self) -> int:
        """Number of pending requests in the queue."""
        with self._lock:
            return sum(1 for r in self._requests.values() if r.status == RequestStatus.PENDING)

    def is_empty(self) -> bool:
        """Whether the queue has no pending requests."""
        return self.size() == 0

    def is_full(self) -> bool:
        """Whether the queue is at capacity."""
        with self._lock:
            return self.size() >= self._max_size

    def get_request(self, request_id: str) -> Optional[QueuedRequest]:
        """Get a request by ID."""
        return self._requests.get(request_id)

    def get_group_requests(self, group: str) -> List[QueuedRequest]:
        """Get all requests in a group.

        Args:
            group: The group label.

        Returns:
            List of QueuedRequest objects.
        """
        with self._lock:
            return [r for r in self._requests.values() if r.group == group]

    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics.

        Returns:
            Dictionary with queue metrics and status breakdown.
        """
        with self._lock:
            status_counts: Dict[str, int] = {}
            for req in self._requests.values():
                status_counts[req.status] = status_counts.get(req.status, 0) + 1

            group_counts: Dict[str, int] = {}
            for req in self._requests.values():
                if req.group and req.status == RequestStatus.PENDING:
                    group_counts[req.group] = group_counts.get(req.group, 0) + 1

            return {
                "pending": status_counts.get(RequestStatus.PENDING, 0),
                "processing": status_counts.get(RequestStatus.PROCESSING, 0),
                "completed": status_counts.get(RequestStatus.COMPLETED, 0),
                "failed": status_counts.get(RequestStatus.FAILED, 0),
                "timeout": status_counts.get(RequestStatus.TIMEOUT, 0),
                "cancelled": status_counts.get(RequestStatus.CANCELLED, 0),
                "max_size": self._max_size,
                "groups": group_counts,
            }

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def clear(self) -> int:
        """Remove all pending requests.

        Returns:
            Number of requests cleared.
        """
        with self._lock:
            count = 0
            for req in self._requests.values():
                if req.status == RequestStatus.PENDING:
                    req.status = RequestStatus.CANCELLED
                    req.completed_at = time.time()
                    count += 1
            self._heap.clear()
            return count

    def purge_completed(self, max_age: float = 3600.0) -> int:
        """Remove completed/failed/timeout records older than max_age.

        Args:
            max_age: Maximum age in seconds.

        Returns:
            Number of records purged.
        """
        with self._lock:
            cutoff = time.time() - max_age
            to_remove = [
                rid for rid, req in self._requests.items()
                if req.completed_at is not None
                and req.completed_at < cutoff
                and req.status in (
                    RequestStatus.COMPLETED,
                    RequestStatus.FAILED,
                    RequestStatus.TIMEOUT,
                    RequestStatus.CANCELLED,
                )
            ]
            for rid in to_remove:
                del self._requests[rid]
            return len(to_remove)

    def shutdown(self) -> None:
        """Shutdown the queue manager, releasing waiting threads."""
        self._running = False
        with self._not_empty:
            self._not_empty.notify_all()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _timeout_check_loop(self) -> None:
        """Periodically check for timed-out requests."""
        while self._running:
            try:
                time.sleep(10.0)
                with self._lock:
                    for req in self._requests.values():
                        if req.status == RequestStatus.PENDING and req.is_expired:
                            req.status = RequestStatus.TIMEOUT
                            req.completed_at = time.time()
            except Exception:
                pass
