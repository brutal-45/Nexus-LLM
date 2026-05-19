"""Request scheduler for Nexus-LLM backend.

Implements priority queues, batching, fair scheduling, and continuous batching
for efficient inference request management.
"""

import time
import threading
from typing import Optional, Dict, List, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import heapq
import logging

logger = logging.getLogger(__name__)


class RequestPriority(Enum):
    """Priority levels for inference requests."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3


class RequestStatus(Enum):
    """Status of a request in the scheduler."""
    QUEUED = "queued"
    PREFILLING = "prefilling"
    DECODING = "decoding"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class InferenceRequest:
    """A single inference request."""
    request_id: str
    prompt: str
    priority: RequestPriority = RequestPriority.NORMAL
    status: RequestStatus = RequestStatus.QUEUED
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    max_tokens: int = 512
    params: Dict[str, Any] = field(default_factory=dict)
    callback: Optional[Callable] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    user_id: Optional[str] = None
    token_count: int = 0

    @property
    def wait_time(self) -> float:
        """Time spent waiting in the queue."""
        start = self.started_at or time.time()
        return start - self.created_at

    @property
    def total_time(self) -> float:
        """Total time from creation to completion."""
        if self.completed_at is None:
            return time.time() - self.created_at
        return self.completed_at - self.created_at

    @property
    def priority_value(self) -> int:
        """Numeric priority for sorting (higher = more urgent)."""
        return self.priority.value

    def __lt__(self, other: "InferenceRequest") -> bool:
        """Compare for priority queue ordering."""
        if self.priority_value != other.priority_value:
            return self.priority_value > other.priority_value
        return self.created_at < other.created_at


@dataclass
class Batch:
    """A batch of requests being processed together."""
    batch_id: str
    requests: List[InferenceRequest]
    created_at: float = field(default_factory=time.time)
    max_batch_size: int = 8

    @property
    def size(self) -> int:
        return len(self.requests)

    @property
    def total_tokens(self) -> int:
        return sum(r.token_count for r in self.requests)

    @property
    def is_full(self) -> bool:
        return self.size >= self.max_batch_size


class RequestScheduler:
    """Scheduler for managing and batching inference requests.

    Supports priority-based scheduling, continuous batching, and fair
    scheduling across users.
    """

    def __init__(
        self,
        max_batch_size: int = 8,
        max_queue_size: int = 1000,
        fair_scheduling: bool = True,
        max_requests_per_user: int = 50,
        scheduling_policy: str = "priority",
    ):
        self.max_batch_size = max_batch_size
        self.max_queue_size = max_queue_size
        self.fair_scheduling = fair_scheduling
        self.max_requests_per_user = max_requests_per_user
        self.scheduling_policy = scheduling_policy

        self._queue: List[InferenceRequest] = []
        self._active_batch: Optional[Batch] = None
        self._completed: Dict[str, InferenceRequest] = {}
        self._user_counts: Dict[str, int] = defaultdict(int)
        self._request_counter: int = 0
        self._lock = threading.RLock()
        self._batch_counter: int = 0

    def submit(self, request: InferenceRequest) -> str:
        """Submit a request to the scheduler.

        Returns the request ID. Raises ValueError if queue is full
        or user has too many pending requests.
        """
        with self._lock:
            if len(self._queue) >= self.max_queue_size:
                raise ValueError(f"Request queue is full (max {self.max_queue_size})")

            if request.user_id and self.fair_scheduling:
                user_pending = self._user_counts.get(request.user_id, 0)
                if user_pending >= self.max_requests_per_user:
                    raise ValueError(
                        f"User '{request.user_id}' has too many pending requests "
                        f"(max {self.max_requests_per_user})"
                    )
                self._user_counts[request.user_id] += 1

            self._request_counter += 1
            if not request.request_id:
                request.request_id = f"req_{self._request_counter}"

            heapq.heappush(self._queue, request)
            request.status = RequestStatus.QUEUED

            logger.debug(f"Request '{request.request_id}' queued (priority={request.priority.name})")
            return request.request_id

    def submit_prompt(
        self,
        prompt: str,
        priority: RequestPriority = RequestPriority.NORMAL,
        max_tokens: int = 512,
        params: Optional[Dict[str, Any]] = None,
        callback: Optional[Callable] = None,
        user_id: Optional[str] = None,
    ) -> str:
        """Convenience method to submit a prompt directly.

        Returns the request ID.
        """
        request = InferenceRequest(
            request_id="",
            prompt=prompt,
            priority=priority,
            max_tokens=max_tokens,
            params=params or {},
            callback=callback,
            user_id=user_id,
        )
        return self.submit(request)

    def form_batch(self) -> Optional[Batch]:
        """Form a batch from queued requests using the configured scheduling policy.

        Returns None if no requests are available.
        """
        with self._lock:
            if not self._queue:
                return None

            batch_requests = []

            if self.scheduling_policy == "priority":
                batch_requests = self._form_priority_batch()
            elif self.scheduling_policy == "fair":
                batch_requests = self._form_fair_batch()
            elif self.scheduling_policy == "fifo":
                batch_requests = self._form_fifo_batch()
            elif self.scheduling_policy == "continuous":
                batch_requests = self._form_continuous_batch()
            else:
                batch_requests = self._form_priority_batch()

            if not batch_requests:
                return None

            self._batch_counter += 1
            batch = Batch(
                batch_id=f"batch_{self._batch_counter}",
                requests=batch_requests,
                max_batch_size=self.max_batch_size,
            )

            for req in batch_requests:
                req.status = RequestStatus.PREFILLING
                req.started_at = time.time()

            self._active_batch = batch
            logger.info(f"Formed batch '{batch.batch_id}' with {batch.size} requests")
            return batch

    def _form_priority_batch(self) -> List[InferenceRequest]:
        """Form a batch using strict priority ordering."""
        batch = []
        temp = []
        while self._queue and len(batch) < self.max_batch_size:
            req = heapq.heappop(self._queue)
            batch.append(req)
            temp.append(req)
        return batch

    def _form_fair_batch(self) -> List[InferenceRequest]:
        """Form a batch with fair scheduling across users.

        Round-robin across users to ensure no single user dominates.
        """
        user_queues: Dict[str, List[InferenceRequest]] = defaultdict(list)

        temp = []
        while self._queue:
            req = heapq.heappop(self._queue)
            user_queues[req.user_id or "anonymous"].append(req)
            temp.append(req)

        batch = []
        users = list(user_queues.keys())
        round_idx = 0

        while len(batch) < self.max_batch_size and any(user_queues[u] for u in users):
            user = users[round_idx % len(users)]
            if user_queues[user]:
                batch.append(user_queues[user].pop(0))
            round_idx += 1

        remaining = []
        for reqs in user_queues.values():
            remaining.extend(reqs)

        for req in remaining:
            heapq.heappush(self._queue, req)

        return batch

    def _form_fifo_batch(self) -> List[InferenceRequest]:
        """Form a batch using first-in-first-out ordering."""
        batch = []
        sorted_queue = sorted(self._queue, key=lambda r: r.created_at)
        self._queue.clear()

        for req in sorted_queue:
            if len(batch) < self.max_batch_size:
                batch.append(req)
            else:
                heapq.heappush(self._queue, req)

        return batch

    def _form_continuous_batch(self) -> List[InferenceRequest]:
        """Form a batch for continuous batching (prefill new requests alongside decoding).

        Includes ongoing decoding requests and new prefill requests up to batch limit.
        """
        batch = []

        ongoing = [r for r in self._queue if r.status == RequestStatus.DECODING]
        for req in ongoing:
            if len(batch) < self.max_batch_size:
                self._queue.remove(req)
                batch.append(req)

        new_requests = [r for r in self._queue if r.status == RequestStatus.QUEUED]
        new_requests.sort(key=lambda r: (-r.priority_value, r.created_at))

        for req in new_requests:
            if len(batch) < self.max_batch_size:
                self._queue.remove(req)
                batch.append(req)
            else:
                break

        heapq.heapify(self._queue)
        return batch

    def mark_completed(self, request_id: str, result: Any = None) -> None:
        """Mark a request as completed."""
        with self._lock:
            if self._active_batch:
                for req in self._active_batch.requests:
                    if req.request_id == request_id:
                        req.status = RequestStatus.COMPLETED
                        req.completed_at = time.time()
                        req.result = result
                        self._completed[request_id] = req
                        self._decrement_user_count(req.user_id)

                        if req.callback:
                            try:
                                req.callback(result)
                            except Exception as e:
                                logger.error(f"Callback error for request '{request_id}': {e}")
                        break

            if self._active_batch and all(
                r.status in (RequestStatus.COMPLETED, RequestStatus.FAILED, RequestStatus.CANCELLED)
                for r in self._active_batch.requests
            ):
                self._active_batch = None

    def mark_failed(self, request_id: str, error: str = "") -> None:
        """Mark a request as failed."""
        with self._lock:
            if self._active_batch:
                for req in self._active_batch.requests:
                    if req.request_id == request_id:
                        req.status = RequestStatus.FAILED
                        req.completed_at = time.time()
                        req.error = error
                        self._completed[request_id] = req
                        self._decrement_user_count(req.user_id)
                        break

    def cancel_request(self, request_id: str) -> bool:
        """Cancel a queued request. Returns True if successfully cancelled."""
        with self._lock:
            for i, req in enumerate(self._queue):
                if req.request_id == request_id:
                    req.status = RequestStatus.CANCELLED
                    req.completed_at = time.time()
                    self._queue.pop(i)
                    heapq.heapify(self._queue)
                    self._decrement_user_count(req.user_id)
                    return True

            if self._active_batch:
                for req in self._active_batch.requests:
                    if req.request_id == request_id:
                        req.status = RequestStatus.CANCELLED
                        req.completed_at = time.time()
                        self._decrement_user_count(req.user_id)
                        return True

            return False

    def _decrement_user_count(self, user_id: Optional[str]) -> None:
        """Decrement the request count for a user."""
        if user_id and user_id in self._user_counts:
            self._user_counts[user_id] = max(0, self._user_counts[user_id] - 1)
            if self._user_counts[user_id] == 0:
                del self._user_counts[user_id]

    def get_request(self, request_id: str) -> Optional[InferenceRequest]:
        """Get a request by its ID."""
        for req in self._queue:
            if req.request_id == request_id:
                return req

        if self._active_batch:
            for req in self._active_batch.requests:
                if req.request_id == request_id:
                    return req

        return self._completed.get(request_id)

    def get_queue_size(self) -> int:
        """Return the number of queued requests."""
        return len(self._queue)

    def get_active_batch_size(self) -> int:
        """Return the size of the currently active batch."""
        if self._active_batch is None:
            return 0
        return self._active_batch.size

    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        priority_counts = defaultdict(int)
        for req in self._queue:
            priority_counts[req.priority.name] += 1

        return {
            "queue_size": len(self._queue),
            "max_queue_size": self.max_queue_size,
            "active_batch_size": self.get_active_batch_size(),
            "completed_requests": len(self._completed),
            "total_submitted": self._request_counter,
            "max_batch_size": self.max_batch_size,
            "scheduling_policy": self.scheduling_policy,
            "fair_scheduling": self.fair_scheduling,
            "priority_counts": dict(priority_counts),
            "active_users": len(self._user_counts),
        }

    def clear_completed(self, max_age_seconds: float = 3600) -> int:
        """Clear old completed requests. Returns number cleared."""
        with self._lock:
            cutoff = time.time() - max_age_seconds
            to_remove = [
                rid for rid, req in self._completed.items()
                if req.completed_at and req.completed_at < cutoff
            ]
            for rid in to_remove:
                del self._completed[rid]
            return len(to_remove)
