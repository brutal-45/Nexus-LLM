"""Load balancer for Nexus-LLM serving."""

import enum
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


class Strategy(enum.Enum):
    """Load-balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    RANDOM = "random"


@dataclass
class WorkerInfo:
    """Metadata about a registered worker."""
    worker_id: str
    address: str
    active_connections: int = 0
    total_requests: int = 0
    last_request_time: Optional[float] = None
    healthy: bool = True
    registered_at: float = field(default_factory=time.time)


class LoadBalancer:
    """Distributes requests across registered workers.

    Supports round-robin, least-connections, and random routing
    strategies with health tracking.
    """

    def __init__(self, strategy: Strategy = Strategy.ROUND_ROBIN) -> None:
        self._strategy = strategy
        self._workers: Dict[str, WorkerInfo] = {}
        self._rr_index: int = 0
        self._lock = threading.Lock()

    # -- Worker management ----------------------------------------------------

    def add_worker(self, worker_id: str, address: str) -> None:
        """Register a new worker.

        Args:
            worker_id: Unique identifier for the worker.
            address: Network address (e.g. ``"http://10.0.0.1:8000"``).

        Raises:
            ValueError: If a worker with the same ID already exists.
        """
        with self._lock:
            if worker_id in self._workers:
                raise ValueError(f"Worker '{worker_id}' already registered")
            self._workers[worker_id] = WorkerInfo(
                worker_id=worker_id,
                address=address,
            )

    def remove_worker(self, worker_id: str) -> None:
        """Unregister a worker.

        Args:
            worker_id: ID of the worker to remove.

        Returns:
            None. Silently ignores unknown worker IDs.
        """
        with self._lock:
            self._workers.pop(worker_id, None)

    def mark_unhealthy(self, worker_id: str) -> None:
        """Mark a worker as unhealthy (excluded from routing)."""
        with self._lock:
            info = self._workers.get(worker_id)
            if info is not None:
                info.healthy = False

    def mark_healthy(self, worker_id: str) -> None:
        """Mark a worker as healthy (included in routing)."""
        with self._lock:
            info = self._workers.get(worker_id)
            if info is not None:
                info.healthy = True

    # -- Request routing ------------------------------------------------------

    def route_request(self, request: Any = None) -> Optional[str]:
        """Select a worker to handle a request.

        Args:
            request: The incoming request (used for future extensions).

        Returns:
            Worker ID of the selected worker, or ``None`` if no
            healthy workers are available.
        """
        with self._lock:
            healthy = [
                w for w in self._workers.values() if w.healthy
            ]

        if not healthy:
            return None

        if self._strategy == Strategy.ROUND_ROBIN:
            return self._route_round_robin(healthy)
        elif self._strategy == Strategy.LEAST_CONNECTIONS:
            return self._route_least_connections(healthy)
        elif self._strategy == Strategy.RANDOM:
            return self._route_random(healthy)
        else:
            return self._route_round_robin(healthy)

    def record_request_start(self, worker_id: str) -> None:
        """Record that a worker has started handling a request."""
        with self._lock:
            info = self._workers.get(worker_id)
            if info is not None:
                info.active_connections += 1
                info.total_requests += 1
                info.last_request_time = time.time()

    def record_request_end(self, worker_id: str) -> None:
        """Record that a worker has finished handling a request."""
        with self._lock:
            info = self._workers.get(worker_id)
            if info is not None:
                info.active_connections = max(0, info.active_connections - 1)

    # -- Stats ----------------------------------------------------------------

    def get_worker_stats(self) -> Dict[str, Dict[str, Any]]:
        """Return statistics for all registered workers.

        Returns:
            Dict mapping worker IDs to their stats.
        """
        with self._lock:
            return {
                wid: {
                    "address": w.address,
                    "active_connections": w.active_connections,
                    "total_requests": w.total_requests,
                    "last_request_time": w.last_request_time,
                    "healthy": w.healthy,
                    "uptime_seconds": round(time.time() - w.registered_at, 2),
                }
                for wid, w in self._workers.items()
            }

    def get_healthy_worker_count(self) -> int:
        """Return the number of currently healthy workers."""
        with self._lock:
            return sum(1 for w in self._workers.values() if w.healthy)

    # -- Private routing implementations --------------------------------------

    def _route_round_robin(self, healthy: List[WorkerInfo]) -> str:
        idx = self._rr_index % len(healthy)
        self._rr_index = idx + 1
        return healthy[idx].worker_id

    @staticmethod
    def _route_least_connections(healthy: List[WorkerInfo]) -> str:
        return min(healthy, key=lambda w: w.active_connections).worker_id

    @staticmethod
    def _route_random(healthy: List[WorkerInfo]) -> str:
        return random.choice(healthy).worker_id
