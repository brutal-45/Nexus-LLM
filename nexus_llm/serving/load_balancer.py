"""Load Balancer for Nexus-LLM.

Provides request load balancing across multiple model serving workers.
Supports round-robin, least-connections, least-latency, and weighted
strategies with health-aware routing and automatic worker exclusion.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class BalancerStrategy(str, Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_LATENCY = "least_latency"
    WEIGHTED = "weighted"
    RANDOM = "random"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class WorkerEndpoint:
    """Represents a model serving worker endpoint."""
    id: str
    address: str = ""  # host:port or URL
    weight: int = 1
    healthy: bool = True
    active_connections: int = 0
    total_requests: int = 0
    total_errors: int = 0
    avg_latency_ms: float = 0.0
    last_request_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "address": self.address,
            "weight": self.weight,
            "healthy": self.healthy,
            "active_connections": self.active_connections,
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "last_request_at": self.last_request_at,
        }


@dataclass
class BalancedRequest:
    """A request to be routed by the load balancer."""
    request_id: str = ""
    payload: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass
class RoutedRequest:
    """A request that has been routed to a specific worker."""
    request: BalancedRequest
    worker: WorkerEndpoint
    routed_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request.request_id,
            "worker_id": self.worker.id,
            "worker_address": self.worker.address,
            "routed_at": self.routed_at,
        }


# ---------------------------------------------------------------------------
# Load Balancer
# ---------------------------------------------------------------------------

class LoadBalancer:
    """Request load balancer for distributing inference across workers.

    Distributes incoming requests across registered worker endpoints
    using configurable strategies.  Automatically excludes unhealthy
    workers and tracks per-worker metrics for intelligent routing.

    Example::

        lb = LoadBalancer(strategy=BalancerStrategy.LEAST_CONNECTIONS)
        lb.add_worker(WorkerEndpoint(id="w1", address="localhost:8001"))
        lb.add_worker(WorkerEndpoint(id="w2", address="localhost:8002"))

        routed = lb.route(BalancedRequest(payload={"prompt": "Hello"}))
        print(f"Routed to {routed.worker.id}")

        lb.mark_complete(routed.worker.id, latency_ms=45.2)
    """

    def __init__(
        self,
        strategy: BalancerStrategy = BalancerStrategy.ROUND_ROBIN,
        health_check_interval: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        self._strategy = strategy
        self._health_check_interval = health_check_interval
        self._max_retries = max_retries
        self._retry_delay = retry_delay

        self._workers: Dict[str, WorkerEndpoint] = {}
        self._worker_order: List[str] = []  # For round-robin
        self._rr_index = 0
        self._lock = threading.RLock()

        self._health_checker: Optional[Callable[[WorkerEndpoint], bool]] = None
        self._health_thread: Optional[threading.Thread] = None
        self._running = False

    @property
    def strategy(self) -> BalancerStrategy:
        """Current balancing strategy."""
        return self._strategy

    @property
    def worker_count(self) -> int:
        """Number of registered workers."""
        return len(self._workers)

    @property
    def healthy_worker_count(self) -> int:
        """Number of healthy workers."""
        return sum(1 for w in self._workers.values() if w.healthy)

    # ------------------------------------------------------------------
    # Worker management
    # ------------------------------------------------------------------

    def add_worker(self, worker: WorkerEndpoint) -> None:
        """Register a new worker endpoint.

        Args:
            worker: WorkerEndpoint to add.
        """
        with self._lock:
            self._workers[worker.id] = worker
            if worker.id not in self._worker_order:
                self._worker_order.append(worker.id)

    def remove_worker(self, worker_id: str) -> bool:
        """Remove a worker from the pool.

        Args:
            worker_id: ID of the worker to remove.

        Returns:
            True if the worker was found and removed.
        """
        with self._lock:
            if worker_id in self._workers:
                del self._workers[worker_id]
                self._worker_order = [wid for wid in self._worker_order if wid != worker_id]
                return True
            return False

    def set_worker_health(self, worker_id: str, healthy: bool) -> None:
        """Manually set the health status of a worker.

        Args:
            worker_id: Worker ID.
            healthy: New health status.
        """
        with self._lock:
            worker = self._workers.get(worker_id)
            if worker:
                worker.healthy = healthy

    def get_worker(self, worker_id: str) -> Optional[WorkerEndpoint]:
        """Get a worker by ID.

        Args:
            worker_id: Worker ID.

        Returns:
            WorkerEndpoint or None.
        """
        return self._workers.get(worker_id)

    def get_all_workers(self) -> List[WorkerEndpoint]:
        """Get all registered workers."""
        return list(self._workers.values())

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def route(self, request: BalancedRequest) -> RoutedRequest:
        """Route a request to an appropriate worker.

        Args:
            request: The request to route.

        Returns:
            RoutedRequest specifying the selected worker.

        Raises:
            RuntimeError: If no healthy workers are available.
        """
        worker = self._select_worker()
        if worker is None:
            raise RuntimeError("No healthy workers available for routing")

        with self._lock:
            worker.active_connections += 1
            worker.total_requests += 1
            worker.last_request_at = time.time()

        return RoutedRequest(request=request, worker=worker)

    def route_with_retry(self, request: BalancedRequest) -> RoutedRequest:
        """Route a request with automatic retry on failure.

        Args:
            request: The request to route.

        Returns:
            RoutedRequest specifying the selected worker.

        Raises:
            RuntimeError: If all retries are exhausted.
        """
        last_error: Optional[Exception] = None
        for attempt in range(self._max_retries):
            try:
                return self.route(request)
            except RuntimeError as e:
                last_error = e
                if attempt < self._max_retries - 1:
                    time.sleep(self._retry_delay)
        raise RuntimeError(f"Routing failed after {self._max_retries} retries: {last_error}")

    def mark_complete(self, worker_id: str, latency_ms: Optional[float] = None, error: bool = False) -> None:
        """Mark a request as completed on a worker.

        Args:
            worker_id: ID of the worker.
            latency_ms: Request latency in milliseconds.
            error: Whether the request resulted in an error.
        """
        with self._lock:
            worker = self._workers.get(worker_id)
            if worker:
                worker.active_connections = max(0, worker.active_connections - 1)
                if latency_ms is not None:
                    # Exponential moving average
                    alpha = 0.3
                    worker.avg_latency_ms = alpha * latency_ms + (1 - alpha) * worker.avg_latency_ms
                if error:
                    worker.total_errors += 1

    # ------------------------------------------------------------------
    # Strategy selection
    # ------------------------------------------------------------------

    def _select_worker(self) -> Optional[WorkerEndpoint]:
        """Select a worker based on the current strategy."""
        healthy = [w for w in self._workers.values() if w.healthy]
        if not healthy:
            return None

        if self._strategy == BalancerStrategy.ROUND_ROBIN:
            return self._select_round_robin(healthy)
        elif self._strategy == BalancerStrategy.LEAST_CONNECTIONS:
            return self._select_least_connections(healthy)
        elif self._strategy == BalancerStrategy.LEAST_LATENCY:
            return self._select_least_latency(healthy)
        elif self._strategy == BalancerStrategy.WEIGHTED:
            return self._select_weighted(healthy)
        elif self._strategy == BalancerStrategy.RANDOM:
            return self._select_random(healthy)
        else:
            return healthy[0]

    def _select_round_robin(self, workers: List[WorkerEndpoint]) -> WorkerEndpoint:
        """Round-robin worker selection."""
        with self._lock:
            idx = self._rr_index % len(workers)
            self._rr_index += 1
            return workers[idx]

    def _select_least_connections(self, workers: List[WorkerEndpoint]) -> WorkerEndpoint:
        """Select the worker with the fewest active connections."""
        return min(workers, key=lambda w: w.active_connections)

    def _select_least_latency(self, workers: List[WorkerEndpoint]) -> WorkerEndpoint:
        """Select the worker with the lowest average latency."""
        return min(workers, key=lambda w: w.avg_latency_ms if w.total_requests > 0 else float("inf"))

    def _select_weighted(self, workers: List[WorkerEndpoint]) -> WorkerEndpoint:
        """Select a worker using weighted random distribution."""
        import random
        total_weight = sum(w.weight for w in workers)
        if total_weight == 0:
            return random.choice(workers)
        r = random.uniform(0, total_weight)
        cumulative = 0.0
        for w in workers:
            cumulative += w.weight
            if r <= cumulative:
                return w
        return workers[-1]

    @staticmethod
    def _select_random(workers: List[WorkerEndpoint]) -> WorkerEndpoint:
        """Select a random worker."""
        import random
        return random.choice(workers)

    # ------------------------------------------------------------------
    # Health checking
    # ------------------------------------------------------------------

    def set_health_checker(self, checker: Callable[[WorkerEndpoint], bool]) -> None:
        """Set a custom health check function.

        Args:
            checker: Callable that takes a WorkerEndpoint and returns True if healthy.
        """
        self._health_checker = checker

    def start_health_checks(self) -> None:
        """Start the periodic health check thread."""
        if self._running:
            return
        self._running = True
        self._health_thread = threading.Thread(
            target=self._health_check_loop,
            daemon=True,
            name="lb-health-check",
        )
        self._health_thread.start()

    def stop_health_checks(self) -> None:
        """Stop the health check thread."""
        self._running = False
        if self._health_thread and self._health_thread.is_alive():
            self._health_thread.join(timeout=5.0)

    def _health_check_loop(self) -> None:
        """Periodically check worker health."""
        while self._running:
            try:
                time.sleep(self._health_check_interval)
                if self._health_checker:
                    for worker in list(self._workers.values()):
                        try:
                            healthy = self._health_checker(worker)
                            worker.healthy = healthy
                        except Exception:
                            worker.healthy = False
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics.

        Returns:
            Dictionary with strategy, worker count, and per-worker stats.
        """
        return {
            "strategy": self._strategy.value,
            "total_workers": self.worker_count,
            "healthy_workers": self.healthy_worker_count,
            "total_requests": sum(w.total_requests for w in self._workers.values()),
            "total_errors": sum(w.total_errors for w in self._workers.values()),
            "workers": {wid: w.to_dict() for wid, w in self._workers.items()},
        }
