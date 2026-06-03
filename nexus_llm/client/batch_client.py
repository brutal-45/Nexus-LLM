"""Nexus-LLM Batch Client.

Provides the BatchClient class for executing multiple API requests
concurrently, with support for rate limiting, retry logic, and
progress tracking.
"""

import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from nexus_llm.client.http_client import HttpClient, HttpClientConfig

logger = logging.getLogger(__name__)


class BatchStatus(Enum):
    """Status of a batch request."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BatchRequest:
    """A single request within a batch.

    Attributes:
        id: Unique request identifier.
        method: HTTP method.
        endpoint: API endpoint.
        params: Request parameters.
        metadata: Additional metadata.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    method: str = "POST"
    endpoint: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchResult:
    """Result of a single batch request.

    Attributes:
        request_id: ID of the corresponding request.
        success: Whether the request succeeded.
        response: Response data.
        error: Error message if failed.
        duration_ms: Request duration in milliseconds.
    """

    request_id: str = ""
    success: bool = True
    response: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0


@dataclass
class BatchConfig:
    """Configuration for batch execution.

    Attributes:
        max_workers: Maximum number of concurrent workers.
        rate_limit: Maximum requests per second.
        max_retries: Maximum retries per failed request.
        retry_delay: Base delay between retries in seconds.
        timeout: Timeout per request in seconds.
    """

    max_workers: int = 4
    rate_limit: float = 10.0
    max_retries: int = 2
    retry_delay: float = 1.0
    timeout: int = 60


class BatchClient:
    """Client for executing multiple API requests concurrently.

    Example::

        client = BatchClient(base_url="http://localhost:8000")
        client.add_request(BatchRequest(endpoint="/chat", params={"message": "Hello"}))
        client.add_request(BatchRequest(endpoint="/chat", params={"message": "World"}))
        results = client.execute()
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        config: Optional[BatchConfig] = None,
    ) -> None:
        self._config = config or BatchConfig()
        self._client = HttpClient(HttpClientConfig(
            base_url=base_url,
            api_key=api_key,
            timeout=self._config.timeout,
        ))
        self._requests: List[BatchRequest] = []
        self._status = BatchStatus.PENDING
        self._results: List[BatchResult] = []
        self._lock = threading.Lock()
        self._cancel_event = threading.Event()
        logger.debug("BatchClient initialized: max_workers=%d", self._config.max_workers)

    @property
    def status(self) -> BatchStatus:
        """Current batch execution status."""
        return self._status

    @property
    def request_count(self) -> int:
        """Number of queued requests."""
        return len(self._requests)

    @property
    def results(self) -> List[BatchResult]:
        """Results from the last batch execution."""
        return list(self._results)

    def add_request(self, request: BatchRequest) -> None:
        """Add a request to the batch.

        Args:
            request: The batch request to add.
        """
        with self._lock:
            self._requests.append(request)
            logger.debug("Added request: %s", request.id)

    def add_requests(self, requests: List[BatchRequest]) -> None:
        """Add multiple requests to the batch.

        Args:
            requests: List of batch requests.
        """
        for req in requests:
            self.add_request(req)

    def clear(self) -> None:
        """Clear all queued requests and results."""
        with self._lock:
            self._requests.clear()
            self._results.clear()
            self._status = BatchStatus.PENDING

    def execute(self) -> List[BatchResult]:
        """Execute all queued requests concurrently.

        Returns:
            List of BatchResult objects, one per request.
        """
        self._status = BatchStatus.RUNNING
        self._cancel_event.clear()
        self._results = []

        with ThreadPoolExecutor(max_workers=self._config.max_workers) as executor:
            futures = {}
            for req in self._requests:
                if self._cancel_event.is_set():
                    break
                future = executor.submit(self._execute_single, req)
                futures[future] = req.id

            for future in as_completed(futures):
                if self._cancel_event.is_set():
                    break
                try:
                    result = future.result()
                    self._results.append(result)
                except Exception as exc:
                    self._results.append(BatchResult(
                        request_id=futures[future],
                        success=False,
                        error=str(exc),
                    ))

        self._status = BatchStatus.CANCELLED if self._cancel_event.is_set() else BatchStatus.COMPLETED
        logger.info("Batch execution completed: %d results", len(self._results))
        return self._results

    def cancel(self) -> None:
        """Cancel the running batch execution."""
        self._cancel_event.set()
        self._status = BatchStatus.CANCELLED

    def _execute_single(self, request: BatchRequest) -> BatchResult:
        """Execute a single batch request with retries.

        Args:
            request: The batch request to execute.

        Returns:
            A BatchResult with the outcome.
        """
        start = time.perf_counter()
        last_error = None

        for attempt in range(self._config.max_retries + 1):
            if self._cancel_event.is_set():
                return BatchResult(
                    request_id=request.id,
                    success=False,
                    error="Cancelled",
                    duration_ms=(time.perf_counter() - start) * 1000,
                )

            try:
                # Use the HTTP client to make the request
                if request.method.upper() == "POST":
                    response = self._client.chat(
                        messages=request.params.get("messages", []),
                        model=request.params.get("model", ""),
                    )
                else:
                    response = self._client.list_models()

                return BatchResult(
                    request_id=request.id,
                    success=True,
                    response=response,
                    duration_ms=(time.perf_counter() - start) * 1000,
                )
            except Exception as exc:
                last_error = str(exc)
                if attempt < self._config.max_retries:
                    time.sleep(self._config.retry_delay * (attempt + 1))

        return BatchResult(
            request_id=request.id,
            success=False,
            error=last_error,
            duration_ms=(time.perf_counter() - start) * 1000,
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the last batch execution.

        Returns:
            Dictionary with success/failure counts and timing.
        """
        total = len(self._results)
        successes = sum(1 for r in self._results if r.success)
        failures = total - successes
        avg_duration = (
            sum(r.duration_ms for r in self._results) / total
            if total > 0 else 0.0
        )
        return {
            "total": total,
            "successes": successes,
            "failures": failures,
            "avg_duration_ms": avg_duration,
            "status": self._status.value,
        }
