"""Nexus-LLM Request Dispatcher.

Provides the Dispatcher class that routes incoming requests to the
appropriate handlers based on request type, priority, and available
capacity.
"""

import enum
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from nexus_llm.exceptions import NexusLLMError

logger = logging.getLogger(__name__)


class DispatchStatus(enum.Enum):
    """Status of a dispatched request."""

    PENDING = "pending"
    ROUTED = "routed"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    REJECTED = "rejected"


@dataclass
class DispatchResult:
    """Result of dispatching a request.

    Attributes:
        request_id: Unique identifier for the dispatched request.
        status: Dispatch status.
        handler: Name of the handler that was selected.
        output: Output from the handler (if completed).
        error: Error message (if failed).
        duration_ms: Total dispatch and execution time in milliseconds.
        metadata: Additional dispatch metadata.
    """

    request_id: str
    status: DispatchStatus = DispatchStatus.PENDING
    handler: Optional[str] = None
    output: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Request:
    """Incoming request to be dispatched.

    Attributes:
        type: Request type string (e.g., 'chat', 'completion', 'embedding').
        payload: Request payload data.
        priority: Priority level (higher = more urgent).
        timeout: Optional timeout in seconds.
        metadata: Additional request metadata.
    """

    type: str
    payload: Dict[str, Any]
    priority: int = 0
    timeout: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def request_id(self) -> str:
        """Generate a unique request ID."""
        return str(uuid.uuid4())


class Dispatcher:
    """Request dispatcher that routes requests to registered handlers.

    The Dispatcher maintains a registry of handlers keyed by request type,
    supports priority-based ordering, and tracks dispatch statistics.

    Attributes:
        handler_count: Number of registered handlers.
    """

    def __init__(self) -> None:
        self._handlers: Dict[str, List[Callable]] = {}
        self._middlewares: List[Callable] = []
        self._stats: Dict[str, int] = {
            "total_dispatched": 0,
            "total_completed": 0,
            "total_failed": 0,
            "total_rejected": 0,
        }

    @property
    def handler_count(self) -> int:
        """Number of registered handler types."""
        return len(self._handlers)

    def register_handler(self, request_type: str, handler: Callable) -> None:
        """Register a handler for a request type.

        Multiple handlers can be registered for the same type; they are
        tried in registration order until one succeeds.

        Args:
            request_type: The type of request this handler handles.
            handler: Callable that accepts a Request and returns a result.
        """
        self._handlers.setdefault(request_type, []).append(handler)
        logger.debug("Registered handler for type '%s': %s", request_type, handler.__name__)

    def unregister_handler(self, request_type: str, handler: Optional[Callable] = None) -> None:
        """Unregister a handler for a request type.

        Args:
            request_type: The request type.
            handler: Specific handler to remove, or None to remove all.
        """
        if handler is None:
            self._handlers.pop(request_type, None)
        elif request_type in self._handlers:
            self._handlers[request_type] = [
                h for h in self._handlers[request_type] if h is not handler
            ]

    def add_middleware(self, middleware: Callable) -> None:
        """Add a middleware function that processes requests before dispatch.

        Middleware functions receive the Request and must return a (possibly
        modified) Request, or raise an exception to reject the request.

        Args:
            middleware: Callable accepting a Request, returning a Request.
        """
        self._middlewares.append(middleware)

    def dispatch(self, request: Request) -> DispatchResult:
        """Dispatch a request to the appropriate handler.

        Args:
            request: The request to dispatch.

        Returns:
            A DispatchResult with execution details.
        """
        start = time.perf_counter()
        request_id = request.request_id
        result = DispatchResult(request_id=request_id)
        self._stats["total_dispatched"] += 1

        # Apply middleware
        processed_request = request
        for mw in self._middlewares:
            try:
                processed_request = mw(processed_request)
            except Exception as exc:
                result.status = DispatchStatus.REJECTED
                result.error = str(exc)
                result.duration_ms = (time.perf_counter() - start) * 1000
                self._stats["total_rejected"] += 1
                return result

        # Find handler
        handlers = self._handlers.get(processed_request.type)
        if not handlers:
            result.status = DispatchStatus.REJECTED
            result.error = f"No handler registered for type '{processed_request.type}'"
            result.duration_ms = (time.perf_counter() - start) * 1000
            self._stats["total_rejected"] += 1
            return result

        # Execute handler chain
        last_error: Optional[str] = None
        for handler in handlers:
            try:
                result.output = handler(processed_request)
                result.handler = handler.__name__
                result.status = DispatchStatus.COMPLETED
                self._stats["total_completed"] += 1
                break
            except Exception as exc:
                last_error = str(exc)
                logger.debug("Handler %s failed: %s", handler.__name__, exc)
        else:
            result.status = DispatchStatus.FAILED
            result.error = last_error
            self._stats["total_failed"] += 1

        result.duration_ms = (time.perf_counter() - start) * 1000
        return result

    def dispatch_batch(self, requests: List[Request]) -> List[DispatchResult]:
        """Dispatch multiple requests, sorted by priority (descending).

        Args:
            requests: List of requests to dispatch.

        Returns:
            List of DispatchResults in dispatch order.
        """
        sorted_requests = sorted(requests, key=lambda r: r.priority, reverse=True)
        return [self.dispatch(req) for req in sorted_requests]

    def stats(self) -> Dict[str, int]:
        """Return dispatch statistics.

        Returns:
            Dictionary with dispatch counts.
        """
        return dict(self._stats)

    def reset_stats(self) -> None:
        """Reset dispatch statistics."""
        for key in self._stats:
            self._stats[key] = 0

    def health_check(self) -> Dict[str, Any]:
        """Return health status of the dispatcher.

        Returns:
            Dictionary with handler info and statistics.
        """
        return {
            "status": "healthy",
            "registered_types": list(self._handlers.keys()),
            "handler_count": self.handler_count,
            "middleware_count": len(self._middlewares),
            "stats": self.stats(),
        }
