"""Nexus-LLM Base Protocol.

Provides the abstract base class that all protocol handlers must implement,
along with common data structures for request/response handling.
"""

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ProtocolType(Enum):
    """Supported protocol types."""

    CHAT = "chat"
    COMPLETION = "completion"
    EMBEDDING = "embedding"
    TRAINING = "training"
    STREAMING = "streaming"


class RequestStatus(Enum):
    """Status of a protocol request."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ProtocolRequest:
    """Base request data structure for all protocols.

    Attributes:
        request_id: Unique identifier for this request.
        protocol_type: The type of protocol.
        model: Target model identifier.
        metadata: Additional request metadata.
        created_at: Timestamp of request creation.
    """

    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    protocol_type: ProtocolType = ProtocolType.CHAT
    model: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "request_id": self.request_id,
            "protocol_type": self.protocol_type.value,
            "model": self.model,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }


@dataclass
class ProtocolResponse:
    """Base response data structure for all protocols.

    Attributes:
        request_id: ID of the corresponding request.
        status: Response status.
        data: Response payload.
        error: Error message if failed.
        duration_ms: Processing duration in milliseconds.
        metadata: Additional response metadata.
    """

    request_id: str = ""
    status: RequestStatus = RequestStatus.COMPLETED
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        """Whether the response indicates success."""
        return self.status == RequestStatus.COMPLETED and self.error is None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "request_id": self.request_id,
            "status": self.status.value,
            "data": self.data,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }


class BaseProtocol(ABC):
    """Abstract base class for all protocol handlers.

    Every protocol handler must implement the `process` method
    and declare its protocol type.

    Example::

        class MyProtocol(BaseProtocol):
            @property
            def protocol_type(self):
                return ProtocolType.CHAT

            def process(self, request):
                return ProtocolResponse(request_id=request.request_id)

            def validate_request(self, request):
                errors = []
                if not request.model:
                    errors.append("Model is required")
                return errors
    """

    @property
    @abstractmethod
    def protocol_type(self) -> ProtocolType:
        """The type of protocol this handler implements."""
        ...

    @abstractmethod
    def process(self, request: ProtocolRequest) -> ProtocolResponse:
        """Process a protocol request and return a response.

        Args:
            request: The incoming protocol request.

        Returns:
            A ProtocolResponse with the processing result.
        """
        ...

    def validate_request(self, request: ProtocolRequest) -> List[str]:
        """Validate a protocol request.

        Args:
            request: The request to validate.

        Returns:
            List of validation errors (empty if valid).
        """
        errors: List[str] = []
        if not request.request_id:
            errors.append("Request ID is required")
        if not request.model:
            errors.append("Model is required")
        return errors

    def handle(self, request: ProtocolRequest) -> ProtocolResponse:
        """Handle a request with validation and timing.

        Args:
            request: The incoming protocol request.

        Returns:
            A ProtocolResponse with the processing result.
        """
        start = time.perf_counter()

        errors = self.validate_request(request)
        if errors:
            return ProtocolResponse(
                request_id=request.request_id,
                status=RequestStatus.FAILED,
                error="Validation errors: " + "; ".join(errors),
                duration_ms=(time.perf_counter() - start) * 1000,
            )

        try:
            response = self.process(request)
            response.duration_ms = (time.perf_counter() - start) * 1000
            return response
        except Exception as exc:
            return ProtocolResponse(
                request_id=request.request_id,
                status=RequestStatus.FAILED,
                error=str(exc),
                duration_ms=(time.perf_counter() - start) * 1000,
            )

    def schema(self) -> Dict[str, Any]:
        """Return the protocol schema description.

        Returns:
            Dictionary describing the protocol.
        """
        return {
            "protocol_type": self.protocol_type.value,
            "name": self.__class__.__name__,
        }
