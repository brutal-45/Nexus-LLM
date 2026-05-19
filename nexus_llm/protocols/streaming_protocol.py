"""Nexus-LLM Streaming Protocol.

Defines the structures and protocol handler for streaming LLM
responses, supporting real-time token delivery.
"""

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional


class StreamEventType(Enum):
    """Types of streaming events."""

    TOKEN = "token"
    START = "start"
    END = "end"
    ERROR = "error"
    PROGRESS = "progress"
    METADATA = "metadata"


@dataclass
class StreamChunk:
    """A single chunk in a streaming response.

    Attributes:
        id: Stream identifier.
        event: Type of stream event.
        data: Chunk data (e.g., token text).
        index: Token index in the sequence.
        model: Model generating the stream.
        finish_reason: Reason for finishing (None if not finished).
        usage: Token usage (available on END events).
        timestamp: Chunk creation time.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event: StreamEventType = StreamEventType.TOKEN
    data: str = ""
    index: int = 0
    model: str = ""
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "event": self.event.value,
            "data": self.data,
            "index": self.index,
            "model": self.model,
            "finish_reason": self.finish_reason,
            "usage": self.usage,
            "timestamp": self.timestamp,
        }

    def to_sse(self) -> str:
        """Format as a Server-Sent Events (SSE) string.

        Returns:
            SSE-formatted string.
        """
        import json
        return f"data: {json.dumps(self.to_dict())}\n\n"


@dataclass
class StreamConfig:
    """Configuration for a streaming session.

    Attributes:
        stream_id: Unique stream identifier.
        buffer_size: Internal buffer size for token accumulation.
        flush_interval_ms: How often to flush the buffer (ms).
        include_usage: Whether to include usage in the final chunk.
    """

    stream_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    buffer_size: int = 1
    flush_interval_ms: float = 50.0
    include_usage: bool = True


class StreamingProtocol(ABC):
    """Abstract protocol handler for streaming LLM interactions.

    Concrete implementations must provide the `stream` method
    that yields StreamChunk objects as they become available.
    """

    @abstractmethod
    def stream(self, request: Any) -> Any:
        """Start a streaming response.

        Args:
            request: The request to stream (chat or completion).

        Returns:
            An iterator/generator of StreamChunk objects.
        """
        ...

    def create_start_chunk(self, model: str = "") -> StreamChunk:
        """Create a START event chunk.

        Args:
            model: Model name.

        Returns:
            A StreamChunk with START event.
        """
        return StreamChunk(
            event=StreamEventType.START,
            data="[STREAM_START]",
            model=model,
        )

    def create_token_chunk(self, token: str, index: int, model: str = "") -> StreamChunk:
        """Create a TOKEN event chunk.

        Args:
            token: The generated token text.
            index: Token index.
            model: Model name.

        Returns:
            A StreamChunk with TOKEN event.
        """
        return StreamChunk(
            event=StreamEventType.TOKEN,
            data=token,
            index=index,
            model=model,
        )

    def create_end_chunk(
        self,
        model: str = "",
        finish_reason: str = "stop",
        usage: Optional[Dict[str, int]] = None,
    ) -> StreamChunk:
        """Create an END event chunk.

        Args:
            model: Model name.
            finish_reason: Reason for finishing.
            usage: Token usage statistics.

        Returns:
            A StreamChunk with END event.
        """
        return StreamChunk(
            event=StreamEventType.END,
            data="[STREAM_END]",
            model=model,
            finish_reason=finish_reason,
            usage=usage,
        )

    def create_error_chunk(self, error: str, model: str = "") -> StreamChunk:
        """Create an ERROR event chunk.

        Args:
            error: Error message.
            model: Model name.

        Returns:
            A StreamChunk with ERROR event.
        """
        return StreamChunk(
            event=StreamEventType.ERROR,
            data=error,
            model=model,
            finish_reason="error",
        )
