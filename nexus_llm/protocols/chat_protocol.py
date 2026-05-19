"""Nexus-LLM Chat Protocol.

Defines the request/response structures and protocol handler for
chat-based LLM interactions, supporting multi-turn conversations.
"""

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from nexus_llm.enums import ChatRole


@dataclass
class ChatMessage:
    """A single message in a chat conversation.

    Attributes:
        role: Sender role (system, user, assistant).
        content: Message text content.
        name: Optional sender name.
        metadata: Additional message metadata.
    """

    role: str
    content: str
    name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"role": self.role, "content": self.content, "name": self.name, "metadata": self.metadata}


@dataclass
class ChatRequest:
    """Request for a chat completion.

    Attributes:
        messages: List of chat messages forming the conversation.
        model: Model identifier to use.
        temperature: Sampling temperature.
        top_p: Top-p sampling parameter.
        max_tokens: Maximum tokens to generate.
        stop: Stop sequences.
        stream: Whether to stream the response.
        request_id: Unique request identifier.
        metadata: Additional request metadata.
    """

    messages: List[ChatMessage] = field(default_factory=list)
    model: str = ""
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 2048
    stop: Optional[List[str]] = None
    stream: bool = False
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "messages": [m.to_dict() for m in self.messages],
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "stop": self.stop,
            "stream": self.stream,
            "request_id": self.request_id,
        }


@dataclass
class ChatResponse:
    """Response from a chat completion.

    Attributes:
        id: Response identifier.
        model: Model that generated the response.
        message: The assistant's response message.
        finish_reason: Reason for finishing (stop, length, content_filter).
        usage: Token usage statistics.
        created: Timestamp of creation.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model: str = ""
    message: Optional[ChatMessage] = None
    finish_reason: str = "stop"
    usage: Dict[str, int] = field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    created: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "model": self.model,
            "message": self.message.to_dict() if self.message else None,
            "finish_reason": self.finish_reason,
            "usage": self.usage,
            "created": self.created,
        }


class ChatProtocol(ABC):
    """Abstract protocol handler for chat interactions.

    Concrete implementations must provide the `chat` method that
    processes a ChatRequest and returns a ChatResponse.
    """

    @abstractmethod
    def chat(self, request: ChatRequest) -> ChatResponse:
        """Process a chat request and return a response.

        Args:
            request: The chat request containing messages and parameters.

        Returns:
            A ChatResponse with the model's reply.
        """
        ...

    def validate_request(self, request: ChatRequest) -> List[str]:
        """Validate a chat request.

        Args:
            request: The request to validate.

        Returns:
            List of validation errors (empty if valid).
        """
        errors: List[str] = []
        if not request.messages:
            errors.append("Messages list is empty")
        for msg in request.messages:
            if not msg.role:
                errors.append("Message missing role")
            if not msg.content and msg.role != "system":
                errors.append(f"Message with role '{msg.role}' missing content")
        if not request.model:
            errors.append("Model not specified")
        return errors
