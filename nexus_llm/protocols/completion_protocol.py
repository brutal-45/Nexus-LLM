"""Nexus-LLM Completion Protocol.

Defines the request/response structures and protocol handler for
text completion interactions.
"""

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CompletionRequest:
    """Request for a text completion.

    Attributes:
        prompt: The input prompt text.
        model: Model identifier to use.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        top_p: Top-p sampling parameter.
        top_k: Top-k sampling parameter.
        stop: Stop sequences.
        n: Number of completions to generate.
        stream: Whether to stream the response.
        echo: Whether to echo the prompt in the response.
        request_id: Unique request identifier.
        metadata: Additional request metadata.
    """

    prompt: str = ""
    model: str = ""
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    stop: Optional[List[str]] = None
    n: int = 1
    stream: bool = False
    echo: bool = False
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "stop": self.stop,
            "n": self.n,
            "stream": self.stream,
            "echo": self.echo,
            "request_id": self.request_id,
        }


@dataclass
class CompletionChoice:
    """A single completion choice.

    Attributes:
        text: The completed text.
        index: Choice index.
        finish_reason: Reason for finishing.
        logprobs: Optional log probabilities.
    """

    text: str = ""
    index: int = 0
    finish_reason: str = "stop"
    logprobs: Optional[Dict[str, Any]] = None


@dataclass
class CompletionResponse:
    """Response from a text completion.

    Attributes:
        id: Response identifier.
        model: Model that generated the response.
        choices: List of completion choices.
        usage: Token usage statistics.
        created: Timestamp of creation.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model: str = ""
    choices: List[CompletionChoice] = field(default_factory=list)
    usage: Dict[str, int] = field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    created: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "model": self.model,
            "choices": [
                {"text": c.text, "index": c.index, "finish_reason": c.finish_reason}
                for c in self.choices
            ],
            "usage": self.usage,
            "created": self.created,
        }


class CompletionProtocol(ABC):
    """Abstract protocol handler for text completion interactions.

    Concrete implementations must provide the `complete` method.
    """

    @abstractmethod
    def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Process a completion request and return a response.

        Args:
            request: The completion request.

        Returns:
            A CompletionResponse with generated text.
        """
        ...

    def validate_request(self, request: CompletionRequest) -> List[str]:
        """Validate a completion request.

        Args:
            request: The request to validate.

        Returns:
            List of validation errors.
        """
        errors: List[str] = []
        if not request.prompt:
            errors.append("Prompt is empty")
        if not request.model:
            errors.append("Model not specified")
        if request.max_tokens <= 0:
            errors.append("max_tokens must be positive")
        if request.temperature < 0:
            errors.append("temperature must be non-negative")
        return errors
