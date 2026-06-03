"""Nexus-LLM Embedding Protocol.

Defines the request/response structures and protocol handler for
text embedding generation.
"""

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EmbeddingRequest:
    """Request for text embeddings.

    Attributes:
        input: Text or list of texts to embed.
        model: Model identifier to use.
        encoding_format: Output format ('float' or 'base64').
        dimensions: Desired embedding dimensions (0 = model default).
        request_id: Unique request identifier.
        metadata: Additional request metadata.
    """

    input: Any = None  # str or List[str]
    model: str = ""
    encoding_format: str = "float"
    dimensions: int = 0
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input": self.input,
            "model": self.model,
            "encoding_format": self.encoding_format,
            "dimensions": self.dimensions,
            "request_id": self.request_id,
        }


@dataclass
class EmbeddingData:
    """A single embedding result.

    Attributes:
        embedding: The embedding vector.
        index: Index of the input text.
        object_type: Object type identifier.
    """

    embedding: List[float] = field(default_factory=list)
    index: int = 0
    object_type: str = "embedding"


@dataclass
class EmbeddingResponse:
    """Response from an embedding request.

    Attributes:
        id: Response identifier.
        model: Model that generated the embeddings.
        data: List of embedding results.
        usage: Token usage statistics.
        created: Timestamp of creation.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model: str = ""
    data: List[EmbeddingData] = field(default_factory=list)
    usage: Dict[str, int] = field(default_factory=lambda: {"prompt_tokens": 0, "total_tokens": 0})
    created: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "model": self.model,
            "data": [
                {"embedding": d.embedding, "index": d.index, "object": d.object_type}
                for d in self.data
            ],
            "usage": self.usage,
            "created": self.created,
        }


class EmbeddingProtocol(ABC):
    """Abstract protocol handler for embedding generation.

    Concrete implementations must provide the `embed` method.
    """

    @abstractmethod
    def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generate embeddings for the given input.

        Args:
            request: The embedding request.

        Returns:
            An EmbeddingResponse with the generated vectors.
        """
        ...

    def validate_request(self, request: EmbeddingRequest) -> List[str]:
        """Validate an embedding request.

        Args:
            request: The request to validate.

        Returns:
            List of validation errors.
        """
        errors: List[str] = []
        if request.input is None:
            errors.append("Input is required")
        if not request.model:
            errors.append("Model not specified")
        if request.encoding_format not in ("float", "base64"):
            errors.append(f"Unsupported encoding format: {request.encoding_format}")
        return errors
