"""Nexus-LLM Protocols Module.

Defines the communication protocols for chat, completion, embedding,
training, and streaming interactions with LLM backends.
"""

from nexus_llm.protocols.chat_protocol import ChatProtocol, ChatRequest, ChatResponse
from nexus_llm.protocols.completion_protocol import CompletionProtocol, CompletionRequest, CompletionResponse
from nexus_llm.protocols.embedding_protocol import EmbeddingProtocol, EmbeddingRequest, EmbeddingResponse
from nexus_llm.protocols.training_protocol import TrainingProtocol, TrainingRequest, TrainingResponse
from nexus_llm.protocols.streaming_protocol import StreamingProtocol, StreamChunk

__all__ = [
    "ChatProtocol",
    "ChatRequest",
    "ChatResponse",
    "CompletionProtocol",
    "CompletionRequest",
    "CompletionResponse",
    "EmbeddingProtocol",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "TrainingProtocol",
    "TrainingRequest",
    "TrainingResponse",
    "StreamingProtocol",
    "StreamChunk",
]
