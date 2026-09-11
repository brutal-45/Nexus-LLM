"""API layer for Nexus-LLM: client, schemas, auth, rate limiting, middleware.

Re-exports the pieces callers use directly (``from nexus_llm.api import
ChatMessage``) so the package exposes a complete public surface.
"""

from nexus_llm.api.auth import APIKey, AuthManager, KeyStore
from nexus_llm.api.client import NexusClient
from nexus_llm.api.cors import CORSConfig, setup_cors
from nexus_llm.api.middleware import setup_middleware
from nexus_llm.api.rate_limit import RateLimiter, TokenBucket
from nexus_llm.api.schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    FinishReason,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    ModelInfoResponse,
    StreamChunk,
)
from nexus_llm.api.websocket import (
    ConnectionManager,
    WebSocketMessageHandler,
    get_connection_manager,
)

__all__ = [
    "APIKey",
    "AuthManager",
    "CORSConfig",
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "ConnectionManager",
    "ErrorResponse",
    "FinishReason",
    "GenerateRequest",
    "GenerateResponse",
    "HealthResponse",
    "KeyStore",
    "ModelInfoResponse",
    "NexusClient",
    "RateLimiter",
    "StreamChunk",
    "TokenBucket",
    "WebSocketMessageHandler",
    "get_connection_manager",
    "setup_cors",
    "setup_middleware",
]
