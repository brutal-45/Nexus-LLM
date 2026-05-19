"""Nexus-LLM API Module.

Provides REST API, WebSocket, authentication, rate limiting,
CORS, middleware, and documentation for LLM serving.
"""

from nexus_llm.api.schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatRole,
    ConfigResponse,
    ConfigUpdateRequest,
    ErrorResponse,
    FinishReason,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    ModelInfoResponse,
    ModelsListResponse,
    StreamChunk,
    TrainingRequest,
    TrainingResponse,
)
from nexus_llm.api.errors import (
    AuthenticationError,
    ContentFilterError,
    GenerationError,
    InsufficientResourcesError,
    InvalidRequestError,
    ModelLoadError,
    ModelNotFoundError,
    ModelNotLoadedError,
    NexusAPIError,
    RateLimitExceededError,
    TimeoutError,
    TrainingError,
    register_error_handlers,
)
from nexus_llm.api.auth import (
    APIKey,
    AuthManager,
    KeyStore,
    get_auth_manager,
    init_auth,
)
from nexus_llm.api.rate_limit import (
    RateLimitConfig,
    RateLimiter,
    SlidingWindowCounter,
    TokenBucket,
    get_rate_limiter,
    init_rate_limiter,
)
from nexus_llm.api.middleware import (
    ErrorHandlingMiddleware,
    LoggingMiddleware,
    RequestIDMiddleware,
    TimingMiddleware,
    setup_middleware,
)
from nexus_llm.api.cors import (
    CORSConfig,
    CORSConfigBuilder,
    create_development_cors,
    create_production_cors,
    setup_cors,
)
from nexus_llm.api.websocket import (
    ConnectionManager,
    WebSocketMessageHandler,
    get_connection_manager,
    websocket_endpoint,
)
from nexus_llm.api.routes import (
    router,
    set_model_manager,
    set_safety_manager,
)
from nexus_llm.api.docs import (
    setup_docs,
    custom_openapi,
)


from typing import Any, Optional


def create_app(
    model_manager: Optional[Any] = None,
    safety_manager: Optional[Any] = None,
    require_auth: bool = False,
    cors_config: Optional[Any] = None,
) -> Any:
    """Create and configure a FastAPI application instance.

    Args:
        model_manager: Optional model manager for serving.
        safety_manager: Optional safety/content filter manager.
        require_auth: Whether to require API key authentication.
        cors_config: Optional CORS configuration.

    Returns:
        Configured FastAPI application.
    """
    from fastapi import FastAPI

    app = FastAPI(
        title="Nexus-LLM API",
        version="1.0.0",
        description="High-performance API for LLM inference and serving.",
    )

    if model_manager:
        set_model_manager(model_manager)
    if safety_manager:
        set_safety_manager(safety_manager)

    setup_middleware(app)
    setup_cors(app, cors_config)
    setup_docs(app)
    register_error_handlers(app)

    app.include_router(router)

    from nexus_llm.api.websocket import websocket_endpoint
    app.websocket("/ws")(websocket_endpoint)

    return app


__all__ = [
    # Schemas
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "ChatRole",
    "ConfigResponse",
    "ConfigUpdateRequest",
    "ErrorResponse",
    "FinishReason",
    "GenerateRequest",
    "GenerateResponse",
    "HealthResponse",
    "ModelInfoResponse",
    "ModelsListResponse",
    "StreamChunk",
    "TrainingRequest",
    "TrainingResponse",
    # Errors
    "AuthenticationError",
    "ContentFilterError",
    "GenerationError",
    "InsufficientResourcesError",
    "InvalidRequestError",
    "ModelLoadError",
    "ModelNotFoundError",
    "ModelNotLoadedError",
    "NexusAPIError",
    "RateLimitExceededError",
    "TimeoutError",
    "TrainingError",
    "register_error_handlers",
    # Auth
    "APIKey",
    "AuthManager",
    "KeyStore",
    "get_auth_manager",
    "init_auth",
    # Rate Limit
    "RateLimitConfig",
    "RateLimiter",
    "SlidingWindowCounter",
    "TokenBucket",
    "get_rate_limiter",
    "init_rate_limiter",
    # Middleware
    "ErrorHandlingMiddleware",
    "LoggingMiddleware",
    "RequestIDMiddleware",
    "TimingMiddleware",
    "setup_middleware",
    # CORS
    "CORSConfig",
    "CORSConfigBuilder",
    "create_development_cors",
    "create_production_cors",
    "setup_cors",
    # WebSocket
    "ConnectionManager",
    "WebSocketMessageHandler",
    "get_connection_manager",
    "websocket_endpoint",
    # Routes
    "router",
    "set_model_manager",
    "set_safety_manager",
    # Docs
    "setup_docs",
    "custom_openapi",
    # App factory
    "create_app",
]
