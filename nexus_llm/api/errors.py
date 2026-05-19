"""Error handling: custom exceptions, error responses, error logging."""

import logging
import traceback
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import Request
from fastapi.responses import JSONResponse

from nexus_llm.api.schemas import ErrorResponse

logger = logging.getLogger("nexus_llm.api.errors")


class NexusAPIError(Exception):
    """Base exception for all Nexus-LLM API errors."""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_type: str = "InternalError",
        detail: Optional[str] = None,
        request_id: Optional[str] = None,
    ):
        self.message = message
        self.status_code = status_code
        self.error_type = error_type
        self.detail = detail
        self.request_id = request_id or str(uuid.uuid4())
        super().__init__(message)

    def to_response(self) -> ErrorResponse:
        return ErrorResponse(
            error=self.message,
            error_type=self.error_type,
            detail=self.detail,
            status_code=self.status_code,
            request_id=self.request_id,
            timestamp=datetime.utcnow().isoformat(),
        )


class ModelNotFoundError(NexusAPIError):
    """Raised when a requested model is not found or not loaded."""

    def __init__(self, model_name: str, request_id: Optional[str] = None):
        super().__init__(
            message=f"Model '{model_name}' not found",
            status_code=404,
            error_type="ModelNotFoundError",
            detail=f"The requested model '{model_name}' is not available. Check /v1/models for available models.",
            request_id=request_id,
        )


class ModelNotLoadedError(NexusAPIError):
    """Raised when a model exists but is not loaded into memory."""

    def __init__(self, model_name: str, request_id: Optional[str] = None):
        super().__init__(
            message=f"Model '{model_name}' is not loaded",
            status_code=503,
            error_type="ModelNotLoadedError",
            detail=f"The model '{model_name}' exists but is not currently loaded. Try again later or contact an administrator.",
            request_id=request_id,
        )


class ModelLoadError(NexusAPIError):
    """Raised when a model fails to load."""

    def __init__(self, model_name: str, reason: str = "", request_id: Optional[str] = None):
        super().__init__(
            message=f"Failed to load model '{model_name}'",
            status_code=500,
            error_type="ModelLoadError",
            detail=reason,
            request_id=request_id,
        )


class GenerationError(NexusAPIError):
    """Raised when text generation fails."""

    def __init__(self, reason: str = "", request_id: Optional[str] = None):
        super().__init__(
            message="Generation failed",
            status_code=500,
            error_type="GenerationError",
            detail=reason,
            request_id=request_id,
        )


class InvalidRequestError(NexusAPIError):
    """Raised when a request is malformed or invalid."""

    def __init__(self, detail: str, request_id: Optional[str] = None):
        super().__init__(
            message="Invalid request",
            status_code=400,
            error_type="InvalidRequestError",
            detail=detail,
            request_id=request_id,
        )


class ContentFilterError(NexusAPIError):
    """Raised when content is blocked by safety filters."""

    def __init__(self, reason: str = "Content blocked by safety filter", request_id: Optional[str] = None):
        super().__init__(
            message="Content filtered",
            status_code=422,
            error_type="ContentFilterError",
            detail=reason,
            request_id=request_id,
        )


class RateLimitExceededError(NexusAPIError):
    """Raised when rate limit is exceeded."""

    def __init__(
        self,
        limit: str = "",
        reset_at: Optional[str] = None,
        request_id: Optional[str] = None,
    ):
        detail = f"Rate limit exceeded: {limit}"
        if reset_at:
            detail += f". Resets at: {reset_at}"
        super().__init__(
            message="Rate limit exceeded",
            status_code=429,
            error_type="RateLimitExceededError",
            detail=detail,
            request_id=request_id,
        )


class AuthenticationError(NexusAPIError):
    """Raised when authentication fails."""

    def __init__(self, detail: str = "Invalid or missing API key", request_id: Optional[str] = None):
        super().__init__(
            message="Authentication failed",
            status_code=401,
            error_type="AuthenticationError",
            detail=detail,
            request_id=request_id,
        )


class InsufficientResourcesError(NexusAPIError):
    """Raised when server resources are insufficient."""

    def __init__(self, detail: str = "", request_id: Optional[str] = None):
        super().__init__(
            message="Insufficient resources",
            status_code=507,
            error_type="InsufficientResourcesError",
            detail=detail,
            request_id=request_id,
        )


class TimeoutError(NexusAPIError):
    """Raised when a request times out."""

    def __init__(self, timeout_seconds: float = 0.0, request_id: Optional[str] = None):
        super().__init__(
            message="Request timed out",
            status_code=504,
            error_type="TimeoutError",
            detail=f"Request exceeded timeout of {timeout_seconds}s",
            request_id=request_id,
        )


class TrainingError(NexusAPIError):
    """Raised when a training job fails."""

    def __init__(self, detail: str = "", request_id: Optional[str] = None):
        super().__init__(
            message="Training job failed",
            status_code=500,
            error_type="TrainingError",
            detail=detail,
            request_id=request_id,
        )


# Mapping of common exceptions to API errors
_EXCEPTION_MAP = {
    ValueError: InvalidRequestError,
    KeyError: InvalidRequestError,
    FileNotFoundError: ModelNotFoundError,
    RuntimeError: GenerationError,
}


async def nexus_error_handler(request: Request, exc: NexusAPIError) -> JSONResponse:
    """FastAPI exception handler for NexusAPIError and its subclasses.

    Args:
        request: The incoming request.
        exc: The raised NexusAPIError.

    Returns:
        JSONResponse with error details.
    """
    request_id = exc.request_id

    logger.error(
        "API Error [%s] %s: %s (detail: %s, request_id: %s)",
        exc.status_code,
        exc.error_type,
        exc.message,
        exc.detail,
        request_id,
    )

    error_response = exc.to_response()
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.model_dump(),
    )


async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all handler for unexpected exceptions.

    Args:
        request: The incoming request.
        exc: The unhandled exception.

    Returns:
        JSONResponse with a generic 500 error.
    """
    request_id = str(uuid.uuid4())
    error_type = type(exc).__name__

    logger.critical(
        "Unhandled exception [%s] %s: %s\n%s",
        request_id,
        error_type,
        str(exc),
        traceback.format_exc(),
    )

    mapped_class = _EXCEPTION_MAP.get(type(exc), NexusAPIError)
    if mapped_class != NexusAPIError:
        mapped_error = mapped_class(str(exc), request_id=request_id)
        return await nexus_error_handler(request, mapped_error)

    error_response = ErrorResponse(
        error="Internal server error",
        error_type="InternalServerError",
        detail="An unexpected error occurred. Please try again later.",
        status_code=500,
        request_id=request_id,
        timestamp=datetime.utcnow().isoformat(),
    )
    return JSONResponse(
        status_code=500,
        content=error_response.model_dump(),
    )


def register_error_handlers(app: Any) -> None:
    """Register all error handlers on a FastAPI application.

    Args:
        app: FastAPI application instance.
    """
    app.add_exception_handler(NexusAPIError, nexus_error_handler)
    app.add_exception_handler(Exception, unhandled_error_handler)

    for exc_class in _EXCEPTION_MAP:
        def make_handler(exc_cls):
            async def handler(request: Request, exc: Exception) -> JSONResponse:
                mapped = _EXCEPTION_MAP[exc_cls](str(exc))
                return await nexus_error_handler(request, mapped)
            return handler
        app.add_exception_handler(exc_class, make_handler(exc_class))
