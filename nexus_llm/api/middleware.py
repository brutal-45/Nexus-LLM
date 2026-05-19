"""Middleware: logging, timing, error handling, request ID."""

import logging
import time
import uuid
from typing import Any, Callable, Optional

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse

logger = logging.getLogger("nexus_llm.api.middleware")

_REQUEST_ID_HEADER = "X-Request-ID"
_PROCESS_TIME_HEADER = "X-Process-Time-Ms"


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Assigns a unique request ID to every incoming request.

    If the client provides an X-Request-ID header, it is preserved.
    Otherwise, a new UUID is generated. The ID is stored in request
    state and added to the response headers.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        request_id = request.headers.get(
            _REQUEST_ID_HEADER, str(uuid.uuid4())
        )
        request.state.request_id = request_id

        response = await call_next(request)
        response.headers[_REQUEST_ID_HEADER] = request_id
        return response


class TimingMiddleware(BaseHTTPMiddleware):
    """Measures and logs request processing time.

    Adds X-Process-Time-Ms header to the response and logs
    timing information for monitoring and debugging.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        start_time = time.perf_counter()

        response = await call_next(request)

        process_time_ms = (time.perf_counter() - start_time) * 1000.0
        response.headers[_PROCESS_TIME_HEADER] = f"{process_time_ms:.2f}"

        request_id = getattr(request.state, "request_id", "unknown")
        logger.info(
            "Request %s %s completed in %.2fms (request_id: %s, status: %s)",
            request.method,
            request.url.path,
            process_time_ms,
            request_id,
            response.status_code,
        )

        return response


class LoggingMiddleware(BaseHTTPMiddleware):
    """Logs request and response details for debugging and auditing.

    Logs method, path, query params, status code, and processing time.
    Supports configurable log levels and path filtering.
    """

    def __init__(
        self,
        app: Any,
        log_level: str = "INFO",
        exclude_paths: Optional[list] = None,
        log_body: bool = False,
        max_body_length: int = 1000,
    ):
        super().__init__(app)
        self.log_level = getattr(logging, log_level.upper(), logging.INFO)
        self.exclude_paths = exclude_paths or ["/v1/health", "/docs", "/openapi.json", "/redoc"]
        self.log_body = log_body
        self.max_body_length = max_body_length

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        if request.url.path in self.exclude_paths:
            return await call_next(request)

        request_id = getattr(request.state, "request_id", "unknown")

        log_data = {
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "query": str(request.query_params),
            "client_ip": request.client.host if request.client else "unknown",
            "user_agent": request.headers.get("user-agent", "unknown"),
        }

        if self.log_body and request.method in ("POST", "PUT", "PATCH"):
            try:
                body = await request.body()
                body_str = body.decode("utf-8", errors="replace")
                if len(body_str) > self.max_body_length:
                    body_str = body_str[:self.max_body_length] + "...[truncated]"
                log_data["body"] = body_str
            except Exception:
                log_data["body"] = "[failed to read body]"

        logger.log(self.log_level, "Request: %s", log_data)

        start_time = time.perf_counter()
        response = await call_next(request)
        process_time_ms = (time.perf_counter() - start_time) * 1000.0

        response_log = {
            "request_id": request_id,
            "status_code": response.status_code,
            "process_time_ms": round(process_time_ms, 2),
        }
        logger.log(self.log_level, "Response: %s", response_log)

        return response


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Catches unhandled exceptions and returns structured error responses.

    Prevents stack traces from leaking to clients and ensures
    all errors are properly logged with request context.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        try:
            response = await call_next(request)
            return response
        except Exception as exc:
            request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

            logger.exception(
                "Unhandled exception for request %s %s (request_id: %s): %s",
                request.method,
                request.url.path,
                request_id,
                str(exc),
            )

            from nexus_llm.api.schemas import ErrorResponse
            from datetime import datetime
            import json

            error_response = ErrorResponse(
                error="Internal server error",
                error_type=type(exc).__name__,
                detail="An unexpected error occurred. Please try again later.",
                status_code=500,
                request_id=request_id,
                timestamp=datetime.utcnow().isoformat(),
            )

            return Response(
                content=json.dumps(error_response.model_dump()),
                status_code=500,
                media_type="application/json",
                headers={_REQUEST_ID_HEADER: request_id},
            )


class CORSPreflightMiddleware(BaseHTTPMiddleware):
    """Handles CORS preflight OPTIONS requests.

    Intercepts OPTIONS requests and returns appropriate CORS headers
    before they reach the router.
    """

    def __init__(
        self,
        app: Any,
        allow_origins: Optional[list] = None,
        allow_methods: Optional[list] = None,
        allow_headers: Optional[list] = None,
    ):
        super().__init__(app)
        self.allow_origins = allow_origins or ["*"]
        self.allow_methods = allow_methods or ["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"]
        self.allow_headers = allow_headers or [
            "Content-Type", "Authorization", "X-API-Key", "X-Request-ID"
        ]

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        if request.method == "OPTIONS":
            origin = request.headers.get("origin", "*")
            response = Response(status_code=204)
            response.headers["Access-Control-Allow-Origin"] = (
                origin if origin in self.allow_origins or "*" in self.allow_origins else ""
            )
            response.headers["Access-Control-Allow-Methods"] = ", ".join(self.allow_methods)
            response.headers["Access-Control-Allow-Headers"] = ", ".join(self.allow_headers)
            response.headers["Access-Control-Max-Age"] = "86400"
            return response

        return await call_next(request)


def setup_middleware(app: Any) -> None:
    """Register all middleware on a FastAPI application.

    Args:
        app: FastAPI application instance.
    """
    app.add_middleware(ErrorHandlingMiddleware)
    app.add_middleware(LoggingMiddleware, log_level="INFO")
    app.add_middleware(TimingMiddleware)
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(CORSPreflightMiddleware)
