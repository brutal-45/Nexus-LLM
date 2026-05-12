"""
Server Core - Inference Server
================================

Async HTTP server for LLM model inference with request handling,
model management, response formatting, and error handling.

All implementations use Python stdlib only (asyncio, json, http.server).
"""

import asyncio
import json
import os
import time
import hashlib
import threading
import traceback
import uuid
import socket
import struct
from collections import deque, defaultdict, OrderedDict
from dataclasses import dataclass, field
from typing import (
    Any, Dict, List, Optional, Tuple, Union, Callable, Awaitable, Set
)
from enum import Enum
import queue


# ============================================================================
# Constants and Error Codes
# ============================================================================

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
DEFAULT_MAX_BATCH_SIZE = 32
DEFAULT_MAX_QUEUE_SIZE = 1024
DEFAULT_TIMEOUT = 30.0
DEFAULT_WORKERS = 1

HTTP_STATUS_OK = 200
HTTP_STATUS_CREATED = 201
HTTP_STATUS_BAD_REQUEST = 400
HTTP_STATUS_UNAUTHORIZED = 401
HTTP_STATUS_FORBIDDEN = 403
HTTP_STATUS_NOT_FOUND = 404
HTTP_STATUS_METHOD_NOT_ALLOWED = 405
HTTP_STATUS_REQUEST_TIMEOUT = 408
HTTP_STATUS_TOO_MANY_REQUESTS = 429
HTTP_STATUS_INTERNAL_ERROR = 500
HTTP_STATUS_NOT_IMPLEMENTED = 501
HTTP_STATUS_BAD_GATEWAY = 502
HTTP_STATUS_SERVICE_UNAVAILABLE = 503
HTTP_STATUS_GATEWAY_TIMEOUT = 504

HTTP_STATUS_TEXT = {
    200: "OK",
    201: "Created",
    400: "Bad Request",
    401: "Unauthorized",
    403: "Forbidden",
    404: "Not Found",
    405: "Method Not Allowed",
    408: "Request Timeout",
    429: "Too Many Requests",
    500: "Internal Server Error",
    501: "Not Implemented",
    502: "Bad Gateway",
    503: "Service Unavailable",
    504: "Gateway Timeout",
}

CONTENT_TYPE_JSON = "application/json"
CONTENT_TYPE_SSE = "text/event-stream"
CONTENT_TYPE_TEXT = "text/plain"
CONTENT_TYPE_HTML = "text/html"


class ServerState(Enum):
    """Server lifecycle states."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    DRAINING = "draining"
    STOPPING = "stopping"
    ERROR = "error"


class ErrorCode(Enum):
    """Standard error codes for the inference server."""
    UNKNOWN = "unknown_error"
    INVALID_REQUEST = "invalid_request"
    MODEL_NOT_LOADED = "model_not_loaded"
    MODEL_LOAD_ERROR = "model_load_error"
    INFERENCE_ERROR = "inference_error"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    QUEUE_FULL = "queue_full"
    INVALID_MODEL = "invalid_model"
    CONTEXT_TOO_LONG = "context_too_long"
    INTERNAL_ERROR = "internal_error"
    NOT_IMPLEMENTED = "not_implemented"
    SERVICE_UNAVAILABLE = "service_unavailable"


# ============================================================================
# ServerConfig
# ============================================================================

@dataclass
class ServerConfig:
    """Configuration for the inference server."""
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    max_batch_size: int = DEFAULT_MAX_BATCH_SIZE
    max_queue_size: int = DEFAULT_MAX_QUEUE_SIZE
    timeout: float = DEFAULT_TIMEOUT
    num_workers: int = DEFAULT_WORKERS
    max_concurrent_requests: int = 128
    request_timeout: float = 60.0
    model_path: Optional[str] = None
    model_name: Optional[str] = None
    dtype: str = "float32"
    device: str = "cpu"
    max_sequence_length: int = 2048
    max_new_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    enable_streaming: bool = True
    enable_caching: bool = True
    enable_batching: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    api_key: Optional[str] = None
    log_level: str = "info"
    health_check_interval: float = 30.0
    graceful_shutdown_timeout: float = 30.0
    chunk_size: int = 16
    backpressure_threshold: float = 0.8
    max_request_size: int = 1024 * 1024  # 1MB


# ============================================================================
# Data Classes for Requests/Responses
# ============================================================================

@dataclass
class InferenceRequest:
    """Parsed inference request."""
    id: str = ""
    prompt: str = ""
    messages: Optional[List[Dict[str, str]]] = None
    model: Optional[str] = None
    max_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[List[str]] = None
    stream: bool = False
    n: int = 1
    logprobs: bool = False
    echo: bool = False
    user: Optional[str] = None
    request_timestamp: float = 0.0
    priority: int = 0

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if self.request_timestamp == 0.0:
            self.request_timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "messages": self.messages,
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "stop": self.stop,
            "stream": self.stream,
            "n": self.n,
            "logprobs": self.logprobs,
            "echo": self.echo,
            "user": self.user,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InferenceRequest":
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


@dataclass
class InferenceResponse:
    """Inference response."""
    id: str = ""
    object: str = "text_completion"
    created: float = 0.0
    model: str = ""
    choices: List[Dict[str, Any]] = field(default_factory=list)
    usage: Dict[str, int] = field(default_factory=dict)
    finish_reason: str = "stop"
    error: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.created == 0.0:
            self.created = time.time()
        if not self.id:
            self.id = str(uuid.uuid4())

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "id": self.id,
            "object": self.object,
            "created": int(self.created),
            "model": self.model,
            "choices": self.choices,
            "usage": self.usage,
        }
        if self.error:
            result["error"] = self.error
        return result

    @classmethod
    def from_generation(cls, text: str, request: InferenceRequest, model: str = "") -> "InferenceResponse":
        """Create a response from generated text."""
        return cls(
            id=request.id,
            model=model,
            choices=[{
                "index": 0,
                "text": text,
                "finish_reason": "stop",
            }],
            usage={
                "prompt_tokens": len(request.prompt.split()) if request.prompt else 0,
                "completion_tokens": len(text.split()),
                "total_tokens": len(request.prompt.split()) + len(text.split()) if request.prompt else len(text.split()),
            },
        )


@dataclass
class StreamChunk:
    """A single chunk in a streaming response."""
    id: str = ""
    object: str = "text_completion"
    created: float = 0.0
    model: str = ""
    choices: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        if self.created == 0.0:
            self.created = time.time()
        if not self.id:
            self.id = str(uuid.uuid4())

    def to_sse(self) -> str:
        """Format as Server-Sent Event."""
        data = json.dumps({
            "id": self.id,
            "object": self.object,
            "created": int(self.created),
            "model": self.model,
            "choices": self.choices,
        })
        return f"data: {data}\n\n"

    def to_done_sse(self) -> str:
        """Format as SSE done event."""
        return "data: [DONE]\n\n"


# ============================================================================
# ResponseFormatter
# ============================================================================

class ResponseFormatter:
    """
    Format inference responses as JSON, SSE stream, or plain text.

    Provides consistent response formatting with proper HTTP headers
    and content type handling.
    """

    def __init__(
        self,
        default_model_name: str = "nexus-llm",
        include_usage: bool = True,
        pretty_print: bool = False,
    ):
        """Initialize the response formatter.

        Args:
            default_model_name: Default model name for responses.
            include_usage: Whether to include token usage info.
            pretty_print: Whether to pretty-print JSON.
        """
        self._default_model = default_model_name
        self._include_usage = include_usage
        self._pretty_print = pretty_print
        self._indent = 2 if pretty_print else None

    def format_completion(
        self,
        text: str,
        request: InferenceRequest,
        model: Optional[str] = None,
        finish_reason: str = "stop",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> InferenceResponse:
        """Format a single completion response.

        Args:
            text: Generated text.
            request: Original request.
            model: Model name.
            finish_reason: Reason for completion.
            prompt_tokens: Number of prompt tokens.
            completion_tokens: Number of completion tokens.

        Returns:
            InferenceResponse object.
        """
        model_name = model or self._default_model
        usage = {}
        if self._include_usage:
            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }

        return InferenceResponse(
            id=request.id,
            model=model_name,
            choices=[{
                "index": 0,
                "text": text,
                "finish_reason": finish_reason,
            }],
            usage=usage,
        )

    def format_chat_completion(
        self,
        text: str,
        request: InferenceRequest,
        model: Optional[str] = None,
        finish_reason: str = "stop",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> InferenceResponse:
        """Format a chat completion response.

        Args:
            text: Generated text.
            request: Original request.
            model: Model name.
            finish_reason: Reason for completion.
            prompt_tokens: Number of prompt tokens.
            completion_tokens: Number of completion tokens.

        Returns:
            InferenceResponse object.
        """
        model_name = model or self._default_model
        response = self.format_completion(
            text, request, model_name, finish_reason,
            prompt_tokens, completion_tokens,
        )
        response.object = "chat.completion"
        response.choices = [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": finish_reason,
        }]
        return response

    def format_stream_chunk(
        self,
        text: str,
        request_id: str,
        model: Optional[str] = None,
        finish_reason: Optional[str] = None,
        chunk_index: int = 0,
    ) -> StreamChunk:
        """Format a streaming response chunk.

        Args:
            text: Generated text chunk.
            request_id: Request ID.
            model: Model name.
            finish_reason: Finish reason or None.
            chunk_index: Chunk index.

        Returns:
            StreamChunk object.
        """
        model_name = model or self._default_model
        choice: Dict[str, Any] = {
            "index": chunk_index,
            "text": text,
        }
        if finish_reason:
            choice["finish_reason"] = finish_reason

        return StreamChunk(
            id=request_id,
            model=model_name,
            choices=[choice],
        )

    def format_error(
        self,
        error_code: ErrorCode,
        message: str,
        status_code: int = 500,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Format an error response.

        Args:
            error_code: Error code enum.
            message: Error message.
            status_code: HTTP status code.
            request_id: Optional request ID.

        Returns:
            Error dictionary.
        """
        return {
            "error": {
                "code": error_code.value,
                "message": message,
                "type": error_code.name,
                "status": status_code,
            },
            "id": request_id or str(uuid.uuid4()),
            "object": "error",
        }

    def format_health_response(
        self,
        status: str,
        uptime: float,
        model_loaded: bool,
        requests_processed: int = 0,
        queue_size: int = 0,
        gpu_memory: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Format a health check response.

        Args:
            status: Health status.
            uptime: Server uptime in seconds.
            model_loaded: Whether model is loaded.
            requests_processed: Total requests processed.
            queue_size: Current queue size.
            gpu_memory: GPU memory usage in GB.

        Returns:
            Health dictionary.
        """
        response = {
            "status": status,
            "uptime": uptime,
            "model_loaded": model_loaded,
            "requests_processed": requests_processed,
            "queue_size": queue_size,
        }
        if gpu_memory is not None:
            response["gpu_memory_gb"] = gpu_memory
        return response

    def to_json_bytes(self, obj: Any) -> bytes:
        """Convert object to JSON bytes.

        Args:
            obj: Object to serialize.

        Returns:
            JSON-encoded bytes.
        """
        if isinstance(obj, InferenceResponse):
            obj = obj.to_dict()
        return json.dumps(obj, ensure_ascii=False, indent=self._indent).encode("utf-8")


# ============================================================================
# ServerErrorHandler
# ============================================================================

class ServerErrorHandler:
    """
    Handle server errors with proper codes, rate limiting, and degradation.

    Provides centralized error handling, rate limit enforcement,
    and graceful degradation when the server is under load.
    """

    def __init__(
        self,
        max_error_rate: float = 0.1,
        error_window: int = 100,
        circuit_breaker_threshold: int = 10,
        circuit_breaker_reset: float = 60.0,
    ):
        """Initialize the error handler.

        Args:
            max_error_rate: Maximum error rate before triggering alerts.
            error_window: Window for error rate calculation.
            circuit_breaker_threshold: Errors before circuit opens.
            circuit_breaker_reset: Seconds before circuit resets.
        """
        self._max_error_rate = max_error_rate
        self._error_window = error_window
        self._circuit_threshold = circuit_breaker_threshold
        self._circuit_reset = circuit_breaker_reset

        self._error_counts: deque = deque(maxlen=10000)
        self._error_types: Dict[str, int] = defaultdict(int)
        self._circuit_open: bool = False
        self._circuit_open_time: float = 0.0
        self._total_errors: int = 0
        self._lock = threading.Lock()

        self._on_error_callbacks: List[Callable[[Dict[str, Any]], None]] = []

    def handle_error(
        self,
        error: Exception,
        request_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Handle an error and return an appropriate response.

        Args:
            error: The exception that occurred.
            request_id: Optional request ID.
            context: Additional context about the error.

        Returns:
            Error response dictionary.
        """
        with self._lock:
            now = time.time()
            self._error_counts.append(now)
            self._total_errors += 1

            error_type = type(error).__name__
            self._error_types[error_type] += 1

            # Classify error
            error_code, status_code, message = self._classify_error(error)

            # Check circuit breaker
            if self._should_open_circuit(now):
                self._circuit_open = True
                self._circuit_open_time = now

        # Build response
        response = {
            "error": {
                "code": error_code.value,
                "message": message,
                "type": error_type,
                "status": status_code,
            },
            "id": request_id or str(uuid.uuid4()),
        }

        if context:
            response["context"] = context

        # Fire callbacks
        for callback in self._on_error_callbacks:
            try:
                callback({
                    "error": error,
                    "error_code": error_code,
                    "status_code": status_code,
                    "request_id": request_id,
                    "context": context,
                    "timestamp": now,
                })
            except Exception:
                pass

        return response

    def check_rate_limit(self) -> bool:
        """Check if the error rate exceeds the threshold.

        Returns:
            True if error rate is too high.
        """
        with self._lock:
            if not self._error_counts:
                return False

            now = time.time()
            cutoff = now - 60.0
            recent_errors = sum(1 for t in self._error_counts if t > cutoff)

            total_requests = recent_errors / max(self._max_error_rate, 0.001)
            error_rate = recent_errors / total_requests if total_requests > 0 else 0

            return error_rate > self._max_error_rate

    def is_circuit_open(self) -> bool:
        """Check if the circuit breaker is open.

        Returns:
            True if circuit breaker is open.
        """
        with self._lock:
            if not self._circuit_open:
                return False

            # Check if circuit should reset
            if time.time() - self._circuit_open_time > self._circuit_reset:
                self._circuit_open = False
                self._error_counts.clear()
                return False

            return True

    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics.

        Returns:
            Dictionary with error statistics.
        """
        with self._lock:
            now = time.time()
            recent = [t for t in self._error_counts if now - t < 60]
            return {
                "total_errors": self._total_errors,
                "errors_last_minute": len(recent),
                "error_types": dict(self._error_types),
                "circuit_breaker_open": self._circuit_open,
                "circuit_open_since": self._circuit_open_time if self._circuit_open else None,
            }

    def should_degrade(self) -> bool:
        """Check if the server should degrade functionality.

        Returns:
            True if degradation is recommended.
        """
        return self.is_circuit_open() or self.check_rate_limit()

    def on_error(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Register error callback.

        Args:
            callback: Function receiving error info dict.
        """
        self._on_error_callbacks.append(callback)

    def _classify_error(self, error: Exception) -> Tuple[ErrorCode, int, str]:
        """Classify an error into an error code and status.

        Args:
            error: Exception to classify.

        Returns:
            Tuple of (error_code, http_status, message).
        """
        error_type = type(error).__name__
        message = str(error) or "An error occurred"

        if isinstance(error, (TimeoutError, asyncio.TimeoutError)):
            return ErrorCode.TIMEOUT, HTTP_STATUS_GATEWAY_TIMEOUT, f"Request timed out: {message}"
        elif "rate" in error_type.lower() or "limit" in error_type.lower():
            return ErrorCode.RATE_LIMITED, HTTP_STATUS_TOO_MANY_REQUESTS, message
        elif "queue" in error_type.lower() or "full" in message.lower():
            return ErrorCode.QUEUE_FULL, HTTP_STATUS_SERVICE_UNAVAILABLE, message
        elif "model" in error_type.lower() and "load" in message.lower():
            return ErrorCode.MODEL_LOAD_ERROR, HTTP_STATUS_INTERNAL_ERROR, message
        elif "not loaded" in message.lower():
            return ErrorCode.MODEL_NOT_LOADED, HTTP_STATUS_SERVICE_UNAVAILABLE, message
        elif "context" in message.lower() and "long" in message.lower():
            return ErrorCode.CONTEXT_TOO_LONG, HTTP_STATUS_BAD_REQUEST, message
        elif "invalid" in message.lower() or "validation" in error_type.lower():
            return ErrorCode.INVALID_REQUEST, HTTP_STATUS_BAD_REQUEST, message
        elif "not implemented" in message.lower():
            return ErrorCode.NOT_IMPLEMENTED, HTTP_STATUS_NOT_IMPLEMENTED, message
        else:
            return ErrorCode.INTERNAL_ERROR, HTTP_STATUS_INTERNAL_ERROR, message

    def _should_open_circuit(self, now: float) -> bool:
        """Check if circuit breaker should open.

        Args:
            now: Current timestamp.

        Returns:
            True if circuit should open.
        """
        if self._circuit_open:
            return False

        cutoff = now - 60.0
        recent_errors = sum(1 for t in self._error_counts if t > cutoff)
        return recent_errors >= self._circuit_threshold

    def reset(self) -> None:
        """Reset error tracking."""
        with self._lock:
            self._error_counts.clear()
            self._error_types.clear()
            self._circuit_open = False
            self._total_errors = 0


# ============================================================================
# RequestHandler
# ============================================================================

class RequestHandler:
    """
    Parse and validate incoming HTTP requests for the inference server.

    Validates request format, extracts parameters, and prepares
    requests for model inference.
    """

    def __init__(
        self,
        config: Optional[ServerConfig] = None,
        error_handler: Optional[ServerErrorHandler] = None,
    ):
        """Initialize the request handler.

        Args:
            config: Server configuration.
            error_handler: Error handler instance.
        """
        self._config = config or ServerConfig()
        self._error_handler = error_handler or ServerErrorHandler()

    def parse_request(
        self,
        body: bytes,
        headers: Optional[Dict[str, str]] = None,
    ) -> Tuple[Optional[InferenceRequest], Optional[Dict[str, Any]], int]:
        """Parse an HTTP request body into an InferenceRequest.

        Args:
            body: Raw request body bytes.
            headers: HTTP headers.

        Returns:
            Tuple of (request, error_response, status_code).
            If parsing succeeds, error_response is None.
        """
        # Check content length
        if len(body) > self._config.max_request_size:
            error = self._error_handler.handle_error(
                ValueError(f"Request too large: {len(body)} bytes")
            )
            return None, error, HTTP_STATUS_BAD_REQUEST

        # Parse JSON
        try:
            data = json.loads(body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            error = self._error_handler.handle_error(
                ValueError(f"Invalid JSON: {e}")
            )
            return None, error, HTTP_STATUS_BAD_REQUEST

        if not isinstance(data, dict):
            error = self._error_handler.handle_error(
                ValueError("Request body must be a JSON object")
            )
            return None, error, HTTP_STATUS_BAD_REQUEST

        # Validate required fields
        request = self._validate_and_build(data)
        if request is None:
            error = self._error_handler.handle_error(
                ValueError("Missing required field: 'prompt' or 'messages'")
            )
            return None, error, HTTP_STATUS_BAD_REQUEST

        return request, None, HTTP_STATUS_OK

    def parse_chat_request(
        self,
        body: bytes,
        headers: Optional[Dict[str, str]] = None,
    ) -> Tuple[Optional[InferenceRequest], Optional[Dict[str, Any]], int]:
        """Parse a chat completion request.

        Args:
            body: Raw request body bytes.
            headers: HTTP headers.

        Returns:
            Tuple of (request, error_response, status_code).
        """
        request, error, status = self.parse_request(body, headers)
        if request is not None and not request.messages:
            error = self._error_handler.handle_error(
                ValueError("Chat completion requires 'messages' field")
            )
            return None, error, HTTP_STATUS_BAD_REQUEST
        return request, error, status

    def validate_request(self, request: InferenceRequest) -> Tuple[bool, Optional[str]]:
        """Validate a parsed request.

        Args:
            request: InferenceRequest to validate.

        Returns:
            Tuple of (is_valid, error_message).
        """
        if not request.prompt and not request.messages:
            return False, "Either 'prompt' or 'messages' must be provided"

        if request.temperature < 0 or request.temperature > 2.0:
            return False, "temperature must be between 0 and 2.0"

        if request.top_p < 0 or request.top_p > 1.0:
            return False, "top_p must be between 0 and 1.0"

        if request.max_tokens < 1:
            return False, "max_tokens must be at least 1"

        if request.max_tokens > self._config.max_new_tokens:
            return False, f"max_tokens exceeds maximum ({self._config.max_new_tokens})"

        if request.n < 1 or request.n > 10:
            return False, "n must be between 1 and 10"

        if request.repetition_penalty < 1.0 or request.repetition_penalty > 2.0:
            return False, "repetition_penalty must be between 1.0 and 2.0"

        if request.messages:
            for msg in request.messages:
                if not isinstance(msg, dict):
                    return False, "Each message must be a JSON object"
                if "role" not in msg:
                    return False, "Each message must have a 'role' field"
                if "content" not in msg:
                    return False, "Each message must have a 'content' field"

        return True, None

    def extract_prompt(self, request: InferenceRequest) -> str:
        """Extract the prompt text from a request.

        Handles both direct prompt and chat messages.

        Args:
            request: InferenceRequest.

        Returns:
            Prompt string.
        """
        if request.prompt:
            return request.prompt

        if request.messages:
            parts = []
            for msg in request.messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    parts.append(f"System: {content}")
                elif role == "user":
                    parts.append(f"User: {content}")
                elif role == "assistant":
                    parts.append(f"Assistant: {content}")
                else:
                    parts.append(f"{role}: {content}")
            parts.append("Assistant:")
            return "\n".join(parts)

        return ""

    def _validate_and_build(self, data: Dict[str, Any]) -> Optional[InferenceRequest]:
        """Build InferenceRequest from validated JSON data.

        Args:
            data: Parsed JSON data.

        Returns:
            InferenceRequest or None if invalid.
        """
        if "prompt" not in data and "messages" not in data:
            return None

        return InferenceRequest.from_dict(data)


# ============================================================================
# ModelWorker
# ============================================================================

class ModelWorker:
    """
    Manage model loading, inference, and GPU memory.

    Handles model lifecycle including loading, unloading, warm-up,
    and batched inference execution.
    """

    def __init__(
        self,
        config: Optional[ServerConfig] = None,
        model_load_fn: Optional[Callable] = None,
        inference_fn: Optional[Callable] = None,
    ):
        """Initialize the model worker.

        Args:
            config: Server configuration.
            model_load_fn: Custom model loading function.
            inference_fn: Custom inference function.
        """
        self._config = config or ServerConfig()
        self._model_load_fn = model_load_fn
        self._inference_fn = inference_fn
        self._model = None
        self._model_loaded = False
        self._load_time = 0.0
        self._inference_count = 0
        self._total_inference_time = 0.0
        self._gpu_memory_used = 0.0
        self._gpu_memory_total = 0.0
        self._lock = threading.Lock()
        self._warmup_done = False

    def load_model(self, model_path: Optional[str] = None) -> bool:
        """Load the model.

        Args:
            model_path: Override model path from config.

        Returns:
            True if model loaded successfully.
        """
        path = model_path or self._config.model_path
        if not path:
            return False

        start = time.time()
        try:
            if self._model_load_fn:
                self._model = self._model_load_fn(path, self._config)
            else:
                self._model = self._default_load(path)

            self._model_loaded = True
            self._load_time = time.time() - start
            return True
        except Exception as e:
            self._model_loaded = False
            self._load_time = time.time() - start
            return False

    def unload_model(self) -> None:
        """Unload the model and free memory."""
        with self._lock:
            if self._model is not None:
                del self._model
                self._model = None
            self._model_loaded = False
            self._warmup_done = False
            self._gpu_memory_used = 0.0

    def is_loaded(self) -> bool:
        """Check if model is loaded.

        Returns:
            True if model is loaded.
        """
        return self._model_loaded

    async def infer(
        self,
        request: InferenceRequest,
    ) -> InferenceResponse:
        """Run inference for a single request.

        Args:
            request: InferenceRequest.

        Returns:
            InferenceResponse.

        Raises:
            RuntimeError: If model is not loaded.
        """
        if not self._model_loaded:
            raise RuntimeError("Model is not loaded")

        start = time.time()
        prompt = self._extract_prompt(request)

        with self._lock:
            if self._inference_fn:
                result = await self._inference_fn(self._model, prompt, request)
            else:
                result = await self._default_infer(prompt, request)

            elapsed = time.time() - start
            self._inference_count += 1
            self._total_inference_time += elapsed

            if isinstance(result, str):
                response = InferenceResponse.from_generation(result, request, self._config.model_name or "")
                return response
            elif isinstance(result, InferenceResponse):
                return result
            elif isinstance(result, dict):
                if "text" in result:
                    response = InferenceResponse.from_generation(
                        result["text"], request, self._config.model_name or ""
                    )
                    if "prompt_tokens" in result:
                        response.usage["prompt_tokens"] = result["prompt_tokens"]
                    if "completion_tokens" in result:
                        response.usage["completion_tokens"] = result["completion_tokens"]
                        response.usage["total_tokens"] = (
                            response.usage["prompt_tokens"] + result["completion_tokens"]
                        )
                    return response

        raise RuntimeError("Invalid inference result type")

    async def infer_batch(
        self,
        requests: List[InferenceRequest],
    ) -> List[InferenceResponse]:
        """Run batched inference for multiple requests.

        Args:
            requests: List of InferenceRequest objects.

        Returns:
            List of InferenceResponse objects.
        """
        if not requests:
            return []

        if not self._model_loaded:
            raise RuntimeError("Model is not loaded")

        responses = []
        for request in requests:
            response = await self.infer(request)
            responses.append(response)
        return responses

    async def warmup(self, num_steps: int = 3) -> bool:
        """Warm up the model with dummy inference.

        Args:
            num_steps: Number of warmup steps.

        Returns:
            True if warmup succeeded.
        """
        if not self._model_loaded:
            return False

        try:
            for _ in range(num_steps):
                dummy_request = InferenceRequest(
                    prompt="Hello",
                    max_tokens=10,
                    temperature=1.0,
                )
                await self.infer(dummy_request)
            self._warmup_done = True
            return True
        except Exception:
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get model worker statistics.

        Returns:
            Dictionary with worker stats.
        """
        avg_latency = (
            self._total_inference_time / self._inference_count
            if self._inference_count > 0
            else 0.0
        )
        return {
            "model_loaded": self._model_loaded,
            "model_path": self._config.model_path,
            "load_time": self._load_time,
            "inference_count": self._inference_count,
            "total_inference_time": self._total_inference_time,
            "avg_inference_time": avg_latency,
            "gpu_memory_used_gb": self._gpu_memory_used,
            "gpu_memory_total_gb": self._gpu_memory_total,
            "warmup_done": self._warmup_done,
        }

    def _extract_prompt(self, request: InferenceRequest) -> str:
        """Extract prompt from request.

        Args:
            request: InferenceRequest.

        Returns:
            Prompt string.
        """
        if request.prompt:
            return request.prompt
        if request.messages:
            parts = []
            for msg in request.messages:
                parts.append(f"{msg.get('role', 'user')}: {msg.get('content', '')}")
            return "\n".join(parts) + "\nAssistant:"
        return ""

    def _default_load(self, path: str) -> Any:
        """Default model loading implementation.

        Args:
            path: Model path.

        Returns:
            Loaded model object.
        """
        # This is a placeholder that simulates model loading
        # In practice, this would load from disk using torch/transformers
        class SimulatedModel:
            """Simulated model for testing."""
            def __init__(self, path):
                self.path = path
                self.loaded = True
            def __call__(self, prompt, **kwargs):
                return f"Response to: {prompt[:50]}"

        return SimulatedModel(path)

    async def _default_infer(self, prompt: str, request: InferenceRequest) -> str:
        """Default inference implementation.

        Args:
            prompt: Input prompt.
            request: InferenceRequest.

        Returns:
            Generated text.
        """
        # Simulate inference latency
        tokens = request.max_tokens
        simulated_time = tokens * 0.01  # ~10ms per token
        await asyncio.sleep(min(simulated_time, 2.0))

        return f"This is a simulated response to your prompt. The model would generate {tokens} tokens based on the input."


# ============================================================================
# InferenceServer
# ============================================================================

class InferenceServer:
    """
    Async HTTP server for LLM model inference.

    Provides a complete inference server with request handling,
    model management, streaming support, and health checks.

    Uses Python stdlib http.server and asyncio for async operation.

    Example:
        config = ServerConfig(host="0.0.0.0", port=8000, model_path="/path/to/model")
        server = InferenceServer(config)
        await server.start()
        # ... serve requests ...
        await server.stop()
    """

    def __init__(
        self,
        config: Optional[ServerConfig] = None,
        model_worker: Optional[ModelWorker] = None,
        model_load_fn: Optional[Callable] = None,
        inference_fn: Optional[Callable] = None,
    ):
        """Initialize the inference server.

        Args:
            config: Server configuration.
            model_worker: Custom ModelWorker instance.
            model_load_fn: Custom model loading function.
            inference_fn: Custom inference function.
        """
        self._config = config or ServerConfig()
        self._state = ServerState.STOPPED
        self._start_time = 0.0
        self._server = None
        self._lock = threading.Lock()

        # Components
        self._error_handler = ServerErrorHandler()
        self._request_handler = RequestHandler(self._config, self._error_handler)
        self._response_formatter = ResponseFormatter(
            default_model_name=self._config.model_name or "nexus-llm"
        )

        if model_worker:
            self._worker = model_worker
        else:
            self._worker = ModelWorker(
                config=self._config,
                model_load_fn=model_load_fn,
                inference_fn=inference_fn,
            )

        # Request queue
        self._request_queue: asyncio.Queue = asyncio.Queue(
            maxsize=self._config.max_queue_size
        )
        self._active_requests = 0
        self._total_requests = 0
        self._total_processed = 0

        # Background tasks
        self._tasks: List[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()

        # Callbacks
        self._on_request_callbacks: List[Callable] = []
        self._on_response_callbacks: List[Callable] = []

    @property
    def state(self) -> ServerState:
        """Get current server state."""
        return self._state

    @property
    def is_running(self) -> bool:
        """Check if server is running."""
        return self._state == ServerState.RUNNING

    async def start(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        load_model: bool = True,
    ) -> None:
        """Start the inference server.

        Args:
            host: Override host from config.
            port: Override port from config.
            load_model: Whether to load model on start.
        """
        if self._state != ServerState.STOPPED:
            raise RuntimeError(f"Server is in {self._state.value} state, cannot start")

        self._state = ServerState.STARTING
        server_host = host or self._config.host
        server_port = port or self._config.port

        try:
            # Load model
            if load_model and self._config.model_path:
                loaded = self._worker.load_model()
                if not loaded:
                    raise RuntimeError(f"Failed to load model from {self._config.model_path}")
                await self._worker.warmup()

            self._start_time = time.time()
            self._state = ServerState.RUNNING

            # Start background request processor
            loop = asyncio.get_event_loop()
            task = loop.create_task(self._process_requests())
            self._tasks.append(task)

        except Exception as e:
            self._state = ServerState.ERROR
            raise

    async def stop(self, timeout: Optional[float] = None) -> None:
        """Stop the inference server gracefully.

        Args:
            timeout: Graceful shutdown timeout in seconds.
        """
        if self._state not in (ServerState.RUNNING, ServerState.ERROR):
            return

        self._state = ServerState.STOPPING
        timeout = timeout or self._config.graceful_shutdown_timeout

        # Signal shutdown
        self._shutdown_event.set()

        # Cancel background tasks
        for task in self._tasks:
            task.cancel()

        # Wait for tasks
        if self._tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._tasks, return_exceptions=True),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                pass

        self._tasks.clear()
        self._worker.unload_model()
        self._state = ServerState.STOPPED

    async def handle_request(
        self,
        prompt: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Handle a direct inference request (non-HTTP).

        Args:
            prompt: Input prompt text.
            params: Optional generation parameters.

        Returns:
            Response dictionary.
        """
        if not self.is_running:
            return self._response_formatter.format_error(
                ErrorCode.SERVICE_UNAVAILABLE,
                "Server is not running",
                HTTP_STATUS_SERVICE_UNAVAILABLE,
            )

        request_data = {"prompt": prompt}
        if params:
            request_data.update(params)

        request = InferenceRequest.from_dict(request_data)
        is_valid, error_msg = self._request_handler.validate_request(request)
        if not is_valid:
            return self._response_formatter.format_error(
                ErrorCode.INVALID_REQUEST, error_msg, HTTP_STATUS_BAD_REQUEST, request.id
            )

        self._total_requests += 1
        self._active_requests += 1

        try:
            response = await self._worker.infer(request)
            self._total_processed += 1
            return response.to_dict()
        except Exception as e:
            error = self._error_handler.handle_error(e, request.id)
            return error
        finally:
            self._active_requests -= 1

    async def handle_stream(
        self,
        prompt: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> "AsyncIterator[str]":
        """Handle a streaming inference request.

        Args:
            prompt: Input prompt text.
            params: Optional generation parameters.

        Yields:
            Server-Sent Event formatted chunks.
        """
        if not self.is_running:
            yield self._format_sse_error("Server is not running")
            return

        request_data = {"prompt": prompt, "stream": True}
        if params:
            request_data.update(params)

        request = InferenceRequest.from_dict(request_data)

        try:
            # Simulate streaming by chunking the response
            prompt_text = self._request_handler.extract_prompt(request)
            full_response = await self._worker.infer(request)

            text = full_response.choices[0]["text"] if full_response.choices else ""
            chunk_size = max(1, len(text) // 20)  # ~20 chunks

            for i in range(0, len(text), chunk_size):
                chunk_text = text[i:i + chunk_size]
                is_final = i + chunk_size >= len(text)
                finish = "stop" if is_final else None

                stream_chunk = self._response_formatter.format_stream_chunk(
                    chunk_text, request.id,
                    model=self._config.model_name,
                    finish_reason=finish,
                )
                yield stream_chunk.to_sse()

            # Done signal
            done_chunk = StreamChunk(id=request.id)
            yield done_chunk.to_done_sse()

        except Exception as e:
            yield self._format_sse_error(str(e))

    async def health_check(self) -> Dict[str, Any]:
        """Perform a health check.

        Returns:
            Health check response dictionary.
        """
        uptime = time.time() - self._start_time if self._start_time > 0 else 0
        return self._response_formatter.format_health_response(
            status=self._state.value,
            uptime=uptime,
            model_loaded=self._worker.is_loaded(),
            requests_processed=self._total_processed,
            queue_size=self._request_queue.qsize(),
            gpu_memory=self._worker._gpu_memory_used if self._worker._gpu_memory_used > 0 else None,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics.

        Returns:
            Dictionary with server stats.
        """
        uptime = time.time() - self._start_time if self._start_time > 0 else 0
        return {
            "state": self._state.value,
            "uptime": uptime,
            "total_requests": self._total_requests,
            "total_processed": self._total_processed,
            "active_requests": self._active_requests,
            "queue_size": self._request_queue.qsize(),
            "model": self._worker.get_stats(),
            "errors": self._error_handler.get_error_stats(),
            "config": {
                "host": self._config.host,
                "port": self._config.port,
                "max_batch_size": self._config.max_batch_size,
                "max_queue_size": self._config.max_queue_size,
            },
        }

    def on_request(self, callback: Callable) -> None:
        """Register callback for incoming requests.

        Args:
            callback: Function called with InferenceRequest.
        """
        self._on_request_callbacks.append(callback)

    def on_response(self, callback: Callable) -> None:
        """Register callback for completed responses.

        Args:
            callback: Function called with InferenceResponse.
        """
        self._on_response_callbacks.append(callback)

    async def _process_requests(self) -> None:
        """Background task for processing queued requests."""
        while not self._shutdown_event.is_set():
            try:
                request = await asyncio.wait_for(
                    self._request_queue.get(),
                    timeout=1.0,
                )
                # Process the request
                try:
                    response = await self._worker.infer(request)
                    self._total_processed += 1
                    for cb in self._on_response_callbacks:
                        try:
                            cb(request, response)
                        except Exception:
                            pass
                except Exception as e:
                    self._error_handler.handle_error(e, request.id)
                finally:
                    self._active_requests -= 1
                    self._request_queue.task_done()
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception:
                continue

    def _format_sse_error(self, message: str) -> str:
        """Format an error as SSE.

        Args:
            message: Error message.

        Returns:
            SSE formatted error string.
        """
        error_data = json.dumps({"error": {"message": message}})
        return f"data: {error_data}\n\n"


# ============================================================================
# HTTP Server Implementation (using stdlib http.server)
# ============================================================================

class _HTTPHandler:
    """Base HTTP handler for the inference server."""

    def __init__(self, server: InferenceServer):
        self._server = server
        self._formatter = server._response_formatter
        self._request_handler = server._request_handler

    def handle_http_request(
        self,
        method: str,
        path: str,
        headers: Dict[str, str],
        body: bytes,
    ) -> Tuple[int, Dict[str, str], bytes]:
        """Handle an HTTP request.

        Args:
            method: HTTP method.
            path: URL path.
            headers: HTTP headers.
            body: Request body.

        Returns:
            Tuple of (status_code, response_headers, response_body).
        """
        response_headers = {"Content-Type": CONTENT_TYPE_JSON}

        # CORS preflight
        if method == "OPTIONS":
            return self._handle_cors(response_headers)

        # Route handling
        if path == "/v1/completions" and method == "POST":
            return self._handle_completions(body, response_headers)
        elif path == "/v1/chat/completions" and method == "POST":
            return self._handle_chat_completions(body, response_headers)
        elif path == "/health" or path == "/v1/health":
            return self._handle_health(response_headers)
        elif path == "/stats" or path == "/v1/stats":
            return self._handle_stats(response_headers)
        elif path == "/" and method == "GET":
            return self._handle_root(response_headers)
        else:
            error = self._formatter.format_error(
                ErrorCode.NOT_FOUND, f"Unknown endpoint: {path}", 404
            )
            return 404, response_headers, json.dumps(error).encode("utf-8")

    def _handle_cors(self, headers: Dict[str, str]) -> Tuple[int, Dict[str, str], bytes]:
        """Handle CORS preflight request.

        Args:
            headers: Response headers.

        Returns:
            Tuple of (status, headers, body).
        """
        headers["Access-Control-Allow-Origin"] = "*"
        headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        return 204, headers, b""

    def _handle_completions(
        self, body: bytes, headers: Dict[str, str]
    ) -> Tuple[int, Dict[str, str], bytes]:
        """Handle /v1/completions endpoint.

        Args:
            body: Request body.
            headers: Response headers.

        Returns:
            Tuple of (status, headers, body).
        """
        request, error, status = self._request_handler.parse_request(body)
        if error:
            return status, headers, json.dumps(error).encode("utf-8")

        is_valid, error_msg = self._request_handler.validate_request(request)
        if not is_valid:
            error = self._formatter.format_error(
                ErrorCode.INVALID_REQUEST, error_msg, 400, request.id
            )
            return 400, headers, json.dumps(error).encode("utf-8")

        try:
            response = self._formatter.to_json_bytes(
                self._formatter.format_completion("Simulated response", request)
            )
            return 200, headers, response
        except Exception as e:
            error = self._server._error_handler.handle_error(e, request.id)
            return 500, headers, json.dumps(error).encode("utf-8")

    def _handle_chat_completions(
        self, body: bytes, headers: Dict[str, str]
    ) -> Tuple[int, Dict[str, str], bytes]:
        """Handle /v1/chat/completions endpoint.

        Args:
            body: Request body.
            headers: Response headers.

        Returns:
            Tuple of (status, headers, body).
        """
        request, error, status = self._request_handler.parse_request(body)
        if error:
            return status, headers, json.dumps(error).encode("utf-8")

        if not request.messages:
            error = self._formatter.format_error(
                ErrorCode.INVALID_REQUEST, "messages field required", 400, request.id
            )
            return 400, headers, json.dumps(error).encode("utf-8")

        try:
            prompt = self._request_handler.extract_prompt(request)
            response = self._formatter.format_chat_completion(
                "Simulated chat response", request
            )
            return 200, headers, self._formatter.to_json_bytes(response)
        except Exception as e:
            error = self._server._error_handler.handle_error(e, request.id)
            return 500, headers, json.dumps(error).encode("utf-8")

    def _handle_health(
        self, headers: Dict[str, str]
    ) -> Tuple[int, Dict[str, str], bytes]:
        """Handle health check endpoint.

        Args:
            headers: Response headers.

        Returns:
            Tuple of (status, headers, body).
        """
        health = self._formatter.format_health_response(
            status="ok",
            uptime=time.time() - self._server._start_time,
            model_loaded=self._server._worker.is_loaded(),
            requests_processed=self._server._total_processed,
        )
        return 200, headers, json.dumps(health, indent=2).encode("utf-8")

    def _handle_stats(
        self, headers: Dict[str, str]
    ) -> Tuple[int, Dict[str, str], bytes]:
        """Handle stats endpoint.

        Args:
            headers: Response headers.

        Returns:
            Tuple of (status, headers, body).
        """
        stats = self._server.get_stats()
        return 200, headers, json.dumps(stats, indent=2).encode("utf-8")

    def _handle_root(
        self, headers: Dict[str, str]
    ) -> Tuple[int, Dict[str, str], bytes]:
        """Handle root endpoint.

        Args:
            headers: Response headers.

        Returns:
            Tuple of (status, headers, body).
        """
        info = {
            "name": "Nexus LLM Inference Server",
            "version": "0.1.0",
            "endpoints": [
                "/v1/completions",
                "/v1/chat/completions",
                "/health",
                "/stats",
            ],
        }
        return 200, headers, json.dumps(info, indent=2).encode("utf-8")
