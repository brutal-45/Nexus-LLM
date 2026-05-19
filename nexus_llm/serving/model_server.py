"""Model Server for Nexus-LLM.

Provides dedicated model serving with lifecycle management, health
monitoring, and graceful shutdown.  Wraps a loaded model and exposes
a simple interface for inference with automatic resource management.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ServerState(str, Enum):
    """Lifecycle states of the model server."""
    INITIALIZING = "initializing"
    READY = "ready"
    BUSY = "busy"
    UNHEALTHY = "unhealthy"
    DRAINING = "draining"
    STOPPED = "stopped"
    ERROR = "error"


class HealthStatus(str, Enum):
    """Health check result."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ServerConfig:
    """Configuration for the model server."""
    model_name: str = ""
    device: str = "auto"
    max_concurrent_requests: int = 10
    request_timeout: float = 120.0
    health_check_interval: float = 30.0
    idle_timeout: float = 300.0  # Unload model after idle
    auto_load: bool = True
    warmup_on_start: bool = True
    max_memory_mb: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "device": self.device,
            "max_concurrent_requests": self.max_concurrent_requests,
            "request_timeout": self.request_timeout,
            "health_check_interval": self.health_check_interval,
            "idle_timeout": self.idle_timeout,
            "auto_load": self.auto_load,
            "warmup_on_start": self.warmup_on_start,
            "max_memory_mb": self.max_memory_mb,
        }


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    status: HealthStatus = HealthStatus.HEALTHY
    model_loaded: bool = False
    memory_used_mb: float = 0.0
    active_requests: int = 0
    uptime_seconds: float = 0.0
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "model_loaded": self.model_loaded,
            "memory_used_mb": round(self.memory_used_mb, 2),
            "active_requests": self.active_requests,
            "uptime_seconds": round(self.uptime_seconds, 2),
            "message": self.message,
            "details": self.details,
        }


@dataclass
class InferenceRequest:
    """A single inference request."""
    request_id: str = ""
    prompt: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    callback: Optional[Callable] = None

    def __post_init__(self) -> None:
        if not self.request_id:
            self.request_id = str(uuid.uuid4())[:12]


@dataclass
class InferenceResponse:
    """Response to an inference request."""
    request_id: str = ""
    text: str = ""
    success: bool = True
    error: Optional[str] = None
    latency_ms: float = 0.0
    tokens_generated: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "text": self.text,
            "success": self.success,
            "error": self.error,
            "latency_ms": round(self.latency_ms, 2),
            "tokens_generated": self.tokens_generated,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Model Server
# ---------------------------------------------------------------------------

class ModelServer:
    """Dedicated model serving with lifecycle management.

    Manages model loading, health monitoring, request counting, and
    graceful shutdown.  Provides a clean interface for inference with
    automatic timeout and concurrency control.

    Example::

        server = ModelServer(ServerConfig(model_name="llama-7b"))
        server.start()

        response = server.infer(InferenceRequest(prompt="Hello, world!"))
        print(response.text)

        health = server.health_check()
        print(health.status)

        server.stop()
    """

    def __init__(self, config: Optional[ServerConfig] = None) -> None:
        self._config = config or ServerConfig()
        self._state = ServerState.INITIALIZING
        self._model = None
        self._tokenizer = None
        self._active_requests = 0
        self._total_requests = 0
        self._total_errors = 0
        self._started_at: Optional[float] = None
        self._last_activity: Optional[float] = None

        self._lock = threading.RLock()
        self._health_thread: Optional[threading.Thread] = None
        self._running = False

        self._health_callbacks: List[Callable[[HealthCheckResult], None]] = []
        self._lifecycle_callbacks: List[Callable[[ServerState], None]] = []

    @property
    def state(self) -> ServerState:
        """Current server state."""
        return self._state

    @property
    def config(self) -> ServerConfig:
        """Server configuration."""
        return self._config

    @property
    def is_ready(self) -> bool:
        """Whether the server is ready to accept requests."""
        return self._state in (ServerState.READY, ServerState.BUSY)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the model server.

        Loads the model, performs warmup if configured, and starts
        the health monitoring thread.
        """
        if self._running:
            return

        self._set_state(ServerState.INITIALIZING)
        self._running = True
        self._started_at = time.time()

        if self._config.auto_load:
            self._load_model()

        # Start health check loop
        self._health_thread = threading.Thread(
            target=self._health_check_loop,
            daemon=True,
            name="model-server-health",
        )
        self._health_thread.start()

        self._set_state(ServerState.READY)

    def stop(self, timeout: float = 30.0, drain: bool = True) -> None:
        """Stop the model server gracefully.

        Args:
            timeout: Seconds to wait for in-flight requests.
            drain: If True, wait for active requests to complete.
        """
        self._running = False

        if drain:
            self._set_state(ServerState.DRAINING)
            deadline = time.time() + timeout
            while self._active_requests > 0 and time.time() < deadline:
                time.sleep(0.5)

        self._unload_model()
        self._set_state(ServerState.STOPPED)

        if self._health_thread and self._health_thread.is_alive():
            self._health_thread.join(timeout=5.0)

    def drain(self, timeout: float = 60.0) -> None:
        """Stop accepting new requests and wait for in-flight requests.

        Args:
            timeout: Maximum seconds to wait.
        """
        self._set_state(ServerState.DRAINING)
        deadline = time.time() + timeout
        while self._active_requests > 0 and time.time() < deadline:
            time.sleep(0.5)
        self._set_state(ServerState.STOPPED)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        """Process an inference request.

        Args:
            request: InferenceRequest with prompt and parameters.

        Returns:
            InferenceResponse with generated text and metadata.

        Raises:
            RuntimeError: If the server is not ready.
            TimeoutError: If the request exceeds the timeout.
        """
        if not self.is_ready:
            return InferenceResponse(
                request_id=request.request_id,
                success=False,
                error=f"Server not ready (state: {self._state.value})",
            )

        with self._lock:
            if self._active_requests >= self._config.max_concurrent_requests:
                return InferenceResponse(
                    request_id=request.request_id,
                    success=False,
                    error="Server at maximum concurrent request capacity",
                )
            self._active_requests += 1
            self._total_requests += 1

        # Update state to BUSY if was READY
        if self._state == ServerState.READY:
            self._set_state(ServerState.BUSY)

        start_time = time.time()
        try:
            result = self._run_inference(request)
            latency = (time.time() - start_time) * 1000

            self._last_activity = time.time()
            return InferenceResponse(
                request_id=request.request_id,
                text=result,
                success=True,
                latency_ms=latency,
                metadata=request.params,
            )
        except Exception as e:
            self._total_errors += 1
            latency = (time.time() - start_time) * 1000
            return InferenceResponse(
                request_id=request.request_id,
                success=False,
                error=str(e),
                latency_ms=latency,
            )
        finally:
            with self._lock:
                self._active_requests -= 1
            if self._active_requests == 0 and self._state == ServerState.BUSY:
                self._set_state(ServerState.READY)

    def infer_async(self, request: InferenceRequest, callback: Optional[Callable] = None) -> str:
        """Submit an inference request asynchronously.

        Args:
            request: InferenceRequest to process.
            callback: Optional callback(response) when complete.

        Returns:
            The request ID for tracking.
        """
        def _worker() -> None:
            response = self.infer(request)
            if callback:
                callback(response)
            elif request.callback:
                request.callback(response)

        t = threading.Thread(target=_worker, daemon=True, name=f"infer-{request.request_id}")
        t.start()
        return request.request_id

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def health_check(self) -> HealthCheckResult:
        """Perform a health check and return the result.

        Returns:
            HealthCheckResult with current status information.
        """
        memory_mb = self._get_memory_usage()
        uptime = (time.time() - self._started_at) if self._started_at else 0.0

        status = HealthStatus.HEALTHY
        message = "Server is healthy"

        if not self._model:
            status = HealthStatus.UNHEALTHY
            message = "Model not loaded"
        elif self._state in (ServerState.ERROR, ServerState.STOPPED):
            status = HealthStatus.UNHEALTHY
            message = f"Server in {self._state.value} state"
        elif self._active_requests >= self._config.max_concurrent_requests:
            status = HealthStatus.DEGRADED
            message = "Server at capacity"
        elif memory_mb > 0 and self._config.max_memory_mb and memory_mb > self._config.max_memory_mb:
            status = HealthStatus.DEGRADED
            message = "Memory usage above threshold"

        result = HealthCheckResult(
            status=status,
            model_loaded=self._model is not None,
            memory_used_mb=memory_mb,
            active_requests=self._active_requests,
            uptime_seconds=uptime,
            message=message,
        )

        # Notify health callbacks
        for cb in self._health_callbacks:
            try:
                cb(result)
            except Exception:
                pass

        return result

    def on_health_check(self, callback: Callable[[HealthCheckResult], None]) -> None:
        """Register a callback for health check results."""
        self._health_callbacks.append(callback)

    def on_state_change(self, callback: Callable[[ServerState], None]) -> None:
        """Register a callback for state changes."""
        self._lifecycle_callbacks.append(callback)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics.

        Returns:
            Dictionary with request counts, state, and performance data.
        """
        return {
            "state": self._state.value,
            "model_name": self._config.model_name,
            "active_requests": self._active_requests,
            "total_requests": self._total_requests,
            "total_errors": self._total_errors,
            "uptime_seconds": round((time.time() - self._started_at), 2) if self._started_at else 0,
            "last_activity": self._last_activity,
            "memory_used_mb": self._get_memory_usage(),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load the model and tokenizer."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            device = self._config.device
            if device == "auto":
                try:
                    import torch
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                except ImportError:
                    device = "cpu"

            self._tokenizer = AutoTokenizer.from_pretrained(self._config.model_name)
            self._model = AutoModelForCausalLM.from_pretrained(self._config.model_name)

            try:
                import torch
                self._model = self._model.to(device)
            except Exception:
                pass

            self._model.eval()

            if self._config.warmup_on_start:
                self._warmup()

        except Exception as e:
            self._set_state(ServerState.ERROR)
            raise RuntimeError(f"Failed to load model: {e}") from e

    def _unload_model(self) -> None:
        """Release model resources."""
        import gc
        self._model = None
        self._tokenizer = None
        gc.collect()

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _warmup(self) -> None:
        """Perform a warmup inference to initialize caches."""
        if self._model and self._tokenizer:
            try:
                inputs = self._tokenizer("Hello", return_tensors="pt")
                with self._no_grad_context():
                    _ = self._model.generate(**inputs, max_new_tokens=5)
            except Exception:
                pass

    @staticmethod
    def _no_grad_context():
        """Return a no-grad context manager."""
        try:
            import torch
            return torch.no_grad()
        except ImportError:
            from contextlib import nullcontext
            return nullcontext()

    def _run_inference(self, request: InferenceRequest) -> str:
        """Execute inference on the loaded model."""
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model or tokenizer not loaded")

        max_tokens = request.params.get("max_tokens", 256)
        temperature = request.params.get("temperature", 0.7)

        inputs = self._tokenizer(request.prompt, return_tensors="pt")
        try:
            import torch
            device = next(self._model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
        except Exception:
            pass

        with self._no_grad_context():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
            )

        generated = self._tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Strip the prompt if echoed
        if generated.startswith(request.prompt):
            generated = generated[len(request.prompt):].strip()

        return generated

    def _set_state(self, state: ServerState) -> None:
        """Update server state and notify callbacks."""
        self._state = state
        for cb in self._lifecycle_callbacks:
            try:
                cb(state)
            except Exception:
                pass

    def _health_check_loop(self) -> None:
        """Periodic health check loop."""
        while self._running:
            try:
                time.sleep(self._config.health_check_interval)
                result = self.health_check()

                if result.status == HealthStatus.UNHEALTHY:
                    self._set_state(ServerState.UNHEALTHY)
                elif self._state == ServerState.UNHEALTHY and result.status == HealthStatus.HEALTHY:
                    self._set_state(ServerState.READY)

                # Auto-unload if idle too long
                if (self._last_activity and
                    self._config.idle_timeout > 0 and
                    time.time() - self._last_activity > self._config.idle_timeout and
                    self._active_requests == 0):
                    self._unload_model()
                    self._set_state(ServerState.DRAINING)

            except Exception:
                pass

    @staticmethod
    def _get_memory_usage() -> float:
        """Get current process memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0
