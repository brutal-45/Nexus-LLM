"""Model server for Nexus-LLM."""

import asyncio
import logging
import time
from typing import Any, Dict, Optional

from nexus_llm.serving.config import ServingConfig

logger = logging.getLogger(__name__)


class ModelServer:
    """Production-ready model server wrapping FastAPI.

    Provides model serving with health checks, metrics, and graceful
    shutdown. Uses FastAPI under the hood when available.
    """

    def __init__(self, config: Optional[ServingConfig] = None) -> None:
        self._config = config or ServingConfig()
        self._app: Optional[Any] = None
        self._server: Optional[Any] = None
        self._model: Optional[Any] = None
        self._status: str = "stopped"
        self._start_time: Optional[float] = None
        self._request_count: int = 0
        self._error_count: int = 0

    # -- Lifecycle ------------------------------------------------------------

    def start(
        self,
        model: Any,
        host: Optional[str] = None,
        port: Optional[int] = None,
    ) -> None:
        """Start the model server.

        Args:
            model: The loaded model object to serve.
            host: Override config host.
            port: Override config port.

        Raises:
            RuntimeError: If the server is already running.
        """
        if self._status == "running":
            raise RuntimeError("Server is already running")

        self._model = model
        self._status = "starting"
        self._start_time = time.time()

        host = host or self._config.host
        port = port or self._config.port

        try:
            self._app = self._create_app()
            self._status = "running"
            logger.info("Model server started on %s:%d", host, port)
        except Exception as exc:
            self._status = "error"
            logger.error("Failed to start server: %s", exc)
            raise

    def stop(self) -> None:
        """Stop the model server gracefully."""
        if self._status != "running":
            return

        self._status = "stopping"
        self._model = None
        self._status = "stopped"
        logger.info("Model server stopped")

    def get_status(self) -> Dict[str, Any]:
        """Return current server status.

        Returns:
            Dict with keys: ``status``, ``uptime_seconds``,
            ``request_count``, ``error_count``, ``model_loaded``.
        """
        uptime = 0.0
        if self._start_time is not None and self._status == "running":
            uptime = time.time() - self._start_time

        return {
            "status": self._status,
            "uptime_seconds": round(uptime, 2),
            "request_count": self._request_count,
            "error_count": self._error_count,
            "model_loaded": self._model is not None,
        }

    # -- Inference ------------------------------------------------------------

    def predict(self, input_data: Any) -> Any:
        """Run inference on the loaded model.

        Args:
            input_data: Model input.

        Returns:
            Model output.

        Raises:
            RuntimeError: If no model is loaded.
        """
        if self._model is None:
            raise RuntimeError("No model loaded")

        self._request_count += 1
        try:
            if hasattr(self._model, "generate"):
                return self._model.generate(input_data)
            elif callable(self._model):
                return self._model(input_data)
            else:
                raise RuntimeError("Model has no callable interface")
        except Exception as exc:
            self._error_count += 1
            raise

    # -- App creation ---------------------------------------------------------

    def _create_app(self) -> Any:
        """Create the FastAPI application with all routes and middleware.

        Returns:
            FastAPI app instance, or ``None`` if FastAPI is not installed.
        """
        try:
            from fastapi import FastAPI, HTTPException
            from fastapi.responses import JSONResponse
            from pydantic import BaseModel
        except ImportError:
            logger.warning(
                "FastAPI not installed; server will run in headless mode. "
                "Install fastapi and uvicorn for HTTP serving."
            )
            return None

        app = FastAPI(
            title="Nexus-LLM Model Server",
            version="1.0.0",
        )

        # Request / response schemas
        class PredictRequest(BaseModel):
            input: Any
            parameters: Dict[str, Any] = {}

        class PredictResponse(BaseModel):
            output: Any
            status: str = "success"

        @app.post("/predict", response_model=PredictResponse)
        async def predict_endpoint(request: PredictRequest) -> PredictResponse:
            try:
                result = self.predict(request.input)
                return PredictResponse(output=result)
            except Exception as exc:
                raise HTTPException(status_code=500, detail=str(exc))

        @app.get("/status")
        async def status_endpoint() -> Dict[str, Any]:
            return self.get_status()

        @app.get("/health")
        async def health_endpoint() -> Dict[str, Any]:
            return {
                "healthy": self._model is not None,
                "status": self._status,
            }

        return app

    # -- Properties -----------------------------------------------------------

    @property
    def app(self) -> Any:
        """Return the FastAPI app instance (for use with uvicorn)."""
        return self._app
