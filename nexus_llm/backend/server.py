"""FastAPI-based LLM server for Nexus-LLM.

Exposes the InferenceEngine, ModelManager, and TokenizerManager via
REST and WebSocket endpoints.

Uses the modern ``lifespan`` context manager for startup/shutdown
(replaces the deprecated ``@app.on_event`` pattern).
"""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, List, Optional, Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from nexus_llm.backend.inference import InferenceEngine
from nexus_llm.backend.model_manager import ModelManager, ModelState
from nexus_llm.backend.tokenizer_utils import TokenizerManager
from nexus_llm.core.exceptions import (
    InferenceError,
    ModelLoadError,
    ModelNotFoundError,
)
from nexus_llm.core.model_catalog import MODEL_CATALOG

logger = logging.getLogger(__name__)


# ==================================================================
# Pydantic request / response models
# ==================================================================

class LoadModelRequest(BaseModel):
    """Request body for ``POST /model/load``."""
    model_id: str = Field(..., description="Short model ID from the catalogue")
    device: str = Field("auto", description="Device: auto|cuda|mps|cpu")
    precision: str = Field("fp32", description="Precision: fp32|fp16|bf16|8bit|4bit")
    cache_dir: Optional[str] = Field(None, description="HuggingFace cache directory")


class GenerateRequest(BaseModel):
    """Request body for ``POST /generate``."""
    prompt: str = Field(..., min_length=1, description="Input prompt")
    max_new_tokens: int = Field(256, ge=1, le=4096)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    top_k: int = Field(50, ge=0)
    repetition_penalty: float = Field(1.1, ge=1.0, le=2.0)
    num_beams: int = Field(1, ge=1, le=8)
    do_sample: bool = Field(True)


class ChatRequest(BaseModel):
    """Request body for ``POST /chat``."""
    messages: List[Dict[str, str]] = Field(
        ..., min_length=1, description="List of {role, content} dicts"
    )
    max_new_tokens: int = Field(512, ge=1, le=4096)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    top_k: int = Field(50, ge=0)
    repetition_penalty: float = Field(1.1, ge=1.0, le=2.0)


class SimpleResponse(BaseModel):
    """Generic status response."""
    status: str
    message: str = ""


class ErrorResponse(BaseModel):
    """Error response body."""
    error: str
    detail: str = ""


# ==================================================================
# Lifespan (startup / shutdown)
# ==================================================================

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: ensure engine is initialised on startup, tear down on shutdown."""
    # Engine is created in create_app(), but we guarantee it here too
    # so the lifespan is self-contained.
    if not hasattr(app.state, "engine") or app.state.engine is None:
        app.state.engine = InferenceEngine()
    logger.info("Nexus-LLM server started — engine initialised")
    yield
    # Shutdown: unload model if loaded
    engine: InferenceEngine = app.state.engine
    if engine.is_ready:
        engine.unload_model()
    logger.info("Nexus-LLM server shut down")


# ==================================================================
# App factory
# ==================================================================

def create_app() -> FastAPI:
    """Build and return the FastAPI application."""
    app = FastAPI(
        title="Nexus-LLM",
        description="Local LLM inference server",
        version="2.0.0",
        lifespan=lifespan,
    )

    # Initialise engine eagerly so routes work even if lifespan
    # hasn't been entered yet (e.g. during testing).
    app.state.engine = InferenceEngine()

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routes
    _register_routes(app)

    return app


# ==================================================================
# Route registration
# ==================================================================

def _register_routes(app: FastAPI) -> None:
    """Attach all REST and WebSocket routes to *app*."""

    # --- Health / info ------------------------------------------------

    @app.get("/", response_model=SimpleResponse)
    async def root() -> SimpleResponse:
        """Root endpoint — server is alive."""
        return SimpleResponse(status="ok", message="Nexus-LLM server is running")

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        """Health check including model state."""
        engine: InferenceEngine = app.state.engine
        return {
            "status": "healthy",
            "model_loaded": engine.model_manager.is_loaded,
            "model_state": engine.model_manager.state.value,
            "model_id": engine.model_manager.model_id,
        }

    # --- Model management ---------------------------------------------

    @app.post("/model/load")
    async def load_model(req: LoadModelRequest):
        """Load a model and its tokenizer."""
        engine: InferenceEngine = app.state.engine
        try:
            engine.load_model(
                model_id=req.model_id,
                device=req.device,
                precision=req.precision,
                cache_dir=req.cache_dir,
            )
            return {"status": "ok", "message": f"Model '{req.model_id}' loaded successfully"}
        except ModelNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except ModelLoadError as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/model/unload")
    async def unload_model():
        """Unload the current model and tokenizer."""
        engine: InferenceEngine = app.state.engine
        if not engine.model_manager.is_loaded:
            raise HTTPException(status_code=400, detail="No model is currently loaded")
        engine.unload_model()
        return {"status": "ok", "message": "Model unloaded"}

    @app.post("/model/reload")
    async def reload_model(
        device: Optional[str] = None,
        precision: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        """Reload the current model (optionally with new settings)."""
        engine: InferenceEngine = app.state.engine
        try:
            engine.model_manager.reload(device=device, precision=precision, cache_dir=cache_dir)
            # Also reload the tokenizer
            model_id = engine.model_manager.model_id
            if model_id:
                engine.tokenizer_manager.load(model_id, cache_dir=cache_dir)
            return {"status": "ok", "message": f"Model '{model_id}' reloaded"}
        except ModelLoadError as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.get("/model/info")
    async def model_info():
        """Return information about the currently loaded model."""
        engine: InferenceEngine = app.state.engine
        if not engine.model_manager.is_loaded:
            raise HTTPException(status_code=400, detail="No model is currently loaded")
        return engine.model_manager.get_info()

    @app.get("/model/memory")
    async def model_memory():
        """Return memory usage information."""
        engine: InferenceEngine = app.state.engine
        return engine.model_manager.get_memory_usage()

    # --- Generation ---------------------------------------------------

    @app.post("/generate")
    async def generate(req: GenerateRequest):
        """Generate a text completion."""
        engine: InferenceEngine = app.state.engine
        if not engine.is_ready:
            raise HTTPException(status_code=400, detail="Model is not loaded")
        try:
            # Run blocking generate in a thread so the event loop stays free
            result = await asyncio.to_thread(
                engine.generate,
                prompt=req.prompt,
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                top_k=req.top_k,
                repetition_penalty=req.repetition_penalty,
                num_beams=req.num_beams,
                do_sample=req.do_sample,
            )
            return result
        except InferenceError as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/generate/stop")
    async def stop_generation():
        """Interrupt the current generation."""
        engine: InferenceEngine = app.state.engine
        engine.stop_generation()
        return {"status": "ok", "message": "Stop signal sent"}

    # --- Chat ---------------------------------------------------------

    @app.post("/chat")
    async def chat(req: ChatRequest):
        """Generate a conversational (chat) response."""
        engine: InferenceEngine = app.state.engine
        if not engine.is_ready:
            raise HTTPException(status_code=400, detail="Model is not loaded")
        try:
            result = await asyncio.to_thread(
                engine.chat,
                messages=req.messages,
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                top_k=req.top_k,
                repetition_penalty=req.repetition_penalty,
            )
            return result
        except InferenceError as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    # --- Stats --------------------------------------------------------

    @app.get("/stats")
    async def stats():
        """Return accumulated generation statistics."""
        engine: InferenceEngine = app.state.engine
        return engine.get_stats()

    # --- WebSocket streaming ------------------------------------------

    @app.websocket("/ws/chat")
    async def ws_chat(websocket: WebSocket):
        """WebSocket endpoint for streaming chat generation.

        Protocol:
        1. Client connects.
        2. Client sends a JSON message matching ChatRequest fields.
        3. Server streams back JSON chunks: ``{"token": "..."}``
        4. When done server sends: ``{"status": "done", "stats": {...}}``
        5. If an error occurs: ``{"error": "..."}``
        """
        await websocket.accept()
        logger.info("WebSocket client connected")

        try:
            while True:
                raw = await websocket.receive_text()
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    await websocket.send_json({"error": "Invalid JSON"})
                    continue

                messages = data.get("messages")
                if not messages:
                    await websocket.send_json({"error": "Field 'messages' is required"})
                    continue

                max_new_tokens = data.get("max_new_tokens", 512)
                temperature = data.get("temperature", 0.7)
                top_p = data.get("top_p", 0.9)
                top_k = data.get("top_k", 50)
                repetition_penalty = data.get("repetition_penalty", 1.1)

                engine: InferenceEngine = app.state.engine
                if not engine.is_ready:
                    await websocket.send_json({"error": "Model is not loaded"})
                    continue

                try:
                    # The generator runs synchronously; we pull tokens
                    # in an executor to avoid blocking the event loop.
                    gen = engine.chat_stream(
                        messages=messages,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        repetition_penalty=repetition_penalty,
                    )

                    for token in gen:
                        await websocket.send_json({"token": token})

                    await websocket.send_json({
                        "status": "done",
                        "stats": engine.get_stats(),
                    })

                except InferenceError as exc:
                    await websocket.send_json({"error": str(exc)})
                except Exception as exc:
                    logger.exception("Unexpected error in ws/chat")
                    await websocket.send_json({"error": f"Internal error: {exc}"})

        except WebSocketDisconnect:
            logger.info("WebSocket client disconnected")
        except Exception as exc:
            logger.exception("WebSocket error")
            try:
                await websocket.send_json({"error": f"Connection error: {exc}"})
            except Exception:
                pass


# ==================================================================
# LLMServer convenience wrapper
# ==================================================================

class LLMServer:
    """Convenience wrapper around the FastAPI application.

    Provides ``run()`` to start uvicorn and direct access to the
    underlying ``FastAPI`` and ``InferenceEngine`` instances.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        cors_origins: Optional[List[str]] = None,
    ) -> None:
        self.host = host
        self.port = port
        self.cors_origins = cors_origins or ["*"]
        self._app: Optional[FastAPI] = None

    @property
    def app(self) -> FastAPI:
        """The FastAPI application instance (lazily created)."""
        if self._app is None:
            self._app = create_app()
        return self._app

    @property
    def engine(self) -> InferenceEngine:
        """The InferenceEngine attached to the running app."""
        return self.app.state.engine

    def run(self, **uvicorn_kwargs: Any) -> None:
        """Start the server with uvicorn.

        Extra keyword arguments are forwarded to ``uvicorn.run()``.
        """
        import uvicorn

        log_level = "info"
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            log_level=log_level,
            **uvicorn_kwargs,
        )
