"""FastAPI server for Nexus-LLM backend.

Provides REST endpoints (/generate, /chat, /models, /health, /config)
and WebSocket for streaming generation.
"""

import asyncio
import json
import time
import uuid
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .model_manager import ModelManager, ModelRegistry
from .inference import InferenceEngine, GenerationResult
from .generation import GenerationConfig, GenerationPresets
from .health import ServiceHealthCheck
from .metrics import BackendMetrics

import logging

logger = logging.getLogger(__name__)


# ── Request/Response Models ──────────────────────────────────────────────────


class GenerateRequest(BaseModel):
    """Request body for /generate endpoint."""
    prompt: str = Field(..., min_length=1, description="Input text prompt")
    max_new_tokens: int = Field(512, ge=1, le=8192)
    temperature: float = Field(1.0, ge=0.0, le=2.0)
    top_p: float = Field(1.0, ge=0.0, le=1.0)
    top_k: int = Field(50, ge=0)
    repetition_penalty: float = Field(1.0, ge=0.1, le=2.0)
    num_beams: int = Field(1, ge=1, le=16)
    num_return_sequences: int = Field(1, ge=1, le=8)
    do_sample: bool = Field(True)
    preset: Optional[str] = Field(None, description="Generation preset name")
    model: Optional[str] = Field(None, description="Model ID to use")
    stop: Optional[List[str]] = Field(None, description="Stop sequences")
    seed: Optional[int] = Field(None, description="Random seed")


class GenerateResponse(BaseModel):
    """Response for /generate endpoint."""
    id: str
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    finish_reason: str
    tokens_per_second: float
    generation_time_seconds: float


class ChatMessage(BaseModel):
    """A single chat message."""
    role: str = Field(..., description="Role: system, user, or assistant")
    content: str = Field(..., min_length=1, description="Message content")


class ChatRequest(BaseModel):
    """Request body for /chat endpoint."""
    messages: List[ChatMessage] = Field(..., min_length=1)
    max_new_tokens: int = Field(512, ge=1, le=8192)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    top_k: int = Field(50, ge=0)
    repetition_penalty: float = Field(1.0, ge=0.1, le=2.0)
    do_sample: bool = Field(True)
    model: Optional[str] = Field(None)
    preset: Optional[str] = Field(None)
    stop: Optional[List[str]] = Field(None)


class ChatChoice(BaseModel):
    """A single chat choice."""
    index: int
    message: ChatMessage
    finish_reason: str


class ChatResponse(BaseModel):
    """Response for /chat endpoint."""
    id: str
    model: str
    choices: List[ChatChoice]
    usage: Dict[str, int]


class ModelInfoResponse(BaseModel):
    """Response for model info."""
    model_id: str
    model_path: str
    status: str
    device: str
    dtype: str
    num_parameters: int
    memory_mb: float
    architecture: str
    max_seq_length: int
    quantization: Optional[str] = None


class HealthResponse(BaseModel):
    """Response for /health endpoint."""
    status: str
    message: str
    timestamp: float
    checks: Dict[str, Any]


class ConfigResponse(BaseModel):
    """Response for /config endpoint."""
    active_model: Optional[str]
    loaded_models: List[str]
    supported_models: List[str]
    device: str
    gpu_available: bool


# ── Application Factory ──────────────────────────────────────────────────────


def create_app(model_manager: Optional[ModelManager] = None) -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="Nexus-LLM Backend",
        description="High-performance LLM inference backend with model management, "
                    "streaming, and multi-model support.",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    mm = model_manager or ModelManager()
    engine = InferenceEngine(model_manager=mm)
    health_checker = ServiceHealthCheck(model_manager=mm)
    metrics = BackendMetrics()

    # ── REST Endpoints ───────────────────────────────────────────────────

    @app.post("/generate", response_model=GenerateResponse)
    async def generate(request: GenerateRequest):
        """Generate text from a prompt."""
        try:
            model_id = request.model or mm.get_active_model_id()

            if request.preset:
                config = GenerationPresets.get_preset(request.preset)
                config.max_new_tokens = request.max_new_tokens
                if request.temperature != 1.0:
                    config.temperature = request.temperature
                if request.top_p != 1.0:
                    config.top_p = request.top_p
                if request.repetition_penalty != 1.0:
                    config.repetition_penalty = request.repetition_penalty
                config.do_sample = request.do_sample
            else:
                config = GenerationConfig(
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k,
                    repetition_penalty=request.repetition_penalty,
                    num_beams=request.num_beams,
                    num_return_sequences=request.num_return_sequences,
                    do_sample=request.do_sample,
                    seed=request.seed,
                )

            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: engine.generate(
                    prompt=request.prompt,
                    config=config,
                    model_id=model_id,
                    stop_strings=request.stop,
                )
            )

            metrics.record_request(
                model=model_id or "unknown",
                latency=result.generation_time_seconds,
                tokens=result.num_tokens,
                endpoint="/generate",
            )

            return GenerateResponse(
                id=f"gen-{uuid.uuid4().hex[:8]}",
                text=result.text,
                model=model_id or "unknown",
                prompt_tokens=result.num_prompt_tokens,
                completion_tokens=result.num_tokens,
                total_tokens=result.num_prompt_tokens + result.num_tokens,
                finish_reason=result.finish_reason,
                tokens_per_second=result.tokens_per_second,
                generation_time_seconds=result.generation_time_seconds,
            )

        except Exception as e:
            logger.error(f"Generate error: {e}")
            metrics.errors.inc(labels={"type": "generate", "model": request.model or "unknown"})
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest):
        """Generate a chat completion from messages."""
        try:
            model_id = request.model or mm.get_active_model_id()
            tokenizer = mm.get_tokenizer(model_id)

            messages_dict = [{"role": m.role, "content": m.content} for m in request.messages]
            prompt = tokenizer.apply_chat_template(
                messages_dict, add_generation_prompt=True, tokenize=False
            )
            if not isinstance(prompt, str):
                prompt = tokenizer.decode(prompt)

            if request.preset:
                config = GenerationPresets.get_preset(request.preset)
                config.max_new_tokens = request.max_new_tokens
                config.temperature = request.temperature
                config.top_p = request.top_p
                config.do_sample = request.do_sample
            else:
                config = GenerationConfig(
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k,
                    repetition_penalty=request.repetition_penalty,
                    do_sample=request.do_sample,
                )

            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: engine.generate(
                    prompt=prompt,
                    config=config,
                    model_id=model_id,
                    stop_strings=request.stop,
                )
            )

            metrics.record_request(
                model=model_id or "unknown",
                latency=result.generation_time_seconds,
                tokens=result.num_tokens,
                endpoint="/chat",
            )

            prompt_tokens = result.num_prompt_tokens
            completion_tokens = result.num_tokens

            return ChatResponse(
                id=f"chat-{uuid.uuid4().hex[:8]}",
                model=model_id or "unknown",
                choices=[
                    ChatChoice(
                        index=0,
                        message=ChatMessage(role="assistant", content=result.text),
                        finish_reason=result.finish_reason,
                    )
                ],
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            )

        except Exception as e:
            logger.error(f"Chat error: {e}")
            metrics.errors.inc(labels={"type": "chat", "model": request.model or "unknown"})
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/models")
    async def list_models():
        """List all loaded and available models."""
        loaded = []
        for model_id in mm.list_loaded_models():
            info = mm.get_model_info(model_id)
            if info:
                loaded.append({
                    "model_id": info.model_id,
                    "status": info.status.value,
                    "device": info.device,
                    "num_parameters": info.num_parameters,
                    "memory_mb": info.memory_mb,
                    "architecture": info.architecture,
                })

        supported = ModelRegistry.list_models()

        return {
            "loaded_models": loaded,
            "supported_models": supported,
            "active_model": mm.get_active_model_id(),
        }

    @app.get("/models/{model_id}")
    async def get_model_info(model_id: str):
        """Get detailed info about a specific model."""
        info = mm.get_model_info(model_id)
        if info is None:
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
        return ModelInfoResponse(
            model_id=info.model_id,
            model_path=info.model_path,
            status=info.status.value,
            device=info.device,
            dtype=info.dtype,
            num_parameters=info.num_parameters,
            memory_mb=info.memory_mb,
            architecture=info.architecture,
            max_seq_length=info.max_seq_length,
            quantization=info.quantization,
        )

    @app.post("/models/{model_id}/load")
    async def load_model(model_id: str, model_path: Optional[str] = None, dtype: str = "float16"):
        """Load a model by ID."""
        try:
            mm.load_model(model_id, model_path=model_path, dtype=dtype)
            return {"status": "loaded", "model_id": model_id}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/models/{model_id}/unload")
    async def unload_model(model_id: str):
        """Unload a model by ID."""
        try:
            mm.unload_model(model_id)
            return {"status": "unloaded", "model_id": model_id}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Full health check of all backend components."""
        report = health_checker.check_all()
        return HealthResponse(
            status=report["status"],
            message=report["message"],
            timestamp=report["timestamp"],
            checks=report["checks"],
        )

    @app.get("/health/live")
    async def liveness():
        """Liveness check: is the service running?"""
        return health_checker.liveness_check()

    @app.get("/health/ready")
    async def readiness():
        """Readiness check: is the service ready to serve?"""
        result = health_checker.readiness_check()
        if result["status"] != "ready":
            raise HTTPException(status_code=503, detail=result)
        return result

    @app.get("/config")
    async def get_config():
        """Get current backend configuration."""
        import torch
        return ConfigResponse(
            active_model=mm.get_active_model_id(),
            loaded_models=mm.list_loaded_models(),
            supported_models=ModelRegistry.list_models(),
            device=mm.device,
            gpu_available=torch.cuda.is_available(),
        )

    @app.get("/metrics")
    async def get_metrics(format: str = Query("json", regex="^(json|prometheus)$")):
        """Get backend metrics in JSON or Prometheus format."""
        metrics.update_gpu_metrics()
        if format == "prometheus":
            from fastapi.responses import PlainTextResponse
            return PlainTextResponse(metrics.get_prometheus(), media_type="text/plain")
        return metrics.get_metrics()

    # ── WebSocket Endpoint ───────────────────────────────────────────────

    @app.websocket("/ws/generate")
    async def websocket_generate(websocket: WebSocket):
        """WebSocket endpoint for streaming text generation."""
        await websocket.accept()
        try:
            while True:
                data = await websocket.receive_text()
                request_data = json.loads(data)

                prompt = request_data.get("prompt", "")
                model_id = request_data.get("model")
                max_new_tokens = request_data.get("max_new_tokens", 512)
                temperature = request_data.get("temperature", 1.0)
                top_p = request_data.get("top_p", 1.0)
                do_sample = request_data.get("do_sample", True)

                if not prompt:
                    await websocket.send_json({"error": "prompt is required"})
                    continue

                config = GenerationConfig(
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                )

                await websocket.send_json({"type": "start", "prompt": prompt})

                try:
                    stream = engine.stream_generate(
                        prompt=prompt,
                        config=config,
                        model_id=model_id,
                    )

                    full_text = ""
                    for token_text in stream:
                        full_text += token_text
                        await websocket.send_json({
                            "type": "token",
                            "text": token_text,
                            "accumulated": full_text,
                        })

                    await websocket.send_json({
                        "type": "done",
                        "text": full_text,
                        "finish_reason": "stop",
                    })

                except Exception as e:
                    await websocket.send_json({
                        "type": "error",
                        "message": str(e),
                    })

        except WebSocketDisconnect:
            logger.info("WebSocket client disconnected")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")

    @app.on_event("startup")
    async def startup():
        logger.info("Nexus-LLM backend server starting up")

    @app.on_event("shutdown")
    async def shutdown():
        mm.shutdown()
        logger.info("Nexus-LLM backend server shutting down")

    return app


# Default app instance
app = create_app()
