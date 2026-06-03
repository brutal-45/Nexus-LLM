"""FastAPI Backend Server - Own backend with REST + WebSocket endpoints."""

import json
import logging
import asyncio
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.model_manager import ModelManager
from backend.inference import InferenceEngine
from backend.tokenizer_utils import TokenizerManager

logger = logging.getLogger(__name__)


# ---- Pydantic Models for API ----

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    system_prompt: Optional[str] = None
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    max_new_tokens: int = 512
    stream: bool = False


class GenerateRequest(BaseModel):
    prompt: str
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.2
    max_new_tokens: int = 512
    num_beams: int = 1


class ModelLoadRequest(BaseModel):
    model_name: str
    model_type: str = "causal"
    device: str = "auto"
    precision: str = "fp32"


class ConfigUpdateRequest(BaseModel):
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_new_tokens: Optional[int] = None
    system_prompt: Optional[str] = None


# ---- FastAPI App ----

class LLMServer:
    """
    Own backend server for the Nexus-LLM application.
    Provides REST API and WebSocket endpoints for model interaction.
    """

    def __init__(
        self,
        model_name: str = "gpt2-medium",
        model_type: str = "causal",
        device: str = "auto",
        precision: str = "fp32",
        host: str = "127.0.0.1",
        port: int = 8765,
        cors_origins: List[str] = None,
    ):
        self.host = host
        self.port = port

        # Initialize model components
        self.model_manager = ModelManager(
            model_name=model_name,
            model_type=model_type,
            device=device,
            precision=precision,
        )
        self.tokenizer_manager: Optional[TokenizerManager] = None
        self.inference_engine: Optional[InferenceEngine] = None

        # Create FastAPI app
        self.app = FastAPI(
            title="Nexus-LLM Backend",
            description="Own backend API for Nexus-LLM - No external cloud services",
            version="1.0.0",
        )

        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins or ["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Register routes
        self._register_routes()

    def _ensure_initialized(self):
        """Lazy initialization of inference engine."""
        if self.inference_engine is None:
            self.model_manager.load_model()
            self.tokenizer_manager = TokenizerManager(
                self.model_manager.tokenizer,
                self.model_manager.model_name,
            )
            self.inference_engine = InferenceEngine(
                self.model_manager, self.tokenizer_manager
            )

    def _register_routes(self):
        """Register all API routes."""

        @self.app.on_event("startup")
        async def startup():
            """Pre-load model on startup."""
            logger.info("Starting LLM Backend Server...")
            self._ensure_initialized()
            logger.info("Model loaded and ready for inference.")

        # ---- Health & Info ----

        @self.app.get("/")
        async def root():
            """Root endpoint - server status."""
            return {
                "name": "Nexus-LLM Backend",
                "version": "1.0.0",
                "status": "running",
                "model_loaded": self.model_manager.is_loaded,
            }

        @self.app.get("/health")
        async def health():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "model_loaded": self.model_manager.is_loaded,
                "is_generating": (
                    self.inference_engine.is_generating
                    if self.inference_engine
                    else False
                ),
            }

        @self.app.get("/model/info")
        async def model_info():
            """Get information about the loaded model."""
            if not self.model_manager.is_loaded:
                raise HTTPException(status_code=503, detail="Model not loaded")
            return self.model_manager.model_info

        @self.app.get("/model/memory")
        async def model_memory():
            """Get memory usage information."""
            return self.model_manager.get_memory_usage()

        # ---- Chat Endpoints ----

        @self.app.post("/chat")
        async def chat(request: ChatRequest):
            """Send a chat message and get a response."""
            self._ensure_initialized()

            messages = [{"role": m.role, "content": m.content} for m in request.messages]

            result = self.inference_engine.chat(
                messages=messages,
                system_prompt=request.system_prompt,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                max_new_tokens=request.max_new_tokens,
            )

            return {
                "response": result["text"],
                "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"],
                "generation_time": result["generation_time"],
                "tokens_per_second": result["tokens_per_second"],
            }

        # ---- Generation Endpoints ----

        @self.app.post("/generate")
        async def generate(request: GenerateRequest):
            """Generate text from a prompt."""
            self._ensure_initialized()

            result = self.inference_engine.generate(
                prompt=request.prompt,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                repetition_penalty=request.repetition_penalty,
                max_new_tokens=request.max_new_tokens,
                num_beams=request.num_beams,
            )

            return {
                "generated_text": result["text"],
                "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"],
                "generation_time": result["generation_time"],
                "tokens_per_second": result["tokens_per_second"],
            }

        # ---- WebSocket for Streaming ----

        @self.app.websocket("/ws/chat")
        async def websocket_chat(websocket: WebSocket):
            """WebSocket endpoint for streaming chat."""
            await websocket.accept()
            logger.info("WebSocket client connected")

            try:
                while True:
                    data = await websocket.receive_text()
                    request = json.loads(data)

                    self._ensure_initialized()

                    messages = request.get("messages", [])
                    system_prompt = request.get("system_prompt")
                    temperature = request.get("temperature", 0.7)
                    top_p = request.get("top_p", 0.9)
                    top_k = request.get("top_k", 50)
                    max_new_tokens = request.get("max_new_tokens", 512)

                    # Stream the response
                    full_response = ""
                    token_count = 0

                    for token in self.inference_engine.chat_stream(
                        messages=messages,
                        system_prompt=system_prompt,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        max_new_tokens=max_new_tokens,
                    ):
                        full_response += token
                        token_count += 1

                        await websocket.send_json({
                            "type": "token",
                            "content": token,
                        })

                    # Send completion message
                    await websocket.send_json({
                        "type": "done",
                        "full_response": full_response,
                        "token_count": token_count,
                    })

            except WebSocketDisconnect:
                logger.info("WebSocket client disconnected")
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                try:
                    await websocket.send_json({
                        "type": "error",
                        "message": str(e),
                    })
                except Exception:
                    pass

        # ---- Model Management ----

        @self.app.post("/model/load")
        async def load_model(request: ModelLoadRequest):
            """Load a different model."""
            try:
                self.model_manager.unload_model()
                self.model_manager.model_name = request.model_name
                self.model_manager.model_type = request.model_type
                self.model_manager.device = self.model_manager._resolve_device(request.device)
                self.model_manager.precision = request.precision

                self._ensure_initialized()

                return {
                    "status": "loaded",
                    "model_info": self.model_manager.model_info,
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/model/unload")
        async def unload_model():
            """Unload the current model."""
            self.model_manager.unload_model()
            self.inference_engine = None
            self.tokenizer_manager = None
            return {"status": "unloaded"}

        @self.app.post("/model/reload")
        async def reload_model():
            """Reload the current model."""
            try:
                self.model_manager.reload_model()
                self._ensure_initialized()
                return {"status": "reloaded", "model_info": self.model_manager.model_info}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        # ---- Inference Stats ----

        @self.app.get("/stats")
        async def get_stats():
            """Get inference statistics."""
            if self.inference_engine:
                return self.inference_engine.stats
            return {"total_generations": 0, "total_tokens_generated": 0}

        @self.app.post("/generate/stop")
        async def stop_generation():
            """Stop the current generation."""
            if self.inference_engine:
                self.inference_engine.stop_generation()
            return {"status": "stopped"}

    def run(self):
        """Start the backend server."""
        import uvicorn
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info",
        )


def create_app(
    model_name: str = "gpt2",
    model_type: str = "causal",
    device: str = "auto",
    precision: str = "fp32",
    host: str = "127.0.0.1",
    port: int = 8765,
) -> LLMServer:
    """Factory function to create the LLM server."""
    return LLMServer(
        model_name=model_name,
        model_type=model_type,
        device=device,
        precision=precision,
        host=host,
        port=port,
    )
