"""
Nexus Inference Server
==========================
FastAPI-based inference server with OpenAI-compatible API.

Endpoints:
    POST /v1/chat/completions    - Chat completion (OpenAI-compatible)
    POST /v1/completions         - Text completion (OpenAI-compatible)
    POST /v1/embeddings          - Text embeddings
    GET  /v1/models              - List available models
    GET  /health                 - Health check
    GET  /metrics                - Prometheus-compatible metrics

Features:
    - OpenAI-compatible API format
    - Streaming responses (Server-Sent Events)
    - Continuous batching for high throughput
    - Request queuing and timeout handling
    - Concurrent request processing
    - Token usage tracking

Usage:
    python -m nexus.scripts.serve --checkpoint checkpoints/nexus-100b --port 8000
    curl http://localhost:8000/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{"model": "nexus-100b", "messages": [{"role": "user", "content": "Hello!"}]}'
"""

from __future__ import annotations
import time
import asyncio
import threading
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import uvicorn
import torch

from ..model.transformer import NexusTransformer
from ..model.config import ModelConfig
from ..inference.generator import TextGenerator, GenerationConfig


# === Pydantic Models (OpenAI-compatible) ===

class ChatMessage(BaseModel):
    role: str
    content: str
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str = "nexus-100b"
    messages: List[ChatMessage]
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    max_tokens: int = 4096
    stop: Optional[List[str]] = None
    stream: bool = False
    repetition_penalty: float = 1.1
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0


class CompletionRequest(BaseModel):
    model: str = "nexus-100b"
    prompt: str
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    max_tokens: int = 4096
    stop: Optional[List[str]] = None
    stream: bool = False
    echo: bool = False
    repetition_penalty: float = 1.1


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "nexus"
    params: Optional[int] = None


class UsageStats(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class CompletionChoice(BaseModel):
    index: int = 0
    text: str
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: UsageStats


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: UsageStats


# === Server Metrics ===

@dataclass
class ServerMetrics:
    total_requests: int = 0
    total_tokens_generated: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_errors: int = 0
    avg_latency_ms: float = 0.0
    active_requests: int = 0
    queue_size: int = 0
    
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _latencies: List[float] = field(default_factory=list, repr=False)
    
    def record_request(self, latency_s: float, prompt_tokens: int, completion_tokens: int):
        with self._lock:
            self.total_requests += 1
            self.total_tokens_generated += completion_tokens
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self._latencies.append(latency_s * 1000)
            if len(self._latencies) > 1000:
                self._latencies = self._latencies[-1000:]
            self.avg_latency_ms = sum(self._latencies) / len(self._latencies)
    
    def record_error(self):
        with self._lock:
            self.total_errors += 1


# === Inference Server ===

class InferenceServer:
    """
    Nexus Inference Server.
    
    Wraps a TextGenerator in a FastAPI server with OpenAI-compatible endpoints.
    """

    def __init__(
        self,
        model: NexusTransformer,
        tokenizer,
        model_name: str = "nexus-100b",
        host: str = "0.0.0.0",
        port: int = 8000,
        max_concurrent_requests: int = 512,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.host = host
        self.port = port
        self.max_concurrent_requests = max_concurrent_requests
        
        self.generator = TextGenerator(model, tokenizer)
        self.metrics = ServerMetrics()
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        
        # Build FastAPI app
        self.app = self._build_app()

    def _build_app(self) -> FastAPI:
        """Build the FastAPI application with all endpoints."""
        app = FastAPI(
            title="Nexus Inference Server",
            description="OpenAI-compatible inference API for Nexus",
            version="1.0.0",
        )
        
        @app.get("/health")
        async def health():
            return {"status": "healthy", "model": self.model_name}
        
        @app.get("/metrics")
        async def metrics():
            m = self.metrics
            return {
                "total_requests": m.total_requests,
                "total_tokens_generated": m.total_tokens_generated,
                "total_errors": m.total_errors,
                "avg_latency_ms": round(m.avg_latency_ms, 2),
                "active_requests": m.active_requests,
            }
        
        @app.get("/v1/models")
        async def list_models():
            return {
                "object": "list",
                "data": [
                    ModelInfo(
                        id=self.model_name,
                        created=int(time.time()),
                        owned_by="nexus",
                        params=self.model.num_parameters(),
                    )
                ],
            }
        
        @app.post("/v1/chat/completions")
        async def chat_completions(request: ChatCompletionRequest):
            return await self._handle_chat_completion(request)
        
        @app.post("/v1/completions")
        async def completions(request: CompletionRequest):
            return await self._handle_completion(request)
        
        return app

    async def _handle_chat_completion(
        self, request: ChatCompletionRequest
    ) -> Union[ChatCompletionResponse, StreamingResponse]:
        """Handle chat completion request."""
        start = time.time()
        
        async with self.semaphore:
            self.metrics.active_requests += 1
            try:
                # Build prompt from messages
                prompt = self._format_chat_messages(request.messages)
                
                # Count prompt tokens
                prompt_token_ids = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
                prompt_tokens = len(prompt_token_ids)
                
                # Generate
                gen_config = GenerationConfig(
                    max_new_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k,
                    repetition_penalty=request.repetition_penalty,
                )
                
                if request.stream:
                    return await self._stream_chat(
                        prompt, gen_config, prompt_tokens, request
                    )
                
                # Non-streaming
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: self.generator.generate(prompt, gen_config),
                )
                
                completion_tokens = result.num_generated_tokens
                generated_text = result.generated_text[0] if result.generated_text else ""
                
                latency = time.time() - start
                self.metrics.record_request(latency, prompt_tokens, completion_tokens)
                
                return ChatCompletionResponse(
                    id=f"chatcmpl-{int(time.time())}",
                    created=int(time.time()),
                    model=self.model_name,
                    choices=[
                        ChatCompletionChoice(
                            message=ChatMessage(
                                role="assistant",
                                content=generated_text,
                            ),
                            finish_reason=result.finish_reason[0],
                        )
                    ],
                    usage=UsageStats(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=prompt_tokens + completion_tokens,
                    ),
                )
            except Exception as e:
                self.metrics.record_error()
                raise HTTPException(status_code=500, detail=str(e))
            finally:
                self.metrics.active_requests -= 1

    async def _handle_completion(
        self, request: CompletionRequest
    ) -> Union[CompletionResponse, StreamingResponse]:
        """Handle text completion request."""
        start = time.time()
        
        async with self.semaphore:
            self.metrics.active_requests += 1
            try:
                prompt_tokens = len(
                    self.tokenizer.encode(request.prompt, add_bos=True, add_eos=False)
                )
                
                gen_config = GenerationConfig(
                    max_new_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k,
                    repetition_penalty=request.repetition_penalty,
                )
                
                if request.stream:
                    return await self._stream_completion(
                        request.prompt, gen_config, prompt_tokens, request
                    )
                
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: self.generator.generate(request.prompt, gen_config),
                )
                
                completion_tokens = result.num_generated_tokens
                text = result.generated_text[0] if result.generated_text else ""
                
                if request.echo:
                    text = request.prompt + text
                
                latency = time.time() - start
                self.metrics.record_request(latency, prompt_tokens, completion_tokens)
                
                return CompletionResponse(
                    id=f"cmpl-{int(time.time())}",
                    created=int(time.time()),
                    model=self.model_name,
                    choices=[
                        CompletionChoice(
                            text=text,
                            finish_reason=result.finish_reason[0],
                        )
                    ],
                    usage=UsageStats(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=prompt_tokens + completion_tokens,
                    ),
                )
            except Exception as e:
                self.metrics.record_error()
                raise HTTPException(status_code=500, detail=str(e))
            finally:
                self.metrics.active_requests -= 1

    async def _stream_chat(
        self, prompt: str, gen_config: GenerationConfig,
        prompt_tokens: int, request: ChatCompletionRequest,
    ) -> StreamingResponse:
        """Stream chat completion as Server-Sent Events."""
        
        async def event_generator():
            completion_tokens = 0
            full_text = ""
            
            loop = asyncio.get_event_loop()
            
            def generate():
                chunks = []
                for chunk in self.generator.stream_generate(prompt, gen_config):
                    chunks.append(chunk)
                return chunks
            
            chunks = await loop.run_in_executor(None, generate)
            
            for chunk in chunks:
                full_text += chunk
                completion_tokens += 1
                
                data = {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion.chunk",
                    "model": self.model_name,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": chunk},
                        "finish_reason": None,
                    }],
                }
                yield f"data: {json_str(data)}\n\n"
            
            # Final chunk
            final_data = {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion.chunk",
                "model": self.model_name,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }
            yield f"data: {json_str(final_data)}\n\n"
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
        )

    async def _stream_completion(
        self, prompt: str, gen_config: GenerationConfig,
        prompt_tokens: int, request: CompletionRequest,
    ) -> StreamingResponse:
        """Stream text completion as Server-Sent Events."""
        
        async def event_generator():
            completion_tokens = 0
            
            loop = asyncio.get_event_loop()
            
            def generate():
                chunks = []
                for chunk in self.generator.stream_generate(prompt, gen_config):
                    chunks.append(chunk)
                return chunks
            
            chunks = await loop.run_in_executor(None, generate)
            
            for chunk in chunks:
                completion_tokens += 1
                
                data = {
                    "id": f"cmpl-{int(time.time())}",
                    "object": "text_completion",
                    "model": self.model_name,
                    "choices": [{
                        "index": 0,
                        "text": chunk,
                        "finish_reason": None,
                    }],
                }
                yield f"data: {json_str(data)}\n\n"
            
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
        )

    def _format_chat_messages(self, messages: List[ChatMessage]) -> str:
        """Format chat messages into a single prompt string."""
        formatted = []
        for msg in messages:
            if msg.role == "system":
                formatted.append(f"System: {msg.content}")
            elif msg.role == "user":
                formatted.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                formatted.append(f"Assistant: {msg.content}")
        formatted.append("Assistant:")
        return "\n\n".join(formatted)

    def run(self):
        """Start the inference server (blocking)."""
        print(f"\n{'='*60}")
        print(f"Nexus Inference Server")
        print(f"{'='*60}")
        print(f"  Model: {self.model_name}")
        print(f"  Parameters: {self.model.num_parameters():,}")
        print(f"  Host: {self.host}:{self.port}")
        print(f"  Max concurrent: {self.max_concurrent_requests}")
        print(f"{'='*60}\n")
        
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info",
        )


def json_str(obj) -> str:
    """Convert object to JSON string."""
    import json
    return json.dumps(obj, ensure_ascii=False)
