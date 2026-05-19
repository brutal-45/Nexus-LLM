"""Pydantic schemas: GenerateRequest, ChatRequest, ChatMessage, ModelInfo, ErrorResponse."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator


class FinishReason(str, Enum):
    STOP = "stop"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
    TOOL_CALL = "tool_call"


class ChatRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"


class ChatMessage(BaseModel):
    """A single message in a chat conversation."""
    role: ChatRole
    content: str
    name: Optional[str] = None

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Message content cannot be empty.")
        return v

    model_config = {"json_schema_extra": {
        "examples": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain quantum computing."},
        ]
    }}


class GenerateRequest(BaseModel):
    """Request schema for text generation."""
    prompt: str = Field(..., min_length=1, description="The input prompt for generation.")
    model: Optional[str] = Field(None, description="Model to use for generation.")
    max_new_tokens: int = Field(512, ge=1, le=32768, description="Maximum tokens to generate.")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature.")
    top_k: int = Field(50, ge=0, description="Top-k sampling parameter.")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Top-p (nucleus) sampling parameter.")
    repetition_penalty: float = Field(1.1, ge=1.0, le=2.0, description="Repetition penalty.")
    num_beams: int = Field(1, ge=1, le=8, description="Number of beams for beam search.")
    do_sample: bool = Field(True, description="Whether to use sampling.")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility.")
    stream: bool = Field(False, description="Whether to stream the response.")
    stop: Optional[List[str]] = Field(None, description="Stop sequences.")
    echo: bool = Field(False, description="Whether to echo the prompt in the response.")
    user: Optional[str] = Field(None, description="User identifier for rate limiting.")

    model_config = {"json_schema_extra": {
        "examples": [{
            "prompt": "Explain machine learning in simple terms.",
            "model": "llama-3.1-8b-instruct",
            "max_new_tokens": 256,
            "temperature": 0.7,
        }]
    }}


class ChatRequest(BaseModel):
    """Request schema for chat completion."""
    messages: List[ChatMessage] = Field(..., min_length=1, description="List of chat messages.")
    model: Optional[str] = Field(None, description="Model to use.")
    max_new_tokens: int = Field(1024, ge=1, le=32768)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    repetition_penalty: float = Field(1.1, ge=1.0, le=2.0)
    do_sample: bool = Field(True)
    stream: bool = Field(False)
    conversation_id: Optional[str] = Field(None, description="Conversation ID for multi-turn.")
    system_prompt: Optional[str] = Field(None, description="Override system prompt.")
    seed: Optional[int] = None
    stop: Optional[List[str]] = None
    user: Optional[str] = None

    @field_validator("messages")
    @classmethod
    def messages_not_empty(cls, v: List[ChatMessage]) -> List[ChatMessage]:
        if not v:
            raise ValueError("At least one message is required.")
        return v

    model_config = {"json_schema_extra": {
        "examples": [{
            "messages": [
                {"role": "user", "content": "Hello! How are you?"},
            ],
            "model": "llama-3.1-8b-instruct",
            "temperature": 0.7,
        }]
    }}


class GenerateResponse(BaseModel):
    """Response schema for text generation."""
    id: str = Field(..., description="Unique response ID.")
    text: str = Field(..., description="Generated text.")
    model: str = Field(..., description="Model used for generation.")
    input_tokens: int = Field(0, ge=0)
    output_tokens: int = Field(0, ge=0)
    total_tokens: int = Field(0, ge=0)
    finish_reason: FinishReason = FinishReason.STOP
    generation_time_ms: float = Field(0.0, ge=0.0)
    tokens_per_second: float = Field(0.0, ge=0.0)
    created: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class ChatResponse(BaseModel):
    """Response schema for chat completion."""
    id: str
    message: ChatMessage
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    finish_reason: FinishReason = FinishReason.STOP
    generation_time_ms: float = 0.0
    tokens_per_second: float = 0.0
    conversation_id: Optional[str] = None
    created: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class StreamChunk(BaseModel):
    """Schema for a single streaming chunk."""
    id: str
    chunk: str
    model: str
    finish_reason: Optional[FinishReason] = None
    created: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class ModelInfoResponse(BaseModel):
    """Response schema for model information."""
    name: str
    model_type: str
    description: str = ""
    parameters: int = 0
    size_bytes: int = 0
    size_gb: float = 0.0
    parameters_billions: float = 0.0
    context_length: int = 0
    vocab_size: int = 0
    hidden_size: int = 0
    num_layers: int = 0
    num_heads: int = 0
    quantization: Optional[str] = None
    device: str = "cpu"
    status: str = "unloaded"
    metadata: Dict[str, Any] = {}


class ModelsListResponse(BaseModel):
    """Response schema for listing available models."""
    models: List[ModelInfoResponse]
    total: int


class HealthResponse(BaseModel):
    """Response schema for health check."""
    status: str = "healthy"
    version: str = "1.0.0"
    uptime_seconds: float = 0.0
    loaded_models: int = 0
    gpu_available: bool = False
    gpu_name: Optional[str] = None
    gpu_memory_total_mb: Optional[float] = None
    gpu_memory_used_mb: Optional[float] = None


class ErrorResponse(BaseModel):
    """Standard error response schema."""
    error: str
    error_type: str
    detail: Optional[str] = None
    status_code: int = 500
    request_id: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    model_config = {"json_schema_extra": {
        "examples": [{
            "error": "Model not found",
            "error_type": "ModelNotFoundError",
            "detail": "The requested model 'gpt-5' is not available.",
            "status_code": 404,
        }]
    }}


class ConfigUpdateRequest(BaseModel):
    """Request schema for configuration updates."""
    config: Dict[str, Any] = Field(..., description="Configuration key-value pairs to update.")


class ConfigResponse(BaseModel):
    """Response schema for configuration."""
    config: Dict[str, Any]
    updated: bool = False


class TrainingRequest(BaseModel):
    """Request schema for training/fine-tuning."""
    model: str = Field(..., description="Base model to train.")
    dataset: str = Field(..., description="Dataset identifier or path.")
    method: str = Field("lora", description="Training method: lora, qlora, full.")
    epochs: int = Field(3, ge=1, le=100)
    learning_rate: float = Field(2e-4, gt=0, le=1.0)
    batch_size: int = Field(4, ge=1, le=256)
    lora_rank: int = Field(16, ge=1, le=256)
    lora_alpha: int = Field(32, ge=1)
    max_seq_length: int = Field(2048, ge=128, le=131072)
    warmup_steps: int = Field(100, ge=0)
    weight_decay: float = Field(0.01, ge=0.0, le=1.0)
    output_dir: Optional[str] = None
    user: Optional[str] = None


class TrainingResponse(BaseModel):
    """Response schema for training job submission."""
    job_id: str
    model: str
    status: str = "queued"
    message: str = ""
    created: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
