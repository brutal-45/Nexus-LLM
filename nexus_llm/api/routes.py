"""REST API routes: /v1/generate, /v1/chat, /v1/models, /v1/health, /v1/config, /v1/training."""

import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from nexus_llm.api.auth import AuthManager, APIKey, get_auth_manager
from nexus_llm.api.errors import (
    ContentFilterError,
    GenerationError,
    ModelNotFoundError,
    NexusAPIError,
)
from nexus_llm.api.rate_limit import RateLimiter, get_rate_limiter
from nexus_llm.api.schemas import (
    ChatRequest,
    ChatResponse,
    ConfigResponse,
    ConfigUpdateRequest,
    ErrorResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    ModelInfoResponse,
    ModelsListResponse,
    StreamChunk,
    TrainingRequest,
    TrainingResponse,
    ChatMessage,
    FinishReason,
)

logger = logging.getLogger("nexus_llm.api.routes")

router = APIRouter()

# Application start time for uptime tracking
_start_time = time.time()

# In-memory model manager reference (set during app initialization)
_model_manager: Optional[Any] = None
_safety_manager: Optional[Any] = None
_training_jobs: Dict[str, Dict[str, Any]] = {}


def set_model_manager(manager: Any) -> None:
    """Set the model manager for route handlers."""
    global _model_manager
    _model_manager = manager


def set_safety_manager(manager: Any) -> None:
    """Set the safety manager for route handlers."""
    global _safety_manager
    _safety_manager = manager


def _get_model(model_name: Optional[str]) -> Any:
    """Resolve and return a model instance by name.

    Args:
        model_name: Model identifier. Uses default if None.

    Returns:
        Model instance.

    Raises:
        ModelNotFoundError: If the model is not available.
    """
    if _model_manager is None:
        raise ModelNotFoundError("default", reason="Model manager not initialized")

    name = model_name or _model_manager.get_default_model()
    model = _model_manager.get_model(name)
    if model is None:
        raise ModelNotFoundError(name)
    return model


def _check_safety(text: str, request_id: str) -> str:
    """Run safety checks on input or output text.

    Args:
        text: Text to check.
        request_id: Request ID for error reporting.

    Returns:
        The original text if safe.

    Raises:
        ContentFilterError: If content is flagged.
    """
    if _safety_manager is None:
        return text

    result = _safety_manager.check_input(text)
    if not result.is_safe:
        raise ContentFilterError(
            reason=result.reason or "Content blocked by safety filter",
            request_id=request_id,
        )
    return text


@router.post(
    "/v1/generate",
    response_model=GenerateResponse,
    responses={
        200: {"description": "Successful generation"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        404: {"model": ErrorResponse, "description": "Model not found"},
        422: {"model": ErrorResponse, "description": "Content filtered"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        500: {"model": ErrorResponse, "description": "Generation error"},
    },
    tags=["Generation"],
)
async def generate(
    request: GenerateRequest,
    auth_key: APIKey = Depends(lambda: get_auth_manager().authenticate_request),
) -> GenerateResponse:
    """Generate text from a prompt using a loaded language model.

    Supports configurable generation parameters including temperature,
    top-k, top-p, and beam search. Optionally streams the response.
    """
    request_id = str(uuid.uuid4())
    rate_limiter = get_rate_limiter()
    rate_limiter.enforce_rate(auth_key.key_hash, tokens=len(request.prompt.split()))

    _check_safety(request.prompt, request_id)

    model = _get_model(request.model)

    from nexus_llm.models.base import GenerationConfig

    gen_config = GenerationConfig(
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        top_k=request.top_k,
        top_p=request.top_p,
        repetition_penalty=request.repetition_penalty,
        num_beams=request.num_beams,
        do_sample=request.do_sample,
        seed=request.seed,
    )

    try:
        result = model.generate(request.prompt, config=gen_config)
    except Exception as e:
        raise GenerationError(reason=str(e), request_id=request_id)

    _check_safety(result.text, request_id)

    return GenerateResponse(
        id=f"gen-{request_id[:8]}",
        text=result.text,
        model=result.model_name,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        total_tokens=result.total_tokens,
        finish_reason=FinishReason(result.finish_reason),
        generation_time_ms=result.generation_time_ms,
        tokens_per_second=result.tokens_per_second,
    )


@router.post(
    "/v1/chat",
    response_model=ChatResponse,
    responses={
        200: {"description": "Successful chat completion"},
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
    },
    tags=["Chat"],
)
async def chat(
    request: ChatRequest,
    auth_key: APIKey = Depends(lambda: get_auth_manager().authenticate_request),
) -> ChatResponse:
    """Generate a chat completion response for a multi-turn conversation.

    Accepts a list of messages with roles (system, user, assistant)
    and returns the assistant's response.
    """
    request_id = str(uuid.uuid4())
    rate_limiter = get_rate_limiter()

    total_tokens_estimate = sum(len(m.content.split()) for m in request.messages)
    rate_limiter.enforce_rate(auth_key.key_hash, tokens=total_tokens_estimate)

    for msg in request.messages:
        if msg.role in ("user", "system"):
            _check_safety(msg.content, request_id)

    model = _get_model(request.model)

    from nexus_llm.models.base import GenerationConfig

    gen_config = GenerationConfig(
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        repetition_penalty=request.repetition_penalty,
        do_sample=request.do_sample,
        seed=request.seed,
    )

    try:
        if hasattr(model, "chat") and request.conversation_id:
            result = model.chat(
                message=request.messages[-1].content,
                conversation_id=request.conversation_id,
                system_prompt=request.system_prompt,
                config=gen_config,
            )
        elif hasattr(model, "chat_single"):
            messages_dicts = [m.model_dump() for m in request.messages]
            result = model.chat_single(messages=messages_dicts, config=gen_config)
        else:
            prompt_parts = []
            for msg in request.messages:
                prompt_parts.append(f"{msg.role.value}: {msg.content}")
            prompt = "\n".join(prompt_parts) + "\nassistant: "
            result = model.generate(prompt, config=gen_config)
    except Exception as e:
        raise GenerationError(reason=str(e), request_id=request_id)

    _check_safety(result.text, request_id)

    return ChatResponse(
        id=f"chat-{request_id[:8]}",
        message=ChatMessage(role="assistant", content=result.text),
        model=result.model_name,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        total_tokens=result.total_tokens,
        finish_reason=FinishReason(result.finish_reason),
        generation_time_ms=result.generation_time_ms,
        tokens_per_second=result.tokens_per_second,
        conversation_id=request.conversation_id,
    )


@router.get(
    "/v1/models",
    response_model=ModelsListResponse,
    tags=["Models"],
)
async def list_models(
    auth_key: APIKey = Depends(lambda: get_auth_manager().authenticate_request),
) -> ModelsListResponse:
    """List all available models and their status."""
    if _model_manager is None:
        return ModelsListResponse(models=[], total=0)

    models = _model_manager.list_models()
    model_responses = []

    for model in models:
        info = model.get_info()
        model_responses.append(ModelInfoResponse(
            name=info.name,
            model_type=info.model_type.value,
            description=info.description,
            parameters=info.parameters,
            size_bytes=info.size_bytes,
            size_gb=info.size_gb,
            parameters_billions=info.parameters_billions,
            context_length=info.context_length,
            vocab_size=info.vocab_size,
            hidden_size=info.hidden_size,
            num_layers=info.num_layers,
            num_heads=info.num_heads,
            quantization=info.quantization,
            device=info.device,
            status=info.status.value,
            metadata=info.metadata,
        ))

    return ModelsListResponse(models=model_responses, total=len(model_responses))


@router.get(
    "/v1/models/{model_name}",
    response_model=ModelInfoResponse,
    responses={404: {"model": ErrorResponse}},
    tags=["Models"],
)
async def get_model_info(
    model_name: str,
    auth_key: APIKey = Depends(lambda: get_auth_manager().authenticate_request),
) -> ModelInfoResponse:
    """Get detailed information about a specific model."""
    model = _get_model(model_name)
    info = model.get_info()

    return ModelInfoResponse(
        name=info.name,
        model_type=info.model_type.value,
        description=info.description,
        parameters=info.parameters,
        size_bytes=info.size_bytes,
        size_gb=info.size_gb,
        parameters_billions=info.parameters_billions,
        context_length=info.context_length,
        vocab_size=info.vocab_size,
        hidden_size=info.hidden_size,
        num_layers=info.num_layers,
        num_heads=info.num_heads,
        quantization=info.quantization,
        device=info.device,
        status=info.status.value,
        metadata=info.metadata,
    )


@router.get(
    "/v1/health",
    response_model=HealthResponse,
    tags=["Health"],
)
async def health_check() -> HealthResponse:
    """Check the health and status of the API server."""
    import torch

    uptime = time.time() - _start_time
    loaded_models = 0
    if _model_manager:
        loaded_models = len(_model_manager.list_loaded_models())

    gpu_available = torch.cuda.is_available()
    gpu_name = None
    gpu_memory_total = None
    gpu_memory_used = None

    if gpu_available:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory_total = torch.cuda.get_device_properties(0).total_mem / (1024 * 1024)
        gpu_memory_used = torch.cuda.memory_allocated(0) / (1024 * 1024)

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        uptime_seconds=round(uptime, 1),
        loaded_models=loaded_models,
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        gpu_memory_total_mb=round(gpu_memory_total, 2) if gpu_memory_total else None,
        gpu_memory_used_mb=round(gpu_memory_used, 2) if gpu_memory_used else None,
    )


@router.get(
    "/v1/config",
    response_model=ConfigResponse,
    tags=["Configuration"],
)
async def get_config(
    auth_key: APIKey = Depends(lambda: get_auth_manager().authenticate_request),
) -> ConfigResponse:
    """Get the current runtime configuration."""
    config_data = {}
    if _model_manager:
        config_data["models"] = _model_manager.get_config()
    if _safety_manager:
        config_data["safety"] = _safety_manager.get_config()
    rate_limiter = get_rate_limiter()
    config_data["rate_limit"] = rate_limiter.config.to_dict() if hasattr(rate_limiter.config, "to_dict") else {}
    return ConfigResponse(config=config_data)


@router.post(
    "/v1/config",
    response_model=ConfigResponse,
    tags=["Configuration"],
)
async def update_config(
    request: ConfigUpdateRequest,
    auth_key: APIKey = Depends(lambda: get_auth_manager().authenticate_request),
) -> ConfigResponse:
    """Update runtime configuration parameters."""
    auth_mgr = get_auth_manager()
    if not auth_mgr.is_admin(auth_key):
        raise HTTPException(status_code=403, detail="Admin access required for configuration changes.")

    updated = False
    for key, value in request.config.items():
        if key == "safety" and _safety_manager:
            _safety_manager.update_config(value)
            updated = True
        elif key == "rate_limit":
            from nexus_llm.api.rate_limit import RateLimitConfig, init_rate_limiter
            new_config = RateLimitConfig(**value) if isinstance(value, dict) else RateLimitConfig()
            init_rate_limiter(new_config)
            updated = True

    config_data = {}
    if _model_manager:
        config_data["models"] = _model_manager.get_config()
    if _safety_manager:
        config_data["safety"] = _safety_manager.get_config()

    return ConfigResponse(config=config_data, updated=updated)


@router.post(
    "/v1/training",
    response_model=TrainingResponse,
    responses={
        200: {"description": "Training job submitted"},
        400: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
    },
    tags=["Training"],
)
async def create_training_job(
    request: TrainingRequest,
    auth_key: APIKey = Depends(lambda: get_auth_manager().authenticate_request),
) -> TrainingResponse:
    """Submit a fine-tuning training job.

    Supports LoRA, QLoRA, and full fine-tuning methods.
    Training runs asynchronously; use the job_id to check status.
    """
    job_id = f"train-{uuid.uuid4().hex[:8]}"

    _training_jobs[job_id] = {
        "job_id": job_id,
        "model": request.model,
        "dataset": request.dataset,
        "method": request.method,
        "status": "queued",
        "config": request.model_dump(),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "submitted_by": auth_key.name,
    }

    logger.info(
        "Training job submitted: %s (model=%s, method=%s, user=%s)",
        job_id, request.model, request.method, auth_key.name,
    )

    return TrainingResponse(
        job_id=job_id,
        model=request.model,
        status="queued",
        message=f"Training job {job_id} has been queued. Use GET /v1/training/{job_id} to check status.",
    )


@router.get(
    "/v1/training/{job_id}",
    tags=["Training"],
)
async def get_training_status(
    job_id: str,
    auth_key: APIKey = Depends(lambda: get_auth_manager().authenticate_request),
) -> Dict[str, Any]:
    """Get the status of a training job."""
    if job_id not in _training_jobs:
        raise HTTPException(status_code=404, detail=f"Training job '{job_id}' not found.")
    return _training_jobs[job_id]


@router.get(
    "/v1/training",
    tags=["Training"],
)
async def list_training_jobs(
    auth_key: APIKey = Depends(lambda: get_auth_manager().authenticate_request),
) -> List[Dict[str, Any]]:
    """List all training jobs."""
    return list(_training_jobs.values())
