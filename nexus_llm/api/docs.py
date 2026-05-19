"""API documentation: OpenAPI schema customization, example responses."""

from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi


API_TITLE = "Nexus-LLM API"
API_VERSION = "1.0.0"
API_DESCRIPTION = """
# Nexus-LLM API

High-performance API for large language model inference, chat, code generation,
and fine-tuning. Compatible with OpenAI API format.

## Features

- **Text Generation**: Generate text from prompts using various LLM models
- **Chat Completions**: Multi-turn conversation with context management
- **Code Completion**: Language-aware code generation and FIM
- **Model Management**: List, load, unload, and benchmark models
- **Streaming**: Real-time token streaming via SSE and WebSocket
- **Training**: Submit fine-tuning jobs with LoRA, QLoRA, or full training
- **Safety**: Built-in content filtering and toxicity detection

## Authentication

All endpoints require an API key via the `X-API-Key` header or
a Bearer token in the `Authorization` header.

## Rate Limits

Default rate limits: 60 requests/minute, 100K tokens/minute.
Higher limits available on request.
"""

API_CONTACT = {
    "name": "Nexus-LLM Support",
    "url": "https://github.com/nexus-llm/nexus-llm",
    "email": "support@nexus-llm.dev",
}

API_LICENSE = {
    "name": "Apache 2.0",
    "url": "https://www.apache.org/licenses/LICENSE-2.0",
}

API_SERVERS = [
    {"url": "http://localhost:8000", "description": "Development server"},
    {"url": "https://api.nexus-llm.dev", "description": "Production server"},
]

API_TAGS_METADATA = [
    {
        "name": "Generation",
        "description": "Text generation endpoints for prompt-based completion.",
    },
    {
        "name": "Chat",
        "description": "Chat completion endpoints with multi-turn conversation support.",
    },
    {
        "name": "Models",
        "description": "Model management: list, load, unload, and get model information.",
    },
    {
        "name": "Health",
        "description": "Health check and system status endpoints.",
    },
    {
        "name": "Configuration",
        "description": "Runtime configuration management.",
    },
    {
        "name": "Training",
        "description": "Fine-tuning and training job management.",
    },
]

# Example responses for OpenAPI documentation
EXAMPLE_GENERATE_REQUEST = {
    "prompt": "Explain quantum computing in simple terms.",
    "model": "llama-3.1-8b-instruct",
    "max_new_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True,
}

EXAMPLE_GENERATE_RESPONSE = {
    "id": "gen-abc123",
    "text": "Quantum computing is a type of computing that uses quantum-mechanical phenomena, such as superposition and entanglement, to perform operations on data. Unlike classical computers that use bits (0 or 1), quantum computers use qubits which can exist in multiple states simultaneously.",
    "model": "llama-3.1-8b-instruct",
    "input_tokens": 8,
    "output_tokens": 42,
    "total_tokens": 50,
    "finish_reason": "stop",
    "generation_time_ms": 1523.45,
    "tokens_per_second": 27.57,
    "created": "2024-01-15T10:30:00",
}

EXAMPLE_CHAT_REQUEST = {
    "messages": [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "Write a Python function to reverse a linked list."},
    ],
    "model": "llama-3.1-8b-instruct",
    "max_new_tokens": 1024,
    "temperature": 0.7,
}

EXAMPLE_CHAT_RESPONSE = {
    "id": "chat-xyz789",
    "message": {
        "role": "assistant",
        "content": "Here's a Python function to reverse a linked list:\n\n```python\nclass ListNode:\n    def __init__(self, val=0, next=None):\n        self.val = val\n        self.next = next\n\ndef reverse_linked_list(head):\n    prev = None\n    current = head\n    while current:\n        next_node = current.next\n        current.next = prev\n        prev = current\n        current = next_node\n    return prev\n```",
    },
    "model": "llama-3.1-8b-instruct",
    "input_tokens": 22,
    "output_tokens": 95,
    "total_tokens": 117,
    "finish_reason": "stop",
    "generation_time_ms": 2341.67,
    "tokens_per_second": 40.56,
    "conversation_id": "conv-123",
    "created": "2024-01-15T10:31:00",
}

EXAMPLE_ERROR_RESPONSE = {
    "error": "Model not found",
    "error_type": "ModelNotFoundError",
    "detail": "The requested model 'gpt-5' is not available.",
    "status_code": 404,
    "request_id": "req-456",
    "timestamp": "2024-01-15T10:32:00",
}

EXAMPLE_HEALTH_RESPONSE = {
    "status": "healthy",
    "version": "1.0.0",
    "uptime_seconds": 86400.0,
    "loaded_models": 2,
    "gpu_available": True,
    "gpu_name": "NVIDIA A100 80GB",
    "gpu_memory_total_mb": 81920.0,
    "gpu_memory_used_mb": 45678.0,
}

EXAMPLE_MODELS_LIST_RESPONSE = {
    "models": [
        {
            "name": "llama-3.1-8b-instruct",
            "model_type": "chat",
            "description": "LLaMA 3.1 8B Instruct",
            "parameters": 8000000000,
            "size_gb": 14.96,
            "status": "loaded",
            "device": "cuda:0",
        }
    ],
    "total": 1,
}


def custom_openapi(app: FastAPI) -> Dict[str, Any]:
    """Generate a customized OpenAPI schema for the Nexus-LLM API.

    Args:
        app: FastAPI application instance.

    Returns:
        OpenAPI schema dictionary.
    """
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=API_TITLE,
        version=API_VERSION,
        description=API_DESCRIPTION,
        routes=app.routes,
        tags=API_TAGS_METADATA,
    )

    openapi_schema["info"]["contact"] = API_CONTACT
    openapi_schema["info"]["license"] = API_LICENSE
    openapi_schema["servers"] = API_SERVERS

    security_scheme = {
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "API key for authentication",
        },
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "Bearer token for authentication",
        },
    }
    openapi_schema.setdefault("components", {}).setdefault("securitySchemes", {}).update(
        security_scheme
    )

    openapi_schema["security"] = [{"ApiKeyAuth": []}, {"BearerAuth": []}]

    schemas = openapi_schema.get("components", {}).get("schemas", {})

    if "GenerateRequest" in schemas:
        schemas["GenerateRequest"]["example"] = EXAMPLE_GENERATE_REQUEST

    if "ChatRequest" in schemas:
        schemas["ChatRequest"]["example"] = EXAMPLE_CHAT_REQUEST

    if "ErrorResponse" in schemas:
        schemas["ErrorResponse"]["example"] = EXAMPLE_ERROR_RESPONSE

    for path, path_item in openapi_schema.get("paths", {}).items():
        for method, operation in path_item.items():
            if not isinstance(operation, dict):
                continue

            if "/v1/generate" in path and method == "post":
                operation.setdefault("responses", {})["200"] = {
                    "description": "Successful generation",
                    "content": {
                        "application/json": {
                            "example": EXAMPLE_GENERATE_RESPONSE,
                        }
                    },
                }
                operation["tags"] = ["Generation"]

            elif "/v1/chat" in path and method == "post":
                operation.setdefault("responses", {})["200"] = {
                    "description": "Successful chat completion",
                    "content": {
                        "application/json": {
                            "example": EXAMPLE_CHAT_RESPONSE,
                        }
                    },
                }
                operation["tags"] = ["Chat"]

            elif "/v1/models" in path:
                operation["tags"] = ["Models"]

            elif "/v1/health" in path:
                operation.setdefault("responses", {})["200"] = {
                    "description": "Health check response",
                    "content": {
                        "application/json": {
                            "example": EXAMPLE_HEALTH_RESPONSE,
                        }
                    },
                }
                operation["tags"] = ["Health"]

            elif "/v1/config" in path:
                operation["tags"] = ["Configuration"]

            elif "/v1/training" in path:
                operation["tags"] = ["Training"]

    app.openapi_schema = openapi_schema
    return app.openapi_schema


def setup_docs(app: FastAPI) -> None:
    """Configure API documentation on a FastAPI application.

    Sets up custom OpenAPI schema, docs URL, and redoc URL.

    Args:
        app: FastAPI application instance.
    """
    app.openapi = lambda: custom_openapi(app)

    app.docs_url = "/docs"
    app.redoc_url = "/redoc"
    app.openapi_url = "/openapi.json"
