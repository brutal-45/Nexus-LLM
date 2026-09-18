# API Endpoint Reference

Base URL: `http://localhost:8000/api/v1`

All endpoints require authentication unless otherwise noted. Include your API key in the `X-API-Key` header or a Bearer token in the `Authorization` header.

---

## Chat

### POST /chat

Generate a response to a chat message.

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model` | string | Yes | Model ID to use |
| `messages` | array | Yes | Array of message objects |
| `temperature` | float | No | Sampling temperature (0.0-2.0, default: 0.7) |
| `top_p` | float | No | Nucleus sampling threshold (0.0-1.0, default: 0.9) |
| `max_tokens` | integer | No | Maximum tokens to generate (default: 512) |
| `stop` | array | No | Stop sequences |
| `stream` | boolean | No | Enable streaming (default: false) |
| `repetition_penalty` | float | No | Repetition penalty (1.0-2.0, default: 1.0) |

**Message Object:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `role` | string | Yes | One of: `system`, `user`, `assistant` |
| `content` | string | Yes | Message content |

**Response (non-streaming):**

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1702896000,
  "model": "nexus-7b-chat",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Response text here"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 25,
    "completion_tokens": 42,
    "total_tokens": 67
  }
}
```

**Response (streaming):**

Returns `text/event-stream` with Server-Sent Events:

```
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","choices":[{"delta":{"content":" world"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","choices":[{"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

---

## Models

### GET /models

List available models.

**Response:**

```json
{
  "object": "list",
  "data": [
    {
      "id": "nexus-7b-chat",
      "object": "model",
      "created": 1702896000,
      "owned_by": "nexus-llm",
      "status": "loaded",
      "max_context": 32768
    }
  ]
}
```

### GET /models/{model_id}

Get details about a specific model.

**Response:**

```json
{
  "id": "nexus-7b-chat",
  "object": "model",
  "created": 1702896000,
  "owned_by": "nexus-llm",
  "status": "loaded",
  "max_context": 32768,
  "architecture": "transformer",
  "parameters": "7B",
  "dtype": "float16",
  "gpu_memory_mb": 14000
}
```

---

## Embeddings

### POST /embeddings

Generate embeddings for text inputs.

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model` | string | Yes | Embedding model ID |
| `input` | string or array | Yes | Text or array of texts |
| `normalize` | boolean | No | L2-normalize embeddings (default: true) |

**Response:**

```json
{
  "object": "list",
  "model": "nexus-embedding-large",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [0.0023, -0.0094, ...]
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "total_tokens": 10
  }
}
```

---

## RAG

### POST /rag/query

Query the RAG pipeline.

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | string | Yes | The question to answer |
| `top_k` | integer | No | Number of chunks to retrieve (default: 5) |
| `similarity_threshold` | float | No | Minimum similarity score (default: 0.7) |
| `include_sources` | boolean | No | Include source documents (default: true) |

**Response:**

```json
{
  "answer": "Nexus-LLM supports 128K context length...",
  "sources": [
    {
      "document_id": "doc_001",
      "chunk_index": 3,
      "score": 0.92,
      "text": "Nexus-LLM supports 128K context length with Flash Attention 2."
    }
  ],
  "confidence": 0.89
}
```

### POST /rag/ingest

Ingest documents into the RAG pipeline.

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `documents` | array | Yes | Array of document objects |
| `metadata` | object | No | Metadata to attach to all documents |

---

## Health

### GET /health

Health check endpoint (no authentication required).

**Response:**

```json
{
  "status": "healthy",
  "version": "2.1.0",
  "models_loaded": 2,
  "gpu_available": true,
  "uptime_seconds": 86400
}
```

---

## Authentication

### POST /auth/token

Obtain a JWT access token.

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `user_id` | string | Yes | User identifier |
| `api_key` | string | Yes | API key for the user |

**Response:**

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

---

## Error Responses

All errors follow a consistent format:

```json
{
  "error": {
    "type": "invalid_request_error",
    "message": "Invalid model ID: unknown-model",
    "code": "model_not_found"
  }
}
```

**Common HTTP Status Codes:**

| Code | Type | Description |
|------|------|-------------|
| 400 | Bad Request | Invalid request parameters |
| 401 | Unauthorized | Missing or invalid authentication |
| 403 | Forbidden | Insufficient permissions |
| 404 | Not Found | Resource not found |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Error | Server-side error |
| 503 | Unavailable | Model not loaded or overloaded |
