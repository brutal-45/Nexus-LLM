# REST API Reference

Nexus-LLM provides a comprehensive REST API that is fully compatible with the OpenAI API format, plus additional endpoints for model management, RAG, and agent operations.

## Base URL

```
http://localhost:8000/v1
```

## Authentication

All API requests require an API key passed via the `Authorization` header:

```bash
Authorization: Bearer YOUR_API_KEY
```

See [Authentication](authentication.md) for details.

---

## Chat Completions

### Create Chat Completion

```http
POST /v1/chat/completions
```

Creates a model response for the given conversation.

**Request Body:**

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `model` | string | Yes | — | Model ID to use |
| `messages` | array | Yes | — | List of message objects |
| `temperature` | float | No | 0.7 | Sampling temperature (0.0 - 2.0) |
| `top_p` | float | No | 1.0 | Nucleus sampling threshold |
| `top_k` | integer | No | -1 | Top-k sampling |
| `max_tokens` | integer | No | 4096 | Maximum tokens to generate |
| `stream` | boolean | No | false | Enable streaming response |
| `stop` | string/array | No | null | Stop sequences |
| `frequency_penalty` | float | No | 0.0 | Frequency penalty (-2.0 to 2.0) |
| `presence_penalty` | float | No | 0.0 | Presence penalty (-2.0 to 2.0) |
| `repetition_penalty` | float | No | 1.0 | Repetition penalty |
| `seed` | integer | No | null | Random seed for reproducibility |
| `response_format` | object | No | null | JSON mode configuration |
| `tools` | array | No | null | Tool/function definitions |
| `tool_choice` | string/object | No | auto | Tool choice strategy |

**Example Request:**

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [
      {"role": "system", "content": "You are a helpful AI assistant."},
      {"role": "user", "content": "Explain quantum computing in simple terms."}
    ],
    "temperature": 0.7,
    "max_tokens": 512
  }'
```

**Example Response:**

```json
{
  "id": "chatcmpl-abc123def456",
  "object": "chat.completion",
  "created": 1703318400,
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Quantum computing is a type of computing that uses quantum mechanics..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 25,
    "completion_tokens": 150,
    "total_tokens": 175
  }
}
```

### Streaming Chat Completion

Set `stream: true` to receive Server-Sent Events (SSE):

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [
      {"role": "user", "content": "Write a poem about the ocean."}
    ],
    "stream": true
  }'
```

**Streaming Response (SSE):**

```
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1703318400,"model":"meta-llama/Llama-3.1-8B-Instruct","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1703318400,"model":"meta-llama/Llama-3.1-8B-Instruct","choices":[{"index":0,"delta":{"content":"The"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1703318400,"model":"meta-llama/Llama-3.1-8B-Instruct","choices":[{"index":0,"delta":{"content":" ocean"},"finish_reason":null}]}

data: [DONE]
```

---

## Text Completions

### Create Completion

```http
POST /v1/completions
```

**Request Body:**

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `model` | string | Yes | — | Model ID |
| `prompt` | string/array | Yes | — | Text prompt(s) |
| `max_tokens` | integer | No | 4096 | Max tokens to generate |
| `temperature` | float | No | 0.7 | Sampling temperature |
| `top_p` | float | No | 1.0 | Nucleus sampling |
| `stream` | boolean | No | false | Stream response |
| `logprobs` | integer | No | null | Include log probabilities |
| `echo` | boolean | No | false | Echo the prompt |
| `n` | integer | No | 1 | Number of completions |

**Example:**

```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "prompt": "The capital of France is",
    "max_tokens": 10
  }'
```

---

## Embeddings

### Create Embeddings

```http
POST /v1/embeddings
```

**Request Body:**

| Parameter | Type | Required | Description |
|---|---|---|---|
| `model` | string | Yes | Embedding model ID |
| `input` | string/array | Yes | Text(s) to embed |
| `encoding_format` | string | No | `float` (default) or `base64` |

**Example:**

```bash
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "BAAI/bge-large-en-v1.5",
    "input": "The food was delicious and the waiter was friendly."
  }'
```

---

## Models

### List Models

```http
GET /v1/models
```

Returns the list of available models.

**Example Response:**

```json
{
  "object": "list",
  "data": [
    {
      "id": "meta-llama/Llama-3.1-8B-Instruct",
      "object": "model",
      "created": 1703318400,
      "owned_by": "meta-llama",
      "permission": [],
      "root": "meta-llama/Llama-3.1-8B-Instruct",
      "parent": null
    }
  ]
}
```

### Get Model

```http
GET /v1/models/{model_id}
```

Returns details for a specific model.

---

## RAG Endpoints

### Upload Documents

```http
POST /v1/rag/documents
```

Upload and index documents for RAG.

**Request (multipart/form-data):**

| Parameter | Type | Required | Description |
|---|---|---|---|
| `files` | file(s) | Yes | Document files (PDF, TXT, MD, DOCX) |
| `collection` | string | No | Collection name (default: "default") |
| `chunk_size` | integer | No | Chunk size in tokens |
| `chunk_overlap` | integer | No | Chunk overlap in tokens |
| `metadata` | JSON | No | Additional metadata |

**Example:**

```bash
curl -X POST http://localhost:8000/v1/rag/documents \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "files=@document.pdf" \
  -F "files=@notes.txt" \
  -F "collection=research" \
  -F 'metadata={"source": "internal"}'
```

### Query Documents

```http
POST /v1/rag/query
```

Query indexed documents with a natural language question.

**Request Body:**

| Parameter | Type | Required | Description |
|---|---|---|---|
| `query` | string | Yes | Search query |
| `collection` | string | No | Collection to search |
| `top_k` | integer | No | Number of results (default: 5) |
| `include_metadata` | boolean | No | Include document metadata |

### List Collections

```http
GET /v1/rag/collections
```

### Delete Collection

```http
DELETE /v1/rag/collections/{collection_name}
```

---

## Agent Endpoints

### Create Agent Session

```http
POST /v1/agents/sessions
```

**Request Body:**

| Parameter | Type | Required | Description |
|---|---|---|---|
| `agent_type` | string | No | Agent type: `react`, `planner`, `custom` |
| `model` | string | No | Model for agent reasoning |
| `tools` | array | No | Enabled tool names |
| `system_prompt` | string | No | Custom system prompt |
| `max_iterations` | integer | No | Max reasoning loops |

### Execute Agent Task

```http
POST /v1/agents/sessions/{session_id}/execute
```

**Request Body:**

```json
{
  "task": "Research the latest developments in fusion energy and write a summary",
  "tools": ["web_search", "code_executor"],
  "stream": true
}
```

### List Agent Sessions

```http
GET /v1/agents/sessions
```

### Get Agent Session

```http
GET /v1/agents/sessions/{session_id}
```

---

## Tokenization

### Tokenize

```http
POST /v1/tokenize
```

**Request Body:**

```json
{
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "text": "Hello, world!"
}
```

### Count Tokens

```http
POST /v1/count_tokens
```

**Request Body:**

```json
{
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "messages": [
    {"role": "user", "content": "Hello!"}
  ]
}
```

---

## Health & Status

### Health Check

```http
GET /v1/health
```

Returns server health status.

**Response:**

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime_seconds": 3600,
  "models_loaded": 1,
  "gpu_available": true,
  "gpu_memory_used_gb": 12.5,
  "gpu_memory_total_gb": 80.0
}
```

### Server Info

```http
GET /v1/info
```

Returns detailed server information including loaded models and configuration.

---

## Rate Limits

All endpoints are subject to rate limiting based on your API key tier:

| Tier | Requests/min | Tokens/min | Requests/day |
|---|---|---|---|
| Free | 20 | 10,000 | 1,000 |
| Standard | 100 | 100,000 | 50,000 |
| Professional | 1,000 | 1,000,000 | Unlimited |

Rate limit headers are included in every response:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1703318460
```

---

## Error Handling

All errors follow the OpenAI error format:

```json
{
  "error": {
    "message": "Model not found: invalid-model",
    "type": "invalid_request_error",
    "param": "model",
    "code": "model_not_found"
  }
}
```

| HTTP Status | Error Type | Description |
|---|---|---|
| 400 | `invalid_request_error` | Malformed request |
| 401 | `authentication_error` | Invalid API key |
| 403 | `permission_error` | Insufficient permissions |
| 404 | `not_found_error` | Resource not found |
| 429 | `rate_limit_error` | Rate limit exceeded |
| 500 | `server_error` | Internal server error |
| 503 | `service_unavailable` | Model overloaded |
