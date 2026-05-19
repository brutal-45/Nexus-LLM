# API Error Codes Reference

This document provides a comprehensive reference for all error codes returned by the Nexus-LLM API. All errors follow the OpenAI-compatible error format and include a unique `request_id` for troubleshooting.

## Error Response Format

All API errors return a JSON response with the following structure:

```json
{
  "error": {
    "message": "Human-readable description of the error",
    "type": "error_type_string",
    "param": "affected_parameter",
    "code": "error_code_string"
  }
}
```

Additionally, Nexus-LLM extends this with extra fields in the response body:

```json
{
  "error": "Human-readable description of the error",
  "error_type": "ErrorTypeClassName",
  "detail": "Additional context about the error",
  "status_code": 400,
  "request_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "timestamp": "2024-01-15T10:30:00.000000"
}
```

---

## Authentication Errors

### `AUTH_INVALID_KEY`

| Field | Value |
|---|---|
| HTTP Status | `401 Unauthorized` |
| Error Type | `AuthenticationError` |
| Description | The provided API key is invalid, malformed, or has been revoked. |

**Example Response:**

```json
{
  "error": {
    "message": "Authentication failed",
    "type": "authentication_error",
    "param": null,
    "code": "auth_invalid_key"
  }
}
```

**Resolution:** Verify your API key is correct. Use `nexus-llm api-key list` to check key status. If revoked, create a new key with `nexus-llm api-key create`.

---

### `AUTH_MISSING_KEY`

| Field | Value |
|---|---|
| HTTP Status | `401 Unauthorized` |
| Error Type | `AuthenticationError` |
| Description | No API key was provided in the request. |

**Example Response:**

```json
{
  "error": {
    "message": "Authentication failed",
    "type": "authentication_error",
    "param": "Authorization",
    "code": "auth_missing_key"
  }
}
```

**Resolution:** Include the `Authorization: Bearer YOUR_API_KEY` header in all requests.

---

### `AUTH_EXPIRED_TOKEN`

| Field | Value |
|---|---|
| HTTP Status | `401 Unauthorized` |
| Error Type | `AuthenticationError` |
| Description | The JWT token has expired. |

**Example Response:**

```json
{
  "error": {
    "message": "Authentication failed",
    "type": "authentication_error",
    "param": null,
    "code": "auth_expired_token"
  }
}
```

**Resolution:** Refresh your JWT token via `POST /v1/auth/refresh` or obtain a new token via `POST /v1/auth/token`.

---

### `AUTH_INSUFFICIENT_PERMISSIONS`

| Field | Value |
|---|---|
| HTTP Status | `403 Forbidden` |
| Error Type | `PermissionError` |
| Description | The API key does not have the required permissions for this operation. |

**Example Response:**

```json
{
  "error": {
    "message": "Insufficient permissions",
    "type": "permission_error",
    "param": null,
    "code": "auth_insufficient_permissions"
  }
}
```

**Resolution:** Use an API key with the required tier or permissions. Admin operations require an `admin` tier key.

---

## Request Validation Errors

### `INVALID_REQUEST`

| Field | Value |
|---|---|
| HTTP Status | `400 Bad Request` |
| Error Type | `InvalidRequestError` |
| Description | The request body is malformed, missing required fields, or contains invalid values. |

**Example Response:**

```json
{
  "error": {
    "message": "Invalid request",
    "type": "invalid_request_error",
    "param": "temperature",
    "code": "invalid_request"
  }
}
```

**Resolution:** Check the request body against the API schema. Ensure all required fields are present and values are within valid ranges.

---

### `INVALID_MODEL_PARAMETER`

| Field | Value |
|---|---|
| HTTP Status | `400 Bad Request` |
| Error Type | `InvalidRequestError` |
| Description | A generation parameter is out of the valid range. |

**Common invalid parameter scenarios:**

| Parameter | Valid Range | Invalid Example |
|---|---|---|
| `temperature` | 0.0 - 2.0 | `3.5` |
| `top_p` | 0.0 - 1.0 | `1.5` |
| `top_k` | -1 or >= 1 | `0` |
| `max_tokens` | 1 - model max | `0` or `100000` |
| `frequency_penalty` | -2.0 - 2.0 | `5.0` |
| `presence_penalty` | -2.0 - 2.0 | `-3.0` |
| `repetition_penalty` | > 0.0 | `0.0` or `-1.0` |

---

### `INVALID_MESSAGE_FORMAT`

| Field | Value |
|---|---|
| HTTP Status | `400 Bad Request` |
| Error Type | `InvalidRequestError` |
| Description | The messages array has an invalid structure. |

**Resolution:** Ensure messages follow the format `{ "role": "system|user|assistant", "content": "..." }`. The first message should typically be a `system` message, and the last should be a `user` message.

---

### `CONTEXT_LENGTH_EXCEEDED`

| Field | Value |
|---|---|
| HTTP Status | `400 Bad Request` |
| Error Type | `InvalidRequestError` |
| Description | The combined token count of the prompt and requested `max_tokens` exceeds the model's context window. |

**Example Response:**

```json
{
  "error": {
    "message": "Invalid request",
    "type": "invalid_request_error",
    "param": "messages",
    "code": "context_length_exceeded"
  }
}
```

**Resolution:** Reduce the number of messages, shorten the prompt, or lower `max_tokens`. Use `POST /v1/count_tokens` to check token counts before sending.

---

## Model Errors

### `MODEL_NOT_FOUND`

| Field | Value |
|---|---|
| HTTP Status | `404 Not Found` |
| Error Type | `ModelNotFoundError` |
| Description | The requested model ID does not exist or is not registered. |

**Example Response:**

```json
{
  "error": {
    "message": "Model 'nonexistent-model' not found",
    "type": "not_found_error",
    "param": "model",
    "code": "model_not_found"
  }
}
```

**Resolution:** Check available models via `GET /v1/models`. Ensure the model ID is spelled correctly, including the namespace prefix (e.g., `meta-llama/Llama-3.1-8B-Instruct`).

---

### `MODEL_NOT_LOADED`

| Field | Value |
|---|---|
| HTTP Status | `503 Service Unavailable` |
| Error Type | `ModelNotLoadedError` |
| Description | The model exists but is not currently loaded into memory. This can happen during model swapping or after an unload operation. |

**Example Response:**

```json
{
  "error": {
    "message": "Model 'meta-llama/Llama-3.1-8B-Instruct' is not loaded",
    "type": "service_unavailable_error",
    "param": "model",
    "code": "model_not_loaded"
  }
}
```

**Resolution:** Wait a moment and retry, or use the CLI to load the model: `nexus-llm model load meta-llama/Llama-3.1-8B-Instruct`.

---

### `MODEL_LOAD_FAILED`

| Field | Value |
|---|---|
| HTTP Status | `500 Internal Server Error` |
| Error Type | `ModelLoadError` |
| Description | The model failed to load due to corrupted files, missing dependencies, or out-of-memory conditions. |

**Common causes:**
- Insufficient GPU/CPU memory
- Corrupted model weights file
- Incompatible model format
- Missing Hugging Face authentication for gated models

**Resolution:** Check server logs for the specific error. Ensure sufficient memory is available and the model files are intact.

---

### `MODEL_QUANTIZATION_FAILED`

| Field | Value |
|---|---|
| HTTP Status | `400 Bad Request` |
| Error Type | `InvalidRequestError` |
| Description | The requested quantization method is not compatible with the model. |

**Resolution:** Verify the quantization method is supported for the model architecture. Supported methods: `gptq`, `awq`, `bitsandbytes` (4-bit and 8-bit).

---

## Generation Errors

### `GENERATION_FAILED`

| Field | Value |
|---|---|
| HTTP Status | `500 Internal Server Error` |
| Error Type | `GenerationError` |
| Description | Text generation failed during inference. This can be caused by CUDA errors, numerical instabilities, or internal model errors. |

**Example Response:**

```json
{
  "error": {
    "message": "Generation failed",
    "type": "server_error",
    "param": null,
    "code": "generation_failed"
  }
}
```

**Resolution:** Retry the request. If the error persists, check GPU health and server logs. Try reducing `max_tokens` or adjusting sampling parameters.

---

### `GENERATION_TIMEOUT`

| Field | Value |
|---|---|
| HTTP Status | `504 Gateway Timeout` |
| Error Type | `TimeoutError` |
| Description | The generation request exceeded the server's timeout limit. |

**Example Response:**

```json
{
  "error": {
    "message": "Request timed out",
    "type": "timeout_error",
    "param": null,
    "code": "generation_timeout"
  }
}
```

**Resolution:** Reduce `max_tokens`, simplify the prompt, or increase the server timeout in the configuration.

---

## Safety & Content Errors

### `CONTENT_FILTERED`

| Field | Value |
|---|---|
| HTTP Status | `422 Unprocessable Entity` |
| Error Type | `ContentFilterError` |
| Description | The input or generated output was blocked by the content safety filter. |

**Example Response:**

```json
{
  "error": {
    "message": "Content filtered",
    "type": "content_filter_error",
    "param": null,
    "code": "content_filtered"
  }
}
```

**Filter categories that can trigger this error:**

| Category | Code | Description |
|---|---|---|
| Profanity | `filter_profanity` | Profane or vulgar language detected |
| Hate Speech | `filter_hate_speech` | Hate speech or discriminatory content detected |
| Violence | `filter_violence` | Violent or threatening content detected |
| Self-Harm | `filter_self_harm` | Self-harm related content detected |
| Sexual | `filter_sexual` | Sexually explicit content detected |
| Harassment | `filter_harassment` | Harassment or bullying content detected |
| Illegal | `filter_illegal` | Content promoting illegal activities detected |
| PII | `filter_pii` | Personally identifiable information detected |
| Spam | `filter_spam` | Spam or deceptive content detected |

**Resolution:** Modify the input to comply with content policies, or adjust safety filter settings in the configuration if appropriate for your use case.

---

### `TOXICITY_THRESHOLD_EXCEEDED`

| Field | Value |
|---|---|
| HTTP Status | `422 Unprocessable Entity` |
| Error Type | `ContentFilterError` |
| Description | The generated output exceeded the configured toxicity score threshold. |

**Resolution:** Adjust the `toxicity_threshold` in `safety_config.yaml` or use a different model with better alignment.

---

### `GUARDRAIL_VIOLATION`

| Field | Value |
|---|---|
| HTTP Status | `422 Unprocessable Entity` |
| Error Type | `ContentFilterError` |
| Description | The output violated one or more configured guardrail rules (e.g., topic restrictions, format constraints). |

**Resolution:** Review guardrail configuration and adjust rules as needed for your application.

---

## Rate Limiting Errors

### `RATE_LIMIT_EXCEEDED`

| Field | Value |
|---|---|
| HTTP Status | `429 Too Many Requests` |
| Error Type | `RateLimitExceededError` |
| Description | The number of requests or tokens has exceeded the rate limit for your API key tier. |

**Example Response:**

```json
{
  "error": {
    "message": "Rate limit exceeded: 100 requests/minute. Retry after 30 seconds.",
    "type": "rate_limit_error",
    "param": null,
    "code": "rate_limit_exceeded"
  }
}
```

**Response Headers:**

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1703318460
Retry-After: 30
```

**Resolution:** Wait for the `Retry-After` period before retrying. Consider upgrading your API key tier for higher limits, or implement exponential backoff in your client.

**Rate limits by tier:**

| Tier | Requests/min | Tokens/min | Requests/day |
|---|---|---|---|
| Free | 20 | 10,000 | 1,000 |
| Standard | 100 | 100,000 | 50,000 |
| Professional | 1,000 | 1,000,000 | Unlimited |

---

## Resource Errors

### `INSUFFICIENT_RESOURCES`

| Field | Value |
|---|---|
| HTTP Status | `507 Insufficient Storage` |
| Error Type | `InsufficientResourcesError` |
| Description | The server does not have enough GPU/CPU memory or disk space to fulfill the request. |

**Example Response:**

```json
{
  "error": {
    "message": "Insufficient resources",
    "type": "server_error",
    "param": null,
    "code": "insufficient_resources"
  }
}
```

**Resolution:** Free up resources by unloading other models (`nexus-llm model unload`), use quantization, or add more hardware resources.

---

### `STORAGE_FULL`

| Field | Value |
|---|---|
| HTTP Status | `507 Insufficient Storage` |
| Error Type | `InsufficientResourcesError` |
| Description | Disk storage is full. Cannot download or cache new models. |

**Resolution:** Free disk space by removing unused model caches: `nexus-llm model cache clean`.

---

## Training Errors

### `TRAINING_JOB_FAILED`

| Field | Value |
|---|---|
| HTTP Status | `500 Internal Server Error` |
| Error Type | `TrainingError` |
| Description | A training job failed during execution. Common causes include OOM, data errors, or convergence failures. |

**Resolution:** Check training logs for details. Common fixes:
- Reduce `per_device_train_batch_size`
- Enable `gradient_checkpointing`
- Use LoRA or QLoRA for memory efficiency
- Verify dataset format and quality

---

### `TRAINING_DATASET_NOT_FOUND`

| Field | Value |
|---|---|
| HTTP Status | `404 Not Found` |
| Error Type | `NotFoundError` |
| Description | The specified training dataset could not be found. |

**Resolution:** Verify the dataset path and ensure the file exists in the configured data directory.

---

### `TRAINING_DATASET_INVALID`

| Field | Value |
|---|---|
| HTTP Status | `400 Bad Request` |
| Error Type | `InvalidRequestError` |
| Description | The training dataset is invalid, empty, or has an unsupported format. |

**Resolution:** Ensure the dataset is in JSONL format with the required fields (`instruction`, `input`, `output` for instruction format, or `messages` for chat format).

---

## RAG Errors

### `COLLECTION_NOT_FOUND`

| Field | Value |
|---|---|
| HTTP Status | `404 Not Found` |
| Error Type | `NotFoundError` |
| Description | The specified document collection does not exist. |

**Resolution:** Verify the collection name. Use `GET /v1/rag/collections` to list available collections.

---

### `DOCUMENT_INDEX_FAILED`

| Field | Value |
|---|---|
| HTTP Status | `500 Internal Server Error` |
| Error Type | `ServerError` |
| Description | Failed to index a document. The file may be corrupted or in an unsupported format. |

**Supported formats:** PDF, TXT, MD, DOCX, HTML

---

### `EMBEDDING_GENERATION_FAILED`

| Field | Value |
|---|---|
| HTTP Status | `500 Internal Server Error` |
| Error Type | `ServerError` |
| Description | Failed to generate embeddings for the provided text or document chunks. |

**Resolution:** Verify the embedding model is loaded and the input text is not empty.

---

## Agent Errors

### `AGENT_SESSION_NOT_FOUND`

| Field | Value |
|---|---|
| HTTP Status | `404 Not Found` |
| Error Type | `NotFoundError` |
| Description | The specified agent session does not exist or has expired. |

**Resolution:** Create a new agent session via `POST /v1/agents/sessions`.

---

### `AGENT_MAX_ITERATIONS`

| Field | Value |
|---|---|
| HTTP Status | `400 Bad Request` |
| Error Type | `InvalidRequestError` |
| Description | The agent reached its maximum number of reasoning iterations without completing the task. |

**Resolution:** Increase `max_iterations` in the agent session configuration, or simplify the task.

---

### `AGENT_TOOL_NOT_FOUND`

| Field | Value |
|---|---|
| HTTP Status | `404 Not Found` |
| Error Type | `NotFoundError` |
| Description | The agent requested a tool that is not available. |

**Resolution:** Verify the tool name. Check available tools via `GET /v1/agents/tools`.

---

### `AGENT_TOOL_EXECUTION_FAILED`

| Field | Value |
|---|---|
| HTTP Status | `500 Internal Server Error` |
| Error Type | `ServerError` |
| Description | A tool execution within the agent's reasoning loop failed. |

**Resolution:** Check tool input parameters and permissions. Some tools may require specific environment configuration.

---

## Server Errors

### `INTERNAL_SERVER_ERROR`

| Field | Value |
|---|---|
| HTTP Status | `500 Internal Server Error` |
| Error Type | `InternalServerError` |
| Description | An unexpected error occurred on the server. |

**Resolution:** Retry the request. If the error persists, check server logs and report the issue with the `request_id`.

---

### `SERVICE_UNAVAILABLE`

| Field | Value |
|---|---|
| HTTP Status | `503 Service Unavailable` |
| Error Type | `ServiceUnavailableError` |
| Description | The server is temporarily unavailable, typically due to model loading or maintenance. |

**Resolution:** Wait and retry. The `Retry-After` header may indicate when to retry.

---

### `SERVER_OVERLOADED`

| Field | Value |
|---|---|
| HTTP Status | `503 Service Unavailable` |
| Error Type | `ServiceUnavailableError` |
| Description | The server is overloaded with requests and cannot accept new ones. |

**Resolution:** Implement exponential backoff and retry. Consider scaling your deployment.

---

## Complete Error Code Summary

| Error Code | HTTP Status | Error Type | Description |
|---|---|---|---|
| `auth_invalid_key` | 401 | `AuthenticationError` | Invalid or revoked API key |
| `auth_missing_key` | 401 | `AuthenticationError` | No API key provided |
| `auth_expired_token` | 401 | `AuthenticationError` | JWT token expired |
| `auth_insufficient_permissions` | 403 | `PermissionError` | Insufficient key tier or permissions |
| `invalid_request` | 400 | `InvalidRequestError` | Malformed or invalid request |
| `invalid_model_parameter` | 400 | `InvalidRequestError` | Generation parameter out of range |
| `invalid_message_format` | 400 | `InvalidRequestError` | Invalid messages array structure |
| `context_length_exceeded` | 400 | `InvalidRequestError` | Token count exceeds model context window |
| `model_not_found` | 404 | `ModelNotFoundError` | Model ID does not exist |
| `model_not_loaded` | 503 | `ModelNotLoadedError` | Model exists but not loaded in memory |
| `model_load_failed` | 500 | `ModelLoadError` | Model failed to load |
| `model_quantization_failed` | 400 | `InvalidRequestError` | Incompatible quantization method |
| `generation_failed` | 500 | `GenerationError` | Text generation error during inference |
| `generation_timeout` | 504 | `TimeoutError` | Generation exceeded timeout limit |
| `content_filtered` | 422 | `ContentFilterError` | Content blocked by safety filter |
| `toxicity_threshold_exceeded` | 422 | `ContentFilterError` | Output toxicity score too high |
| `guardrail_violation` | 422 | `ContentFilterError` | Output violated guardrail rules |
| `rate_limit_exceeded` | 429 | `RateLimitExceededError` | Request or token rate limit exceeded |
| `insufficient_resources` | 507 | `InsufficientResourcesError` | Not enough GPU/CPU memory |
| `storage_full` | 507 | `InsufficientResourcesError` | Disk storage is full |
| `training_job_failed` | 500 | `TrainingError` | Training job execution failed |
| `training_dataset_not_found` | 404 | `NotFoundError` | Training dataset not found |
| `training_dataset_invalid` | 400 | `InvalidRequestError` | Dataset format invalid or empty |
| `collection_not_found` | 404 | `NotFoundError` | RAG collection does not exist |
| `document_index_failed` | 500 | `ServerError` | Document indexing failed |
| `embedding_generation_failed` | 500 | `ServerError` | Embedding generation failed |
| `agent_session_not_found` | 404 | `NotFoundError` | Agent session does not exist |
| `agent_max_iterations` | 400 | `InvalidRequestError` | Agent reached max reasoning iterations |
| `agent_tool_not_found` | 404 | `NotFoundError` | Requested tool not available |
| `agent_tool_execution_failed` | 500 | `ServerError` | Tool execution within agent failed |
| `internal_server_error` | 500 | `InternalServerError` | Unexpected server error |
| `service_unavailable` | 503 | `ServiceUnavailableError` | Server temporarily unavailable |
| `server_overloaded` | 503 | `ServiceUnavailableError` | Server overloaded with requests |

---

## Error Handling Best Practices

### 1. Implement Retry Logic with Exponential Backoff

```python
import time
import random

def api_call_with_retry(client, request, max_retries=5):
    """Make an API call with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(**request)
        except Exception as e:
            if getattr(e, 'status_code', 0) not in (429, 500, 503, 504):
                raise
            if attempt == max_retries - 1:
                raise
            delay = (2 ** attempt) + random.uniform(0, 1)
            time.sleep(delay)
```

### 2. Always Check the `code` Field

The `code` field provides a machine-readable error identifier that is more specific than the HTTP status code. Use it to implement targeted error handling:

```python
try:
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[{"role": "user", "content": "Hello"}]
    )
except NexusAPIError as e:
    if e.code == "model_not_found":
        # Switch to a fallback model
        response = client.chat.completions.create(
            model="fallback-model",
            messages=[{"role": "user", "content": "Hello"}]
        )
    elif e.code == "content_filtered":
        # Log and notify user
        logger.warning(f"Content filtered: {e.detail}")
    elif e.code == "rate_limit_exceeded":
        # Wait and retry
        time.sleep(int(e.headers.get("Retry-After", 30)))
    else:
        raise
```

### 3. Log the `request_id` for Support

When contacting support about an error, always include the `request_id`. This allows the team to trace the exact request in the server logs.

### 4. Handle Streaming Errors Gracefully

In SSE streaming, errors are sent as a final event before the `[DONE]` sentinel:

```
data: {"error":{"message":"Content filtered","type":"content_filter_error","code":"content_filtered"}}

data: [DONE]
```

Always check for error objects in stream chunks.

---

## Related Documentation

- [REST API Reference](rest.md) — Complete endpoint documentation
- [Authentication](authentication.md) — API key and JWT authentication
- [Safety Configuration Guide](../guides/safety.md) — Content filter and moderation settings
- [Monitoring Guide](../guides/monitoring.md) — Setting up alerts for error rates
