# Models API Reference

Complete reference for the Nexus-LLM Models API. These endpoints allow you to list, inspect, download, load, unload, and manage LLM models.

## Base URL

```
http://localhost:8000/v1
```

## Authentication

All endpoints require authentication via the `Authorization` header:

```bash
Authorization: Bearer YOUR_API_KEY
```

See [Authentication](authentication.md) for details.

---

## List Models

Returns all models available on the server, including both loaded and unloaded models.

```http
GET /v1/models
```

### Request

| Parameter | Type | In | Required | Description |
|---|---|---|---|---|
| `status` | string | query | No | Filter by status: `loaded`, `unloaded`, `downloading`, `all` (default: `all`) |
| `type` | string | query | No | Filter by model type: `causal_lm`, `seq2seq`, `embedding` |
| `limit` | integer | query | No | Maximum number of results (default: 50) |
| `offset` | integer | query | No | Pagination offset (default: 0) |

### Example Request

```bash
curl -X GET http://localhost:8000/v1/models \
  -H "Authorization: Bearer nxs_sk_abc123" \
  -H "Content-Type: application/json"
```

### Example Response

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
      "parent": null,
      "status": "loaded",
      "model_type": "causal_lm",
      "parameters_billions": 8.0,
      "context_length": 131072,
      "quantization": null,
      "device": "cuda:0",
      "size_gb": 14.96
    },
    {
      "id": "Qwen/Qwen2.5-1.5B-Instruct",
      "object": "model",
      "created": 1703318400,
      "owned_by": "qwen",
      "permission": [],
      "root": "Qwen/Qwen2.5-1.5B-Instruct",
      "parent": null,
      "status": "unloaded",
      "model_type": "causal_lm",
      "parameters_billions": 1.5,
      "context_length": 32768,
      "quantization": null,
      "device": null,
      "size_gb": 3.04
    }
  ],
  "total": 2
}
```

---

## Get Model

Returns detailed information about a specific model.

```http
GET /v1/models/{model_id}
```

### Path Parameters

| Parameter | Type | Required | Description |
|---|---|---|---|
| `model_id` | string | Yes | The model identifier (URL-encoded) |

### Example Request

```bash
curl -X GET "http://localhost:8000/v1/models/meta-llama%2FLlama-3.1-8B-Instruct" \
  -H "Authorization: Bearer nxs_sk_abc123"
```

### Example Response

```json
{
  "id": "meta-llama/Llama-3.1-8B-Instruct",
  "object": "model",
  "created": 1703318400,
  "owned_by": "meta-llama",
  "permission": [],
  "root": "meta-llama/Llama-3.1-8B-Instruct",
  "parent": null,
  "status": "loaded",
  "model_type": "causal_lm",
  "description": "Meta Llama 3.1 8B Instruct model",
  "parameters_billions": 8.0,
  "context_length": 131072,
  "vocab_size": 128256,
  "hidden_size": 4096,
  "num_layers": 32,
  "num_heads": 32,
  "quantization": null,
  "device": "cuda:0",
  "size_bytes": 16060577280,
  "size_gb": 14.96,
  "metadata": {
    "architecture": "LlamaForCausalLM",
    "hf_url": "https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct",
    "license": "llama3.1",
    "tags": ["instruct", "chat"]
  }
}
```

### Error Responses

| Status | Code | Description |
|---|---|---|
| 404 | `model_not_found` | Model ID not found |

---

## Download Model

Download a model from Hugging Face Hub to the local cache.

```http
POST /v1/models/download
```

### Request Body

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `model_id` | string | Yes | — | Hugging Face model ID |
| `revision` | string | No | `main` | Model revision/branch |
| `quantization` | string | No | null | Download specific quantized variant (`gptq`, `awq`, `gguf`) |
| `include_tokenizer` | boolean | No | `true` | Download tokenizer files |
| `cache_dir` | string | No | default | Override default cache directory |
| `hf_token` | string | No | null | Hugging Face token for gated models |

### Example Request

```bash
curl -X POST http://localhost:8000/v1/models/download \
  -H "Authorization: Bearer nxs_sk_abc123" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "meta-llama/Llama-3.1-8B-Instruct",
    "revision": "main",
    "include_tokenizer": true
  }'
```

### Example Response

```json
{
  "model_id": "meta-llama/Llama-3.1-8B-Instruct",
  "status": "downloading",
  "download_id": "dl-a1b2c3d4",
  "progress": 0.0,
  "size_bytes": 16060577280,
  "size_gb": 14.96,
  "files_total": 5,
  "files_completed": 0,
  "estimated_time_seconds": 300,
  "message": "Download started. Use GET /v1/models/download/dl-a1b2c3d4 to monitor progress."
}
```

---

## Get Download Status

Check the progress of a model download.

```http
GET /v1/models/download/{download_id}
```

### Example Response

```json
{
  "download_id": "dl-a1b2c3d4",
  "model_id": "meta-llama/Llama-3.1-8B-Instruct",
  "status": "downloading",
  "progress": 0.65,
  "size_bytes": 16060577280,
  "downloaded_bytes": 10439376512,
  "speed_mbps": 42.5,
  "files_total": 5,
  "files_completed": 3,
  "estimated_time_seconds": 105,
  "started_at": "2024-01-15T10:30:00Z",
  "error": null
}
```

### Download Status Values

| Status | Description |
|---|---|
| `queued` | Download is queued but not yet started |
| `downloading` | Download in progress |
| `completed` | Download finished successfully |
| `failed` | Download failed (check `error` field) |
| `cancelled` | Download was cancelled |

---

## Cancel Download

Cancel an in-progress model download.

```http
DELETE /v1/models/download/{download_id}
```

### Example Response

```json
{
  "download_id": "dl-a1b2c3d4",
  "status": "cancelled",
  "message": "Download cancelled. Partial files have been removed."
}
```

---

## Load Model

Load a downloaded model into GPU/CPU memory for inference.

```http
POST /v1/models/load
```

### Request Body

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `model_id` | string | Yes | — | Model identifier to load |
| `device` | string | No | `auto` | Device: `auto`, `cpu`, `cuda`, `cuda:0`, `cuda:1`, `mps` |
| `dtype` | string | No | `auto` | Data type: `auto`, `float32`, `float16`, `bfloat16` |
| `quantization` | object | No | null | Quantization settings (see below) |
| `max_memory` | object | No | null | Per-device memory limits, e.g. `{"cuda:0": "20GiB"}` |
| `trust_remote_code` | boolean | No | `false` | Allow custom model code execution |
| `gpu_memory_utilization` | float | No | `0.9` | Fraction of GPU memory to use (0.0 - 1.0) |
| `enforce_eager` | boolean | No | `false` | Disable CUDA graphs (debugging) |

### Quantization Object

| Parameter | Type | Required | Description |
|---|---|---|---|
| `method` | string | Yes | Quantization method: `gptq`, `awq`, `bitsandbytes` |
| `bits` | integer | No | Quantization bits: `4` or `8` (default: `4`) |
| `group_size` | integer | No | Group size for GPTQ (default: `128`) |

### Example Request: Load with Default Settings

```bash
curl -X POST http://localhost:8000/v1/models/load \
  -H "Authorization: Bearer nxs_sk_abc123" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "meta-llama/Llama-3.1-8B-Instruct",
    "device": "auto",
    "dtype": "auto"
  }'
```

### Example Request: Load with Quantization

```bash
curl -X POST http://localhost:8000/v1/models/load \
  -H "Authorization: Bearer nxs_sk_abc123" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "meta-llama/Llama-3.1-8B-Instruct",
    "device": "cuda:0",
    "dtype": "float16",
    "quantization": {
      "method": "bitsandbytes",
      "bits": 4
    },
    "gpu_memory_utilization": 0.85
  }'
```

### Example Response

```json
{
  "model_id": "meta-llama/Llama-3.1-8B-Instruct",
  "status": "loaded",
  "device": "cuda:0",
  "dtype": "float16",
  "quantization": "4bit_bitsandbytes",
  "memory_used_gb": 4.2,
  "memory_total_gb": 24.0,
  "load_time_seconds": 12.5,
  "message": "Model loaded successfully on cuda:0"
}
```

### Error Responses

| Status | Code | Description |
|---|---|---|
| 404 | `model_not_found` | Model not downloaded |
| 500 | `model_load_failed` | Failed to load model |
| 507 | `insufficient_resources` | Not enough memory |

---

## Unload Model

Unload a model from memory to free resources.

```http
POST /v1/models/unload
```

### Request Body

| Parameter | Type | Required | Description |
|---|---|---|---|
| `model_id` | string | Yes | Model identifier to unload |
| `force` | boolean | No | Force unload even if in use (default: `false`) |

### Example Request

```bash
curl -X POST http://localhost:8000/v1/models/unload \
  -H "Authorization: Bearer nxs_sk_abc123" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "meta-llama/Llama-3.1-8B-Instruct"
  }'
```

### Example Response

```json
{
  "model_id": "meta-llama/Llama-3.1-8B-Instruct",
  "status": "unloaded",
  "memory_freed_gb": 14.2,
  "message": "Model unloaded successfully"
}
```

---

## Switch Model

Atomically switch the active model. The current model is unloaded after the new model is loaded, ensuring zero downtime.

```http
POST /v1/models/switch
```

### Request Body

| Parameter | Type | Required | Description |
|---|---|---|---|
| `from_model_id` | string | No | Current model to unload (optional, unloads default if omitted) |
| `to_model_id` | string | Yes | New model to load |
| `device` | string | No | Device for the new model |
| `dtype` | string | No | Data type for the new model |
| `quantization` | object | No | Quantization settings for the new model |

### Example Request

```bash
curl -X POST http://localhost:8000/v1/models/switch \
  -H "Authorization: Bearer nxs_sk_abc123" \
  -H "Content-Type: application/json" \
  -d '{
    "from_model_id": "meta-llama/Llama-3.1-8B-Instruct",
    "to_model_id": "Qwen/Qwen2.5-72B-Instruct-AWQ",
    "device": "cuda:0",
    "dtype": "float16"
  }'
```

### Example Response

```json
{
  "from_model_id": "meta-llama/Llama-3.1-8B-Instruct",
  "to_model_id": "Qwen/Qwen2.5-72B-Instruct-AWQ",
  "status": "switched",
  "load_time_seconds": 18.3,
  "message": "Model switched successfully"
}
```

---

## Quantize Model

Apply quantization to a loaded or downloaded model and save the result.

```http
POST /v1/models/quantize
```

### Request Body

| Parameter | Type | Required | Description |
|---|---|---|---|
| `model_id` | string | Yes | Source model identifier |
| `method` | string | Yes | Quantization method: `gptq`, `awq`, `gguf` |
| `bits` | integer | No | Target bits: `4` or `8` (default: `4`) |
| `group_size` | integer | No | Group size (default: `128`) |
| `output_id` | string | No | Output model ID (auto-generated if omitted) |
| `calibration_dataset` | string | No | Path to calibration data for GPTQ |
| `calibration_samples` | integer | No | Number of calibration samples (default: `128`) |

### Example Request

```bash
curl -X POST http://localhost:8000/v1/models/quantize \
  -H "Authorization: Bearer nxs_sk_abc123" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "meta-llama/Llama-3.1-8B-Instruct",
    "method": "gptq",
    "bits": 4,
    "group_size": 128,
    "calibration_samples": 128
  }'
```

### Example Response

```json
{
  "job_id": "quant-a1b2c3d4",
  "model_id": "meta-llama/Llama-3.1-8B-Instruct",
  "method": "gptq",
  "bits": 4,
  "status": "running",
  "progress": 0.0,
  "output_id": "meta-llama/Llama-3.1-8B-Instruct-GPTQ-4bit",
  "estimated_time_seconds": 600,
  "message": "Quantization started. Use GET /v1/models/quantize/quant-a1b2c3d4 to monitor progress."
}
```

---

## Delete Model

Remove a model from the server, unloading it if necessary and deleting cached files.

```http
DELETE /v1/models/{model_id}
```

### Query Parameters

| Parameter | Type | Required | Description |
|---|---|---|---|
| `keep_cache` | boolean | No | Keep downloaded model files (default: `false`) |

### Example Request

```bash
curl -X DELETE "http://localhost:8000/v1/models/Qwen%2FQwen2.5-1.5B-Instruct?keep_cache=false" \
  -H "Authorization: Bearer nxs_sk_abc123"
```

### Example Response

```json
{
  "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
  "status": "deleted",
  "cache_deleted": true,
  "space_freed_gb": 3.04,
  "message": "Model deleted successfully"
}
```

---

## Get Model Memory Estimate

Estimate the memory required to load a model with given settings, without actually loading it.

```http
POST /v1/models/estimate-memory
```

### Request Body

| Parameter | Type | Required | Description |
|---|---|---|---|
| `model_id` | string | Yes | Model identifier |
| `dtype` | string | No | Data type (default: `float16`) |
| `quantization` | object | No | Quantization settings |
| `kv_cache_size` | integer | No | Estimated KV cache size in tokens |

### Example Request

```bash
curl -X POST http://localhost:8000/v1/models/estimate-memory \
  -H "Authorization: Bearer nxs_sk_abc123" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "meta-llama/Llama-3.1-8B-Instruct",
    "dtype": "float16",
    "kv_cache_size": 4096
  }'
```

### Example Response

```json
{
  "model_id": "meta-llama/Llama-3.1-8B-Instruct",
  "dtype": "float16",
  "quantization": null,
  "model_weights_gb": 14.96,
  "kv_cache_gb": 2.0,
  "overhead_gb": 1.5,
  "total_min_gb": 16.96,
  "total_recommended_gb": 18.46,
  "fits_in_gpu": true,
  "available_gpu_memory_gb": 24.0
}
```

---

## Set Default Model

Set the default model used when no model is specified in generation requests.

```http
PUT /v1/models/default
```

### Request Body

| Parameter | Type | Required | Description |
|---|---|---|---|
| `model_id` | string | Yes | Model to set as default |

### Example Request

```bash
curl -X PUT http://localhost:8000/v1/models/default \
  -H "Authorization: Bearer nxs_sk_abc123" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "meta-llama/Llama-3.1-8B-Instruct"
  }'
```

### Example Response

```json
{
  "model_id": "meta-llama/Llama-3.1-8B-Instruct",
  "message": "Default model set successfully"
}
```

---

## List Model Cache

View the contents of the local model cache directory.

```http
GET /v1/models/cache
```

### Example Response

```json
{
  "cache_dir": "/home/user/.cache/nexus-llm/models",
  "total_size_gb": 42.8,
  "models": [
    {
      "model_id": "meta-llama/Llama-3.1-8B-Instruct",
      "size_gb": 14.96,
      "last_accessed": "2024-01-15T10:30:00Z",
      "downloaded_at": "2024-01-10T08:00:00Z"
    },
    {
      "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
      "size_gb": 3.04,
      "last_accessed": "2024-01-14T15:20:00Z",
      "downloaded_at": "2024-01-12T12:00:00Z"
    }
  ]
}
```

---

## Clean Model Cache

Remove unused model files from the cache.

```http
DELETE /v1/models/cache
```

### Query Parameters

| Parameter | Type | Required | Description |
|---|---|---|---|
| `model_id` | string | No | Remove specific model cache only |
| `older_than_days` | integer | No | Remove caches not accessed in N days |
| `dry_run` | boolean | No | Preview what would be deleted (default: `false`) |

### Example Request

```bash
curl -X DELETE "http://localhost:8000/v1/models/cache?older_than_days=30&dry_run=true" \
  -H "Authorization: Bearer nxs_sk_abc123"
```

### Example Response

```json
{
  "deleted_models": [],
  "space_freed_gb": 0,
  "dry_run": true,
  "would_delete": [
    {
      "model_id": "old-model/old-model-v1",
      "size_gb": 8.5,
      "last_accessed": "2023-12-01T10:00:00Z"
    }
  ],
  "would_free_gb": 8.5,
  "message": "Dry run: 1 model(s) would be deleted, freeing 8.5 GB"
}
```

---

## CLI Equivalents

All Models API operations have corresponding CLI commands:

| API Operation | CLI Command |
|---|---|
| `GET /v1/models` | `nexus-llm model list` |
| `GET /v1/models/{id}` | `nexus-llm model info meta-llama/Llama-3.1-8B-Instruct` |
| `POST /v1/models/download` | `nexus-llm model download meta-llama/Llama-3.1-8B-Instruct` |
| `POST /v1/models/load` | `nexus-llm model load meta-llama/Llama-3.1-8B-Instruct` |
| `POST /v1/models/unload` | `nexus-llm model unload meta-llama/Llama-3.1-8B-Instruct` |
| `POST /v1/models/switch` | `nexus-llm model switch --to Qwen/Qwen2.5-72B-Instruct-AWQ` |
| `POST /v1/models/quantize` | `nexus-llm model quantize meta-llama/Llama-3.1-8B-Instruct --method gptq --bits 4` |
| `DELETE /v1/models/{id}` | `nexus-llm model delete Qwen/Qwen2.5-1.5B-Instruct` |
| `PUT /v1/models/default` | `nexus-llm model default meta-llama/Llama-3.1-8B-Instruct` |
| `GET /v1/models/cache` | `nexus-llm model cache list` |
| `DELETE /v1/models/cache` | `nexus-llm model cache clean --older-than 30` |

---

## Related Documentation

- [REST API Reference](rest.md) — Chat completions and generation endpoints
- [Model Management Guide](../guides/models.md) — Comprehensive guide to managing models
- [Error Codes Reference](errors.md) — Complete error code documentation
