# Model Management Guide

This guide covers the complete lifecycle of model management in Nexus-LLM: downloading, loading, switching, quantizing, and maintaining LLM models.

## Overview

Nexus-LLM supports any model compatible with the Hugging Face Transformers library. The model management workflow follows these stages:

```
Download → Load → Use → Unload → Delete
              ↑                   ↓
              └─── Switch ────────┘
                    ↓
              Quantize (optional)
```

---

## Downloading Models

### Using the CLI

```bash
# Download a model from Hugging Face Hub
nexus-llm model download meta-llama/Llama-3.1-8B-Instruct

# Download a specific revision
nexus-llm model download meta-llama/Llama-3.1-8B-Instruct --revision v1.0

# Download a quantized variant (GPTQ, AWQ, GGUF)
nexus-llm model download TheBloke/Llama-3.1-8B-Instruct-GPTQ --quantization gptq

# Download to a custom directory
nexus-llm model download meta-llama/Llama-3.1-8B-Instruct --cache-dir /data/models
```

### Using the API

```bash
curl -X POST http://localhost:8000/v1/models/download \
  -H "Authorization: Bearer nxs_sk_abc123" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "meta-llama/Llama-3.1-8B-Instruct"
  }'
```

### Using the Python SDK

```python
from nexus_llm import NexusClient

client = NexusClient()

# Download a model (does not load it into memory)
client.models.download("meta-llama/Llama-3.1-8B-Instruct")

# Check download progress
status = client.models.download_status("dl-a1b2c3d4")
print(f"Progress: {status.progress * 100:.1f}%")
```

### Gated Models

Some models (e.g., Llama, Mistral) require Hugging Face authentication. Set your token before downloading:

```bash
# Option 1: Environment variable
export HF_TOKEN=hf_your_token_here

# Option 2: Hugging Face CLI login
huggingface-cli login

# Option 3: Pass token in the API request
curl -X POST http://localhost:8000/v1/models/download \
  -H "Authorization: Bearer nxs_sk_abc123" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "meta-llama/Llama-3.1-8B-Instruct",
    "hf_token": "hf_your_token_here"
  }'
```

### Recommended Models by Use Case

| Use Case | Model | Size | Context |
|---|---|---|---|
| General Chat | `meta-llama/Llama-3.1-8B-Instruct` | 8B | 128K |
| Lightweight Chat | `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | 32K |
| Code Generation | `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct` | 16B | 128K |
| Embeddings | `BAAI/bge-large-en-v1.5` | 335M | 512 |
| High Quality Chat | `Qwen/Qwen2.5-72B-Instruct` | 72B | 128K |
| Multilingual | `mistralai/Mixtral-8x7B-Instruct-v0.1` | 47B | 32K |

---

## Loading Models

### Automatic Loading

When you start the server or use the chat CLI, models are automatically loaded on first use:

```bash
# Start server with a model (auto-loads)
nexus-llm serve --model meta-llama/Llama-3.1-8B-Instruct

# Chat CLI (auto-downloads and loads if needed)
nexus-llm chat --model meta-llama/Llama-3.1-8B-Instruct
```

### Manual Loading

```bash
# Load a model with default settings (auto device selection)
nexus-llm model load meta-llama/Llama-3.1-8B-Instruct

# Load to specific device
nexus-llm model load meta-llama/Llama-3.1-8B-Instruct --device cuda:0

# Load with specific data type
nexus-llm model load meta-llama/Llama-3.1-8B-Instruct --dtype bfloat16

# Load with 4-bit quantization
nexus-llm model load meta-llama/Llama-3.1-8B-Instruct --quantize bitsandbytes --bits 4

# Load with GPU memory limit
nexus-llm model load meta-llama/Llama-3.1-8B-Instruct --gpu-memory-utilization 0.7
```

### Loading via API

```bash
curl -X POST http://localhost:8000/v1/models/load \
  -H "Authorization: Bearer nxs_sk_abc123" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "meta-llama/Llama-3.1-8B-Instruct",
    "device": "auto",
    "dtype": "float16",
    "gpu_memory_utilization": 0.85
  }'
```

### Loading via Python SDK

```python
from nexus_llm import NexusClient

client = NexusClient()

# Load a model with default settings
client.models.load("meta-llama/Llama-3.1-8B-Instruct")

# Load with custom settings
client.models.load(
    "meta-llama/Llama-3.1-8B-Instruct",
    device="cuda:0",
    dtype="float16",
    quantization={"method": "bitsandbytes", "bits": 4}
)
```

### Device Selection

The `auto` device selection strategy works as follows:

1. If CUDA is available → use `cuda:0`
2. If Apple Silicon (MPS) is available → use `mps`
3. Fallback → use `cpu`

For multi-GPU setups, specify devices explicitly:

```bash
# Load on specific GPU
nexus-llm model load meta-llama/Llama-3.1-70B-Instruct --device cuda:0

# Pipeline parallelism across GPUs (for very large models)
nexus-llm model load meta-llama/Llama-3.1-70B-Instruct \
  --max-memory '{"cuda:0": "40GiB", "cuda:1": "40GiB"}'
```

### Memory Requirements

Approximate memory requirements for common model sizes:

| Model Size | FP32 | FP16 | 8-bit | 4-bit |
|---|---|---|---|---|
| 1.5B | 6 GB | 3 GB | 1.5 GB | 1 GB |
| 7B/8B | 28 GB | 14 GB | 7 GB | 4 GB |
| 13B/14B | 52 GB | 26 GB | 13 GB | 7 GB |
| 70B/72B | 280 GB | 140 GB | 70 GB | 36 GB |

These are model weights only. Add 1-4 GB for KV cache and runtime overhead depending on context length and batch size.

---

## Switching Models

When you need to change the active model without downtime, use the atomic switch operation. This loads the new model before unloading the old one:

```bash
# Switch to a different model
nexus-llm model switch --to Qwen/Qwen2.5-72B-Instruct-AWQ

# Switch with specific settings
nexus-llm model switch \
  --from meta-llama/Llama-3.1-8B-Instruct \
  --to Qwen/Qwen2.5-72B-Instruct-AWQ \
  --device cuda:0 \
  --dtype float16
```

### API Switch

```bash
curl -X POST http://localhost:8000/v1/models/switch \
  -H "Authorization: Bearer nxs_sk_abc123" \
  -H "Content-Type: application/json" \
  -d '{
    "from_model_id": "meta-llama/Llama-3.1-8B-Instruct",
    "to_model_id": "Qwen/Qwen2.5-72B-Instruct-AWQ",
    "device": "cuda:0"
  }'
```

> **Note:** Atomic switching requires enough memory to hold both models simultaneously during the transition. If memory is insufficient, unload the first model first and then load the new one.

### Memory-Constrained Switching

For memory-constrained environments, use unload-then-load instead:

```bash
# Step 1: Unload the current model
nexus-llm model unload meta-llama/Llama-3.1-8B-Instruct

# Step 2: Load the new model
nexus-llm model load Qwen/Qwen2.5-72B-Instruct-AWQ --device cuda:0
```

---

## Quantizing Models

Quantization reduces model size and memory usage at the cost of a small accuracy loss. Nexus-LLM supports multiple quantization methods.

### Quantization Methods Comparison

| Method | Bits | Compression | Accuracy Loss | Speed | Use Case |
|---|---|---|---|---|---|
| None (FP16) | 16 | 1x | None | Baseline | Maximum quality |
| BitsAndBytes 8-bit | 8 | ~2x | Negligible | Fast | Slight memory savings |
| BitsAndBytes 4-bit | 4 | ~4x | Very small | Fast | Good balance |
| GPTQ | 4 | ~4x | Small | Fast | Pre-quantized models |
| AWQ | 4 | ~4x | Very small | Fastest | Production serving |
| GGUF | 2-8 | 2-8x | Varies | Moderate | CPU-only serving |

### On-the-Fly Quantization (BitsAndBytes)

BitsAndBytes quantization is applied during model loading — no separate quantization step needed:

```bash
# Load with 4-bit quantization
nexus-llm model load meta-llama/Llama-3.1-8B-Instruct \
  --quantize bitsandbytes --bits 4

# Load with 8-bit quantization
nexus-llm model load meta-llama/Llama-3.1-8B-Instruct \
  --quantize bitsandbytes --bits 8
```

### Pre-Quantized Models (GPTQ / AWQ)

Download and load pre-quantized models from Hugging Face Hub:

```bash
# Download a GPTQ model
nexus-llm model download TheBloke/Llama-3.1-8B-Instruct-GPTQ

# Load it (quantization is detected automatically)
nexus-llm model load TheBloke/Llama-3.1-8B-Instruct-GPTQ

# Download an AWQ model
nexus-llm model download TheBloke/Llama-3.1-8B-Instruct-AWQ

# Load AWQ model
nexus-llm model load TheBloke/Llama-3.1-8B-Instruct-AWQ
```

### Quantizing a Model Yourself

Create your own quantized model from a full-precision source:

```bash
# GPTQ quantization
nexus-llm model quantize meta-llama/Llama-3.1-8B-Instruct \
  --method gptq \
  --bits 4 \
  --group-size 128 \
  --calibration-samples 128

# AWQ quantization
nexus-llm model quantize meta-llama/Llama-3.1-8B-Instruct \
  --method awq \
  --bits 4 \
  --group-size 128

# GGUF quantization (for llama.cpp / CPU serving)
nexus-llm model quantize meta-llama/Llama-3.1-8B-Instruct \
  --method gguf \
  --gguf-type Q4_K_M
```

### Quantization via API

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

Quantization is an asynchronous process. Monitor progress:

```bash
curl -X GET http://localhost:8000/v1/models/quantize/quant-a1b2c3d4 \
  -H "Authorization: Bearer nxs_sk_abc123"
```

---

## Model Configuration

### Configuration File

Model defaults are configured in `nexus_llm/config/model_config.yaml`:

```yaml
model:
  name: "gpt2-medium"
  device: "auto"
  dtype: "float32"
  trust_remote_code: false

generation:
  temperature: 0.7
  top_p: 0.9
  top_k: 50
  max_tokens: 512
  repetition_penalty: 1.1
  do_sample: true

quantization:
  enabled: false
  bits: 4
  group_size: 128
  quant_method: "gptq"
```

### Runtime Configuration

Override configuration at runtime via the API:

```bash
# Update generation defaults
curl -X POST http://localhost:8000/v1/config \
  -H "Authorization: Bearer nxs_sk_abc123" \
  -H "Content-Type: application/json" \
  -d '{
    "config": {
      "generation": {
        "temperature": 0.5,
        "max_tokens": 1024
      }
    }
  }'
```

---

## Multi-Model Serving

Nexus-LLM can serve multiple models simultaneously when sufficient memory is available.

### Loading Multiple Models

```bash
# Load a chat model on GPU 0
nexus-llm model load meta-llama/Llama-3.1-8B-Instruct --device cuda:0

# Load an embedding model on GPU 1 (or CPU)
nexus-llm model load BAAI/bge-large-en-v1.5 --device cuda:1

# Load a code model on GPU 0 with quantization (shares GPU)
nexus-llm model load deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct \
  --device cuda:0 --quantize bitsandbytes --bits 4
```

### Specifying the Model in Requests

When multiple models are loaded, specify which one to use:

```bash
# Chat with a specific model
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer nxs_sk_abc123" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# Get embeddings from a specific model
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Authorization: Bearer nxs_sk_abc123" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "BAAI/bge-large-en-v1.5",
    "input": "Hello, world!"
  }'
```

### Memory Management

Monitor memory usage across loaded models:

```bash
# Check GPU memory
nvidia-smi

# Check model memory via API
curl http://localhost:8000/v1/health | python -m json.tool

# Estimate memory before loading
curl -X POST http://localhost:8000/v1/models/estimate-memory \
  -H "Authorization: Bearer nxs_sk_abc123" \
  -H "Content-Type: application/json" \
  -d '{"model_id": "meta-llama/Llama-3.1-8B-Instruct", "dtype": "float16"}'
```

---

## Cache Management

### Viewing the Cache

```bash
# List all cached models
nexus-llm model cache list

# Via API
curl http://localhost:8000/v1/models/cache \
  -H "Authorization: Bearer nxs_sk_abc123"
```

### Cleaning the Cache

```bash
# Remove models not accessed in 30 days
nexus-llm model cache clean --older-than 30

# Preview what would be deleted (dry run)
nexus-llm model cache clean --older-than 30 --dry-run

# Remove a specific model's cache
nexus-llm model cache clean --model-id old-model/old-model-v1

# Clean everything except currently loaded models
nexus-llm model cache clean --all
```

---

## Troubleshooting

### Out of Memory (OOM)

```
Error: InsufficientResourcesError - CUDA out of memory
```

**Solutions:**
1. Use quantization: `--quantize bitsandbytes --bits 4`
2. Reduce GPU memory utilization: `--gpu-memory-utilization 0.7`
3. Use a smaller model
4. Offload layers to CPU: `--device auto --max-memory '{"cuda:0": "20GiB", "cpu": "32GiB"}'`

### Model Download Failures

```
Error: Download failed - Connection timeout
```

**Solutions:**
1. Check internet connection and retry
2. Use a Hugging Face mirror: `--mirror https://hf-mirror.com`
3. Download manually and specify local path: `--path /local/path/to/model`

### Slow Loading

Model loading can be slow due to disk I/O and weight conversion. Tips to speed up:

1. **Use SSD storage** for model cache
2. **Pre-quantize models** instead of on-the-fly quantization
3. **Use safetensors format** (faster than PyTorch binaries)
4. **Warm up the model** after server start to avoid cold-start latency

### Incompatible Model Format

```
Error: ModelLoadError - Unsupported model architecture
```

**Solutions:**
1. Ensure the model uses a supported architecture (Llama, Mistral, Qwen, GPT-2, T5, etc.)
2. Enable `trust_remote_code` if the model uses custom code: `--trust-remote-code`
3. Check the model's Hugging Face page for compatibility notes

---

## Related Documentation

- [Models API Reference](../api/models.md) — Full API endpoint documentation
- [Configuration Guide](configuration.md) — Configuration file reference
- [Training Guide](training.md) — Fine-tuning and LoRA
- [Deployment Guide](deployment.md) — Production model serving
