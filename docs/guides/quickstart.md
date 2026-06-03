# Quick Start Guide

Get up and running with Nexus-LLM in under 5 minutes.

## Prerequisites

- Python 3.9 or higher
- pip package manager
- (Optional) NVIDIA GPU with CUDA 12.x for GPU acceleration

## Step 1: Install Nexus-LLM

```bash
# Basic CPU installation
pip install nexus-llm

# With GPU support (CUDA 12.x)
pip install nexus-llm[gpu]

# With all optional dependencies
pip install nexus-llm[all]
```

## Step 2: Configure Environment

```bash
# Copy the example environment file
cp .env.example .env

# At minimum, set your Hugging Face token for gated models
echo 'HF_TOKEN=hf_your_token_here' >> .env
```

Get a Hugging Face token at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

## Step 3: Start a Chat

### Using the Python SDK

```python
from nexus_llm import NexusClient

# Initialize client (will download model on first use)
client = NexusClient(model="meta-llama/Llama-3.1-8B-Instruct")

# Simple chat
response = client.chat("What is the meaning of life?")
print(response.content)

# Streaming response
for chunk in client.chat("Tell me a story", stream=True):
    print(chunk.text, end="", flush=True)
print()
```

### Using the CLI

```bash
# Interactive chat
nexus-llm chat --model meta-llama/Llama-3.1-8B-Instruct

# One-shot query
nexus-llm chat --message "Explain quantum computing" --model meta-llama/Llama-3.1-8B-Instruct
```

## Step 4: Start the API Server

```bash
# Start server with a model
nexus-llm serve --model meta-llama/Llama-3.1-8B-Instruct --port 8000

# With GPU
nexus-llm serve --model meta-llama/Llama-3.1-8B-Instruct --gpus 0 --quantize awq
```

### Test the Server

```bash
# Health check
curl http://localhost:8000/v1/health

# Chat completion (OpenAI-compatible)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Using with OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="nexus-llm"
)

response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

## Step 5: Explore More

- **[Training Guide](training.md)** — Fine-tune models with LoRA and RLHF
- **[RAG Guide](rag.md)** — Build retrieval-augmented generation pipelines
- **[Agents Guide](agents.md)** — Create AI agents with tool-use capabilities
- **[Deployment Guide](deployment.md)** — Deploy to production with Docker and Kubernetes
- **[Configuration Guide](configuration.md)** — Full configuration reference

## Common Issues

### Out of Memory

If you run out of GPU memory:

```bash
# Use quantization
nexus-llm serve --model meta-llama/Llama-3.1-8B-Instruct --quantize awq

# Reduce GPU memory utilization
nexus-llm serve --model meta-llama/Llama-3.1-8B-Instruct --gpu-memory-utilization 0.7

# Use a smaller model
nexus-llm serve --model Qwen/Qwen2.5-1.5B-Instruct
```

### Model Access

Some models require Hugging Face authentication:

```bash
# Set your HF token
export HF_TOKEN=hf_your_token_here

# Or log in via CLI
huggingface-cli login
```

### Slow First Load

The first time you use a model, it will be downloaded from Hugging Face Hub. This can take several minutes depending on your internet speed and the model size. Subsequent loads will use the cached version.

## Next Steps

- Follow the [Beginner Tutorial](../tutorials/beginner.md) for a hands-on walkthrough
- Read the [REST API Reference](../api/rest.md) for the complete API documentation
- Join our [Discord](https://discord.gg/nexus-llm) for community support
