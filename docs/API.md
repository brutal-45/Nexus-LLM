# Nexus-LLM API Documentation

This document provides comprehensive reference for the Nexus-LLM REST API and WebSocket interface. Whether you're building a chat application, integrating LLM capabilities into your product, or automating workflows, this guide covers everything you need.

---

## 1. Starting the Server

The Nexus-LLM API server is built on FastAPI and Uvicorn, providing high-performance async request handling with automatic OpenAPI documentation.

### Basic Server Start

```bash
# Start with default settings (127.0.0.1:8000)
nexus-llm serve

# Start with custom host and port
nexus-llm serve --host 0.0.0.0 --port 8080

# Start with a specific model
nexus-llm serve --model mistral-7b-instruct

# Start with multiple workers for production
nexus-llm serve --workers 4
```

### Production Deployment

For production deployments, we recommend using Gunicorn as the process manager:

```bash
# Production setup with Gunicorn
gunicorn nexus_llm.backend.server:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 120 \
  --access-logfile - \
  --error-logfile -
```

### Docker Deployment

```bash
# Build the Docker image
docker build -t nexus-llm .

# Run with GPU support
docker run --gpus all -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  nexus-llm

# Run with custom configuration
docker run --gpus all -p 8000:8000 \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/models:/app/models \
  nexus-llm serve --config /app/config/default_config.yaml
```

### Configuration Options

Server behavior can be configured via CLI flags, environment variables, or the config file:

| Setting | CLI Flag | Environment Variable | Default |
|---------|----------|---------------------|---------|
| Host | `--host` | `NEXUS_HOST` | `127.0.0.1` |
| Port | `--port` | `NEXUS_PORT` | `8000` |
| Workers | `--workers` | `NEXUS_WORKERS` | `1` |
| API Key | `--api-key` | `NEXUS_API_KEY` | `null` |
| CORS Origins | `--cors` | `NEXUS_CORS_ORIGINS` | `*` |
| Model | `--model` | `NEXUS_MODEL` | `gpt2-medium` |
| Device | `--device` | `NEXUS_DEVICE` | `auto` |

### Verifying the Server

Once started, verify the server is running:

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "model": "gpt2-medium",
  "device": "cuda:0",
  "uptime_seconds": 42.5,
  "gpu_memory_used_gb": 2.1,
  "gpu_memory_total_gb": 16.0
}
```

### Interactive API Docs

Nexus-LLM automatically generates interactive API documentation:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI Schema**: `http://localhost:8000/openapi.json`

---

## 2. REST Endpoints

### Health Check

Check server status and model information.

**`GET /health`**

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "model": "gpt2-medium",
  "device": "cuda:0",
  "uptime_seconds": 42.5,
  "gpu_memory_used_gb": 2.1,
  "gpu_memory_total_gb": 16.0
}
```

---

### Text Generation

Generate text from a prompt. This is the primary endpoint for single-turn generation tasks.

**`POST /v1/generate`**

**Request Body:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `prompt` | string | Yes | - | The input text prompt |
| `model` | string | No | default | Model to use (overrides server default) |
| `max_length` | integer | No | 512 | Maximum tokens to generate |
| `temperature` | float | No | 0.7 | Sampling temperature (0.0-2.0) |
| `top_p` | float | No | 0.9 | Nucleus sampling threshold |
| `top_k` | integer | No | 50 | Top-K sampling parameter |
| `repetition_penalty` | float | No | 1.1 | Penalty for repeated tokens |
| `num_beams` | integer | No | 1 | Number of beams for beam search |
| `do_sample` | boolean | No | true | Whether to use sampling |
| `stop_sequences` | array | No | [] | Stop generation on these strings |
| `seed` | integer | No | null | Random seed for reproducibility |

**Example:**

```bash
curl -X POST http://localhost:8000/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain the concept of neural networks in simple terms.",
    "max_length": 200,
    "temperature": 0.7,
    "top_p": 0.9
  }'
```

**Response:**

```json
{
  "id": "gen_abc123",
  "generated_text": "Neural networks are computing systems inspired by the human brain. They consist of layers of interconnected nodes (called neurons) that process information...",
  "prompt": "Explain the concept of neural networks in simple terms.",
  "model": "gpt2-medium",
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 87,
    "total_tokens": 99
  },
  "finish_reason": "stop",
  "created": 1705312800
}
```

---

### Chat Completion

Generate a response in a multi-turn conversation. Supports system messages and conversation history.

**`POST /v1/chat/completions`**

**Request Body:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `messages` | array | Yes | - | Array of message objects with `role` and `content` |
| `model` | string | No | default | Model to use |
| `max_length` | integer | No | 512 | Maximum tokens to generate |
| `temperature` | float | No | 0.7 | Sampling temperature |
| `top_p` | float | No | 0.9 | Nucleus sampling threshold |
| `top_k` | integer | No | 50 | Top-K sampling |
| `repetition_penalty` | float | No | 1.1 | Repetition penalty |
| `stop_sequences` | array | No | [] | Stop generation sequences |
| `stream` | boolean | No | false | Enable Server-Sent Events streaming |

**Message Object:**

| Field | Type | Description |
|-------|------|-------------|
| `role` | string | One of: `system`, `user`, `assistant` |
| `content` | string | The message text |

**Example:**

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful coding assistant."},
      {"role": "user", "content": "Write a Python function to reverse a linked list."}
    ],
    "max_length": 500,
    "temperature": 0.7
  }'
```

**Response:**

```json
{
  "id": "chat_abc123",
  "object": "chat.completion",
  "model": "gpt2-medium",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Here's a Python function to reverse a linked list:\n\n```python\nclass ListNode:\n    def __init__(self, val=0, next=None):\n        self.val = val\n        self.next = next\n\ndef reverse_linked_list(head):\n    prev = None\n    current = head\n    while current:\n        next_node = current.next\n        current.next = prev\n        prev = current\n        current = next_node\n    return prev\n```"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 18,
    "completion_tokens": 95,
    "total_tokens": 113
  },
  "created": 1705312800
}
```

---

### Streaming Chat Completion

Stream responses token-by-token using Server-Sent Events (SSE).

**`POST /v1/chat/completions`** with `"stream": true`

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Count from 1 to 5."}
    ],
    "stream": true
  }'
```

**Stream Response (SSE):**

```
data: {"id":"chat_abc123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"id":"chat_abc123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"1,"},"finish_reason":null}]}

data: {"id":"chat_abc123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":" 2,"},"finish_reason":null}]}

data: {"id":"chat_abc123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":" 3,"},"finish_reason":null}]}

data: {"id":"chat_abc123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":" 4,"},"finish_reason":null}]}

data: {"id":"chat_abc123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":" 5."},"finish_reason":null}]}

data: {"id":"chat_abc123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

---

### List Models

List all available and loaded models.

**`GET /v1/models`**

```bash
curl http://localhost:8000/v1/models
```

**Response:**

```json
{
  "object": "list",
  "data": [
    {
      "id": "gpt2-medium",
      "object": "model",
      "status": "loaded",
      "parameters": "355M",
      "vram_gb": 2.1,
      "owned_by": "openai"
    },
    {
      "id": "phi-2",
      "object": "model",
      "status": "available",
      "parameters": "2.7B",
      "vram_gb": null,
      "owned_by": "microsoft"
    }
  ]
}
```

---

### Load Model

Load a model into memory for inference.

**`POST /v1/models/load`**

```bash
curl -X POST http://localhost:8000/v1/models/load \
  -H "Content-Type: application/json" \
  -d '{
    "model": "phi-2",
    "device": "auto",
    "quantization": null
  }'
```

**Response:**

```json
{
  "id": "phi-2",
  "status": "loaded",
  "parameters": "2.7B",
  "vram_gb": 5.8,
  "load_time_seconds": 12.4
}
```

---

### Unload Model

Unload a model from memory to free GPU resources.

**`POST /v1/models/unload`**

```bash
curl -X POST http://localhost:8000/v1/models/unload \
  -H "Content-Type: application/json" \
  -d '{"model": "phi-2"}'
```

**Response:**

```json
{
  "id": "phi-2",
  "status": "unloaded",
  "vram_freed_gb": 5.8
}
```

---

### Tokenize

Tokenize text without running inference. Useful for counting tokens before generation.

**`POST /v1/tokenize`**

```bash
curl -X POST http://localhost:8000/v1/tokenize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, how are you today?",
    "model": "gpt2-medium"
  }'
```

**Response:**

```json
{
  "tokens": [15496, 11, 577, 389, 345, 1909, 30],
  "token_strings": ["Hello", ",", " how", " are", " you", " today", "?"],
  "count": 7
}
```

---

### Get Model Info

Get detailed information about a specific model.

**`GET /v1/models/{model_id}`**

```bash
curl http://localhost:8000/v1/models/gpt2-medium
```

**Response:**

```json
{
  "id": "gpt2-medium",
  "object": "model",
  "status": "loaded",
  "parameters": "355M",
  "architecture": "GPT2LMHeadModel",
  "vocab_size": 50257,
  "hidden_size": 1024,
  "num_layers": 24,
  "num_attention_heads": 16,
  "max_position_embeddings": 1024,
  "vram_used_gb": 2.1,
  "quantization": null,
  "owned_by": "openai",
  "config": {
    "temperature": 0.7,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
    "max_length": 512
  }
}
```

---

## 3. WebSocket Streaming

For real-time, low-latency applications, Nexus-LLM provides a WebSocket interface that supports bidirectional communication with streaming token output.

### Connecting

Connect to the WebSocket endpoint:

```javascript
const ws = new WebSocket("ws://localhost:8000/v1/ws/chat");
```

### Sending Messages

Send chat messages as JSON:

```javascript
ws.send(JSON.stringify({
  type: "chat",
  messages: [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Tell me a story about a robot."}
  ],
  max_length: 500,
  temperature: 0.8,
  stream: true
}));
```

### Receiving Streaming Tokens

Tokens are streamed one at a time as they're generated:

```javascript
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);

  switch (data.type) {
    case "token":
      // Individual token received
      process.stdout.write(data.content);
      break;

    case "done":
      // Generation complete
      console.log("\n--- Generation complete ---");
      console.log("Tokens:", data.usage.completion_tokens);
      console.log("Time:", data.timing.total_seconds, "seconds");
      break;

    case "error":
      console.error("Error:", data.message);
      break;
  }
};
```

### Full WebSocket Example

```javascript
// Complete browser-based chat example
class NexusChat {
  constructor(url = "ws://localhost:8000/v1/ws/chat") {
    this.url = url;
    this.ws = null;
    this.messages = [];
  }

  connect() {
    this.ws = new WebSocket(this.url);

    this.ws.onopen = () => {
      console.log("Connected to Nexus-LLM");
    };

    this.ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === "token") {
        this.onToken(data.content);
      } else if (data.type === "done") {
        this.onComplete(data);
      } else if (data.type === "error") {
        this.onError(data.message);
      }
    };

    this.ws.onclose = () => {
      console.log("Disconnected");
      // Auto-reconnect after 3 seconds
      setTimeout(() => this.connect(), 3000);
    };
  }

  send(content) {
    this.messages.push({"role": "user", "content": content});
    this.ws.send(JSON.stringify({
      type: "chat",
      messages: this.messages,
      max_length: 512,
      temperature: 0.7,
      stream: true
    }));
  }

  onToken(token) { /* Override */ }
  onComplete(data) { /* Override */ }
  onError(message) { /* Override */ }
}

// Usage
const chat = new NexusChat();
chat.onToken = (token) => process.stdout.write(token);
chat.onComplete = (data) => {
  this.messages.push({"role": "assistant", "content": data.full_text});
};
chat.connect();
chat.send("Hello! Tell me about yourself.");
```

### WebSocket Message Types

**Client → Server:**

| Type | Description | Fields |
|------|-------------|--------|
| `chat` | Start a chat completion | `messages`, `max_length`, `temperature`, `top_p`, `stream` |
| `generate` | Start a text generation | `prompt`, `max_length`, `temperature`, `top_p` |
| `stop` | Stop current generation | - |
| `ping` | Keep-alive ping | - |

**Server → Client:**

| Type | Description | Fields |
|------|-------------|--------|
| `token` | A single generated token | `content`, `token_id` |
| `done` | Generation completed | `full_text`, `usage`, `timing` |
| `error` | An error occurred | `message`, `code` |
| `pong` | Response to ping | - |
| `model_info` | Model loaded/changed | `model`, `status`, `vram_gb` |

---

## 4. Authentication

Nexus-LLM supports API key authentication to protect your server from unauthorized access.

### Enabling Authentication

Set an API key when starting the server:

```bash
# Via CLI flag
nexus-llm serve --api-key "sk-your-secret-key-here"

# Via environment variable
export NEXUS_API_KEY="sk-your-secret-key-here"
nexus-llm serve

# Via config file
nexus-llm config --set server.api_key "sk-your-secret-key-here"
nexus-llm serve
```

### Using API Keys

Include the API key in the `Authorization` header:

```bash
curl -X POST http://localhost:8000/v1/generate \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-your-secret-key-here" \
  -d '{
    "prompt": "Hello, world!",
    "max_length": 50
  }'
```

### Authentication Error

If an API key is required but missing or invalid:

```json
{
  "error": {
    "type": "authentication_error",
    "message": "Invalid or missing API key. Include 'Authorization: Bearer <key>' header."
  }
}
```

### Health Endpoint

The `/health` endpoint does not require authentication, allowing monitoring systems to check server status without credentials.

### Best Practices

- **Use strong keys**: Generate keys with at least 32 characters of randomness
- **Rotate keys regularly**: Change API keys periodically
- **Use HTTPS**: Always use TLS in production to protect keys in transit
- **Never commit keys**: Store keys in environment variables or secret managers
- **Restrict by IP**: Combine API keys with network-level access controls

---

## 5. Python Client

Nexus-LLM provides a Python client library for programmatic access to the API. It handles connection management, retries, streaming, and error handling automatically.

### Installation

```bash
pip install nexus-llm
```

The client is included in the main package — no separate installation needed.

### Basic Usage

```python
from nexus_llm.api.client import NexusClient

# Initialize the client
client = NexusClient(
    base_url="http://localhost:8000",
    api_key="sk-your-secret-key-here"  # Optional
)

# Simple text generation
response = client.generate(
    prompt="Write a poem about the ocean.",
    max_length=200,
    temperature=0.8
)
print(response.generated_text)
```

### Chat Completions

```python
# Multi-turn conversation
response = client.chat_completion(
    messages=[
        {"role": "system", "content": "You are a helpful math tutor."},
        {"role": "user", "content": "Explain the Pythagorean theorem."},
    ],
    max_length=500,
    temperature=0.7
)
print(response.choices[0].message.content)

# Continue the conversation
response = client.chat_completion(
    messages=[
        {"role": "system", "content": "You are a helpful math tutor."},
        {"role": "user", "content": "Explain the Pythagorean theorem."},
        {"role": "assistant", "content": response.choices[0].message.content},
        {"role": "user", "content": "Can you give me an example?"},
    ],
    max_length=500
)
print(response.choices[0].message.content)
```

### Streaming

```python
# Stream tokens as they're generated
for chunk in client.chat_completion_stream(
    messages=[
        {"role": "user", "content": "Tell me a long story."}
    ],
    max_length=1000
):
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()  # Newline after completion
```

### Model Management

```python
# List available models
models = client.list_models()
for model in models.data:
    print(f"{model.id}: {model.status} ({model.parameters})")

# Load a model
client.load_model("phi-2")

# Get model info
info = client.get_model("phi-2")
print(f"Architecture: {info.architecture}")
print(f"VRAM: {info.vram_used_gb} GB")

# Unload a model
client.unload_model("phi-2")
```

### Tokenization

```python
# Count tokens before generation
result = client.tokenize("Hello, how are you?", model="gpt2-medium")
print(f"Token count: {result.count}")
print(f"Tokens: {result.token_strings}")
```

### Error Handling

```python
from nexus_llm.api.client import NexusClient, NexusAPIError, ConnectionError

client = NexusClient(base_url="http://localhost:8000")

try:
    response = client.generate(
        prompt="Hello!",
        max_length=50
    )
except NexusAPIError as e:
    print(f"API Error: {e.status_code} - {e.message}")
except ConnectionError as e:
    print(f"Connection Error: {e}")
```

### Async Client

For high-throughput applications, use the async client:

```python
import asyncio
from nexus_llm.api.client import AsyncNexusClient

async def main():
    async with AsyncNexusClient(base_url="http://localhost:8000") as client:
        # Concurrent requests
        tasks = [
            client.generate(prompt=f"Write about topic {i}", max_length=100)
            for i in range(5)
        ]
        results = await asyncio.gather(*tasks)
        for i, result in enumerate(results):
            print(f"Result {i}: {result.generated_text[:80]}...")

asyncio.run(main())
```

---

## 6. cURL Examples

A collection of ready-to-use cURL commands for testing and scripting against the Nexus-LLM API.

### Basic Generation

```bash
curl -X POST http://localhost:8000/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The meaning of life is",
    "max_length": 100
  }'
```

### Chat with System Prompt

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a sarcastic but helpful assistant."},
      {"role": "user", "content": "What is the best programming language?"}
    ],
    "temperature": 0.9,
    "max_length": 300
  }'
```

### Code Generation

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are an expert Python developer. Write clean, documented code."},
      {"role": "user", "content": "Write a function that finds all anagrams of a word in a list."}
    ],
    "max_length": 500,
    "temperature": 0.3,
    "repetition_penalty": 1.15
  }'
```

### Creative Writing with High Temperature

```bash
curl -X POST http://localhost:8000/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "In a world where gravity reverses every 24 hours,",
    "max_length": 500,
    "temperature": 1.2,
    "top_p": 0.95,
    "repetition_penalty": 1.2
  }'
```

### Deterministic Output (Low Temperature)

```bash
curl -X POST http://localhost:8000/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is 2+2?",
    "max_length": 20,
    "temperature": 0.0,
    "do_sample": false
  }'
```

### Streaming Response

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Explain quantum entanglement step by step."}
    ],
    "stream": true,
    "max_length": 500
  }'
```

### With Authentication

```bash
curl -X POST http://localhost:8000/v1/generate \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-your-secret-key-here" \
  -d '{
    "prompt": "Hello!",
    "max_length": 50
  }'
```

### Using a Specific Model

```bash
curl -X POST http://localhost:8000/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain machine learning in one sentence.",
    "model": "phi-2",
    "max_length": 100,
    "temperature": 0.5
  }'
```

### Stop Sequences

```bash
curl -X POST http://localhost:8000/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Question: What is photosynthesis?\nAnswer:",
    "max_length": 200,
    "stop_sequences": ["\nQuestion:", "\n\n"]
  }'
```

### Beam Search

```bash
curl -X POST http://localhost:8000/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Translate to French: The cat sat on the mat.",
    "max_length": 50,
    "num_beams": 5,
    "do_sample": false
  }'
```

### Reproducible Output with Seed

```bash
curl -X POST http://localhost:8000/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Once upon a time",
    "max_length": 100,
    "temperature": 0.7,
    "seed": 42
  }'
```

### Load and Use a Model

```bash
# Load a model
curl -X POST http://localhost:8000/v1/models/load \
  -H "Content-Type: application/json" \
  -d '{"model": "mistral-7b-instruct"}'

# Use it
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistral-7b-instruct",
    "messages": [
      {"role": "user", "content": "What are the benefits of functional programming?"}
    ],
    "max_length": 400
  }'

# Unload when done
curl -X POST http://localhost:8000/v1/models/unload \
  -H "Content-Type: application/json" \
  -d '{"model": "mistral-7b-instruct"}'
```

### Token Count Estimation

```bash
curl -X POST http://localhost:8000/v1/tokenize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This is a sample text to count tokens for.",
    "model": "gpt2-medium"
  }'
```

### Batch Requests with jq

Process multiple prompts from a JSON file:

```bash
# prompts.json: ["Prompt 1", "Prompt 2", "Prompt 3"]
jq -c '.[]' prompts.json | while read -r prompt; do
  curl -s -X POST http://localhost:8000/v1/generate \
    -H "Content-Type: application/json" \
    -d "{\"prompt\": $prompt, \"max_length\": 100}" | \
    jq -r '.generated_text'
  echo "---"
done
```
