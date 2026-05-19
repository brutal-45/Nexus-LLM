# WebSocket API Reference

Nexus-LLM provides a WebSocket API for low-latency, bidirectional communication. This is ideal for interactive chat applications, real-time streaming, and agent workflows.

## Connection

### Endpoint

```
ws://localhost:8000/v1/ws/chat
```

For secure connections:

```
wss://your-domain.com/v1/ws/chat
```

### Authentication

Authenticate by passing your API key as a query parameter or in the initial connection message:

**Query Parameter:**

```javascript
const ws = new WebSocket('ws://localhost:8000/v1/ws/chat?token=YOUR_API_KEY');
```

**Connection Message:**

```json
{
  "type": "auth",
  "api_key": "YOUR_API_KEY"
}
```

---

## Message Format

All WebSocket messages use JSON format with a `type` field for routing.

### Client → Server Messages

| Type | Description |
|---|---|
| `auth` | Authenticate the connection |
| `chat` | Send a chat message |
| `chat.stream` | Send a chat message with streaming response |
| `completion` | Request a text completion |
| `abort` | Abort the current generation |
| `rag.query` | Query RAG pipeline |
| `agent.execute` | Execute an agent task |
| `ping` | Keep-alive ping |

### Server → Client Messages

| Type | Description |
|---|---|
| `auth.success` | Authentication successful |
| `auth.error` | Authentication failed |
| `chat.response` | Complete chat response |
| `chat.chunk` | Streaming chunk |
| `chat.done` | Generation complete |
| `chat.error` | Error during generation |
| `rag.results` | RAG query results |
| `agent.thinking` | Agent reasoning step |
| `agent.action` | Agent tool execution |
| `agent.response` | Agent final response |
| `pong` | Keep-alive response |
| `error` | General error |

---

## Chat Messages

### Complete Response

Send a chat message and receive the full response at once:

```json
{
  "type": "chat",
  "id": "msg-001",
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is machine learning?"}
  ],
  "temperature": 0.7,
  "max_tokens": 512
}
```

**Response:**

```json
{
  "type": "chat.response",
  "id": "msg-001",
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "message": {
    "role": "assistant",
    "content": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed..."
  },
  "usage": {
    "prompt_tokens": 20,
    "completion_tokens": 150,
    "total_tokens": 170
  },
  "finish_reason": "stop"
}
```

### Streaming Response

Receive tokens as they are generated:

```json
{
  "type": "chat.stream",
  "id": "msg-002",
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "messages": [
    {"role": "user", "content": "Write a haiku about coding."}
  ],
  "temperature": 0.8,
  "max_tokens": 100
}
```

**Streaming Chunks:**

```json
{"type": "chat.chunk", "id": "msg-002", "delta": {"role": "assistant", "content": ""}}
{"type": "chat.chunk", "id": "msg-002", "delta": {"content": "Semicolons"}}
{"type": "chat.chunk", "id": "msg-002", "delta": {"content": " dance"}}
{"type": "chat.chunk", "id": "msg-002", "delta": {"content": " in"}}
{"type": "chat.chunk", "id": "msg-002", "delta": {"content": " loops"}}
{"type": "chat.chunk", "id": "msg-002", "delta": {"content": "\n"}}
{"type": "chat.chunk", "id": "msg-002", "delta": {"content": "Logic"}}
{"type": "chat.chunk", "id": "msg-002", "delta": {"content": " flows"}}
{"type": "chat.chunk", "id": "msg-002", "delta": {"content": " like"}}
{"type": "chat.chunk", "id": "msg-002", "delta": {"content": " water"}}
{"type": "chat.chunk", "id": "msg-002", "delta": {"content": "\n"}}
{"type": "chat.chunk", "id": "msg-002", "delta": {"content": "Bugs"}}
{"type": "chat.chunk", "id": "msg-002", "delta": {"content": " hide"}}
{"type": "chat.chunk", "id": "msg-002", "delta": {"content": " in"}}
{"type": "chat.chunk", "id": "msg-002", "delta": {"content": " plain"}}
{"type": "chat.chunk", "id": "msg-002", "delta": {"content": " sight"}}

{"type": "chat.done", "id": "msg-002", "usage": {"prompt_tokens": 15, "completion_tokens": 18, "total_tokens": 33}, "finish_reason": "stop"}
```

---

## Multi-Turn Conversations

Maintain conversation state across multiple messages:

```json
// Message 1
{
  "type": "chat.stream",
  "id": "msg-003",
  "conversation_id": "conv-abc123",
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "messages": [
    {"role": "user", "content": "My name is Alice."}
  ]
}

// Message 2 (conversation persists)
{
  "type": "chat.stream",
  "id": "msg-004",
  "conversation_id": "conv-abc123",
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "messages": [
    {"role": "user", "content": "What is my name?"}
  ]
}

// Response will remember "Alice"
```

---

## RAG Queries

Query the RAG pipeline via WebSocket:

```json
{
  "type": "rag.query",
  "id": "rag-001",
  "query": "What are the company's remote work policies?",
  "collection": "company-docs",
  "top_k": 5,
  "include_sources": true,
  "stream_answer": true
}
```

**Response:**

```json
{
  "type": "rag.results",
  "id": "rag-001",
  "answer": "Based on the company documents, the remote work policy allows...",
  "sources": [
    {
      "document": "employee_handbook.pdf",
      "page": 42,
      "content": "Employees may work remotely up to 3 days per week...",
      "score": 0.94
    },
    {
      "document": "remote_work_policy.pdf",
      "page": 5,
      "content": "All remote workers must have a stable internet connection...",
      "score": 0.89
    }
  ],
  "usage": {
    "prompt_tokens": 350,
    "completion_tokens": 120,
    "total_tokens": 470
  }
}
```

---

## Agent Execution

Run agent tasks with real-time status updates:

```json
{
  "type": "agent.execute",
  "id": "agent-001",
  "task": "Find the current weather in San Francisco and suggest outdoor activities",
  "agent_type": "react",
  "tools": ["web_search"],
  "max_iterations": 5,
  "stream": true
}
```

**Streaming Agent Events:**

```json
{"type": "agent.thinking", "id": "agent-001", "thought": "I need to search for the current weather in San Francisco."}
{"type": "agent.action", "id": "agent-001", "tool": "web_search", "input": {"query": "current weather San Francisco"}, "observation": "Current weather: 65°F, partly cloudy, wind 10mph"}
{"type": "agent.thinking", "id": "agent-001", "thought": "The weather is nice - 65°F and partly cloudy. Good for outdoor activities."}
{"type": "agent.response", "id": "agent-001", "response": "The current weather in San Francisco is 65°F and partly cloudy with light winds. Great conditions for outdoor activities! I'd suggest:\n1. Golden Gate Park\n2. Walking the Embarcadero\n3. Ferry to Sausalito", "iterations": 2}
```

---

## Aborting Generation

Abort an in-progress generation:

```json
{
  "type": "abort",
  "id": "msg-002"
}
```

**Response:**

```json
{
  "type": "chat.done",
  "id": "msg-002",
  "finish_reason": "abort",
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 8,
    "total_tokens": 23
  }
}
```

---

## Keep-Alive

```json
{"type": "ping"}
```

```json
{"type": "pong", "timestamp": 1703318400}
```

Send a ping every 30 seconds to keep the connection alive. Connections idle for more than 60 seconds will be closed.

---

## Error Handling

```json
{
  "type": "error",
  "id": "msg-005",
  "error": {
    "code": "rate_limit_exceeded",
    "message": "Rate limit of 100 requests/minute exceeded. Retry after 30 seconds.",
    "retry_after": 30
  }
}
```

| Error Code | Description |
|---|---|
| `authentication_failed` | Invalid or missing API key |
| `rate_limit_exceeded` | Too many requests |
| `model_not_found` | Requested model not available |
| `context_length_exceeded` | Input exceeds model context length |
| `server_error` | Internal server error |
| `connection_closed` | Connection unexpectedly closed |

---

## JavaScript Client Example

```javascript
const ws = new WebSocket('ws://localhost:8000/v1/ws/chat?token=YOUR_API_KEY');

ws.onopen = () => {
  console.log('Connected to Nexus-LLM');

  // Send a streaming chat message
  ws.send(JSON.stringify({
    type: 'chat.stream',
    id: 'msg-001',
    model: 'meta-llama/Llama-3.1-8B-Instruct',
    messages: [
      { role: 'user', content: 'Hello, who are you?' }
    ],
    temperature: 0.7,
    max_tokens: 256
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);

  switch (data.type) {
    case 'chat.chunk':
      process.stdout.write(data.delta.content || '');
      break;
    case 'chat.done':
      console.log('\nGeneration complete:', data.usage);
      break;
    case 'error':
      console.error('Error:', data.error.message);
      break;
  }
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};

ws.onclose = () => {
  console.log('Connection closed');
};

// Keep-alive
setInterval(() => {
  if (ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'ping' }));
  }
}, 30000);
```

---

## Python Client Example

```python
import asyncio
import json
import websockets

async def chat_stream():
    uri = "ws://localhost:8000/v1/ws/chat?token=YOUR_API_KEY"
    
    async with websockets.connect(uri) as ws:
        # Send message
        await ws.send(json.dumps({
            "type": "chat.stream",
            "id": "msg-001",
            "model": "meta-llama/Llama-3.1-8B-Instruct",
            "messages": [
                {"role": "user", "content": "Explain quantum computing."}
            ],
            "max_tokens": 256
        }))
        
        # Receive chunks
        async for message in ws:
            data = json.loads(message)
            
            if data["type"] == "chat.chunk":
                print(data["delta"].get("content", ""), end="", flush=True)
            elif data["type"] == "chat.done":
                print(f"\nTokens: {data['usage']['total_tokens']}")
                break
            elif data["type"] == "error":
                print(f"Error: {data['error']['message']}")
                break

asyncio.run(chat_stream())
```
