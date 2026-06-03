# WebSocket API Reference

The Nexus-LLM WebSocket API enables real-time, bidirectional communication for streaming chat, agent interactions, and live updates.

## Connection

```
ws://localhost:8000/api/v1/ws
```

Connect with authentication by including a token in the query string or as the first message:

```
ws://localhost:8000/api/v1/ws?token=eyJhbGciOiJIUzI1NiIs...
```

## Message Format

All messages use JSON with a `type` field for routing:

```json
{
  "type": "message_type",
  "id": "unique-message-id",
  "payload": { ... }
}
```

## Client → Server Messages

### chat.request

Start a chat completion with streaming.

```json
{
  "type": "chat.request",
  "id": "msg-001",
  "payload": {
    "model": "nexus-7b-chat",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello!"}
    ],
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 512
  }
}
```

### chat.cancel

Cancel an in-progress chat request.

```json
{
  "type": "chat.cancel",
  "id": "msg-002",
  "payload": {
    "request_id": "msg-001"
  }
}
```

### agent.request

Start an agent task with tool access.

```json
{
  "type": "agent.request",
  "id": "msg-003",
  "payload": {
    "task": "Calculate 15^3 and find the weather in London",
    "tools": ["calculator", "weather"],
    "max_iterations": 10
  }
}
```

### rag.query

Query the RAG pipeline.

```json
{
  "type": "rag.query",
  "id": "msg-004",
  "payload": {
    "query": "What GPUs are supported?",
    "top_k": 5,
    "include_sources": true
  }
}
```

### subscribe

Subscribe to server events (metrics, model status, etc.).

```json
{
  "type": "subscribe",
  "id": "msg-005",
  "payload": {
    "channels": ["metrics", "model_status"]
  }
}
```

### ping

Keep-alive ping.

```json
{
  "type": "ping",
  "id": "msg-006"
}
```

## Server → Client Messages

### chat.stream

Streaming token from a chat request.

```json
{
  "type": "chat.stream",
  "id": "msg-001",
  "payload": {
    "token": "Hello",
    "token_id": 12345,
    "is_final": false
  }
}
```

### chat.complete

Final message when chat generation is complete.

```json
{
  "type": "chat.complete",
  "id": "msg-001",
  "payload": {
    "full_text": "Hello! How can I help you today?",
    "usage": {
      "prompt_tokens": 25,
      "completion_tokens": 42,
      "total_tokens": 67
    },
    "finish_reason": "stop",
    "elapsed_seconds": 1.23
  }
}
```

### chat.error

Error during chat generation.

```json
{
  "type": "chat.error",
  "id": "msg-001",
  "payload": {
    "error": "model_overloaded",
    "message": "The model is currently overloaded. Please try again.",
    "retry_after_seconds": 5
  }
}
```

### agent.step

Agent reasoning step.

```json
{
  "type": "agent.step",
  "id": "msg-003",
  "payload": {
    "step_type": "tool_call",
    "tool_name": "calculator",
    "tool_input": "15**3",
    "step_number": 1
  }
}
```

### agent.tool_result

Result from an agent tool execution.

```json
{
  "type": "agent.tool_result",
  "id": "msg-003",
  "payload": {
    "tool_name": "calculator",
    "result": "{\"result\": 3375}",
    "step_number": 2
  }
}
```

### agent.complete

Agent task completed.

```json
{
  "type": "agent.complete",
  "id": "msg-003",
  "payload": {
    "answer": "15 cubed is 3375. The weather in London is 59°F and rainy.",
    "steps_taken": 4,
    "tools_used": ["calculator", "weather"]
  }
}
```

### rag.result

RAG query result.

```json
{
  "type": "rag.result",
  "id": "msg-004",
  "payload": {
    "answer": "Nexus-LLM supports NVIDIA A100, H100, and consumer GPUs.",
    "sources": [...],
    "confidence": 0.89
  }
}
```

### metrics.update

Subscribed metrics update (if subscribed).

```json
{
  "type": "metrics.update",
  "payload": {
    "inference_latency_p50": 0.45,
    "inference_latency_p99": 2.1,
    "tokens_per_second": 85.3,
    "gpu_utilization": 0.72,
    "active_requests": 5
  }
}
```

### pong

Response to a ping.

```json
{
  "type": "pong",
  "id": "msg-006"
}
```

## Error Handling

### Connection Errors

| Code | Meaning | Action |
|------|---------|--------|
| 4001 | Authentication failed | Check your token |
| 4002 | Rate limit exceeded | Wait and retry |
| 4003 | Model not available | Check model status |
| 4004 | Invalid message format | Validate your JSON |

### Reconnection

Implement exponential backoff reconnection:

```python
import asyncio
import websockets

async def connect_with_retry(url, max_retries=5):
    for attempt in range(max_retries):
        try:
            ws = await websockets.connect(url)
            return ws
        except Exception as e:
            wait_time = min(2 ** attempt, 30)  # Max 30 seconds
            print(f"Connection failed, retrying in {wait_time}s...")
            await asyncio.sleep(wait_time)
    raise ConnectionError("Max retries exceeded")
```

## Example: Full Chat Session

```javascript
const ws = new WebSocket('ws://localhost:8000/api/v1/ws?token=YOUR_TOKEN');

ws.onopen = () => {
  // Send a chat request
  ws.send(JSON.stringify({
    type: 'chat.request',
    id: 'msg-1',
    payload: {
      model: 'nexus-7b-chat',
      messages: [{role: 'user', content: 'Hello!'}],
      stream: true
    }
  }));
};

ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  switch (msg.type) {
    case 'chat.stream':
      process.stdout.write(msg.payload.token);
      break;
    case 'chat.complete':
      console.log('\nDone!', msg.payload.usage);
      break;
    case 'chat.error':
      console.error('Error:', msg.payload.message);
      break;
  }
};
```
