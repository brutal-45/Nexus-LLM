# Getting Started Tutorial

This tutorial guides you through installing Nexus-LLM, loading your first model, and generating responses.

## Prerequisites

- Python 3.9 or higher
- pip package manager
- (Recommended) NVIDIA GPU with CUDA support and 16GB+ VRAM

## Step 1: Install Nexus-LLM

```bash
pip install nexus-llm
```

For GPU support with CUDA 12:

```bash
pip install nexus-llm[cuda12]
```

Verify the installation:

```bash
python -c "import nexus_llm; print(nexus_llm.__version__)"
```

## Step 2: Initialize the Engine

Create a Python file called `hello.py`:

```python
from nexus_llm import InferenceEngine, Conversation

# Initialize the inference engine
engine = InferenceEngine(
    model_name="nexus-7b-chat",
    device="auto",  # Automatically uses GPU if available
)

print("Model loaded successfully!")
```

Run it:

```bash
python hello.py
```

The first run will download the model weights. Subsequent runs will use the cached version.

## Step 3: Your First Chat

Add a conversation to your script:

```python
from nexus_llm import InferenceEngine, Conversation

engine = InferenceEngine(model_name="nexus-7b-chat", device="auto")

# Create a conversation with a system prompt
conversation = Conversation(
    system_prompt="You are a helpful assistant. Be concise and accurate."
)

# Send a message
conversation.add_user_message("What is the capital of Japan?")
response = engine.chat(conversation)

print(f"Assistant: {response.text}")
print(f"Tokens: {response.token_count} | Time: {response.elapsed_time:.2f}s")
```

## Step 4: Multi-Turn Conversation

Conversations maintain context across turns:

```python
# Continue the conversation
conversation.add_user_message("What is its population?")
response = engine.chat(conversation)
print(f"Assistant: {response.text}")

# Check conversation stats
print(f"Turns: {conversation.num_turns}")
print(f"Total tokens: {conversation.total_tokens}")
```

## Step 5: Adjust Generation Parameters

Control the model's behavior with parameters:

```python
# Deterministic, focused response
response = engine.chat(
    conversation,
    temperature=0.1,    # Low temperature = more predictable
    top_p=0.9,          # Nucleus sampling
    max_tokens=200,     # Limit response length
)

# Creative, diverse response
response = engine.chat(
    conversation,
    temperature=0.9,    # High temperature = more creative
    top_p=0.95,
    max_tokens=512,
)
```

## Step 6: Stream Responses

For real-time output, use streaming:

```python
conversation.add_user_message("Tell me about artificial intelligence.")

print("Assistant: ", end="", flush=True)
for chunk in engine.chat_stream(conversation, iterate=True):
    print(chunk.token, end="", flush=True)
    if chunk.is_final:
        print()  # Newline after completion
```

## Step 7: Save and Load Conversations

```python
# Save
conversation.save("my_chat.json")

# Load later
loaded = Conversation.load("my_chat.json")

# Continue the loaded conversation
loaded.add_user_message("Can you elaborate?")
response = engine.chat(loaded)
```

## Next Steps

Now that you have the basics, explore more advanced features:

- **[Fine-Tuning Tutorial](fine_tuning.md)** - Customize a model for your use case
- **[RAG Setup Tutorial](rag_setup.md)** - Build a knowledge-grounded Q&A system
- **[Custom Plugins Tutorial](custom_plugins.md)** - Extend Nexus-LLM with plugins
- **[Model Selection Guide](../guides/model_selection.md)** - Choose the right model
- **[API Reference](../api/endpoints.md)** - Full API documentation
