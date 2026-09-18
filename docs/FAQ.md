# Frequently Asked Questions (FAQ)

## General

### What is Nexus-LLM?

Nexus-LLM is an open-source framework for building, fine-tuning, deploying, and monitoring large language model applications. It provides a unified API for inference, RAG pipelines, agent systems, and model serving.

### Which models are supported?

Nexus-LLM supports any HuggingFace-compatible transformer model, including:
- LLaMA 2/3 and variants
- Mistral and Mixtral
- Qwen series
- Phi series
- Any model with a compatible tokenizer and architecture

Custom architectures (e.g., Mamba) are supported via the plugin system.

### What hardware do I need?

- **Minimum**: 8GB RAM, any modern CPU (for 3B models with quantization)
- **Recommended**: 16GB+ RAM, NVIDIA GPU with 16GB+ VRAM (e.g., RTX 4090, A100)
- **Multi-GPU**: Supported for models larger than 13B parameters

### Is Nexus-LLM free to use?

Yes, Nexus-LLM is open-source and released under the Apache 2.0 license. You are free to use, modify, and distribute it for both personal and commercial purposes.

## Installation & Setup

### How do I install Nexus-LLM?

```bash
pip install nexus-llm
```

For GPU support with CUDA 12:
```bash
pip install nexus-llm[cuda12]
```

### I'm getting CUDA out-of-memory errors. What can I do?

1. Enable quantization: `InferenceEngine(model_name="...", dtype="int8")`
2. Reduce batch size
3. Enable gradient checkpointing during training
4. Use a smaller model
5. Try Flash Attention 2: `InferenceEngine(model_name="...", flash_attention=True)`

### Can I run Nexus-LLM without a GPU?

Yes, but performance will be significantly slower. Use quantized models and reduce `max_tokens` for acceptable CPU performance.

## Inference

### How do I stream responses?

Use `engine.chat_stream()` instead of `engine.chat()`:

```python
for chunk in engine.chat_stream(conversation, iterate=True):
    print(chunk.token, end="", flush=True)
```

### How do I use a fine-tuned LoRA adapter?

```python
engine = InferenceEngine(
    model_name="nexus-7b-chat",
    adapter_path="./path/to/lora/adapter",
)
```

### Can I run multiple models simultaneously?

Yes, use `ModelRegistry` to manage multiple models:

```python
registry = ModelRegistry()
registry.register(name="fast", model_name="nexus-3b-chat")
registry.register(name="quality", model_name="nexus-13b-chat")
```

## RAG

### What vector stores are supported?

- FAISS (default, in-memory)
- HNSWLib (in-memory, faster for large datasets)
- ChromaDB (persistent, with metadata filtering)

### How do I update the vector index?

Use incremental updates without rebuilding the entire index:

```python
new_chunks = doc_store.ingest_text("New document text...")
vector_index.add(new_chunks)
```

### What chunking strategies are available?

- **Recursive**: Splits on paragraph/line/sentence boundaries (recommended)
- **Fixed-size**: Splits at exact token counts
- **Semantic**: Splits based on semantic coherence (slower but higher quality)

## Fine-Tuning

### How much data do I need for LoRA fine-tuning?

- **Minimum**: 100-500 high-quality examples
- **Recommended**: 1,000-10,000 examples
- **Quality > quantity**: Clean, well-formatted data produces better results

### What LoRA rank should I use?

| Rank | Use Case | Trainable Params |
|------|----------|-----------------|
| 8 | Simple tasks, limited data | ~4M |
| 16 | General purpose (recommended) | ~8M |
| 32 | Complex tasks, more data | ~16M |
| 64 | Very complex tasks | ~32M |

## Agents

### How do I define custom tools?

Subclass the `Tool` class and implement the `run` method:

```python
class MyTool(Tool):
    name = "my_tool"
    description = "What this tool does"

    def run(self, input: str) -> str:
        return "result"
```

### How do I prevent infinite agent loops?

Set `max_iterations` in `AgentConfig`:

```python
agent = Agent(
    inference_engine=engine,
    tool_registry=registry,
    config=AgentConfig(max_iterations=10),
)
```

## Deployment

### How do I deploy Nexus-LLM in production?

See the [Deployment Guide](guides/deployment.md) for detailed instructions covering Docker, Kubernetes, and cloud deployments.

### Does Nexus-LLM support Kubernetes?

Yes, we provide Helm charts and Kubernetes manifests in the `deploy/` directory. Horizontal Pod Autoscaling is supported with GPU utilization metrics.

### How do I monitor my deployment?

Use the built-in Prometheus exporter and monitoring dashboard. See the [Monitoring Setup Guide](guides/monitoring_setup.md) for details.
