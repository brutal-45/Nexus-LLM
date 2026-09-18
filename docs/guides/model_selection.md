# Model Selection Guide

Choosing the right model is critical for balancing quality, speed, and cost. This guide helps you select the best model for your use case.

## Available Models

| Model | Parameters | Context Length | VRAM Required | Speed | Quality |
|-------|-----------|----------------|---------------|-------|---------|
| nexus-3b-chat | 3B | 8K | 6 GB | ★★★★★ | ★★☆☆☆ |
| nexus-7b-chat | 7B | 32K | 14 GB | ★★★★☆ | ★★★★☆ |
| nexus-13b-chat | 13B | 32K | 26 GB | ★★★☆☆ | ★★★★★ |
| nexus-7b-code | 7B | 16K | 14 GB | ★★★★☆ | ★★★★☆ (code) |
| nexus-embedding-large | — | 512 | 2 GB | ★★★★★ | ★★★★☆ |
| nexus-vision-large | — | 4K | 18 GB | ★★★☆☆ | ★★★★☆ |

## Decision Framework

### By Use Case

#### General Chat & Q&A
- **Best**: `nexus-7b-chat` — Good balance of speed and quality
- **Fast**: `nexus-3b-chat` — When latency matters more than depth
- **Quality**: `nexus-13b-chat` — When accuracy is paramount

#### Code Generation
- **Best**: `nexus-7b-code` — Specialized for code, supports 30+ languages
- **Fallback**: `nexus-7b-chat` — General purpose, decent at code

#### Embeddings & Search
- **Only choice**: `nexus-embedding-large` — 1024-dimensional embeddings

#### Multimodal (Image + Text)
- **Only choice**: `nexus-vision-large` — Image understanding with text generation

### By Hardware

| GPU | VRAM | Recommended Model | Quantization |
|-----|------|-------------------|-------------|
| RTX 3060 | 12 GB | nexus-3b-chat | None |
| RTX 4070 | 12 GB | nexus-7b-chat | int8 |
| RTX 4090 | 24 GB | nexus-7b-chat | None |
| A100 | 40 GB | nexus-13b-chat | None |
| 2x A100 | 80 GB | nexus-13b-chat + services | None |
| CPU only | — | nexus-3b-chat | int4 |

### By Traffic Volume

| Requests/Day | Recommended Setup |
|-------------|-------------------|
| < 1,000 | Single `nexus-7b-chat` instance |
| 1,000-10,000 | `nexus-7b-chat` with continuous batching |
| 10,000-100,000 | `nexus-3b-fast` for simple queries + `nexus-7b-chat` for complex |
| > 100,000 | Multiple replicas behind a load balancer |

## Quantization Options

Quantization reduces memory usage and increases speed at a minor cost to quality:

| Quantization | Memory Savings | Speed Gain | Quality Impact |
|-------------|---------------|------------|----------------|
| float16 | Baseline | Baseline | None |
| int8 | 50% | 20-40% faster | Minimal |
| int4 (GPTQ) | 75% | 30-60% faster | Slight |
| int4 (AWQ) | 75% | 30-60% faster | Slight |

```python
# Use quantization
engine = InferenceEngine(model_name="nexus-7b-chat", dtype="int8")
```

## Model Routing

For applications with diverse query types, use a model router to automatically select the best model:

```python
from nexus_llm import ModelRegistry

registry = ModelRegistry()
registry.register("fast", model_name="nexus-3b-chat")
registry.register("quality", model_name="nexus-13b-chat")
registry.register("code", model_name="nexus-7b-code")

# Task-based routing
router = registry.create_router(strategy="task-classifier")
engine = router.route("Write a Python function to sort a list")
# Returns the code model
```

## Benchmarks

### MMLU (Knowledge)

| Model | float16 | int8 | int4 |
|-------|---------|------|------|
| nexus-3b-chat | 58.2% | 57.8% | 56.1% |
| nexus-7b-chat | 72.4% | 71.9% | 70.2% |
| nexus-13b-chat | 78.1% | 77.5% | 75.8% |

### HumanEval (Code)

| Model | pass@1 | pass@10 |
|-------|--------|---------|
| nexus-7b-chat | 42.7% | 68.3% |
| nexus-7b-code | 58.1% | 82.4% |

### Throughput (tokens/second on A100)

| Model | float16 | int8 | int4 |
|-------|---------|------|------|
| nexus-3b-chat | 165 | 210 | 245 |
| nexus-7b-chat | 95 | 130 | 155 |
| nexus-13b-chat | 52 | 72 | 88 |

## Recommendations

### For Prototyping
Start with `nexus-7b-chat` in float16. It provides the best quality-to-speed ratio and fits on a single consumer GPU.

### For Production
1. Benchmark your specific workload with your actual prompts.
2. Start with quantization (int8) if you need to reduce costs.
3. Use model routing for mixed workloads.
4. Consider multiple smaller instances instead of one large instance for better availability.

### For Fine-Tuning
Always start with `nexus-7b-chat` as the base. It's large enough to adapt well but small enough to train efficiently with LoRA.
