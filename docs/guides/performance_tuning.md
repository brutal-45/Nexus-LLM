# Performance Tuning Guide

Optimize Nexus-LLM for maximum throughput, lowest latency, and efficient resource utilization.

## Inference Optimization

### Flash Attention 2

Flash Attention 2 reduces memory usage and speeds up attention computation:

```python
engine = InferenceEngine(
    model_name="nexus-7b-chat",
    flash_attention=True,    # Enable Flash Attention 2
)
```

**Expected improvement**: 2-3x faster attention, 50% less memory for long sequences.

### Quantization

Reduce memory and increase throughput with quantization:

```python
# INT8 quantization (minimal quality loss)
engine = InferenceEngine(model_name="nexus-7b-chat", dtype="int8")

# INT4 quantization (slight quality loss, large memory savings)
engine = InferenceEngine(model_name="nexus-7b-chat", dtype="int4")
```

| Quantization | Memory | Speed | Quality |
|-------------|--------|-------|---------|
| float16 | 14 GB | 95 tok/s | Baseline |
| int8 | 7 GB | 130 tok/s | ~99% of fp16 |
| int4 | 4 GB | 155 tok/s | ~97% of fp16 |

### KV-Cache Optimization

```python
engine = InferenceEngine(
    model_name="nexus-7b-chat",
    kv_cache_dtype="float8",      # Reduce KV cache memory by 50%
    max_batch_size=32,             # Pre-allocate for expected batch size
)
```

### Continuous Batching

For server deployments, continuous batching significantly improves throughput:

```python
from nexus_llm.server import NexusServer, ServerConfig

server_config = ServerConfig(
    continuous_batching=True,
    max_batch_size=32,
    scheduling_policy="fcfs",      # First-come-first-served
)
```

## Memory Optimization

### Gradient Checkpointing (Training)

Trade compute for memory during fine-tuning:

```python
trainer = Trainer(
    training_args={
        "gradient_checkpointing": True,   # ~30% memory savings
        "gradient_accumulation_steps": 4, # Effective batch size = 4 * per_device_batch_size
    },
)
```

### CPU Offloading

Offload model weights to CPU when GPU memory is limited:

```python
engine = InferenceEngine(
    model_name="nexus-13b-chat",
    device_map="auto",           # Automatically distribute across GPU/CPU
    max_memory={0: "20GiB", "cpu": "32GiB"},
)
```

### Paged Attention

Reduce memory fragmentation for long-context inference:

```python
engine = InferenceEngine(
    model_name="nexus-7b-chat",
    paged_attention=True,
    gpu_memory_utilization=0.9,   # Use 90% of GPU memory
)
```

## Server Tuning

### Worker Configuration

```yaml
server:
  workers: 4                        # Match CPU core count
  max_concurrent_requests: 100      # Queue size
  request_timeout: 120              # Seconds
```

### Batch Size Optimization

| Scenario | Recommended Batch Size |
|----------|----------------------|
| Low latency (chat) | 1-4 |
| Balanced | 8-16 |
| High throughput (batch) | 32-64 |

### Thread Pool Tuning

```python
server_config = ServerConfig(
    inference_threads=4,          # Threads for model inference
    io_threads=2,                 # Threads for I/O operations
)
```

## RAG Pipeline Optimization

### Chunking

```python
# Smaller chunks = faster retrieval, more precise matching
chunking = ChunkingStrategy(chunk_size=256, chunk_overlap=32)

# Larger chunks = more context per result, fewer lookups
chunking = ChunkingStrategy(chunk_size=512, chunk_overlap=64)
```

### Embedding Batch Size

```python
# Increase batch size for faster indexing (uses more GPU memory)
embedder = EmbeddingEngine(batch_size=64)

# Decrease if running out of memory
embedder = EmbeddingEngine(batch_size=16)
```

### Vector Index Optimization

```python
# FAISS with GPU acceleration
vector_index = VectorIndex(index_type="faiss", use_gpu=True)

# HNSWLib for fast approximate search on CPU
vector_index = VectorIndex(
    index_type="hnswlib",
    ef_construction=200,   # Higher = better recall, slower build
    M=32,                  # Higher = better recall, more memory
)
```

### Caching

Enable response caching for frequently asked questions:

```python
pipeline = RAGPipeline(
    ...,
    cache_responses=True,
    cache_ttl_seconds=3600,     # Cache for 1 hour
    cache_max_size=1000,
)
```

## Benchmarking Your Setup

Always benchmark with your actual workload:

```python
from nexus_llm.benchmark import BenchmarkRunner, BenchmarkConfig

config = BenchmarkConfig(
    warmup_iterations=3,
    benchmark_iterations=10,
    batch_sizes=[1, 4, 8],
    sequence_lengths=[128, 512, 2048],
)

runner = BenchmarkRunner(config=config)
results = runner.run(engine, model_label="my-setup")
results.print_summary()
```

## Performance Checklist

- [ ] Enable Flash Attention 2
- [ ] Use appropriate quantization (int8 for production)
- [ ] Set `dtype="auto"` to use float16 by default
- [ ] Enable continuous batching for multi-user scenarios
- [ ] Pre-allocate KV-cache with expected batch size
- [ ] Use gradient checkpointing during training
- [ ] Tune batch sizes for your GPU and workload
- [ ] Enable response caching for repeated queries
- [ ] Monitor GPU utilization (target 80-95%)
- [ ] Benchmark with your actual prompts and sequence lengths
