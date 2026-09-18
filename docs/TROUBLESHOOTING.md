# Troubleshooting Guide

This guide covers common issues and their solutions when using Nexus-LLM.

## Installation Issues

### pip install fails with compilation errors

**Symptoms:** Build errors during `pip install nexus-llm` related to CUDA or C++ extensions.

**Solutions:**
1. Ensure CUDA toolkit matches your PyTorch version:
   ```bash
   python -c "import torch; print(torch.version.cuda)"
   ```
2. Install the correct PyTorch version first:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   ```
3. If you don't need GPU support, install CPU-only:
   ```bash
   pip install nexus-llm[cpu]
   ```

### Import errors after installation

**Symptoms:** `ModuleNotFoundError` or `ImportError` when importing nexus_llm.

**Solutions:**
1. Verify installation:
   ```bash
   python -c "import nexus_llm; print(nexus_llm.__version__)"
   ```
2. Check you're using the correct virtual environment.
3. Reinstall in editable mode:
   ```bash
   pip install -e .
   ```

## Model Loading Issues

### CUDA out of memory when loading a model

**Symptoms:** `torch.cuda.OutOfMemoryError` during model initialization.

**Solutions:**
1. Use quantization:
   ```python
   engine = InferenceEngine(model_name="nexus-7b-chat", dtype="int8")
   # or 4-bit:
   engine = InferenceEngine(model_name="nexus-7b-chat", dtype="int4")
   ```
2. Enable CPU offloading:
   ```python
   engine = InferenceEngine(model_name="nexus-7b-chat", device_map="auto")
   ```
3. Specify max memory:
   ```python
   engine = InferenceEngine(
       model_name="nexus-7b-chat",
       max_memory={0: "12GiB", "cpu": "24GiB"},
   )
   ```

### Model download fails or is slow

**Symptoms:** Timeout or slow download when loading a model for the first time.

**Solutions:**
1. Set a mirror endpoint:
   ```bash
   export HF_ENDPOINT=https://hf-mirror.com
   ```
2. Pre-download the model:
   ```bash
   nexus-llm download nexus-7b-chat
   ```
3. Load from a local path:
   ```python
   engine = InferenceEngine(model_name="./models/nexus-7b-chat")
   ```

### "Model not found" error

**Symptoms:** `ValueError: Model 'xxx' not found in registry`.

**Solutions:**
1. Check the model name is correct (case-sensitive).
2. Register a custom model path:
   ```python
   from nexus_llm import ModelRegistry
   registry = ModelRegistry()
   registry.register_path("my-model", "/path/to/model/files")
   ```

## Inference Issues

### Slow inference speed

**Symptoms:** Low tokens/second throughput.

**Solutions:**
1. Enable Flash Attention 2:
   ```python
   engine = InferenceEngine(model_name="...", flash_attention=True)
   ```
2. Use `float16` instead of `float32` (default with `dtype="auto"`).
3. Enable continuous batching for multi-request scenarios.
4. Check GPU utilization — if low, you may be CPU-bound (check tokenization).

### Repetitive or low-quality outputs

**Symptoms:** Model repeats itself, generates nonsensical text, or gives poor answers.

**Solutions:**
1. Adjust temperature:
   - Lower (0.1-0.3) for factual, deterministic responses
   - Higher (0.7-1.0) for creative, diverse responses
2. Set `top_p` to 0.9-0.95 to filter unlikely tokens.
3. Add repetition penalty:
   ```python
   response = engine.chat(conversation, repetition_penalty=1.15)
   ```
4. Improve your prompt — be specific about the expected format and content.

### Streaming stops unexpectedly

**Symptoms:** Stream ends before the response is complete.

**Solutions:**
1. Increase `max_tokens`:
   ```python
   response = engine.chat_stream(conversation, max_tokens=1024)
   ```
2. Check for timeout settings on the server/client side.
3. Look for errors in the server logs.

## RAG Issues

### Poor retrieval quality

**Symptoms:** Retrieved chunks are not relevant to the query.

**Solutions:**
1. Enable reranking:
   ```python
   retrieval_config = RetrievalConfig(reranking=True, rerank_top_k=3)
   ```
2. Adjust chunk size — smaller chunks (256 tokens) often improve precision.
3. Increase `top_k` to retrieve more candidates.
4. Lower the similarity threshold.

### Vector index build is slow

**Symptoms:** Building the index takes a long time for large document sets.

**Solutions:**
1. Use larger batch sizes for embedding:
   ```python
   embedder = EmbeddingEngine(batch_size=64)
   ```
2. Use GPU-accelerated FAISS:
   ```python
   vector_index = VectorIndex(index_type="faiss", use_gpu=True)
   ```
3. Build incrementally rather than from scratch.

## Server Issues

### Server won't start

**Symptoms:** Server fails to bind to port or crashes on startup.

**Solutions:**
1. Check the port isn't already in use:
   ```bash
   lsof -i :8000
   ```
2. Ensure the model can be loaded before starting the server.
3. Check server logs for specific errors.

### Authentication errors

**Symptoms:** 401 or 403 responses from API calls.

**Solutions:**
1. Verify your API key is correct and active.
2. Check the API key has the required permissions for the endpoint.
3. Ensure the key header name matches the server configuration (default: `X-API-Key`).

## Getting Help

If your issue isn't covered here:

1. Search [GitHub Issues](https://github.com/nexus-llm/Nexus-LLM/issues) for similar problems.
2. Check the [FAQ](FAQ.md) for common questions.
3. Open a new issue with:
   - Nexus-LLM version (`pip show nexus-llm`)
   - Python version (`python --version`)
   - OS and GPU information
   - Minimal reproduction code
   - Full error traceback
