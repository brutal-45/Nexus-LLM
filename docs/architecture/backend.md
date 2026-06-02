# Backend Architecture

Deep dive into the backend systems: model management, inference pipeline, caching, and the API layer.

---

## Model Manager

The Model Manager is responsible for loading, unloading, and caching language models. It abstracts the complexity of dealing with different model formats, quantization methods, and device placement.

### Architecture

```
┌─────────────────────────────────────────────────┐
│                  Model Manager                  │
│                                                 │
│  ┌──────────────────────────────────────────┐   │
│  │           Model Registry                 │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐  │   │
│  │  │ Model A  │ │ Model B  │ │ Model C  │  │   │
│  │  │ (loaded) │ │ (loaded) │ │ (cached) │  │   │
│  │  └──────────┘ └──────────┘ └──────────┘  │   │
│  └──────────────────────────────────────────┘   │
│                                                 │
│  ┌──────────┐  ┌───────────┐  ┌──────────────┐  │
│  │  Loader  │  │  Evictor  │  │   Profiler   │  │
│  └──────────┘  └───────────┘  └──────────────┘  │ 
│                                                 │
│  ┌──────────┐  ┌───────────┐  ┌──────────────┐  │
│  │ Quantizer│  │  Sharder  │  │  Downloader  │  │
│  └──────────┘  └───────────┘  └──────────────┘  │ 
└─────────────────────────────────────────────────┘
```

### Key Components

#### Model Registry

Maintains the state of all known models:

```python
class ModelRegistry:
    """Registry tracking all known models and their states."""

    # Model states:
    # - "not_downloaded"  : Model ID known but not locally available
    # - "downloading"     : Currently downloading
    # - "downloaded"      : Downloaded but not loaded in memory
    # - "loading"         : Currently loading into memory
    # - "loaded"          : In memory and ready for inference
    # - "unloading"       : Being removed from memory
    # - "error"           : Failed to load

    def register(self, model_id: str, metadata: ModelMetadata) -> None: ...
    def get_state(self, model_id: str) -> ModelState: ...
    def get_loaded_models(self) -> list[ModelInfo]: ...
    def get_model_info(self, model_id: str) -> ModelInfo: ...
```

#### Loader

Handles the complex process of loading a model:

```python
class ModelLoader:
    """Loads models with support for quantization, sharding, and offloading."""

    def load(self, model_id: str, config: LoadConfig) -> LoadedModel:
        """
        Loading pipeline:
        1. Resolve model path (local cache or HuggingFace Hub)
        2. Read model config (config.json)
        3. Determine device map (auto, sequential, custom)
        4. Apply quantization config (4bit, 8bit, GPTQ, AWQ)
        5. Load model weights
        6. Load tokenizer
        7. Warm up model (optional forward pass)
        8. Register in ModelRegistry
        """
        ...
```

#### Evictor (LRU Cache)

When GPU memory is full, the evictor removes the least-recently-used model:

```python
class ModelEvictor:
    """LRU-based model eviction when GPU memory is constrained."""

    def __init__(self, max_memory_gb: float):
        self.max_memory = max_memory_gb
        self.usage_map: OrderedDict[str, float] = OrderedDict()

    def touch(self, model_id: str) -> None:
        """Mark a model as recently used."""
        self.usage_map.move_to_end(model_id)

    def evict_if_needed(self, required_gb: float) -> list[str]:
        """Evict models until enough memory is free. Returns evicted model IDs."""
        evicted = []
        while self._available_memory() < required_gb and self.usage_map:
            model_id, _ = self.usage_map.popitem(last=False)  # Remove LRU
            evicted.append(model_id)
        return evicted
```

---

## Inference Engine

The Inference Engine manages the generation process, from tokenization to sampling to detokenization.

### Architecture

```
┌───────────────────────────────────────────────────────┐
│                   Inference Engine                     │
│                                                        │
│  Input                                                 │
│    │                                                   │
│    ▼                                                   │
│  ┌──────────┐                                         │
│  │ Tokenizer │  Convert text → token IDs              │
│  └────┬─────┘                                         │
│       │                                                │
│       ▼                                                │
│  ┌──────────────────────────────────────┐             │
│  │         Model Forward Pass            │             │
│  │  ┌────────┐  ┌────────┐  ┌────────┐ │             │
│  │  │Layer 0 │→ │Layer 1 │→ │...Layer│ │             │
│  │  │(GPU:0) │  │(GPU:0) │  │ N      │ │             │
│  │  └────────┘  └────────┘  └────────┘ │             │
│  └──────────────┬───────────────────────┘             │
│                 │                                      │
│                 ▼                                      │
│  ┌──────────────────────────────────────┐             │
│  │           Sampling Layer              │             │
│  │  ┌─────────┐ ┌──────┐ ┌───────────┐ │             │
│  │  │Temp/TopP│ │Top-K │ │Repetition │ │             │
│  │  │ Filter  │ │Filter│ │Penalty    │ │             │
│  │  └─────────┘ └──────┘ └───────────┘ │             │
│  └──────────────┬───────────────────────┘             │
│                 │                                      │
│                 ▼                                      │
│  ┌──────────┐                                         │
│  │Stopping  │  Check: stop tokens, max length, etc.  │
│  │Criteria  │                                         │
│  └────┬─────┘                                         │
│       │                                                │
│       ▼                                                │
│  ┌──────────┐                                         │
│  │Detokenize│  Convert token IDs → text               │
│  └────┬─────┘                                         │
│       │                                                │
│       ▼                                                │
│  Output                                                │
└───────────────────────────────────────────────────────┘
```

### Inference Modes

#### Synchronous (Batch)

```python
class SyncInference:
    """Standard synchronous inference for single requests."""

    def generate(self, prompt: str, params: GenerateParams) -> str:
        input_ids = self.tokenizer.encode(prompt)
        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=params.max_new_tokens,
            temperature=params.temperature,
            top_p=params.top_p,
            top_k=params.top_k,
            repetition_penalty=params.repetition_penalty,
        )
        return self.tokenizer.decode(output_ids[0][len(input_ids[0]):])
```

#### Streaming (Token-by-Token)

```python
class StreamInference:
    """Streaming inference that yields tokens as they're generated."""

    async def generate_stream(self, prompt: str, params: GenerateParams):
        input_ids = self.tokenizer.encode(prompt)

        async for token_id in self.model.stream_generate(input_ids, params):
            token_text = self.tokenizer.decode([token_id])
            yield StreamToken(content=token_text, token_id=token_id)

            if self._should_stop(token_id, params.stop_tokens):
                break
```

#### Batched (Multiple Requests)

```python
class BatchedInference:
    """Continuous batching for high-throughput serving."""

    def __init__(self, model, max_batch_size=32):
        self.model = model
        self.max_batch_size = max_batch_size
        self.pending_queue = asyncio.Queue()

    async def submit(self, request: InferenceRequest) -> AsyncIterator[StreamToken]:
        """Submit a request and get a stream of tokens back."""
        future = asyncio.Future()
        await self.pending_queue.put((request, future))
        return self._stream_from_future(future)

    async def _batch_loop(self):
        """Continuously process batches of requests."""
        while True:
            batch = await self._collect_batch()
            results = await self.model.batch_generate(batch)
            self._dispatch_results(results)
```

### KV Cache Management

```python
class KVCacheManager:
    """Manages the Key-Value cache for efficient inference."""

    def __init__(self, config: CacheConfig):
        self.cache_type = config.type  # "paged", "unified"
        self.block_size = config.block_size
        self.gpu_memory_utilization = config.gpu_memory_utilization

    def allocate(self, sequence_length: int) -> CacheBlocks:
        """Allocate cache blocks for a new sequence."""
        ...

    def free(self, blocks: CacheBlocks) -> None:
        """Free cache blocks when a sequence completes."""
        ...

    def get_memory_usage(self) -> MemoryInfo:
        """Return current cache memory usage statistics."""
        ...
```

---

## Caching Layer

### Response Cache

Cache identical requests to avoid redundant inference:

```python
class ResponseCache:
    """LRU cache for inference responses."""

    def __init__(self, max_size_mb: int = 512):
        self.cache: OrderedDict[str, CachedResponse] = OrderedDict()
        self.max_size = max_size_mb * 1024 * 1024
        self.current_size = 0

    def _cache_key(self, request: ChatRequest) -> str:
        """Generate a deterministic cache key from the request."""
        key_parts = [
            request.model,
            json.dumps(request.messages, sort_keys=True),
            str(request.temperature),
            str(request.top_p),
            str(request.max_tokens),
            str(request.seed) if request.seed else "random",
        ]
        return hashlib.sha256("|".join(key_parts).encode()).hexdigest()

    def get(self, request: ChatRequest) -> Optional[CachedResponse]:
        """Check cache for a matching response."""
        key = self._cache_key(request)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, request: ChatRequest, response: ChatResponse) -> None:
        """Cache a response."""
        key = self._cache_key(request)
        response_size = len(json.dumps(response.dict()).encode())
        self._evict_until_fits(response_size)
        self.cache[key] = CachedResponse(response=response, size=response_size)
        self.current_size += response_size
```

### Model Weight Cache

HuggingFace models are cached locally to avoid re-downloading:

```
cache/models/
├── models--meta-llama--Llama-3.1-8B-Instruct/
│   ├── refs/
│   │   └── main
│   ├── blobs/
│   │   ├── abc123...  (model weights)
│   │   ├── def456...  (tokenizer)
│   │   └── ghi789...  (config)
│   └── snapshots/
│       └── abc123.../
│           ├── config.json -> ../../blobs/ghi789...
│           ├── model.safetensors -> ../../blobs/abc123...
│           └── tokenizer.json -> ../../blobs/def456...
```

---

## API Layer

### Request Lifecycle

```
1. HTTP Request arrives at Uvicorn
       │
       ▼
2. ASGI middleware chain
       ├── CORS middleware
       ├── Request logging middleware
       ├── Authentication middleware
       │   ├── Extract Bearer token or API key
       │   ├── Validate against key store
       │   └── Attach user context to request
       └── Rate limiting middleware
           ├── Check rate limit counters
           └── Add X-RateLimit-* headers
       │
       ▼
3. FastAPI route handler
       ├── Pydantic validates request body
       ├── Business logic execution
       └── Pydantic serializes response
       │
       ▼
4. HTTP Response returned
```

### Authentication Middleware

```python
class AuthMiddleware:
    """Validates API keys and JWT tokens on every request."""

    async def __call__(self, request: Request, call_next):
        # Skip auth for health endpoint
        if request.url.path == "/health":
            return await call_next(request)

        # Extract credentials
        auth_header = request.headers.get("Authorization", "")
        api_key = self._extract_api_key(auth_header)

        if not api_key:
            return JSONResponse(status_code=401, content={
                "error": {"type": "authentication_error", "code": "auth_required"}
            })

        # Validate
        key_info = await self.key_store.validate(api_key)
        if not key_info:
            return JSONResponse(status_code=401, content={
                "error": {"type": "authentication_error", "code": "invalid_api_key"}
            })

        # Check permissions
        required_perm = self._get_required_permission(request)
        if required_perm not in key_info.permissions:
            return JSONResponse(status_code=403, content={
                "error": {"type": "permission_error", "code": "insufficient_permissions"}
            })

        # Attach context and proceed
        request.state.user = key_info
        response = await call_next(request)
        return response
```

### Rate Limiter

```python
class RateLimiter:
    """Token bucket rate limiter per API key."""

    def __init__(self, config: RateLimitConfig):
        self.buckets: dict[str, TokenBucket] = {}
        self.config = config

    async def check(self, api_key: str) -> RateLimitResult:
        bucket = self.buckets.get(api_key)
        if not bucket:
            bucket = TokenBucket(
                rate=self.config.requests_per_minute / 60,
                capacity=self.config.requests_per_minute,
            )
            self.buckets[api_key] = bucket

        allowed = bucket.consume(1)
        return RateLimitResult(
            allowed=allowed,
            limit=self.config.requests_per_minute,
            remaining=int(bucket.tokens),
            reset_at=bucket.refill_at,
        )
```

---

## Error Handling

### Error Hierarchy

```
NexusError (base)
├── ModelNotFoundError
├── ModelNotLoadedError
├── ModelOverloadedError
├── InferenceError
│   ├── ContextLengthExceededError
│   ├── GenerationTimeoutError
│   └── CudaOutOfMemoryError
├── AuthenticationError
│   ├── InvalidAPIKeyError
│   ├── TokenExpiredError
│   └── InsufficientPermissionsError
├── TrainingError
│   ├── DatasetError
│   └── TrainingJobError
├── RAGError
│   ├── CollectionNotFoundError
│   └── IndexError
└── ConfigError
```

### Error Recovery

The backend implements automatic recovery for common issues:

| Error | Recovery Strategy |
|-------|-------------------|
| CUDA OOM | Unload least-recently-used model, retry |
| Model not loaded | Auto-load if configured, otherwise return error |
| Inference timeout | Retry with reduced max_tokens |
| Rate limit exceeded | Return 429 with Retry-After header |
| Network error (HF Hub) | Retry with exponential backoff |
