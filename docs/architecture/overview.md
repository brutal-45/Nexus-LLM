# Architecture Overview

Understand the design, module structure, and data flow of Nexus-LLM.

---

## System Design

Nexus-LLM follows a modular, layered architecture that separates concerns and allows each component to be used independently or together as a complete system.

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Client Layer                                 │
│  ┌──────────┐  ┌──────────────┐  ┌──────────┐  ┌──────────────┐   │
│  │ Terminal  │  │  REST API    │  │ WebSocket│  │ Python SDK   │   │
│  │   UI     │  │  (FastAPI)   │  │  Server  │  │              │   │
│  └────┬─────┘  └──────┬───────┘  └────┬─────┘  └──────┬───────┘   │
│       │               │               │               │            │
├───────┴───────────────┴───────────────┴───────────────┴────────────┤
│                       Application Layer                              │
│  ┌──────────┐  ┌──────────────┐  ┌──────────┐  ┌──────────────┐   │
│  │  Chat     │  │  Agent       │  │ Training │  │    RAG       │   │
│  │  Engine   │  │  Framework   │  │ Pipeline │  │  Pipeline    │   │
│  └────┬──────┘  └──────┬───────┘  └────┬─────┘  └──────┬───────┘   │
│       │               │               │               │            │
├───────┴───────────────┴───────────────┴───────────────┴────────────┤
│                        Core Layer                                    │
│  ┌──────────┐  ┌──────────────┐  ┌──────────┐  ┌──────────────┐   │
│  │  Model    │  │  Inference   │  │  Plugin  │  │    Config    │   │
│  │ Manager   │  │  Engine      │  │ System   │  │  Manager     │   │
│  └────┬──────┘  └──────┬───────┘  └────┬─────┘  └──────┬───────┘   │
│       │               │               │               │            │
├───────┴───────────────┴───────────────┴───────────────┴────────────┤
│                      Infrastructure Layer                            │
│  ┌──────────┐  ┌──────────────┐  ┌──────────┐  ┌──────────────┐   │
│  │  CUDA /   │  │  Vector DB   │  │  Cache   │  │   Logging    │   │
│  │  Hardware │  │  (Chroma/    │  │  Layer   │  │   System     │   │
│  │  Abstraction│ │   FAISS)    │  │          │  │              │   │
│  └───────────┘  └─────────────┘  └──────────┘  └─────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### Design Principles

1. **Modularity** — Each component has a clear interface and can be replaced independently
2. **Layered abstraction** — Higher layers depend on lower layers, never the reverse
3. **Plugin-first** — Core functionality is extensible without modifying source code
4. **Async by default** — I/O-bound operations use async/await for concurrency
5. **Configuration-driven** — Behavior is controlled by config files, not code changes

---

## Module Structure

```
nexus-llm/
├── main.py                    # Application entry point
├── nexus_llm/                 # Core Python package
│   ├── __init__.py
│   ├── cli/                   # Command-line interface
│   │   ├── __init__.py
│   │   ├── main.py            # CLI entry point (Click)
│   │   ├── chat.py            # Chat mode commands
│   │   ├── train.py           # Training commands
│   │   ├── rag.py             # RAG management commands
│   │   └── auth.py            # Authentication commands
│   │
│   ├── core/                  # Core engine
│   │   ├── __init__.py
│   │   ├── config.py          # Configuration manager
│   │   ├── model_manager.py   # Model loading, unloading, caching
│   │   ├── inference.py       # Inference engine
│   │   └── tokenizer.py       # Tokenizer management
│   │
│   ├── api/                   # API layer
│   │   ├── __init__.py
│   │   ├── app.py             # FastAPI application
│   │   ├── routes/
│   │   │   ├── chat.py        # /chat/completions
│   │   │   ├── completions.py # /completions
│   │   │   ├── embeddings.py  # /embeddings
│   │   │   ├── models.py      # /models
│   │   │   ├── training.py    # /training
│   │   │   ├── rag.py         # /rag
│   │   │   ├── auth.py        # /auth
│   │   │   └── system.py      # /health, /info, /metrics
│   │   ├── middleware/
│   │   │   ├── auth.py        # Authentication middleware
│   │   │   ├── rate_limit.py  # Rate limiting
│   │   │   └── logging.py     # Request logging
│   │   └── websocket.py       # WebSocket handler
│   │
│   ├── training/              # Training pipeline
│   │   ├── __init__.py
│   │   ├── trainer.py         # Main training orchestrator
│   │   ├── data.py            # Dataset loading and preprocessing
│   │   ├── lora.py            # LoRA configuration and training
│   │   ├── qlora.py           # QLoRA-specific logic
│   │   └── callbacks.py       # Training callbacks (logging, checkpointing)
│   │
│   ├── rag/                   # RAG pipeline
│   │   ├── __init__.py
│   │   ├── indexer.py         # Document indexing
│   │   ├── retriever.py       # Vector search and retrieval
│   │   ├── chunker.py         # Document chunking strategies
│   │   ├── embeddings.py      # Embedding generation
│   │   └── pipeline.py        # End-to-end RAG pipeline
│   │
│   ├── agents/                # Agent framework
│   │   ├── __init__.py
│   │   ├── base.py            # Base agent class
│   │   ├── tools.py           # Tool definitions and registry
│   │   ├── planner.py         # Planning and reasoning
│   │   ├── memory.py          # Agent memory systems
│   │   └── orchestrator.py    # Multi-agent orchestration
│   │
│   ├── plugins/               # Plugin system
│   │   ├── __init__.py
│   │   ├── manager.py         # Plugin loader and lifecycle
│   │   ├── hooks.py           # Hook definitions and registry
│   │   └── builtin/           # Built-in plugins
│   │       ├── content_filter.py
│   │       ├── token_counter.py
│   │       ├── conversation_logger.py
│   │       └── auto_summary.py
│   │
│   └── utils/                 # Shared utilities
│       ├── __init__.py
│       ├── logging.py         # Logging configuration
│       ├── crypto.py          # API key hashing, JWT
│       ├── gpu.py             # GPU detection and monitoring
│       └── formatting.py      # Output formatting utilities
│
├── config/                    # Configuration files
│   ├── default.yaml
│   ├── user.yaml
│   ├── profiles/
│   ├── agents/
│   └── prompts/
│
├── scripts/                   # Shell scripts
├── docs/                      # Documentation
├── tests/                     # Test suite
│   ├── unit/
│   ├── integration/
│   └── e2e/
│
├── plugins/                   # User plugins directory
├── data/                      # Data files
├── checkpoints/               # Training checkpoints
├── logs/                      # Log files
└── cache/                     # Model and index cache
```

---

## Data Flow

### Chat Request Flow

```
1. Client sends request
       │
       ▼
2. API Router receives request
       │
       ├── Authentication middleware validates API key
       ├── Rate limiter checks quotas
       └── Request body is validated (Pydantic)
       │
       ▼
3. Chat Engine processes the request
       │
       ├── Pre-inference hooks (plugins)
       │   ├── Content filtering
       │   ├── Token counting
       │   └── Custom preprocessing
       │
       ├── Build prompt from messages
       │   ├── Apply chat template
       │   ├── Inject system prompt
       │   └── RAG context injection (if enabled)
       │
       ├── Tokenize input
       │
       ▼
4. Inference Engine generates response
       │
       ├── Model Manager selects loaded model
       ├── Forward pass through model
       ├── Sampling (temperature, top-p, top-k)
       ├── Per-token hooks (plugins)
       └── Stopping criteria (stop tokens, max length)
       │
       ▼
5. Post-processing
       │
       ├── Post-inference hooks (plugins)
       │   ├── Sentiment analysis
       │   ├── Content filtering
       │   └── Custom postprocessing
       │
       ├── Detokenize output
       ├── Calculate usage metrics
       └── Format response
       │
       ▼
6. Response sent to client
       │
       ├── REST: JSON response
       ├── WebSocket: Token stream
       └── Terminal: Rich-formatted output
```

### Training Data Flow

```
1. Raw dataset (JSONL, CSV, HuggingFace)
       │
       ▼
2. Data Loader
       ├── Parse format
       ├── Validate schema
       └── Split train/eval
       │
       ▼
3. Preprocessing
       ├── Tokenize
       ├── Apply chat template
       ├── Truncate to max_seq_length
       └── Group by length (optional)
       │
       ▼
4. Training Loop
       ├── Forward pass
       ├── Loss computation
       ├── Backward pass
       ├── Gradient accumulation
       ├── Optimizer step
       ├── LR scheduler step
       └── Logging & checkpointing
       │
       ▼
5. Evaluation
       ├── Compute eval loss
       ├── Generate samples
       └── Compute metrics
       │
       ▼
6. Output
       ├── Save checkpoint
       ├── Merge LoRA (if applicable)
       └── Push to Hub (if configured)
```

### RAG Data Flow

```
1. Document input (PDF, TXT, MD, etc.)
       │
       ▼
2. Document Processor
       ├── Extract text
       ├── Clean and normalize
       └── Extract metadata
       │
       ▼
3. Chunker
       ├── Split into chunks
       ├── Add overlap
       └── Preserve metadata
       │
       ▼
4. Embedding Generator
       ├── Encode chunks to vectors
       └── Batch processing
       │
       ▼
5. Vector Store
       ├── Store embeddings
       ├── Index for fast search
       └── Persist to disk
       │
       ▼
6. Query Time
       ├── Embed query
       ├── Similarity search
       ├── Re-rank (if enabled)
       └── Return top-K chunks
       │
       ▼
7. Prompt Builder
       ├── Inject retrieved context
       └── Format with query
```

---

## Key Design Decisions

### Why FastAPI?

FastAPI was chosen for the API layer because:
- **Async-native** — Handles concurrent inference requests efficiently
- **OpenAPI generation** — Auto-generates API documentation
- **Pydantic validation** — Type-safe request/response handling
- **WebSocket support** — Built-in streaming capability
- **OpenAI compatibility** — Easy to integrate with existing tools

### Why Plugin System over Inheritance?

The hook-based plugin system was chosen over class inheritance because:
- **Open/Closed Principle** — Extend without modifying core code
- **Composition** — Multiple plugins can stack their behavior
- **Loose coupling** — Plugins don't depend on each other
- **Dynamic loading** — Enable/disable at runtime

### Why ChromaDB as Default Vector Store?

ChromaDB is the default RAG vector store because:
- **Zero-config** — Works out of the box with no external services
- **Embedded** — Runs in-process, no network overhead
- **Persistent** — Data survives restarts
- **Queryable** — Supports metadata filtering alongside vector search

Users can switch to FAISS (performance) or Qdrant (scale) as needed.

---

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| API Framework | FastAPI + Uvicorn | HTTP/WebSocket server |
| ML Framework | PyTorch + Transformers | Model loading and inference |
| Training | HuggingFace TRL + PEFT | Fine-tuning (LoRA/QLoRA) |
| Quantization | bitsandbytes, auto-gptq, auto-awq | Model compression |
| Embeddings | sentence-transformers | RAG embeddings |
| Vector DB | ChromaDB / FAISS / Qdrant | RAG storage and retrieval |
| CLI | Click + Rich | Terminal interface |
| Configuration | PyYAML + Pydantic | Config management |
| Auth | python-jose + passlib | JWT and API key management |
| Database | SQLite / PostgreSQL | Persistent storage |
| Monitoring | Prometheus client | Metrics export |
