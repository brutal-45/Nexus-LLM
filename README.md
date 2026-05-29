<div align="center">

# 🧠 Nexus-LLM

**A Terminal-Based LLM Framework — Train, Chat, Serve, and Deploy Local Language Models**

[![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)](https://github.com/nexus-llm/nexus-llm)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![CI](https://img.shields.io/github/actions/workflow/status/nexus-llm/nexus-llm/ci.yml?branch=main)](https://github.com/nexus-llm/nexus-llm/actions)
[![Docker](https://img.shields.io/badge/docker-supported-2496ED.svg)](https://hub.docker.com/r/nexusllm/nexus-llm)
[![Docs](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://nexus-llm.readthedocs.io)

</div>

---

Nexus-LLM is a production-grade, open-source terminal-based LLM framework that works like Claude AI in your terminal — with its **own backend**, no cloud APIs required. It unifies the entire LLM lifecycle: from model loading and inference to training, fine-tuning, RAG, agents, and deployment. Built with modularity at its core, Nexus-LLM supports **39+ models** across all major families.

## 🌟 Features

### Core Engine
- **Terminal Chat Interface** — Claude AI-like experience in your terminal with streaming, markdown rendering, and syntax highlighting
- **Own Backend** — No cloud APIs. Runs locally with FastAPI server (REST + WebSocket)
- **39+ Supported Models** — GPT-2, GPT-Neo, LLaMA, Phi, Pythia, BLOOM, OPT, Qwen, TinyLlama, CodeGen, and more
- **Streaming Inference** — Real-time token streaming with multiple sampling strategies
- **25+ Slash Commands** — `/help`, `/model`, `/switch`, `/config`, `/save`, `/load`, `/history`, `/search`, and more
- **Multiple Sampling** — Greedy, top-k, top-p (nucleus), typical, eta, epsilon, beam search

### Training & Fine-Tuning
- **LoRA / QLoRA Fine-Tuning** — Parameter-efficient fine-tuning with configurable rank, alpha, and target modules
- **Full Training Loop** — Gradient accumulation, mixed precision (fp16/bf16), gradient clipping
- **6 LR Schedulers** — Linear, cosine, cosine with restarts, polynomial, constant, inverse sqrt
- **Curriculum Learning** — Progressive difficulty scheduling for improved convergence
- **Distributed Training** — DDP and FSDP support for multi-GPU training
- **Model Export** — Export to HuggingFace, ONNX, GGML, safetensors, TorchScript formats

### RAG & Knowledge
- **Built-in RAG Pipeline** — End-to-end retrieve-augment-generate with query expansion
- **FAISS Vector Store** — Cosine similarity and L2 distance search
- **Hybrid Retrieval** — BM25 + dense retrieval with reciprocal rank fusion
- **Multiple Chunking Strategies** — Fixed-size, sentence, paragraph, and semantic chunking
- **Embedding Models** — SentenceTransformers and HuggingFace embeddings
- **Result Reranking** — Cross-encoder, MMR diversity, and multi-signal reranking

### Agents & Tools
- **Agent Framework** — Think/Act/Observe loop with 5 agent types (chat, tool, code, research, base)
- **Built-in Tools** — Calculator, web search, file manager, code runner, weather, note taker, system monitor
- **Task Planner** — Decompose tasks, create execution plans, manage subtasks
- **Agent Memory** — Short-term (LRU + TTL), long-term (persistent), and episodic memory

### API & Serving
- **REST API** — Full-featured FastAPI server with `/v1/generate`, `/v1/chat`, `/v1/models`, `/v1/health` 
- **WebSocket API** — Low-latency bidirectional streaming for interactive applications
- **Authentication** — API key and token-based auth with role-based permissions
- **Rate Limiting** — Token bucket and sliding window rate limiting
- **Model Serving** — Dedicated model server with load balancing and batch processing
- **Batched Inference** — Efficient batch processing for high-throughput workloads

### Safety & Content
- **Content Filtering** — Keyword, regex, and category-based filtering with PII redaction
- **Content Moderation** — Severity-based action policies with audit logging
- **Toxicity Detection** — 9 toxicity categories with weighted compound scoring
- **Safety Guardrails** — Input/output validation, topic restrictions, prompt injection detection
- **Configurable Policies** — Default, strict, and permissive policy presets

### Infrastructure
- **Quantization** — 4-bit/8-bit quantization with bitsandbytes, GPTQ, AWQ, and FP8 support
- **KV Cache** — PagedAttention-style cache management with eviction policies
- **CPU Offloading** — Layer-wise CPU and disk offloading with prefetching
- **Model Adapters** — LoRA adapter loading, switching, merging, and multi-adapter serving
- **Plugin System** — Extensible architecture with lifecycle hooks, 7 built-in plugins
- **Docker Support** — Multi-stage Docker build with optional CUDA support
- **Internationalization** — English, Chinese, Japanese, and Spanish localizations
- **Monitoring** — Terminal dashboard, alerts, resource tracking, and metrics
- **Experiment Tracking** — MLflow-style experiment management with hyperparameter search and ablation studies

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/nexus-llm/nexus-llm.git
cd nexus-llm

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Or use the install script
chmod +x scripts/install.sh
./scripts/install.sh
```

### Chat (Terminal)

```bash
# Start interactive chat with default model (GPT-2 Medium)
python main.py chat

# Use a specific model
python main.py chat --model gpt2-large

# List available models
python main.py models

# Download a model
python main.py download --model gpt2-medium
```

### Server

```bash
# Start API server
python main.py serve --host 0.0.0.0 --port 8000

# Or use the runner script
python run_server.py --model gpt2-medium --port 8000

# Docker
docker-compose up
```

### Training

```bash
# LoRA fine-tuning
python main.py train \
  --model gpt2-medium \
  --data data/training_data.jsonl \
  --method lora \
  --lora-rank 16 \
  --lora-alpha 32 \
  --epochs 3 \
  --learning-rate 2e-4

# Full training
python run_train.py \
  --model gpt2-medium \
  --data data/training_data.jsonl \
  --epochs 5 \
  --batch-size 4
```

### API Usage

```python
import requests

# Chat completion
response = requests.post("http://localhost:8000/v1/chat", json={
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ],
    "temperature": 0.7,
    "max_tokens": 512
})
print(response.json())
```

---

## 🤖 Supported Models (39+)

| Category | Models | Sizes |
|---|---|---|
| **GPT-2** | gpt2, gpt2-medium, gpt2-large, gpt2-xl | 124M - 1.5B |
| **GPT-Neo** | gpt-neo-125M, gpt-neo-1.3B, gpt-neo-2.7B, gpt-j-6B | 125M - 6B |
| **LLaMA** | Llama-2-7b, Llama-2-13b | 7B - 13B |
| **Phi** | phi-1, phi-1.5, phi-2 | 1.3B - 2.7B |
| **Pythia** | pythia-70m, pythia-160m, pythia-410m, pythia-1b | 70M - 1B |
| **BLOOM** | bloom-560m, bloom-1b1, bloom-1b7, bloom-3b | 560M - 3B |
| **OPT** | opt-125m, opt-350m, opt-1.3b, opt-2.7b | 125M - 2.7B |
| **Chat** | DialoGPT-medium, DialoGPT-large, Qwen1.5-0.5B, Qwen1.5-1.8B | 355M - 1.8B |
| **Tiny** | TinyLlama-1.1B-Chat-v1.0, llama-2-7b-finetuned | 1.1B - 7B |
| **Code** | tiny_starcoder_py, codegen-350M-mono, codegen-2B-mono | 164M - 2B |

---

## 📁 Project Structure

```
Nexus-LLM/
├── main.py                          # CLI entry point (Click)
├── setup.py                         # Package setup
├── pyproject.toml                   # Modern Python config
├── requirements.txt                 # Core dependencies
├── requirements-dev.txt             # Dev dependencies
├── requirements-gpu.txt             # GPU dependencies
├── Dockerfile                       # Multi-stage Docker build
├── docker-compose.yml               # Docker Compose config
├── Makefile                         # Build automation
├── download_model.py                # Model downloader (39+ models)
├── run_server.py                    # Server runner
├── run_chat.py                      # Chat runner
├── run_train.py                     # Training runner
├── run_eval.py                      # Evaluation runner
├── run_benchmark.py                 # Benchmark runner
├── README.md                        # This file
├── CHANGELOG.md                     # Version history
├── CONTRIBUTING.md                  # Contributing guide
├── LICENSE                          # MIT License
│
├── nexus_llm/                       # Main package
│   ├── __init__.py                  # Package init with public API
│   ├── __main__.py                  # python -m nexus_llm entry
│   ├── __version__.py               # Version info
│   ├── app.py                       # Main Application class
│   ├── cli.py                       # Full Click CLI
│   ├── constants.py                 # All constants
│   ├── exceptions.py                # Custom exceptions
│   ├── types.py                     # Type definitions
│   ├── enums.py                     # Enumerations
│   ├── registry.py                  # Component registry
│   ├── events.py                    # Event system
│   ├── plugins.py                   # Plugin interface
│   ├── signals.py                   # Signal handling
│   ├── context.py                   # Application context
│   ├── state.py                     # State management
│   ├── config_loader.py             # Configuration loading
│   │
│   ├── backend/                     # Backend engine (22 files)
│   │   ├── model_manager.py         # Model loading & device management
│   │   ├── inference.py             # Generation engine
│   │   ├── tokenizer_utils.py       # Tokenization utilities
│   │   ├── server.py                # FastAPI REST + WebSocket server
│   │   ├── cache.py                 # KV cache management
│   │   ├── scheduler.py             # Request scheduling & batching
│   │   ├── quantization.py          # 4-bit/8-bit quantization
│   │   ├── pipeline.py              # Pipeline management
│   │   ├── generation.py            # Generation config presets
│   │   ├── sampling.py              # 9 sampling strategies
│   │   ├── stopping.py              # Stopping criteria
│   │   ├── logits_process.py        # 14 logits processors
│   │   ├── beam_search.py           # Beam search implementation
│   │   ├── streamer.py              # Output streaming
│   │   ├── adapter.py               # LoRA adapter management
│   │   ├── loader.py                # Model loading (HF, safetensors)
│   │   ├── offload.py               # CPU/disk offloading
│   │   ├── memory.py                # Memory management
│   │   ├── benchmark.py             # Backend benchmarking
│   │   ├── health.py                # Health checks
│   │   └── metrics.py               # Prometheus-style metrics
│   │
│   ├── terminal/                    # Terminal UI (21 files)
│   │   ├── chat.py                  # Interactive chat loop
│   │   ├── formatter.py             # Rich output formatter
│   │   ├── commands.py              # 25+ slash commands
│   │   ├── history.py               # Session persistence
│   │   ├── themes.py                # Terminal themes
│   │   ├── prompts.py               # Input prompts
│   │   ├── renderer.py              # Text renderer
│   │   ├── syntax.py                # Syntax highlighting
│   │   ├── markdown_ext.py          # Markdown extensions
│   │   ├── widgets.py               # UI widgets
│   │   ├── layout.py                # Layout manager
│   │   ├── progress.py              # Progress indicators
│   │   ├── spinner.py               # Loading spinners
│   │   ├── table.py                 # Table display
│   │   ├── panel.py                 # Panel display
│   │   ├── status.py                # Status bar
│   │   ├── autocomplete.py          # Tab completion
│   │   ├── keybinds.py              # Key bindings
│   │   ├── multiline.py             # Multi-line input
│   │   └── ansi.py                  # ANSI utilities
│   │
│   ├── training/                    # Training pipeline (17 files)
│   │   ├── trainer.py               # Training loop
│   │   ├── dataset.py               # Dataset handling
│   │   ├── fine_tune.py             # LoRA/PEFT fine-tuning
│   │   ├── collator.py              # Data collation
│   │   ├── scheduler.py             # LR schedulers
│   │   ├── callbacks.py             # Training callbacks
│   │   ├── metrics.py               # Training metrics
│   │   ├── checkpoint.py            # Checkpointing
│   │   ├── distributed.py           # Distributed training
│   │   ├── augmentation.py          # Data augmentation
│   │   ├── preprocessing.py         # Data preprocessing
│   │   ├── evaluation.py            # Training evaluation
│   │   ├── optimizer.py             # Optimizer configs
│   │   ├── loss.py                  # Loss functions
│   │   ├── curriculum.py            # Curriculum learning
│   │   └── export.py                # Model export
│   │
│   ├── config/                      # Configuration (10 files)
│   │   ├── settings.py              # Settings loader
│   │   ├── model_config.yaml        # Model configuration
│   │   ├── training_config.yaml     # Training configuration
│   │   ├── server_config.yaml       # Server configuration
│   │   ├── ui_config.yaml           # UI configuration
│   │   ├── defaults.py              # Default values
│   │   ├── validators.py            # Config validation
│   │   ├── schema.py                # Config schemas
│   │   └── profiles.py              # Config profiles
│   │
│   ├── utils/                       # Utilities (18 files)
│   │   ├── logger.py                # Logging system
│   │   ├── helpers.py               # MODEL_CATALOG (39 models)
│   │   ├── io.py                    # I/O utilities
│   │   ├── system.py                # System information
│   │   ├── network.py               # Network utilities
│   │   ├── crypto.py                # Hash & verification
│   │   ├── text.py                  # Text processing
│   │   ├── validation.py            # Input validation
│   │   ├── decorators.py            # Python decorators
│   │   ├── retry.py                 # Retry logic
│   │   ├── threading.py             # Threading utilities
│   │   ├── process.py               # Process management
│   │   ├── timer.py                 # Timing utilities
│   │   ├── hash.py                  # Hashing utilities
│   │   ├── format.py                # Format conversion
│   │   ├── sanitize.py              # Input sanitization
│   │   └── profiler.py              # Profiling utilities
│   │
│   ├── models/                      # Model management (14 files)
│   │   ├── registry.py              # Model registry
│   │   ├── catalog.py               # 39+ model catalog
│   │   ├── base.py                  # BaseModel abstract class
│   │   ├── causal_lm.py             # CausalLM implementation
│   │   ├── seq2seq.py               # Seq2Seq implementation
│   │   ├── chat_model.py            # Chat model wrapper
│   │   ├── code_model.py            # Code model wrapper
│   │   ├── adapter_model.py         # Adapter model (PEFT)
│   │   ├── download.py              # Model downloader
│   │   ├── convert.py               # Model conversion
│   │   ├── verify.py                # Model verification
│   │   ├── benchmark.py             # Model benchmarking
│   │   └── quantize.py              # Model quantization
│   │
│   ├── api/                         # API layer (10 files)
│   │   ├── routes.py                # REST API routes
│   │   ├── websocket.py             # WebSocket handlers
│   │   ├── middleware.py            # API middleware
│   │   ├── auth.py                  # Authentication
│   │   ├── rate_limit.py            # Rate limiting
│   │   ├── schemas.py               # Pydantic schemas
│   │   ├── errors.py                # Error handling
│   │   ├── cors.py                  # CORS configuration
│   │   └── docs.py                  # API documentation
│   │
│   ├── safety/                      # Safety module (6 files)
│   │   ├── content_filter.py        # Content filtering
│   │   ├── moderation.py            # Content moderation
│   │   ├── toxicity.py              # Toxicity detection
│   │   ├── guardrails.py            # Safety guardrails
│   │   └── policies.py              # Safety policies
│   │
│   ├── rag/                         # RAG pipeline (8 files)
│   │   ├── retriever.py             # Document retrieval
│   │   ├── indexer.py               # Document indexing
│   │   ├── embeddings.py            # Embedding models
│   │   ├── vector_store.py          # FAISS vector store
│   │   ├── chunker.py               # Text chunking
│   │   ├── reranker.py              # Result reranking
│   │   └── pipeline.py              # RAG pipeline
│   │
│   ├── agents/                      # Agent framework (10 files)
│   │   ├── base.py                  # Base agent
│   │   ├── chat_agent.py            # Chat agent
│   │   ├── tool_agent.py            # Tool-using agent
│   │   ├── code_agent.py            # Code agent
│   │   ├── research_agent.py        # Research agent
│   │   ├── tools.py                 # Built-in tools
│   │   ├── planner.py               # Task planner
│   │   ├── memory.py                # Agent memory
│   │   └── executor.py              # Action executor
│   │
│   ├── plugins/                     # Plugin system (15 files)
│   │   ├── manager.py               # Plugin manager
│   │   ├── loader.py                # Plugin loader
│   │   ├── hook.py                  # Hook system
│   │   ├── builtin/                 # Built-in plugins
│   │   │   ├── weather.py           # Weather plugin
│   │   │   ├── calculator.py        # Calculator plugin
│   │   │   ├── web_search.py        # Web search plugin
│   │   │   ├── file_manager.py      # File manager plugin
│   │   │   ├── code_runner.py       # Code runner plugin
│   │   │   ├── note_taker.py        # Note taker plugin
│   │   │   └── system_monitor.py    # System monitor plugin
│   │   └── examples/               # Example plugins
│   │       ├── custom_greet.py      # Custom greeting
│   │       └── echo.py              # Echo plugin
│   │
│   ├── evaluation/                  # Evaluation (7 files)
│   │   ├── evaluator.py             # Main evaluator
│   │   ├── benchmarks.py            # Benchmark runner
│   │   ├── metrics.py               # BLEU, ROUGE, F1, etc.
│   │   ├── perplexity.py            # Perplexity calculator
│   │   ├── generation_eval.py       # Generation quality
│   │   └── report.py                # Evaluation reports
│   │
│   ├── data/                        # Data handling (7 files)
│   │   ├── loader.py                # Data loading
│   │   ├── processor.py             # Data processing
│   │   ├── splitter.py              # Data splitting
│   │   ├── tokenizer_data.py        # Tokenizer data
│   │   ├── converter.py             # Format conversion
│   │   └── validator.py             # Data validation
│   │
│   ├── templates/                   # Templates (13 files)
│   │   ├── prompts/                 # Prompt templates
│   │   │   ├── chat.yaml            # Chat templates
│   │   │   ├── code.yaml            # Code templates
│   │   │   ├── summary.yaml         # Summary templates
│   │   │   ├── translate.yaml       # Translation templates
│   │   │   ├── creative.yaml        # Creative templates
│   │   │   ├── analysis.yaml        # Analysis templates
│   │   │   └── default.yaml         # Default template
│   │   └── system/                  # System prompts
│   │       ├── default.txt          # Default system prompt
│   │       ├── coding.txt           # Coding assistant
│   │       ├── creative.txt         # Creative writing
│   │       ├── assistant.txt        # General assistant
│   │       └── researcher.txt       # Research assistant
│   │
│   ├── storage/                     # Storage (5 files)
│   │   ├── database.py              # SQLite database
│   │   ├── conversation_store.py    # Conversation persistence
│   │   ├── model_store.py           # Model metadata
│   │   └── cache_store.py           # Response cache
│   │
│   ├── monitoring/                  # Monitoring (5 files)
│   │   ├── dashboard.py             # Terminal dashboard
│   │   ├── alerts.py                # Alert system
│   │   ├── tracker.py               # Resource tracking
│   │   └── reporter.py              # Status reporting
│   │
│   ├── multimodal/                  # Multimodal (6 files)
│   │   ├── image_processor.py       # Image processing
│   │   ├── audio_processor.py       # Audio processing
│   │   ├── document_processor.py    # Document processing
│   │   ├── vision_model.py          # Vision model wrapper
│   │   └── ocr.py                   # OCR text extraction
│   │
│   ├── serving/                     # Model serving (5 files)
│   │   ├── model_server.py          # Model server
│   │   ├── load_balancer.py         # Load balancing
│   │   ├── queue_manager.py         # Request queue
│   │   └── batch_processor.py       # Batch processing
│   │
│   ├── experiments/                 # Experiments (6 files)
│   │   ├── experiment.py            # Experiment tracking
│   │   ├── tracker.py               # ML experiment tracker
│   │   ├── comparison.py            # Experiment comparison
│   │   ├── hyperparams.py           # Hyperparameter search
│   │   └── ablation.py              # Ablation studies
│   │
│   ├── migrations/                  # Database migrations (5 files)
│   │   ├── migrate.py               # Migration runner
│   │   ├── v001_initial.py          # Initial schema
│   │   ├── v002_conversations.py    # Conversations table
│   │   └── v002_training_jobs.py    # Training jobs table
│   │
│   ├── presets/                     # Presets (5 files)
│   │   ├── preset_manager.py        # Preset manager
│   │   ├── chat_presets.yaml        # Chat presets
│   │   ├── training_presets.yaml    # Training presets
│   │   └── server_presets.yaml      # Server presets
│   │
│   ├── i18n/                        # Internationalization (6 files)
│   │   ├── localizer.py             # Localization manager
│   │   ├── en.yaml                  # English
│   │   ├── zh.yaml                  # Chinese
│   │   ├── ja.yaml                  # Japanese
│   │   └── es.yaml                  # Spanish
│   │
│   ├── auth/                        # Authentication (4 files)
│   │   ├── manager.py               # Auth manager
│   │   ├── tokens.py                # Token management
│   │   └── permissions.py           # Permission system
│   │
│   ├── cli_ext/                     # CLI extensions (4 files)
│   │   ├── completions.py           # Shell completions
│   │   ├── formatters.py            # Output formatters
│   │   └── validators.py            # CLI validators
│   │
│   └── workers/                     # Worker processes (5 files)
│       ├── inference_worker.py      # Inference worker
│       ├── training_worker.py       # Training worker
│       ├── worker_pool.py           # Worker pool
│       └── task_queue.py            # Task queue
│
├── tests/                           # Test suite (160+ files)
│   ├── conftest.py                  # Shared fixtures
│   ├── test_*.py                    # Unit & integration tests
│   └── ...                          # Covering all modules
│
├── scripts/                         # Shell scripts (8 files)
│   ├── install.sh                   # Installation
│   ├── run.sh                       # Run server
│   ├── train.sh                     # Training
│   ├── benchmark.sh                 # Benchmarking
│   ├── setup_env.sh                 # Environment setup
│   ├── download_models.sh           # Model download
│   ├── clean.sh                     # Cleanup
│   └── update.sh                    # Update
│
├── data/                            # Sample data
│   ├── training_data.jsonl          # Training examples
│   ├── eval_data.jsonl              # Evaluation examples
│   ├── sample_conversations.json    # Sample conversations
│   ├── prompt_templates.json        # Prompt templates
│   └── datasets/                    # Format examples
│       ├── alpaca_format.jsonl      # Alpaca format
│       ├── chat_format.jsonl        # Chat format
│       └── instruction_format.jsonl # Instruction format
│
├── docs/                            # Documentation (18 files)
│   ├── api/                         # API documentation
│   ├── guides/                      # User guides
│   ├── tutorials/                   # Tutorials
│   └── architecture/               # Architecture docs
│
├── .github/                         # GitHub CI/CD
│   ├── workflows/                   # GitHub Actions
│   ├── ISSUE_TEMPLATE/              # Issue templates
│   ├── dependabot.yml               # Dependabot config
│   └── PULL_REQUEST_TEMPLATE.md     # PR template
│
├── .gitignore                       # Git ignore rules
├── .env                             # Environment config
├── .env.example                     # Environment example
├── .pre-commit-config.yaml          # Pre-commit hooks
├── .flake8                          # Flake8 config
├── .isort.cfg                       # Isort config
├── .mypy.ini                        # MyPy config
├── .pylintrc                        # Pylint config
├── codecov.yml                      # Codecov config
├── tox.ini                          # Tox config
├── MANIFEST.in                      # Package manifest
└── setup.cfg                        # Setup configuration
```

---

## ⚙️ Configuration

Nexus-LLM can be configured via environment variables, YAML config files, or CLI arguments.

### Environment Variables

```bash
# Core
NEXUS_MODEL=gpt2-medium
NEXUS_HOST=0.0.0.0
NEXUS_PORT=8000
NEXUS_DEVICE=auto              # auto, cpu, cuda, mps

# Generation
NEXUS_TEMPERATURE=0.7
NEXUS_TOP_P=0.9
NEXUS_TOP_K=50
NEXUS_MAX_TOKENS=512

# Training
NEXUS_LEARNING_RATE=2e-4
NEXUS_BATCH_SIZE=4
NEXUS_EPOCHS=3

# Logging
NEXUS_LOG_LEVEL=INFO
```

### Config File (YAML)

```yaml
# nexus_llm/config/model_config.yaml
model:
  name: gpt2-medium
  device: auto
  max_length: 1024

generation:
  temperature: 0.7
  top_p: 0.9
  top_k: 50
  max_new_tokens: 512
  repetition_penalty: 1.1

server:
  host: 0.0.0.0
  port: 8000
  workers: 1

training:
  learning_rate: 2e-4
  batch_size: 4
  epochs: 3
  method: lora
  lora_rank: 16
  lora_alpha: 32
```

---

## 🔌 Plugin Development

Create custom plugins by implementing the `PluginInterface`:

```python
from nexus_llm.plugins import PluginInterface

class MyPlugin(PluginInterface):
    name = "my_plugin"
    version = "1.0.0"
    description = "My custom plugin"

    def on_load(self):
        print("Plugin loaded!")

    def on_unload(self):
        print("Plugin unloaded!")

    def on_message(self, message):
        # Process messages
        return message
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=nexus_llm --cov-report=html

# Run specific module tests
pytest tests/test_inference.py -v

# Using Make
make test
```

---

## 🐳 Docker

```bash
# Build image
docker build -t nexus-llm .

# Run container
docker run -p 8000:8000 nexus-llm

# With GPU
docker run --gpus all -p 8000:8000 nexus-llm

# Using Docker Compose
docker-compose up
```

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📊 Stats

| Metric | Count |
|--------|-------|
| **Source Files** | 495+ |
| **Python Modules** | 20+ |
| **Test Files** | 160+ |
| **Supported Models** | 39+ |
| **Slash Commands** | 25+ |
| **Built-in Plugins** | 7 |
| **Documentation Pages** | 18 |
| **Shell Scripts** | 8 |

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with ❤️ by the Nexus-LLM Team**

[Documentation](docs/) · [Contributing](CONTRIBUTING.md) · [Changelog](CHANGELOG.md)

</div>
