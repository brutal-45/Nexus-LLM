# Configuration Guide

Complete reference for configuring Nexus-LLM through YAML files, environment variables, and CLI flags.

---

## Configuration Hierarchy

Nexus-LLM loads configuration in the following priority order (highest to lowest):

1. **CLI flags** — `./scripts/run.sh --port 9000`
2. **Environment variables** — `NEXUS_PORT=9000`
3. **User config** — `config/user.yaml`
4. **Default config** — `config/default.yaml`
5. **Built-in defaults** — Hardcoded in the application

This means CLI flags always win, and user config overrides default config.

---

## Configuration Files

### Default Configuration

`config/default.yaml` contains all available settings with their default values. This file is tracked by git and should not be modified directly.

### User Configuration

`config/user.yaml` is where you put your custom settings. Only include the settings you want to override:

```yaml
# config/user.yaml
model:
  default_model: "mistralai/Mistral-7B-Instruct-v0.3"

inference:
  temperature: 0.5
  max_new_tokens: 4096

server:
  port: 9000
```

### Profile-Based Configuration

Create different profiles for different use cases:

```yaml
# config/profiles/production.yaml
server:
  host: "0.0.0.0"
  port: 8000
  workers: 4

logging:
  level: "WARNING"

auth:
  enabled: true
```

```yaml
# config/profiles/development.yaml
server:
  host: "127.0.0.1"
  port: 8000
  workers: 1

logging:
  level: "DEBUG"

auth:
  enabled: false
```

Select a profile at startup:

```bash
./scripts/run.sh --config config/profiles/production.yaml
# OR
export NEXUS_CONFIG_DIR=config/profiles/production
```

---

## Complete Configuration Reference

### Server Settings

```yaml
server:
  host: "127.0.0.1"          # Bind address (0.0.0.0 for all interfaces)
  port: 8000                   # API server port
  workers: 1                   # Number of Uvicorn workers
  reload: false                # Auto-reload on code changes (dev only)
  cors_origins:                # Allowed CORS origins
    - "*"
  ssl: false                   # Enable HTTPS
  ssl_certfile: null           # Path to SSL certificate
  ssl_keyfile: null            # Path to SSL private key
  timeout_keep_alive: 5        # Keep-alive timeout (seconds)
  limit_max_request_size: 10485760  # Max request size (10MB)
```

### Model Settings

```yaml
model:
  default_model: "meta-llama/Llama-3.1-8B-Instruct"  # Default model ID
  device: "auto"              # auto, cpu, cuda, cuda:0, mps
  dtype: "float16"            # float32, float16, bfloat16
  max_memory: null            # Per-device memory limit
  trust_remote_code: true     # Trust remote code in model repos
  attn_implementation: null   # eager, sdpa, flash_attention_2
  device_map: "auto"          # auto, balanced, sequential, or custom
  offload_folder: "offload"   # Folder for disk offloading
  rope_scaling: null          # RoPE scaling config for long context

  # Model download/cache
  cache_dir: "./cache/models"
  resume_download: true
  force_download: false

  # Quantization
  quantization: null          # null, 4bit, 8bit, gptq, awq
  quantization_config:
    load_in_4bit: false
    load_in_8bit: false
    bnb_4bit_compute_dtype: "bfloat16"
    bnb_4bit_quant_type: "nf4"
    bnb_4bit_use_double_quant: true
```

### Inference Settings

```yaml
inference:
  max_new_tokens: 2048        # Maximum tokens to generate
  temperature: 0.7            # Sampling temperature (0.0–2.0)
  top_p: 0.9                  # Nucleus sampling threshold
  top_k: 50                   # Top-K sampling
  repetition_penalty: 1.1     # Repetition penalty (1.0 = disabled)
  frequency_penalty: 0.0      # Frequency penalty
  presence_penalty: 0.0       # Presence penalty
  streaming: true             # Stream tokens by default
  stop: null                  # Default stop sequences
  seed: null                  # Random seed for reproducibility

  # Context management
  max_context_length: null    # Override model's max context (null = model default)
  truncation: true            # Truncate input if it exceeds context length
  system_prompt: null         # Default system prompt

  # Batch inference
  batch_size: 1               # Batch size for non-interactive inference
```

### Training Settings

```yaml
training:
  output_dir: "./checkpoints"
  num_train_epochs: 3
  per_device_train_batch_size: 4
  per_device_eval_batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 2.0e-5
  weight_decay: 0.01
  warmup_ratio: 0.1
  warmup_steps: 0
  lr_scheduler_type: "cosine"   # linear, cosine, cosine_with_restarts, polynomial, constant
  max_grad_norm: 1.0
  fp16: true
  bf16: false
  gradient_checkpointing: false
  max_seq_length: 2048

  # Saving and evaluation
  logging_steps: 10
  logging_dir: "./logs"
  save_steps: 500
  save_total_limit: 3
  eval_steps: 500
  eval_strategy: "steps"        # no, steps, epoch
  load_best_model_at_end: true
  metric_for_best_model: "eval_loss"

  # Reporting
  report_to: ["tensorboard"]    # tensorboard, wandb, mlflow, none
  run_name: null

  # LoRA specific
  lora:
    r: 8
    lora_alpha: 16
    lora_dropout: 0.05
    target_modules: ["q_proj", "v_proj"]
    bias: "none"                # none, all, lora_only
    task_type: "CAUSAL_LM"
    modules_to_save: []

  # Data
  data:
    train_file: null
    eval_file: null
    dataset_name: null
    dataset_split: "train"
    validation_split: 0.1
    preprocessing_num_workers: 4
    group_by_length: true
```

### RAG Settings

```yaml
rag:
  enabled: false
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  vector_store: "chromadb"
  persist_directory: "./cache/vector_store"

  # Chunking
  chunk_size: 512
  chunk_overlap: 64
  chunking_strategy: "fixed"    # fixed, semantic, sentence, parent_child

  # Retrieval
  retrieval_top_k: 5
  similarity_threshold: 0.7
  retrieval_strategy: "semantic"  # semantic, keyword, hybrid

  # Re-ranking
  reranking:
    enabled: false
    model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_n: 20
    rerank_top_k: 5

  # Template for augmented prompt
  prompt_template: |
    Use the following context to answer the question. If you cannot find the answer
    in the context, say so. Always cite your sources.

    Context:
    {context}

    Question: {question}

    Answer:
```

### Authentication Settings

```yaml
auth:
  enabled: true
  api_keys:
    enabled: true
    hash_algorithm: "sha256"
    auto_create_admin: true
    max_keys_per_user: 10
  jwt:
    enabled: false
    algorithm: "RS256"
    public_key_path: "./keys/public.pem"
    private_key_path: "./keys/private.pem"
    access_token_expire_minutes: 60
    refresh_token_expire_days: 30
    issuer: "nexus-llm"
    audience: "nexus-llm-api"
  oauth2:
    enabled: false
    providers: {}
  rate_limits:
    default:
      requests_per_minute: 60
      tokens_per_minute: 200000
```

### Logging Settings

```yaml
logging:
  level: "INFO"               # DEBUG, INFO, WARNING, ERROR, CRITICAL
  file: "./logs/nexus-llm.log"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  max_size_mb: 100             # Max log file size before rotation
  backup_count: 5              # Number of rotated log files to keep
  console: true                # Also log to console
  log_requests: true           # Log API requests
  log_responses: false         # Log API responses (verbose)
  log_token_counts: true       # Log token usage
```

### Plugin Settings

```yaml
plugins:
  directories:
    - "./plugins"
    - "~/.nexus/plugins"
  enabled: []
  disabled: []
```

---

## Environment Variables

All configuration options can be set via environment variables using the `NEXUS_` prefix. Nested keys use underscores:

| Environment Variable | Config Path | Example |
|---------------------|-------------|---------|
| `NEXUS_HOST` | `server.host` | `0.0.0.0` |
| `NEXUS_PORT` | `server.port` | `9000` |
| `NEXUS_LOG_LEVEL` | `logging.level` | `DEBUG` |
| `NEXUS_MODEL` | `model.default_model` | `mistralai/Mistral-7B-Instruct-v0.3` |
| `NEXUS_DEVICE` | `model.device` | `cuda:0` |
| `NEXUS_HF_TOKEN` | — | `hf_abc123...` |
| `NEXUS_API_KEY` | — | `nexus_live_abc123...` |
| `NEXUS_OPENAI_API_KEY` | — | `sk-abc123...` |
| `NEXUS_DATABASE_URL` | — | `sqlite:///./nexus.db` |
| `NEXUS_CONFIG_DIR` | — | `./config` |
| `NEXUS_RAG_ENABLED` | `rag.enabled` | `true` |
| `NEXUS_RAG_EMBEDDING_MODEL` | `rag.embedding_model` | `sentence-transformers/all-mpnet-base-v2` |

### .env File

Environment variables can be stored in a `.env` file at the project root:

```bash
# .env
NEXUS_HF_TOKEN=hf_abc123def456
NEXUS_API_KEY=nexus_live_xyz789
NEXUS_PORT=8000
NEXUS_LOG_LEVEL=INFO
NEXUS_RAG_ENABLED=true
```

> **Important:** Never commit `.env` to version control. It's included in `.gitignore` by default.

---

## CLI Flags

Common settings can be overridden via CLI flags:

```bash
./scripts/run.sh \
  --mode server \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --host 0.0.0.0 \
  --port 9000 \
  --config config/profiles/production.yaml \
  --log-level WARNING
```

| Flag | Config Path | Description |
|------|-------------|-------------|
| `--mode` | — | Run mode: chat, server, interactive |
| `--model` | `model.default_model` | Model to use |
| `--host` | `server.host` | Server bind address |
| `--port` | `server.port` | Server port |
| `--config` | — | Config file path |
| `--log-level` | `logging.level` | Logging level |
| `--device` | `model.device` | Device to use |
| `--adapter` | — | LoRA adapter path |

---

## Dynamic Configuration

Some settings can be changed at runtime without restarting:

```bash
# Change the loaded model
curl -X POST http://localhost:8000/api/v1/models/mistralai%2FMistral-7B-Instruct-v0.3/load \
  -H "Authorization: Bearer nexus_your_api_key"

# Update inference parameters
curl -X PATCH http://localhost:8000/api/v1/config \
  -H "Authorization: Bearer nexus_your_api_key" \
  -d '{"inference": {"temperature": 0.5, "max_new_tokens": 4096}}'
```
