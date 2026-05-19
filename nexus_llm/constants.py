"""Nexus-LLM Constants Module.

All application-wide constants, default values, and configuration
parameters used throughout the Nexus-LLM framework.
"""

import os
from pathlib import Path

# ============================================================
# Application Identity
# ============================================================
APP_NAME: str = "Nexus-LLM"
APP_DESCRIPTION: str = "A powerful LLM framework for training, serving, and chatting"
VERSION: str = "0.1.0"
APP_AUTHOR: str = "Nexus-LLM Team"
APP_LICENSE: str = "MIT"

# ============================================================
# Default Model Settings
# ============================================================
DEFAULT_MODEL: str = os.environ.get("NEXUS_LLM_DEFAULT_MODEL", "gpt2-medium")
DEFAULT_TOKENIZER: str = DEFAULT_MODEL
DEFAULT_MODEL_SOURCE: str = "huggingface"
MODEL_CACHE_DIR: str = os.environ.get("NEXUS_LLM_MODEL_CACHE_DIR", str(Path.home() / ".cache" / "nexus_llm" / "models"))

# ============================================================
# Generation Defaults
# ============================================================
MAX_TOKENS: int = int(os.environ.get("NEXUS_LLM_MAX_TOKENS", "2048"))
MIN_TOKENS: int = 1
TEMPERATURE: float = float(os.environ.get("NEXUS_LLM_TEMPERATURE", "0.7"))
TOP_P: float = float(os.environ.get("NEXUS_LLM_TOP_P", "0.9"))
TOP_K: int = int(os.environ.get("NEXUS_LLM_TOP_K", "50"))
REPETITION_PENALTY: float = 1.1
LENGTH_PENALTY: float = 1.0
NO_REPEAT_NGRAM_SIZE: int = 0
NUM_BEAMS: int = 1
DO_SAMPLE: bool = True

# ============================================================
# Server Defaults
# ============================================================
DEFAULT_HOST: str = os.environ.get("NEXUS_LLM_HOST", "0.0.0.0")
DEFAULT_PORT: int = int(os.environ.get("NEXUS_LLM_PORT", "8000"))
DEFAULT_WORKERS: int = int(os.environ.get("NEXUS_LLM_WORKERS", "1"))
DEFAULT_API_PREFIX: str = "/api/v1"
HEALTH_CHECK_ENDPOINT: str = "/health"
CORS_ORIGINS: list = ["*"]

# ============================================================
# Training Defaults
# ============================================================
DEFAULT_BATCH_SIZE: int = int(os.environ.get("NEXUS_LLM_TRAIN_BATCH_SIZE", "8"))
DEFAULT_LEARNING_RATE: float = float(os.environ.get("NEXUS_LLM_TRAIN_LEARNING_RATE", "2e-5"))
DEFAULT_EPOCHS: int = int(os.environ.get("NEXUS_LLM_TRAIN_EPOCHS", "3"))
DEFAULT_WARMUP_STEPS: int = 100
DEFAULT_WEIGHT_DECAY: float = 0.01
DEFAULT_GRADIENT_ACCUMULATION: int = 1
DEFAULT_MAX_SEQ_LENGTH: int = int(os.environ.get("NEXUS_LLM_MAX_SEQ_LENGTH", "2048"))
DEFAULT_SAVE_STEPS: int = 500
DEFAULT_EVAL_STEPS: int = 500
DEFAULT_LOGGING_STEPS: int = 10
DEFAULT_LORA_RANK: int = 8
DEFAULT_LORA_ALPHA: int = 16
DEFAULT_LORA_DROPOUT: float = 0.05
DEFAULT_SEED: int = 42

# ============================================================
# Data Defaults
# ============================================================
DEFAULT_DATA_DIR: str = os.environ.get("NEXUS_LLM_DATA_DIR", "./data")
DEFAULT_OUTPUT_DIR: str = os.environ.get("NEXUS_LLM_TRAIN_OUTPUT_DIR", "./output")
DEFAULT_TRAIN_SPLIT: float = 0.9
DEFAULT_VAL_SPLIT: float = 0.05
DEFAULT_TEST_SPLIT: float = 0.05
DEFAULT_DATA_FORMAT: str = "jsonl"

# ============================================================
# Device and Precision
# ============================================================
DEFAULT_DEVICE: str = os.environ.get("NEXUS_LLM_DEVICE", "auto")
DEFAULT_PRECISION: str = os.environ.get("NEXUS_LLM_PRECISION", "fp16")

# ============================================================
# Logging
# ============================================================
LOG_LEVEL: str = os.environ.get("NEXUS_LLM_LOG_LEVEL", "INFO")
LOG_FORMAT: str = os.environ.get("NEXUS_LLM_LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
LOG_DIR: str = os.environ.get("NEXUS_LLM_LOGS_DIR", "./logs")
LOG_MAX_BYTES: int = 10 * 1024 * 1024  # 10 MB
LOG_BACKUP_COUNT: int = 5

# ============================================================
# File Paths
# ============================================================
CONFIG_DIR: str = os.environ.get("NEXUS_LLM_CONFIG_PATH", "./config")
CONFIG_FILENAME: str = "nexus_llm.yaml"
MODELS_DIR: str = os.environ.get("NEXUS_LLM_MODELS_DIR", "./models")
CACHE_DIR: str = os.environ.get("MODEL_CACHE_DIR", "./cache")

# ============================================================
# Chat Settings
# ============================================================
CHAT_HISTORY_LIMIT: int = 100
SYSTEM_PROMPT_DEFAULT: str = "You are a helpful, respectful, and honest assistant."
CHAT_WELCOME_MESSAGE: str = "Welcome to Nexus-LLM Chat! Type 'exit' or 'quit' to end the session."
CHAT_EXIT_COMMANDS: list = ["exit", "quit", ":q", ":quit"]

# ============================================================
# Plugin Settings
# ============================================================
PLUGIN_DIR: str = "plugins"
PLUGIN_CONFIG_FILE: str = "plugins.yaml"
PLUGIN_AUTO_DISCOVER: bool = True

# ============================================================
# Event Settings
# ============================================================
EVENT_MAX_HISTORY: int = 1000
EVENT_HANDLER_TIMEOUT: float = 30.0

# ============================================================
# API Rate Limiting
# ============================================================
API_RATE_LIMIT: str = "60/minute"
API_MAX_CONCURRENT: int = 10

# ============================================================
# Benchmark Settings
# ============================================================
BENCHMARK_DEFAULT_BATCH_SIZES: list = [1, 2, 4, 8]
BENCHMARK_DEFAULT_SEQ_LENGTHS: list = [128, 256, 512, 1024]
BENCHMARK_WARMUP_ITERATIONS: int = 3
BENCHMARK_NUM_ITERATIONS: int = 10

# ============================================================
# Download Settings
# ============================================================
DOWNLOAD_MAX_WORKERS: int = 4
DOWNLOAD_CHUNK_SIZE: int = 8192
DOWNLOAD_TIMEOUT: int = 300
DOWNLOAD_RETRIES: int = 3
DOWNLOAD_RETRY_DELAY: float = 5.0

# ============================================================
# Environment Variable Prefixes
# ============================================================
ENV_PREFIX: str = "NEXUS_LLM_"
HF_CACHE_DIR: str = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
TRANSFORMERS_CACHE: str = os.environ.get(
    "TRANSFORMERS_CACHE",
    os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")),
)
