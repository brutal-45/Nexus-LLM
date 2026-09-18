"""
Nexus-LLM Default Configuration Values

Provides default values for all settings across all configuration
sections (model, training, server, UI).
"""

from __future__ import annotations

from typing import Any


class Defaults:
    """Central repository of default configuration values.

    All defaults are organized by section and can be accessed
    individually or as a complete configuration dictionary.
    """

    # ── Model Defaults ──────────────────────────────────────────

    MODEL: dict[str, Any] = {
        "model": {
            "name": "gpt2-medium",
            "path": "",
            "device": "auto",
            "dtype": "float32",
            "trust_remote_code": False,
            "use_auth_token": False,
            "revision": "main",
            "mirror": "",
        }
    }

    GENERATION: dict[str, Any] = {
        "generation": {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "max_tokens": 512,
            "min_tokens": 1,
            "repetition_penalty": 1.1,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "do_sample": True,
            "num_beams": 1,
            "num_return_sequences": 1,
            "length_penalty": 1.0,
            "early_stopping": False,
            "stop_sequences": ["\n\n\n"],
            "bad_words_ids": [],
            "no_repeat_ngram_size": 0,
            "seed": None,
        }
    }

    QUANTIZATION: dict[str, Any] = {
        "quantization": {
            "enabled": False,
            "bits": 4,
            "group_size": 128,
            "quant_method": "gptq",
        }
    }

    CACHE: dict[str, Any] = {
        "cache": {
            "enabled": True,
            "max_size": 1000,
            "ttl": 3600,
        }
    }

    TOKENIZER: dict[str, Any] = {
        "tokenizer": {
            "padding_side": "left",
            "truncation": True,
            "max_length": 1024,
            "add_special_tokens": True,
        }
    }

    # ── Training Defaults ──────────────────────────────────────

    TRAINING: dict[str, Any] = {
        "training": {
            "output_dir": "./outputs",
            "overwrite_output_dir": False,
            "seed": 42,
            "deterministic": False,
            "num_train_epochs": 3,
            "max_steps": -1,
            "per_device_train_batch_size": 4,
            "per_device_eval_batch_size": 4,
            "gradient_accumulation_steps": 1,
            "warmup_steps": 0,
            "warmup_ratio": 0.0,
            "learning_rate": 5.0e-5,
            "lr_scheduler_type": "cosine",
            "lr_scheduler_kwargs": {},
            "weight_decay": 0.01,
            "optimizer": "adamw_torch",
            "fp16": False,
            "bf16": False,
            "fp16_full_eval": False,
            "bf16_full_eval": False,
            "max_grad_norm": 1.0,
            "gradient_checkpointing": False,
            "gradient_checkpointing_kwargs": {},
            "evaluation_strategy": "no",
            "eval_steps": 500,
            "eval_delay": 0,
            "eval_accumulation_steps": 1,
            "logging_dir": "./logs",
            "logging_strategy": "steps",
            "logging_steps": 10,
            "logging_nan_inf_filter": True,
            "log_level": "info",
            "report_to": "tensorboard",
            "save_strategy": "steps",
            "save_steps": 500,
            "save_total_limit": 2,
            "save_safetensors": True,
            "load_best_model_at_end": False,
            "metric_for_best_model": "loss",
            "dataloader_drop_last": False,
            "dataloader_num_workers": 0,
            "dataloader_pin_memory": True,
            "dataloader_persistent_workers": False,
            "group_by_length": False,
            "ddp_find_unused_parameters": False,
            "fsdp": "",
            "deepspeed": None,
        }
    }

    LORA: dict[str, Any] = {
        "lora": {
            "enabled": False,
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "target_modules": ["q_proj", "v_proj"],
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "fan_in_fan_out": False,
            "modules_to_save": [],
            "layers_to_transform": None,
            "layers_pattern": None,
            "rank_pattern": {},
            "alpha_pattern": {},
        }
    }

    PEFT: dict[str, Any] = {
        "peft": {
            "enabled": False,
            "method": "lora",
            "config_path": None,
        }
    }

    # ── Server Defaults ────────────────────────────────────────

    SERVER: dict[str, Any] = {
        "server": {
            "host": "0.0.0.0",
            "port": 8000,
            "workers": 1,
            "reload": False,
            "debug": False,
            "access_log": True,
            "log_level": "info",
            "app_name": "Nexus-LLM",
            "api_prefix": "/api/v1",
            "docs_url": "/docs",
            "redoc_url": "/redoc",
            "openapi_url": "/openapi.json",
            "ssl_certfile": None,
            "ssl_keyfile": None,
            "ssl_keyfile_password": None,
            "request_timeout": 120,
            "keep_alive_timeout": 5,
            "graceful_shutdown_timeout": 30,
            "max_request_size": 10485760,
            "max_connections": 1000,
            "max_concurrent_requests": 100,
            "backlog": 2048,
        }
    }

    CORS: dict[str, Any] = {
        "cors": {
            "enabled": True,
            "allow_origins": ["*"],
            "allow_methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["*"],
            "allow_credentials": False,
            "max_age": 600,
            "expose_headers": [],
        }
    }

    RATE_LIMITING: dict[str, Any] = {
        "rate_limiting": {
            "enabled": True,
            "requests_per_minute": 60,
            "requests_per_hour": 1000,
            "burst_size": 10,
            "storage": "memory",
            "redis_url": "redis://localhost:6379/0",
        }
    }

    AUTH: dict[str, Any] = {
        "auth": {
            "enabled": False,
            "secret_key": "change-me-in-production",
            "algorithm": "HS256",
            "access_token_expire_minutes": 30,
            "refresh_token_expire_days": 7,
            "api_key_header": "X-API-Key",
            "api_keys": [],
        }
    }

    SERVER_CACHING: dict[str, Any] = {
        "caching": {
            "enabled": True,
            "backend": "memory",
            "ttl": 300,
            "max_size": 1000,
            "redis_url": "redis://localhost:6379/1",
        }
    }

    STREAMING: dict[str, Any] = {
        "streaming": {
            "enabled": True,
            "chunk_size": 1024,
            "heartbeat_interval": 15,
            "max_duration": 300,
        }
    }

    HEALTH: dict[str, Any] = {
        "health": {
            "enabled": True,
            "endpoint": "/health",
            "check_interval": 30,
            "models_check": True,
            "memory_threshold": 90,
        }
    }

    METRICS: dict[str, Any] = {
        "metrics": {
            "enabled": True,
            "endpoint": "/metrics",
            "prometheus": False,
            "collect_interval": 10,
        }
    }

    # ── UI Defaults ────────────────────────────────────────────

    UI: dict[str, Any] = {
        "ui": {
            "theme": "dark",
            "color_support": "auto",
            "force_color": False,
            "respect_no_color": True,
            "terminal_width": 0,
            "terminal_height": 0,
            "use_alt_screen": False,
            "show_cursor": True,
            "prompt_char": "❯",
            "prompt_style": "bold cyan",
            "continuation_char": "...",
            "continuation_style": "dim",
            "show_timestamp": False,
            "input_mode": "emacs",
            "multiline": True,
            "auto_indent": True,
            "auto_close_brackets": True,
            "smart_enter": True,
            "indent_str": "    ",
            "history_file": "~/.nexus_llm/prompt_history",
            "max_history": 10000,
            "show_welcome": True,
            "show_status_bar": True,
            "status_bar_position": "bottom",
            "show_token_count": True,
            "show_latency": True,
            "show_memory": True,
            "show_model_name": True,
            "markdown_rendering": True,
            "syntax_highlighting": True,
            "code_line_numbers": True,
            "code_theme": "monokai",
            "word_wrap": True,
            "wrap_width": 0,
            "panel_style": "rounded",
            "panel_border_color": "cyan",
            "panel_padding": 1,
            "table_style": "rounded",
            "table_show_lines": False,
            "spinner_style": "dots",
            "spinner_color": "cyan",
            "progress_bar_width": 30,
            "progress_show_eta": True,
            "progress_show_rate": True,
        }
    }

    KEYBINDINGS: dict[str, Any] = {
        "keybindings": {
            "mode": "emacs",
            "custom": {},
        }
    }

    FORMATTING: dict[str, Any] = {
        "formatting": {
            "date_format": "%Y-%m-%d %H:%M:%S",
            "number_format": ",.2f",
            "time_format": "relative",
            "truncate_length": 80,
        }
    }

    HISTORY: dict[str, Any] = {
        "history": {
            "enabled": True,
            "directory": "~/.nexus_llm/history",
            "auto_save": True,
            "save_interval": 30,
            "max_entries": 10000,
            "max_sessions": 100,
        }
    }

    EXPORT: dict[str, Any] = {
        "export": {
            "default_format": "json",
            "directory": "~/.nexus_llm/exports",
        }
    }

    # ── Combined defaults ──────────────────────────────────────

    ALL_DEFAULTS: dict[str, Any] = {}

    @classmethod
    def _build_all_defaults(cls) -> dict[str, Any]:
        """Build the combined defaults dictionary by merging all sections.

        Returns:
            Complete defaults dictionary.
        """
        from nexus_llm.config.settings import _deep_merge

        result: dict[str, Any] = {}
        sections = [
            cls.MODEL,
            cls.GENERATION,
            cls.QUANTIZATION,
            cls.CACHE,
            cls.TOKENIZER,
            cls.TRAINING,
            cls.LORA,
            cls.PEFT,
            cls.SERVER,
            cls.CORS,
            cls.RATE_LIMITING,
            cls.AUTH,
            cls.SERVER_CACHING,
            cls.STREAMING,
            cls.HEALTH,
            cls.METRICS,
            cls.UI,
            cls.KEYBINDINGS,
            cls.FORMATTING,
            cls.HISTORY,
            cls.EXPORT,
        ]
        for section in sections:
            result = _deep_merge(result, section)
        return result


# Build the combined defaults at module load time
Defaults.ALL_DEFAULTS = Defaults._build_all_defaults()
