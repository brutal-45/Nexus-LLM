"""
Nexus-LLM Configuration Schema Definitions

Provides dataclass-based schema definitions for all configuration
sections with type annotations, defaults, and documentation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class GenerationSchema:
    """Schema for text generation parameters."""
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    max_tokens: int = 512
    min_tokens: int = 1
    repetition_penalty: float = 1.1
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    do_sample: bool = True
    num_beams: int = 1
    num_return_sequences: int = 1
    length_penalty: float = 1.0
    early_stopping: bool = False
    stop_sequences: list[str] = field(default_factory=lambda: ["\n\n\n"])
    bad_words_ids: list[list[int]] = field(default_factory=list)
    no_repeat_ngram_size: int = 0
    seed: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_tokens": self.max_tokens,
            "min_tokens": self.min_tokens,
            "repetition_penalty": self.repetition_penalty,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "do_sample": self.do_sample,
            "num_beams": self.num_beams,
            "num_return_sequences": self.num_return_sequences,
            "length_penalty": self.length_penalty,
            "early_stopping": self.early_stopping,
            "stop_sequences": self.stop_sequences,
            "bad_words_ids": self.bad_words_ids,
            "no_repeat_ngram_size": self.no_repeat_ngram_size,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GenerationSchema:
        """Deserialize from dictionary."""
        known_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_keys}
        return cls(**filtered)


@dataclass
class ModelSchema:
    """Schema for model configuration."""
    name: str = "gpt2-medium"
    path: str = ""
    device: str = "auto"
    dtype: str = "float32"
    trust_remote_code: bool = False
    use_auth_token: bool = False
    revision: str = "main"
    mirror: str = ""
    generation: GenerationSchema = field(default_factory=GenerationSchema)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        result = {
            "name": self.name,
            "path": self.path,
            "device": self.device,
            "dtype": self.dtype,
            "trust_remote_code": self.trust_remote_code,
            "use_auth_token": self.use_auth_token,
            "revision": self.revision,
            "mirror": self.mirror,
        }
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelSchema:
        """Deserialize from dictionary."""
        gen_data = data.pop("generation", {})
        known_keys = {f.name for f in cls.__dataclass_fields__.values() if f != "generation"}
        filtered = {k: v for k, v in data.items() if k in known_keys}
        instance = cls(**filtered)
        if gen_data:
            instance.generation = GenerationSchema.from_dict(gen_data)
        return instance


@dataclass
class LoraSchema:
    """Schema for LoRA (Low-Rank Adaptation) parameters."""
    enabled: bool = False
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    fan_in_fan_out: bool = False
    modules_to_save: list[str] = field(default_factory=list)
    layers_to_transform: Optional[list[int]] = None
    layers_pattern: Optional[str] = None
    rank_pattern: dict[str, int] = field(default_factory=dict)
    alpha_pattern: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "enabled": self.enabled,
            "r": self.r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": self.target_modules,
            "bias": self.bias,
            "task_type": self.task_type,
            "fan_in_fan_out": self.fan_in_fan_out,
            "modules_to_save": self.modules_to_save,
            "layers_to_transform": self.layers_to_transform,
            "layers_pattern": self.layers_pattern,
            "rank_pattern": self.rank_pattern,
            "alpha_pattern": self.alpha_pattern,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LoraSchema:
        """Deserialize from dictionary."""
        known_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_keys}
        return cls(**filtered)


@dataclass
class OptimizerSchema:
    """Schema for optimizer configuration."""
    name: str = "adamw_torch"
    learning_rate: float = 5.0e-5
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    lr_scheduler_kwargs: dict[str, Any] = field(default_factory=dict)
    warmup_steps: int = 0
    warmup_ratio: float = 0.0
    max_grad_norm: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "lr_scheduler_type": self.lr_scheduler_type,
            "lr_scheduler_kwargs": self.lr_scheduler_kwargs,
            "warmup_steps": self.warmup_steps,
            "warmup_ratio": self.warmup_ratio,
            "max_grad_norm": self.max_grad_norm,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OptimizerSchema:
        """Deserialize from dictionary."""
        known_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_keys}
        return cls(**filtered)


@dataclass
class TrainingSchema:
    """Schema for training configuration."""
    output_dir: str = "./outputs"
    overwrite_output_dir: bool = False
    seed: int = 42
    deterministic: bool = False
    num_train_epochs: int = 3
    max_steps: int = -1
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    fp16: bool = False
    bf16: bool = False
    gradient_checkpointing: bool = False
    evaluation_strategy: str = "no"
    eval_steps: int = 500
    save_strategy: str = "steps"
    save_steps: int = 500
    save_total_limit: int = 2
    save_safetensors: bool = True
    load_best_model_at_end: bool = False
    metric_for_best_model: str = "loss"
    logging_dir: str = "./logs"
    logging_steps: int = 10
    log_level: str = "info"
    report_to: str = "tensorboard"
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = True
    group_by_length: bool = False
    optimizer: OptimizerSchema = field(default_factory=OptimizerSchema)
    lora: LoraSchema = field(default_factory=LoraSchema)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "output_dir": self.output_dir,
            "overwrite_output_dir": self.overwrite_output_dir,
            "seed": self.seed,
            "deterministic": self.deterministic,
            "num_train_epochs": self.num_train_epochs,
            "max_steps": self.max_steps,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "per_device_eval_batch_size": self.per_device_eval_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "fp16": self.fp16,
            "bf16": self.bf16,
            "gradient_checkpointing": self.gradient_checkpointing,
            "evaluation_strategy": self.evaluation_strategy,
            "eval_steps": self.eval_steps,
            "save_strategy": self.save_strategy,
            "save_steps": self.save_steps,
            "save_total_limit": self.save_total_limit,
            "save_safetensors": self.save_safetensors,
            "load_best_model_at_end": self.load_best_model_at_end,
            "metric_for_best_model": self.metric_for_best_model,
            "logging_dir": self.logging_dir,
            "logging_steps": self.logging_steps,
            "log_level": self.log_level,
            "report_to": self.report_to,
            "dataloader_num_workers": self.dataloader_num_workers,
            "dataloader_pin_memory": self.dataloader_pin_memory,
            "group_by_length": self.group_by_length,
            "optimizer": self.optimizer.to_dict(),
            "lora": self.lora.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingSchema:
        """Deserialize from dictionary."""
        opt_data = data.pop("optimizer", {})
        lora_data = data.pop("lora", {})
        known_keys = {f.name for f in cls.__dataclass_fields__.values()
                      if f not in ("optimizer", "lora")}
        filtered = {k: v for k, v in data.items() if k in known_keys}
        instance = cls(**filtered)
        if opt_data:
            instance.optimizer = OptimizerSchema.from_dict(opt_data)
        if lora_data:
            instance.lora = LoraSchema.from_dict(lora_data)
        return instance


@dataclass
class CorsSchema:
    """Schema for CORS configuration."""
    enabled: bool = True
    allow_origins: list[str] = field(default_factory=lambda: ["*"])
    allow_methods: list[str] = field(default_factory=lambda: ["GET", "POST", "PUT", "DELETE", "OPTIONS"])
    allow_headers: list[str] = field(default_factory=lambda: ["*"])
    allow_credentials: bool = False
    max_age: int = 600
    expose_headers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "enabled": self.enabled,
            "allow_origins": self.allow_origins,
            "allow_methods": self.allow_methods,
            "allow_headers": self.allow_headers,
            "allow_credentials": self.allow_credentials,
            "max_age": self.max_age,
            "expose_headers": self.expose_headers,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CorsSchema:
        """Deserialize from dictionary."""
        known_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_keys}
        return cls(**filtered)


@dataclass
class RateLimitSchema:
    """Schema for rate limiting configuration."""
    enabled: bool = True
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    burst_size: int = 10
    storage: str = "memory"
    redis_url: str = "redis://localhost:6379/0"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "enabled": self.enabled,
            "requests_per_minute": self.requests_per_minute,
            "requests_per_hour": self.requests_per_hour,
            "burst_size": self.burst_size,
            "storage": self.storage,
            "redis_url": self.redis_url,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RateLimitSchema:
        """Deserialize from dictionary."""
        known_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_keys}
        return cls(**filtered)


@dataclass
class LoggingSchema:
    """Schema for logging configuration."""
    level: str = "info"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = ""
    rotation: str = "10 MB"
    retention: str = "30 days"
    console: bool = True
    file_enabled: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "level": self.level,
            "format": self.format,
            "file": self.file,
            "rotation": self.rotation,
            "retention": self.retention,
            "console": self.console,
            "file_enabled": self.file_enabled,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LoggingSchema:
        """Deserialize from dictionary."""
        known_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_keys}
        return cls(**filtered)


@dataclass
class ServerSchema:
    """Schema for server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False
    debug: bool = False
    access_log: bool = True
    log_level: str = "info"
    app_name: str = "Nexus-LLM"
    api_prefix: str = "/api/v1"
    docs_url: str = "/docs"
    redoc_url: str = "/redoc"
    openapi_url: str = "/openapi.json"
    request_timeout: int = 120
    keep_alive_timeout: int = 5
    graceful_shutdown_timeout: int = 30
    max_request_size: int = 10485760
    max_connections: int = 1000
    max_concurrent_requests: int = 100
    backlog: int = 2048
    ssl_certfile: Optional[str] = None
    ssl_keyfile: Optional[str] = None
    cors: CorsSchema = field(default_factory=CorsSchema)
    rate_limiting: RateLimitSchema = field(default_factory=RateLimitSchema)
    logging: LoggingSchema = field(default_factory=LoggingSchema)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        result = {
            "host": self.host,
            "port": self.port,
            "workers": self.workers,
            "reload": self.reload,
            "debug": self.debug,
            "access_log": self.access_log,
            "log_level": self.log_level,
            "app_name": self.app_name,
            "api_prefix": self.api_prefix,
            "docs_url": self.docs_url,
            "redoc_url": self.redoc_url,
            "openapi_url": self.openapi_url,
            "request_timeout": self.request_timeout,
            "keep_alive_timeout": self.keep_alive_timeout,
            "graceful_shutdown_timeout": self.graceful_shutdown_timeout,
            "max_request_size": self.max_request_size,
            "max_connections": self.max_connections,
            "max_concurrent_requests": self.max_concurrent_requests,
            "backlog": self.backlog,
            "ssl_certfile": self.ssl_certfile,
            "ssl_keyfile": self.ssl_keyfile,
            "cors": self.cors.to_dict(),
            "rate_limiting": self.rate_limiting.to_dict(),
            "logging": self.logging.to_dict(),
        }
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ServerSchema:
        """Deserialize from dictionary."""
        cors_data = data.pop("cors", {})
        rate_data = data.pop("rate_limiting", {})
        log_data = data.pop("logging", {})
        nested_keys = {"cors", "rate_limiting", "logging"}
        known_keys = {f.name for f in cls.__dataclass_fields__.values()} - nested_keys
        filtered = {k: v for k, v in data.items() if k in known_keys}
        instance = cls(**filtered)
        if cors_data:
            instance.cors = CorsSchema.from_dict(cors_data)
        if rate_data:
            instance.rate_limiting = RateLimitSchema.from_dict(rate_data)
        if log_data:
            instance.logging = LoggingSchema.from_dict(log_data)
        return instance


@dataclass
class UISchema:
    """Schema for UI configuration."""
    theme: str = "dark"
    color_support: str = "auto"
    force_color: bool = False
    respect_no_color: bool = True
    terminal_width: int = 0
    terminal_height: int = 0
    use_alt_screen: bool = False
    show_cursor: bool = True
    prompt_char: str = "❯"
    prompt_style: str = "bold cyan"
    continuation_char: str = "..."
    continuation_style: str = "dim"
    show_timestamp: bool = False
    input_mode: str = "emacs"
    multiline: bool = True
    auto_indent: bool = True
    auto_close_brackets: bool = True
    smart_enter: bool = True
    indent_str: str = "    "
    history_file: str = "~/.nexus_llm/prompt_history"
    max_history: int = 10000
    show_welcome: bool = True
    show_status_bar: bool = True
    status_bar_position: str = "bottom"
    show_token_count: bool = True
    show_latency: bool = True
    show_memory: bool = True
    show_model_name: bool = True
    markdown_rendering: bool = True
    syntax_highlighting: bool = True
    code_line_numbers: bool = True
    code_theme: str = "monokai"
    word_wrap: bool = True
    wrap_width: int = 0
    panel_style: str = "rounded"
    panel_border_color: str = "cyan"
    panel_padding: int = 1
    table_style: str = "rounded"
    table_show_lines: bool = False
    spinner_style: str = "dots"
    spinner_color: str = "cyan"
    progress_bar_width: int = 30
    progress_show_eta: bool = True
    progress_show_rate: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {f.name: getattr(self, f.name) for f in self.__dataclass_fields__.values()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UISchema:
        """Deserialize from dictionary."""
        known_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_keys}
        return cls(**filtered)
