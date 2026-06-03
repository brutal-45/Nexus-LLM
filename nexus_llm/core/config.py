"""Configuration management for Nexus-LLM."""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path

import yaml

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default_config.yaml"


@dataclass
class ModelSettings:
    """Model-related settings."""
    name: str = "gpt2-medium"
    device: str = "auto"  # auto, cpu, cuda, mps
    precision: str = "fp32"  # fp32, fp16, bf16, 8bit, 4bit
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    num_beams: int = 1
    do_sample: bool = True
    cache_dir: str = str(PROJECT_ROOT / "models")


@dataclass
class ServerSettings:
    """Server-related settings."""
    host: str = "127.0.0.1"
    port: int = 8000
    workers: int = 1
    cors_origins: list = field(default_factory=lambda: ["*"])
    api_key: Optional[str] = None


@dataclass
class TerminalSettings:
    """Terminal-related settings."""
    theme: str = "dark"
    show_tokens: bool = True
    show_timing: bool = True
    streaming: bool = True
    markdown: bool = True
    syntax_highlight: bool = True
    history_file: str = str(PROJECT_ROOT / ".nexus_history")
    max_history: int = 1000


@dataclass
class TrainingSettings:
    """Training-related settings."""
    output_dir: str = str(PROJECT_ROOT / "models" / "fine-tuned")
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    save_steps: int = 500
    eval_steps: int = 500
    max_seq_length: int = 512


@dataclass
class Settings:
    """Main settings for Nexus-LLM."""
    model: ModelSettings = field(default_factory=ModelSettings)
    server: ServerSettings = field(default_factory=ServerSettings)
    terminal: TerminalSettings = field(default_factory=TerminalSettings)
    training: TrainingSettings = field(default_factory=TrainingSettings)
    debug: bool = False
    log_level: str = "INFO"
    log_file: str = str(PROJECT_ROOT / "logs" / "nexus_llm.log")

    @classmethod
    def from_yaml(cls, path: str) -> "Settings":
        """Load settings from a YAML file."""
        settings = cls()
        if not os.path.exists(path):
            return settings
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}

        if "model" in data:
            for k, v in data["model"].items():
                if hasattr(settings.model, k):
                    setattr(settings.model, k, v)
        if "server" in data:
            for k, v in data["server"].items():
                if hasattr(settings.server, k):
                    setattr(settings.server, k, v)
        if "terminal" in data:
            for k, v in data["terminal"].items():
                if hasattr(settings.terminal, k):
                    setattr(settings.terminal, k, v)
        if "training" in data:
            for k, v in data["training"].items():
                if hasattr(settings.training, k):
                    setattr(settings.training, k, v)
        if "debug" in data:
            settings.debug = data["debug"]
        if "log_level" in data:
            settings.log_level = data["log_level"]
        return settings

    def to_yaml(self, path: str) -> None:
        """Save settings to a YAML file."""
        import dataclasses
        data = dataclasses.asdict(self)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)


_settings_instance: Optional[Settings] = None


def get_settings(config_path: Optional[str] = None) -> Settings:
    """Get or create the global settings instance."""
    global _settings_instance
    if _settings_instance is None:
        path = config_path or os.environ.get("NEXUS_CONFIG", str(DEFAULT_CONFIG_PATH))
        _settings_instance = Settings.from_yaml(path)
    return _settings_instance


def reset_settings() -> None:
    """Reset settings (useful for testing)."""
    global _settings_instance
    _settings_instance = None
