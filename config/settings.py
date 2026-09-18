"""Configuration settings loader and manager."""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str = "gpt2-medium"
    type: str = "causal"
    device: str = "auto"
    precision: str = "fp32"
    max_length: int = 1024
    max_new_tokens: int = 512


@dataclass
class InferenceConfig:
    """Inference configuration."""
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.2
    num_beams: int = 1
    streaming: bool = True
    system_prompt: str = (
        "You are a helpful, harmless, and honest AI assistant. "
        "You provide clear, well-structured answers."
    )


@dataclass
class BackendConfig:
    """Backend server configuration."""
    host: str = "127.0.0.1"
    port: int = 8765
    workers: int = 1
    cors_origins: List[str] = field(default_factory=lambda: ["*"])


@dataclass
class TrainingConfig:
    """Training pipeline configuration."""
    base_model: str = "gpt2-medium"
    output_dir: str = "./models/fine-tuned"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    learning_rate: float = 5e-5
    warmup_steps: int = 500
    weight_decay: float = 0.01
    save_strategy: str = "steps"
    save_steps: int = 500
    logging_steps: int = 10
    dataset_path: str = "./data/training_data.jsonl"
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1


@dataclass
class ChatConfig:
    """Chat interface configuration."""
    history_size: int = 20
    save_history: bool = True
    history_dir: str = "./chat_history"
    show_stats: bool = True


class Settings:
    """Main settings class that loads and manages all configuration."""

    _instance: Optional["Settings"] = None

    def __init__(self, config_path: Optional[str] = None):
        self.model = ModelConfig()
        self.inference = InferenceConfig()
        self.backend = BackendConfig()
        self.training = TrainingConfig()
        self.chat = ChatConfig()

        if config_path:
            self.load_from_yaml(config_path)

        # Override with environment variables
        self._load_env_overrides()

    def load_from_yaml(self, config_path: str) -> None:
        """Load configuration from a YAML file."""
        path = Path(config_path)
        if not path.exists():
            return

        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}

        if "model" in data:
            self._update_dataclass(self.model, data["model"])
        if "inference" in data:
            self._update_dataclass(self.inference, data["inference"])
        if "backend" in data:
            self._update_dataclass(self.backend, data["backend"])
        if "training" in data:
            self._update_dataclass(self.training, data["training"])
        if "chat" in data:
            self._update_dataclass(self.chat, data["chat"])

    def _update_dataclass(self, obj: Any, data: Dict[str, Any]) -> None:
        """Update a dataclass instance with dictionary data."""
        for key, value in data.items():
            if hasattr(obj, key):
                setattr(obj, key, value)

    def _load_env_overrides(self) -> None:
        """Override settings with environment variables."""
        env_mapping = {
            "LLM_MODEL_NAME": ("model", "name"),
            "LLM_DEVICE": ("model", "device"),
            "LLM_PRECISION": ("model", "precision"),
            "LLM_TEMPERATURE": ("inference", "temperature"),
            "LLM_TOP_P": ("inference", "top_p"),
            "LLM_TOP_K": ("inference", "top_k"),
            "LLM_BACKEND_HOST": ("backend", "host"),
            "LLM_BACKEND_PORT": ("backend", "port"),
            "LLM_SYSTEM_PROMPT": ("inference", "system_prompt"),
        }

        for env_var, (section, attr) in env_mapping.items():
            value = os.environ.get(env_var)
            if value is not None:
                obj = getattr(self, section)
                current_type = type(getattr(obj, attr))
                try:
                    setattr(obj, attr, current_type(value))
                except (ValueError, TypeError):
                    setattr(obj, attr, value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        from dataclasses import asdict
        return {
            "model": asdict(self.model),
            "inference": asdict(self.inference),
            "backend": asdict(self.backend),
            "training": asdict(self.training),
            "chat": asdict(self.chat),
        }

    def save_to_yaml(self, config_path: str) -> None:
        """Save current configuration to a YAML file."""
        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


def get_settings(config_path: Optional[str] = None) -> Settings:
    """Get or create the settings singleton."""
    if Settings._instance is None:
        Settings._instance = Settings(config_path)
    return Settings._instance


def reset_settings() -> None:
    """Reset the settings singleton (useful for testing)."""
    Settings._instance = None
