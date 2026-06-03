"""
Nexus-LLM Configuration Profiles

Provides predefined configuration profiles for different use cases:
minimal, standard, high_performance, development, and production.
Each profile overrides specific default values.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field
from typing import Any

from nexus_llm.config.defaults import Defaults
from nexus_llm.config.settings import _deep_merge


@dataclass
class Profile:
    """A named configuration profile with overrides."""
    name: str
    display_name: str = ""
    description: str = ""
    overrides: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.display_name:
            self.display_name = self.name.replace("_", " ").title()

    def apply(self, base_config: dict[str, Any]) -> dict[str, Any]:
        """Apply this profile's overrides to a base configuration.

        Args:
            base_config: Base configuration dictionary.

        Returns:
            Merged configuration with profile overrides.
        """
        return _deep_merge(base_config, self.overrides)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "overrides": self.overrides,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Profile:
        """Deserialize from dictionary."""
        return cls(
            name=data.get("name", ""),
            display_name=data.get("display_name", ""),
            description=data.get("description", ""),
            overrides=data.get("overrides", {}),
            tags=data.get("tags", []),
        )


# ── Built-in Profiles ──────────────────────────────────────────


def _make_minimal_profile() -> Profile:
    """Create the minimal profile - lowest resource usage."""
    return Profile(
        name="minimal",
        display_name="Minimal",
        description="Minimal configuration for resource-constrained environments. Uses CPU, small batch sizes, and no streaming.",
        overrides={
            "model": {
                "name": "distilgpt2",
                "device": "cpu",
                "dtype": "float32",
            },
            "generation": {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 10,
                "max_tokens": 128,
                "do_sample": False,
                "num_beams": 1,
            },
            "training": {
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 8,
                "fp16": False,
                "bf16": False,
                "gradient_checkpointing": False,
                "dataloader_num_workers": 0,
                "save_steps": 1000,
                "logging_steps": 50,
            },
            "lora": {
                "enabled": True,
                "r": 4,
                "lora_alpha": 8,
            },
            "server": {
                "workers": 1,
                "max_connections": 50,
                "max_concurrent_requests": 10,
            },
            "rate_limiting": {
                "requests_per_minute": 10,
                "burst_size": 3,
            },
            "ui": {
                "syntax_highlighting": False,
                "code_line_numbers": False,
                "markdown_rendering": False,
                "show_status_bar": False,
                "show_memory": False,
                "spinner_style": "line",
            },
        },
        tags=["cpu", "low-memory", "minimal"],
    )


def _make_standard_profile() -> Profile:
    """Create the standard profile - balanced defaults."""
    return Profile(
        name="standard",
        display_name="Standard",
        description="Standard configuration with balanced performance and resource usage. Good for most use cases.",
        overrides={
            "model": {
                "name": "gpt2-medium",
                "device": "auto",
                "dtype": "float32",
            },
            "generation": {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50,
                "max_tokens": 512,
                "do_sample": True,
            },
            "training": {
                "per_device_train_batch_size": 4,
                "gradient_accumulation_steps": 1,
                "learning_rate": 5.0e-5,
                "num_train_epochs": 3,
            },
            "lora": {
                "enabled": False,
                "r": 16,
                "lora_alpha": 32,
            },
            "server": {
                "workers": 1,
                "max_connections": 1000,
                "max_concurrent_requests": 100,
            },
            "rate_limiting": {
                "requests_per_minute": 60,
                "burst_size": 10,
            },
            "ui": {
                "theme": "dark",
                "syntax_highlighting": True,
                "markdown_rendering": True,
                "show_status_bar": True,
            },
        },
        tags=["balanced", "default", "standard"],
    )


def _make_high_performance_profile() -> Profile:
    """Create the high performance profile - maximum throughput."""
    return Profile(
        name="high_performance",
        display_name="High Performance",
        description="Maximum performance configuration for powerful hardware. Uses GPU, large batch sizes, and optimized settings.",
        overrides={
            "model": {
                "name": "gpt2-xl",
                "device": "cuda",
                "dtype": "float16",
            },
            "generation": {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 100,
                "max_tokens": 2048,
                "do_sample": True,
                "num_beams": 1,
            },
            "quantization": {
                "enabled": True,
                "bits": 4,
                "group_size": 128,
                "quant_method": "gptq",
            },
            "training": {
                "per_device_train_batch_size": 16,
                "gradient_accumulation_steps": 1,
                "fp16": True,
                "bf16": False,
                "gradient_checkpointing": True,
                "dataloader_num_workers": 4,
                "dataloader_pin_memory": True,
                "dataloader_persistent_workers": True,
                "learning_rate": 1.0e-4,
                "num_train_epochs": 5,
                "save_steps": 200,
                "logging_steps": 5,
            },
            "lora": {
                "enabled": True,
                "r": 64,
                "lora_alpha": 128,
                "lora_dropout": 0.01,
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
            },
            "server": {
                "workers": 4,
                "max_connections": 5000,
                "max_concurrent_requests": 500,
                "keep_alive_timeout": 30,
            },
            "rate_limiting": {
                "requests_per_minute": 600,
                "burst_size": 50,
            },
            "caching": {
                "enabled": True,
                "backend": "redis",
                "ttl": 600,
                "max_size": 10000,
            },
            "ui": {
                "theme": "monokai",
                "syntax_highlighting": True,
                "code_line_numbers": True,
                "markdown_rendering": True,
                "show_status_bar": True,
                "progress_bar_width": 40,
            },
        },
        tags=["gpu", "high-performance", "cuda", "production"],
    )


def _make_development_profile() -> Profile:
    """Create the development profile - fast iteration."""
    return Profile(
        name="development",
        display_name="Development",
        description="Development configuration optimized for fast iteration, debugging, and testing.",
        overrides={
            "model": {
                "name": "gpt2",
                "device": "auto",
                "dtype": "float32",
            },
            "generation": {
                "temperature": 0.5,
                "top_p": 0.9,
                "top_k": 20,
                "max_tokens": 256,
                "do_sample": True,
            },
            "training": {
                "per_device_train_batch_size": 2,
                "gradient_accumulation_steps": 2,
                "fp16": False,
                "bf16": False,
                "gradient_checkpointing": False,
                "num_train_epochs": 1,
                "max_steps": 100,
                "learning_rate": 1.0e-4,
                "logging_steps": 1,
                "save_steps": 50,
                "save_total_limit": 5,
                "evaluation_strategy": "steps",
                "eval_steps": 25,
                "log_level": "debug",
            },
            "lora": {
                "enabled": True,
                "r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.1,
            },
            "server": {
                "workers": 1,
                "reload": True,
                "debug": True,
                "access_log": True,
                "log_level": "debug",
            },
            "rate_limiting": {
                "enabled": False,
            },
            "auth": {
                "enabled": False,
            },
            "ui": {
                "theme": "dracula",
                "show_timestamp": True,
                "show_status_bar": True,
                "show_memory": True,
                "show_latency": True,
                "spinner_style": "dots",
            },
        },
        tags=["dev", "debug", "fast-iteration"],
    )


def _make_production_profile() -> Profile:
    """Create the production profile - security and reliability."""
    return Profile(
        name="production",
        display_name="Production",
        description="Production configuration prioritizing security, reliability, and monitoring.",
        overrides={
            "model": {
                "name": "gpt2-medium",
                "device": "auto",
                "dtype": "float16",
            },
            "generation": {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50,
                "max_tokens": 512,
                "do_sample": True,
            },
            "quantization": {
                "enabled": True,
                "bits": 8,
                "group_size": 128,
                "quant_method": "bitsandbytes",
            },
            "training": {
                "fp16": True,
                "gradient_checkpointing": True,
                "save_total_limit": 3,
                "save_safetensors": True,
                "load_best_model_at_end": True,
                "logging_steps": 10,
                "log_level": "warning",
            },
            "lora": {
                "enabled": True,
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
            },
            "server": {
                "workers": 4,
                "reload": False,
                "debug": False,
                "access_log": True,
                "log_level": "warning",
                "graceful_shutdown_timeout": 60,
            },
            "cors": {
                "enabled": True,
                "allow_origins": [],  # Must be explicitly configured
                "allow_credentials": True,
            },
            "rate_limiting": {
                "enabled": True,
                "requests_per_minute": 30,
                "burst_size": 5,
            },
            "auth": {
                "enabled": True,
                "access_token_expire_minutes": 15,
            },
            "health": {
                "enabled": True,
                "check_interval": 15,
            },
            "metrics": {
                "enabled": True,
                "prometheus": True,
            },
            "ui": {
                "theme": "nord",
                "show_timestamp": False,
                "show_status_bar": True,
            },
        },
        tags=["production", "secure", "monitored", "reliable"],
    )


class ProfileManager:
    """Manages configuration profiles with creation, loading, and application.

    Supports built-in profiles (minimal, standard, high_performance,
    development, production) and custom user-defined profiles stored
    as JSON files.
    """

    DEFAULT_PROFILES_DIR = os.path.expanduser("~/.nexus_llm/profiles")

    def __init__(self, profiles_dir: str | None = None) -> None:
        self._profiles: dict[str, Profile] = {}
        self._profiles_dir = profiles_dir or self.DEFAULT_PROFILES_DIR
        self._active_profile: str | None = None

        # Register built-in profiles
        self._register_builtins()
        # Load custom profiles
        self._load_custom_profiles()

    def _register_builtins(self) -> None:
        """Register all built-in profiles."""
        builtins = [
            _make_minimal_profile(),
            _make_standard_profile(),
            _make_high_performance_profile(),
            _make_development_profile(),
            _make_production_profile(),
        ]
        for profile in builtins:
            self._profiles[profile.name] = profile

    def _load_custom_profiles(self) -> None:
        """Load custom profiles from the profiles directory."""
        profiles_path = os.path.expanduser(self._profiles_dir)
        if not os.path.exists(profiles_path):
            return

        for profile_file in os.listdir(profiles_path):
            if profile_file.endswith(".json"):
                try:
                    filepath = os.path.join(profiles_path, profile_file)
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    profile = Profile.from_dict(data)
                    self._profiles[profile.name] = profile
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue

    @property
    def active_profile(self) -> str | None:
        """Get the name of the active profile."""
        return self._active_profile

    @property
    def active_profile_obj(self) -> Profile | None:
        """Get the active Profile object."""
        if self._active_profile:
            return self._profiles.get(self._active_profile)
        return None

    def get_profile(self, name: str) -> Profile | None:
        """Get a profile by name.

        Args:
            name: Profile name.

        Returns:
            The Profile object, or None if not found.
        """
        return self._profiles.get(name)

    def list_profiles(self) -> list[Profile]:
        """List all available profiles.

        Returns:
            List of Profile objects.
        """
        return list(self._profiles.values())

    def list_profile_names(self) -> list[str]:
        """List all available profile names.

        Returns:
            List of profile name strings.
        """
        return list(self._profiles.keys())

    def apply_profile(self, name: str, base_config: dict[str, Any] | None = None) -> dict[str, Any]:
        """Apply a profile's overrides to a base configuration.

        Args:
            name: Profile name to apply.
            base_config: Base configuration (defaults to Defaults.ALL_DEFAULTS).

        Returns:
            Configuration dictionary with profile overrides applied.

        Raises:
            ValueError: If the profile is not found.
        """
        profile = self._profiles.get(name)
        if not profile:
            raise ValueError(f"Profile '{name}' not found. Available: {self.list_profile_names()}")

        base = base_config if base_config is not None else copy.deepcopy(Defaults.ALL_DEFAULTS)
        self._active_profile = name
        return profile.apply(base)

    def create_profile(
        self,
        name: str,
        display_name: str = "",
        description: str = "",
        overrides: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> Profile:
        """Create a new custom profile.

        Args:
            name: Unique profile name.
            display_name: Human-readable name.
            description: Profile description.
            overrides: Configuration overrides.
            tags: Optional tags.

        Returns:
            The created Profile object.

        Raises:
            ValueError: If a profile with the same name already exists.
        """
        if name in self._profiles:
            raise ValueError(f"Profile '{name}' already exists")

        profile = Profile(
            name=name,
            display_name=display_name,
            description=description,
            overrides=overrides or {},
            tags=tags or [],
        )
        self._profiles[name] = profile
        return profile

    def delete_profile(self, name: str) -> bool:
        """Delete a custom profile.

        Built-in profiles cannot be deleted.

        Args:
            name: Profile name to delete.

        Returns:
            True if the profile was deleted.
        """
        builtin_names = {"minimal", "standard", "high_performance", "development", "production"}
        if name in builtin_names:
            return False
        if name not in self._profiles:
            return False

        del self._profiles[name]
        # Delete the file
        profile_path = os.path.join(os.path.expanduser(self._profiles_dir), f"{name}.json")
        if os.path.exists(profile_path):
            os.remove(profile_path)

        if self._active_profile == name:
            self._active_profile = None
        return True

    def save_profile(self, name: str) -> str:
        """Save a profile to disk as a JSON file.

        Args:
            name: Profile name to save.

        Returns:
            The file path the profile was saved to.

        Raises:
            ValueError: If the profile is not found.
        """
        profile = self._profiles.get(name)
        if not profile:
            raise ValueError(f"Profile '{name}' not found")

        profiles_dir = os.path.expanduser(self._profiles_dir)
        os.makedirs(profiles_dir, exist_ok=True)

        filepath = os.path.join(profiles_dir, f"{name}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(profile.to_dict(), f, indent=2, ensure_ascii=False)

        return filepath

    def export_profile(self, name: str) -> str:
        """Export a profile as a JSON string.

        Args:
            name: Profile name to export.

        Returns:
            JSON string of the profile.

        Raises:
            ValueError: If the profile is not found.
        """
        profile = self._profiles.get(name)
        if not profile:
            raise ValueError(f"Profile '{name}' not found")
        return json.dumps(profile.to_dict(), indent=2, ensure_ascii=False)

    def import_profile(self, json_str: str) -> Profile:
        """Import a profile from a JSON string.

        Args:
            json_str: JSON string containing the profile definition.

        Returns:
            The imported Profile object.
        """
        data = json.loads(json_str)
        profile = Profile.from_dict(data)
        self._profiles[profile.name] = profile
        return profile

    def get_help_text(self) -> str:
        """Generate help text describing all available profiles.

        Returns:
            Formatted help string.
        """
        lines = ["Available Configuration Profiles:", ""]
        for name in sorted(self._profiles):
            profile = self._profiles[name]
            active_marker = " ← active" if name == self._active_profile else ""
            lines.append(f"  {profile.display_name} ({name}){active_marker}")
            lines.append(f"    {profile.description}")
            if profile.tags:
                lines.append(f"    Tags: {', '.join(profile.tags)}")
            lines.append("")
        return "\n".join(lines)
