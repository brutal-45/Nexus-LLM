"""
Nexus-LLM Settings Loader Module

Provides a layered configuration system that merges settings from
YAML files, environment variables, and CLI arguments with priority
ordering: CLI args > env vars > YAML config > defaults.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from nexus_llm.config.defaults import Defaults
from nexus_llm.config.validators import ConfigValidator


# Environment variable prefix
ENV_PREFIX = "NEXUS_LLM_"

# Config directory
DEFAULT_CONFIG_DIR = os.path.join(os.path.dirname(__file__))


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dictionaries, with override taking priority.

    Args:
        base: Base dictionary.
        override: Override dictionary (higher priority).

    Returns:
        Merged dictionary.
    """
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _flatten_dict(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten a nested dictionary with dot-notation keys.

    Args:
        data: Nested dictionary.
        prefix: Key prefix for recursion.

    Returns:
        Flat dictionary with dot-separated keys.
    """
    items: dict[str, Any] = {}
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            items.update(_flatten_dict(value, full_key))
        else:
            items[full_key] = value
    return items


def _unflatten_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Unflatten a dot-notation dictionary into a nested structure.

    Args:
        data: Flat dictionary with dot-separated keys.

    Returns:
        Nested dictionary.
    """
    result: dict[str, Any] = {}
    for key, value in data.items():
        parts = key.split(".")
        current = result
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    return result


class Settings:
    """Holds all application settings with attribute-style access.

    Settings are organized into sections (model, training, server, ui)
    and support dot-notation access for nested values.
    """

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        self._data: dict[str, Any] = data or {}

    def get(self, key: str, default: Any = None) -> Any:
        """Get a setting value by dot-notation key.

        Args:
            key: Dot-separated key (e.g., 'model.name').
            default: Default value if key is not found.

        Returns:
            The setting value, or default.
        """
        parts = key.split(".")
        current = self._data
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        return current

    def set(self, key: str, value: Any) -> None:
        """Set a setting value by dot-notation key.

        Args:
            key: Dot-separated key.
            value: Value to set.
        """
        parts = key.split(".")
        current = self._data
        for part in parts[:-1]:
            if part not in current or not isinstance(current[part], dict):
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value

    def has(self, key: str) -> bool:
        """Check if a setting key exists.

        Args:
            key: Dot-separated key.

        Returns:
            True if the key exists.
        """
        return self.get(key, _SENTINEL) is not _SENTINEL

    def section(self, name: str) -> dict[str, Any]:
        """Get all settings in a section.

        Args:
            name: Section name.

        Returns:
            Dictionary of settings in that section.
        """
        return self._data.get(name, {})

    def to_dict(self) -> dict[str, Any]:
        """Get all settings as a dictionary.

        Returns:
            Complete settings dictionary.
        """
        return dict(self._data)

    def update(self, data: dict[str, Any]) -> None:
        """Update settings by deep-merging with the provided data.

        Args:
            data: Dictionary of settings to merge.
        """
        self._data = _deep_merge(self._data, data)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return self._data.get(name, {})

    def __repr__(self) -> str:
        return f"Settings({self._data!r})"


class _Sentinel:
    """Sentinel value for 'key not found' detection."""
    pass


_SENTINEL = _Sentinel()


class SettingsLoader:
    """Loads and merges settings from multiple sources with priority.

    Priority order (highest to lowest):
    1. CLI arguments
    2. Environment variables
    3. YAML configuration files
    4. Default values

    Supports loading from the bundled YAML configs, custom config files,
    and environment variable overrides.
    """

    def __init__(self, config_dir: str | None = None) -> None:
        self._config_dir = config_dir or DEFAULT_CONFIG_DIR
        self._validator = ConfigValidator()
        self._sources: dict[str, dict[str, Any]] = {}

    def load_yaml(self, path: str) -> dict[str, Any]:
        """Load settings from a YAML file.

        Args:
            path: Path to the YAML file.

        Returns:
            Parsed settings dictionary.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the YAML is invalid.
        """
        if not HAS_YAML:
            # Simple YAML-like parser for basic configs
            return self._parse_simple_yaml(path)

        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Expected dict in {path}, got {type(data).__name__}")

        return data

    def _parse_simple_yaml(self, path: str) -> dict[str, Any]:
        """Simple YAML parser for basic key: value configs.

        Used as a fallback when PyYAML is not installed.

        Args:
            path: Path to the YAML file.

        Returns:
            Parsed settings dictionary.
        """
        result: dict[str, Any] = {}
        current_section = result

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip()

                # Skip comments and empty lines
                if not line or line.startswith("#"):
                    continue

                # Section header
                if not line.startswith(" ") and line.endswith(":"):
                    section_name = line[:-1].strip()
                    result[section_name] = {}
                    current_section = result[section_name]
                    continue

                # Key-value pair
                if ":" in line:
                    key, _, value = line.partition(":")
                    key = key.strip()
                    value = value.strip()

                    # Remove inline comments
                    if " #" in value:
                        value = value[: value.index(" #")].strip()

                    # Parse value type
                    parsed_value = self._parse_value(value)
                    current_section[key] = parsed_value

        return result

    @staticmethod
    def _parse_value(value: str) -> Any:
        """Parse a string value into the appropriate Python type.

        Args:
            value: String value from YAML.

        Returns:
            Parsed value (int, float, bool, list, or string).
        """
        if not value:
            return ""

        # Boolean
        if value.lower() in ("true", "yes", "on"):
            return True
        if value.lower() in ("false", "no", "off"):
            return False

        # None
        if value.lower() in ("null", "none", "~"):
            return None

        # Integer
        try:
            return int(value)
        except ValueError:
            pass

        # Float
        try:
            return float(value)
        except ValueError:
            pass

        # List
        if value.startswith("[") and value.endswith("]"):
            items = value[1:-1].split(",")
            return [SettingsLoader._parse_value(i.strip()) for i in items if i.strip()]

        # String (remove quotes)
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            return value[1:-1]

        return value

    def load_env_vars(self, prefix: str = ENV_PREFIX) -> dict[str, Any]:
        """Load settings from environment variables.

        Environment variables are mapped to settings by:
        1. Removing the prefix
        2. Converting to lowercase
        3. Replacing double underscores with dots
        4. Replacing single underscores with underscores

        Example: NEXUS_LLM_MODEL__NAME=gpt2 → model.name = "gpt2"

        Args:
            prefix: Environment variable prefix.

        Returns:
            Dictionary of settings from environment variables.
        """
        env_settings: dict[str, Any] = {}
        prefix_len = len(prefix)

        for env_key, env_value in os.environ.items():
            if env_key.startswith(prefix):
                # Convert env var name to settings key
                setting_key = env_key[prefix_len:].lower()
                # Double underscore → dot (for nested keys)
                setting_key = setting_key.replace("__", ".")
                env_settings[setting_key] = self._parse_value(env_value)

        return _unflatten_dict(env_settings) if env_settings else env_settings

    def load_cli_args(self, args: list[str] | None = None) -> dict[str, Any]:
        """Load settings from CLI arguments.

        Parses --key=value and --key value style arguments.

        Args:
            args: Argument list (defaults to sys.argv[1:]).

        Returns:
            Dictionary of settings from CLI arguments.
        """
        if args is None:
            args = sys.argv[1:]

        cli_settings: dict[str, Any] = {}

        i = 0
        while i < len(args):
            arg = args[i]
            if arg.startswith("--"):
                key = arg[2:]
                if "=" in key:
                    key, _, value = key.partition("=")
                    cli_settings[key.replace("-", "_")] = self._parse_value(value)
                elif i + 1 < len(args) and not args[i + 1].startswith("--"):
                    i += 1
                    cli_settings[key.replace("-", "_")] = self._parse_value(args[i])
                else:
                    cli_settings[key.replace("-", "_")] = True
            i += 1

        return _unflatten_dict(cli_settings) if cli_settings else cli_settings

    def load(
        self,
        config_files: list[str] | None = None,
        cli_args: list[str] | None = None,
        include_defaults: bool = True,
        include_env: bool = True,
        validate: bool = True,
    ) -> Settings:
        """Load settings from all sources with priority merging.

        Args:
            config_files: Additional YAML config file paths.
            cli_args: CLI argument list.
            include_defaults: Whether to include default values.
            include_env: Whether to include environment variables.
            validate: Whether to validate the final settings.

        Returns:
            Merged Settings object.
        """
        # Start with defaults (lowest priority)
        if include_defaults:
            merged = dict(Defaults.ALL_DEFAULTS)
        else:
            merged = {}

        # Load bundled config files
        bundled_configs = [
            os.path.join(self._config_dir, "model_config.yaml"),
            os.path.join(self._config_dir, "training_config.yaml"),
            os.path.join(self._config_dir, "server_config.yaml"),
            os.path.join(self._config_dir, "ui_config.yaml"),
        ]

        for config_path in bundled_configs:
            if os.path.exists(config_path):
                try:
                    data = self.load_yaml(config_path)
                    merged = _deep_merge(merged, data)
                except (OSError, ValueError):
                    pass

        # Load custom config files
        if config_files:
            for config_path in config_files:
                try:
                    data = self.load_yaml(config_path)
                    merged = _deep_merge(merged, data)
                except (OSError, ValueError) as exc:
                    print(f"Warning: Failed to load config {config_path}: {exc}", file=sys.stderr)

        # Environment variables (higher priority)
        if include_env:
            env_data = self.load_env_vars()
            if env_data:
                merged = _deep_merge(merged, env_data)

        # CLI arguments (highest priority)
        if cli_args:
            cli_data = self.load_cli_args(cli_args)
            if cli_data:
                merged = _deep_merge(merged, cli_data)

        # Validate
        if validate:
            errors = self._validator.validate(merged)
            if errors:
                for error in errors:
                    print(f"Config warning: {error}", file=sys.stderr)

        self._sources = {
            "defaults": Defaults.ALL_DEFAULTS,
            "merged": merged,
        }

        return Settings(merged)

    def get_loaded_sources(self) -> dict[str, dict[str, Any]]:
        """Get the raw data from each loaded source.

        Returns:
            Dictionary mapping source names to their data.
        """
        return dict(self._sources)
