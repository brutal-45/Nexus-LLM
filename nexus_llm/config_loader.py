"""Nexus-LLM Configuration Loader Module.

Provides configuration loading from multiple sources with a priority system:
1. CLI arguments (highest priority)
2. Environment variables
3. Configuration file (YAML/JSON)
4. Default values (lowest priority)

Supports deep merging of configuration from different sources,
variable interpolation, and configuration validation.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from nexus_llm.constants import CONFIG_DIR, CONFIG_FILENAME, ENV_PREFIX
from nexus_llm.exceptions import ConfigError

logger = logging.getLogger(__name__)

# Default configuration values
_DEFAULTS: Dict[str, Any] = {
    "app_name": "Nexus-LLM",
    "version": "0.1.0",
    "debug": False,
    "log_level": "INFO",
    "default_model": "gpt2-medium",
    "model_cache_dir": "./cache",
    "max_tokens": 2048,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "host": "0.0.0.0",
    "port": 8000,
    "workers": 1,
    "device": "auto",
    "precision": "fp16",
    "train_batch_size": 8,
    "train_learning_rate": 2e-5,
    "train_epochs": 3,
    "train_output_dir": "./output",
    "train_fp16": False,
    "train_bf16": False,
    "data_dir": "./data",
    "max_seq_length": 2048,
    "cors_enabled": False,
    "api_key": None,
}


class ConfigSource:
    """Represents a configuration source with metadata.

    Attributes:
        name: Name of the configuration source.
        priority: Priority (higher = takes precedence).
        values: The configuration values from this source.
    """

    def __init__(self, name: str, priority: int, values: Dict[str, Any]) -> None:
        self.name = name
        self.priority = priority
        self.values = values

    def __repr__(self) -> str:
        return f"ConfigSource(name={self.name!r}, priority={self.priority}, keys={len(self.values)})"


class ConfigLoader:
    """Configuration loader that merges settings from multiple sources.

    Loads configuration from files, environment variables, and defaults,
    merging them with a priority system.

    Example:
        >>> loader = ConfigLoader()
        >>> config = loader.load("config.yaml")
        >>> model = loader.get("default_model")
        >>> loader.set("default_model", "mistral-7b")
    """

    def __init__(
        self,
        env_prefix: str = ENV_PREFIX,
        defaults: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the configuration loader.

        Args:
            env_prefix: Prefix for environment variables.
            defaults: Override default configuration values.
        """
        self._env_prefix = env_prefix
        self._defaults = defaults or _DEFAULTS.copy()
        self._sources: List[ConfigSource] = []
        self._merged: Dict[str, Any] = {}
        self._overrides: Dict[str, Any] = {}
        self._file_path: Optional[str] = None

        # Add defaults as lowest priority source
        self._sources.append(
            ConfigSource(name="defaults", priority=0, values=self._defaults.copy())
        )

    def load(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from all sources.

        Args:
            config_path: Optional path to a configuration file.

        Returns:
            The merged configuration dictionary.
        """
        self._sources = [self._sources[0]]  # Keep defaults

        # Load from file
        if config_path:
            file_config = self._load_file(config_path)
            self._sources.append(
                ConfigSource(name=f"file:{config_path}", priority=10, values=file_config)
            )
            self._file_path = config_path

        # Load from default config location
        if not config_path:
            default_config_path = Path(CONFIG_DIR) / CONFIG_FILENAME
            if default_config_path.exists():
                file_config = self._load_file(str(default_config_path))
                self._sources.append(
                    ConfigSource(name=f"file:{default_config_path}", priority=10, values=file_config)
                )
                self._file_path = str(default_config_path)

        # Load from environment variables
        env_config = self._load_env()
        if env_config:
            self._sources.append(
                ConfigSource(name="environment", priority=20, values=env_config)
            )

        # Merge all sources
        self._merge()

        return self._merged

    def _load_file(self, path: str) -> Dict[str, Any]:
        """Load configuration from a file.

        Args:
            path: Path to the configuration file.

        Returns:
            Configuration dictionary.

        Raises:
            ConfigError: If the file cannot be loaded or parsed.
        """
        file_path = Path(path)
        if not file_path.exists():
            raise ConfigError(
                message=f"Configuration file not found: {path}",
                config_source=path,
            )

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            if file_path.suffix in (".yaml", ".yml"):
                try:
                    import yaml
                    config = yaml.safe_load(content) or {}
                except ImportError:
                    raise ConfigError(
                        message="PyYAML is required to load YAML configuration files. "
                                "Install with: pip install pyyaml",
                        config_source=path,
                    )
            elif file_path.suffix == ".json":
                config = json.loads(content)
            else:
                raise ConfigError(
                    message=f"Unsupported configuration file format: {file_path.suffix}. "
                            f"Use .yaml, .yml, or .json.",
                    config_source=path,
                )

            if not isinstance(config, dict):
                raise ConfigError(
                    message="Configuration file must contain a mapping/dictionary at the top level.",
                    config_source=path,
                )

            # Flatten nested config for consistent key access
            return self._flatten(config)

        except ConfigError:
            raise
        except json.JSONDecodeError as exc:
            raise ConfigError(
                message=f"Invalid JSON in configuration file: {exc}",
                config_source=path,
            ) from exc
        except Exception as exc:
            raise ConfigError(
                message=f"Failed to load configuration file: {exc}",
                config_source=path,
            ) from exc

    def _load_env(self) -> Dict[str, Any]:
        """Load configuration from environment variables.

        Looks for variables with the configured prefix (e.g., NEXUS_LLM_).

        Returns:
            Configuration dictionary from environment variables.
        """
        env_config: Dict[str, Any] = {}
        prefix = self._env_prefix

        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower()
                # Convert double underscores to dots for nested keys
                config_key = config_key.replace("__", ".")
                # Remove leading underscore if present
                config_key = config_key.lstrip("_")
                # Convert value types
                env_config[config_key] = self._parse_env_value(value)

        return env_config

    @staticmethod
    def _parse_env_value(value: str) -> Any:
        """Parse an environment variable value into the appropriate Python type.

        Args:
            value: The string value from the environment variable.

        Returns:
            The parsed value as bool, int, float, or str.
        """
        # Boolean handling
        if value.lower() in ("true", "1", "yes", "on"):
            return True
        if value.lower() in ("false", "0", "no", "off"):
            return False

        # None/null handling
        if value.lower() in ("none", "null", ""):
            return None

        # Integer handling
        try:
            return int(value)
        except ValueError:
            pass

        # Float handling
        try:
            return float(value)
        except ValueError:
            pass

        return value

    def _flatten(self, d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
        """Flatten a nested dictionary into dot-separated keys.

        Args:
            d: Dictionary to flatten.
            parent_key: Prefix for keys.
            sep: Separator between key levels.

        Returns:
            Flattened dictionary.
        """
        items: Dict[str, Any] = {}
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(self._flatten(v, new_key, sep))
            else:
                items[new_key] = v
        return items

    def _unflatten(self, d: Dict[str, Any], sep: str = ".") -> Dict[str, Any]:
        """Unflatten a dictionary with dot-separated keys into nested structure.

        Args:
            d: Flattened dictionary.
            sep: Separator used in keys.

        Returns:
            Nested dictionary.
        """
        result: Dict[str, Any] = {}
        for key, value in d.items():
            parts = key.split(sep)
            current = result
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
        return result

    def _merge(self) -> None:
        """Merge all configuration sources by priority."""
        # Sort sources by priority (lowest first, so higher priority overwrites)
        sorted_sources = sorted(self._sources, key=lambda s: s.priority)

        merged: Dict[str, Any] = {}
        for source in sorted_sources:
            merged.update(source.values)

        # Apply runtime overrides
        merged.update(self._overrides)

        self._merged = merged

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value.

        Args:
            key: Dot-separated configuration key.
            default: Default value if key not found.

        Returns:
            The configuration value.
        """
        if not self._merged:
            self.load()
        return self._merged.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a configuration value at runtime.

        This sets an override that takes highest priority.

        Args:
            key: Dot-separated configuration key.
            value: The value to set.
        """
        self._overrides[key] = value
        self._merged[key] = value

    def unset(self, key: str) -> bool:
        """Remove a runtime override.

        Args:
            key: Configuration key to unset.

        Returns:
            True if the key was found and removed.
        """
        if key in self._overrides:
            del self._overrides[key]
            # Re-merge to get the value from lower priority sources
            self._merge()
            return True
        return False

    def get_all(self) -> Dict[str, Dict[str, Any]]:
        """Get all configuration values with source information.

        Returns:
            Dictionary mapping keys to {value, source} dictionaries.
        """
        if not self._merged:
            self.load()

        result: Dict[str, Dict[str, Any]] = {}
        sorted_sources = sorted(self._sources, key=lambda s: -s.priority)

        all_keys = set()
        for source in self._sources:
            all_keys.update(source.keys())
        all_keys.update(self._overrides.keys())

        for key in sorted(all_keys):
            # Find the source with highest priority that has this key
            source_name = "default"
            value = self._defaults.get(key)

            for source in sorted_sources:
                if key in source.values:
                    source_name = source.name
                    value = source.values[key]
                    break

            if key in self._overrides:
                source_name = "override"
                value = self._overrides[key]

            result[key] = {
                "value": value,
                "source": source_name,
            }

        return result

    def get_nested(self) -> Dict[str, Any]:
        """Get the configuration as a nested dictionary.

        Returns:
            Nested configuration dictionary.
        """
        if not self._merged:
            self.load()
        return self._unflatten(self._merged)

    def save(self, path: Optional[str] = None) -> None:
        """Save current configuration to a file.

        Args:
            path: File path to save to. Uses the loaded file path if not specified.
        """
        save_path = path or self._file_path
        if save_path is None:
            raise ConfigError(message="No file path specified for saving configuration.")

        file_path = Path(save_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        nested_config = self.get_nested()

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                if file_path.suffix in (".yaml", ".yml"):
                    try:
                        import yaml
                        yaml.dump(nested_config, f, default_flow_style=False, sort_keys=True)
                    except ImportError:
                        raise ConfigError(message="PyYAML required for YAML export.")
                elif file_path.suffix == ".json":
                    json.dump(nested_config, f, indent=2, ensure_ascii=False)
                else:
                    raise ConfigError(
                        message=f"Unsupported format for saving: {file_path.suffix}"
                    )

            logger.info("Configuration saved to: %s", save_path)

        except ConfigError:
            raise
        except Exception as exc:
            raise ConfigError(
                message=f"Failed to save configuration: {exc}",
                config_source=save_path,
            ) from exc

    def reset(self) -> None:
        """Reset configuration to defaults."""
        self._sources = [self._sources[0]]  # Keep defaults
        self._overrides.clear()
        self._merged.clear()
        self._file_path = None
        logger.info("Configuration reset to defaults.")

    def validate(self, schema: Optional[Dict[str, Any]] = None) -> List[str]:
        """Validate the current configuration.

        Args:
            schema: Optional validation schema. If not provided, basic
                   validation is performed.

        Returns:
            List of validation error messages (empty if valid).
        """
        if not self._merged:
            self.load()

        errors: List[str] = []

        if schema:
            for key, rules in schema.items():
                value = self._merged.get(key)
                if rules.get("required") and value is None:
                    errors.append(f"Required configuration key '{key}' is missing.")
                if rules.get("type") and value is not None:
                    expected_type = rules["type"]
                    if not isinstance(value, expected_type):
                        errors.append(
                            f"Configuration key '{key}' expected type {expected_type.__name__}, "
                            f"got {type(value).__name__}."
                        )
                if "choices" in rules and value is not None:
                    if value not in rules["choices"]:
                        errors.append(
                            f"Configuration key '{key}' value '{value}' not in "
                            f"allowed choices: {rules['choices']}."
                        )
                if "range" in rules and value is not None:
                    min_val, max_val = rules["range"]
                    if not (min_val <= value <= max_val):
                        errors.append(
                            f"Configuration key '{key}' value {value} out of "
                            f"range [{min_val}, {max_val}]."
                        )

        # Basic validation
        if not isinstance(self._merged.get("port", 8000), int):
            errors.append("Port must be an integer.")
        if not isinstance(self._merged.get("temperature", 0.7), (int, float)):
            errors.append("Temperature must be a number.")
        if not isinstance(self._merged.get("max_tokens", 2048), int):
            errors.append("Max tokens must be an integer.")

        return errors

    def __repr__(self) -> str:
        return f"ConfigLoader(sources={len(self._sources)}, keys={len(self._merged)})"
