"""
Preset Manager for Nexus-LLM

Loads, validates, and applies configuration presets from YAML files.
Supports chat presets, training presets, and server presets.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class PresetError(Exception):
    """Base exception for preset-related errors."""


class PresetNotFoundError(PresetError):
    """Raised when a requested preset cannot be found."""

    def __init__(self, preset_name: str, category: str) -> None:
        self.preset_name = preset_name
        self.category = category
        super().__init__(
            f"Preset '{preset_name}' not found in category '{category}'"
        )


class PresetValidationError(PresetError):
    """Raised when a preset fails validation."""

    def __init__(self, preset_name: str, errors: List[str]) -> None:
        self.preset_name = preset_name
        self.errors = errors
        super().__init__(
            f"Preset '{preset_name}' validation failed: {'; '.join(errors)}"
        )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PRESETS_DIR = Path(__file__).parent

PRESET_CATEGORIES = {
    "chat": PRESETS_DIR / "chat_presets.yaml",
    "training": PRESETS_DIR / "training_presets.yaml",
    "server": PRESETS_DIR / "server_presets.yaml",
}


# ---------------------------------------------------------------------------
# Preset Manager
# ---------------------------------------------------------------------------

class PresetManager:
    """Manages loading, caching, and applying configuration presets.

    Presets are organized by category (chat, training, server) and stored
    as YAML files. The manager loads them on first access and caches them
    for subsequent requests.

    Example::

        manager = PresetManager()
        creative_preset = manager.load("chat", "creative")
        manager.apply("chat", "creative", my_config_dict)
    """

    def __init__(self, presets_dir: Optional[Union[str, Path]] = None) -> None:
        """Initialize the PresetManager.

        Args:
            presets_dir: Optional custom directory for preset YAML files.
                         Defaults to the bundled presets directory.
        """
        if presets_dir is not None:
            self._presets_dir = Path(presets_dir)
        else:
            self._presets_dir = PRESETS_DIR

        self._cache: Dict[str, Dict[str, Any]] = {}
        self._category_files = {
            "chat": self._presets_dir / "chat_presets.yaml",
            "training": self._presets_dir / "training_presets.yaml",
            "server": self._presets_dir / "server_presets.yaml",
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_categories(self) -> List[str]:
        """Return a list of available preset categories.

        Returns:
            List of category names (e.g., ['chat', 'training', 'server']).
        """
        return list(self._category_files.keys())

    def list_presets(self, category: str) -> List[str]:
        """Return a list of preset names within a given category.

        Args:
            category: The preset category (e.g., 'chat', 'training', 'server').

        Returns:
            List of preset names.

        Raises:
            PresetNotFoundError: If the category does not exist.
        """
        data = self._load_category(category)
        return list(data.keys())

    def load(self, category: str, name: str) -> Dict[str, Any]:
        """Load a specific preset by category and name.

        Args:
            category: The preset category.
            name: The preset name within that category.

        Returns:
            Deep copy of the preset dictionary.

        Raises:
            PresetNotFoundError: If the category or preset name does not exist.
        """
        data = self._load_category(category)
        if name not in data:
            available = list(data.keys())
            raise PresetNotFoundError(name, category)
        return copy.deepcopy(data[name])

    def apply(
        self,
        category: str,
        name: str,
        config: Dict[str, Any],
        *,
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        """Apply a preset to an existing configuration dictionary.

        The preset values are merged into the config. By default, existing
        values in *config* are preserved (not overwritten). Set *overwrite*
        to True to allow preset values to replace existing config values.

        Args:
            category: The preset category.
            name: The preset name.
            config: The configuration dictionary to merge into.
            overwrite: Whether preset values should overwrite existing values.

        Returns:
            The merged configuration dictionary.

        Raises:
            PresetNotFoundError: If the preset does not exist.
        """
        preset = self.load(category, name)
        return self._deep_merge(config, preset, overwrite=overwrite)

    def get_description(self, category: str, name: str) -> str:
        """Get the description of a preset.

        Args:
            category: The preset category.
            name: The preset name.

        Returns:
            The preset description string.
        """
        preset = self.load(category, name)
        return preset.get("description", "")

    def validate(self, category: str, name: str) -> List[str]:
        """Validate a preset and return a list of issues found.

        Args:
            category: The preset category.
            name: The preset name.

        Returns:
            List of validation error strings. Empty if valid.
        """
        preset = self.load(category, name)
        errors: List[str] = []

        if "description" not in preset:
            errors.append("Missing required field 'description'")

        if category == "chat":
            errors.extend(self._validate_chat_preset(name, preset))
        elif category == "training":
            errors.extend(self._validate_training_preset(name, preset))
        elif category == "server":
            errors.extend(self._validate_server_preset(name, preset))

        return errors

    def reload(self, category: Optional[str] = None) -> None:
        """Clear cached preset data.

        Args:
            category: If specified, only clear the cache for that category.
                      Otherwise, clear the entire cache.
        """
        if category is not None:
            self._cache.pop(category, None)
        else:
            self._cache.clear()

    def register_category(
        self, category: str, filepath: Union[str, Path]
    ) -> None:
        """Register a custom preset category.

        Args:
            category: Category name.
            filepath: Path to the YAML file containing presets.
        """
        self._category_files[category] = Path(filepath)
        self._cache.pop(category, None)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_category(self, category: str) -> Dict[str, Any]:
        """Load and cache a preset category from its YAML file.

        Args:
            category: The category to load.

        Returns:
            Dictionary of preset name -> preset data.

        Raises:
            PresetNotFoundError: If the category is not registered.
        """
        if category not in self._category_files:
            raise PresetNotFoundError(category, "__category__")

        if category in self._cache:
            return self._cache[category]

        filepath = self._category_files[category]
        if not filepath.exists():
            raise PresetNotFoundError(
                category, f"__file__ ({filepath})"
            )

        with open(filepath, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if data is None:
            data = {}

        self._cache[category] = data
        return data

    @staticmethod
    def _deep_merge(
        base: Dict[str, Any],
        override: Dict[str, Any],
        *,
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        """Recursively merge *override* into *base*.

        When *overwrite* is False (default), values already present in *base*
        are preserved. When True, *override* values take precedence.
        """
        result = copy.deepcopy(base)
        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = PresetManager._deep_merge(
                    result[key], value, overwrite=overwrite
                )
            elif key not in result or overwrite:
                result[key] = copy.deepcopy(value)
        return result

    @staticmethod
    def _validate_chat_preset(
        name: str, preset: Dict[str, Any]
    ) -> List[str]:
        """Validate a chat preset."""
        errors: List[str] = []
        gen = preset.get("generation", {})
        required = ["temperature", "top_p", "max_new_tokens"]
        for field in required:
            if field not in gen:
                errors.append(
                    f"Chat preset '{name}' missing generation field '{field}'"
                )
        temp = gen.get("temperature")
        if temp is not None and not (0.0 <= temp <= 2.0):
            errors.append(
                f"Chat preset '{name}' temperature {temp} out of range [0, 2]"
            )
        return errors

    @staticmethod
    def _validate_training_preset(
        name: str, preset: Dict[str, Any]
    ) -> List[str]:
        """Validate a training preset."""
        errors: List[str] = []
        train = preset.get("training", {})
        required = ["learning_rate", "num_train_epochs", "per_device_train_batch_size"]
        for field in required:
            if field not in train:
                errors.append(
                    f"Training preset '{name}' missing training field '{field}'"
                )
        lr = train.get("learning_rate")
        if lr is not None and lr <= 0:
            errors.append(
                f"Training preset '{name}' has invalid learning_rate: {lr}"
            )
        return errors

    @staticmethod
    def _validate_server_preset(
        name: str, preset: Dict[str, Any]
    ) -> List[str]:
        """Validate a server preset."""
        errors: List[str] = []
        server = preset.get("server", {})
        required = ["host", "port"]
        for field in required:
            if field not in server:
                errors.append(
                    f"Server preset '{name}' missing server field '{field}'"
                )
        port = server.get("port")
        if port is not None and not (1 <= port <= 65535):
            errors.append(
                f"Server preset '{name}' port {port} out of valid range"
            )
        return errors


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

_default_manager: Optional[PresetManager] = None


def _get_manager() -> PresetManager:
    """Return the default PresetManager singleton."""
    global _default_manager
    if _default_manager is None:
        _default_manager = PresetManager()
    return _default_manager


def list_presets(category: str) -> List[str]:
    """List available presets in a category using the default manager."""
    return _get_manager().list_presets(category)


def load_preset(category: str, name: str) -> Dict[str, Any]:
    """Load a preset by category and name using the default manager."""
    return _get_manager().load(category, name)


def apply_preset(
    category: str,
    name: str,
    config: Dict[str, Any],
    *,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Apply a preset to a config dict using the default manager."""
    return _get_manager().apply(category, name, config, overwrite=overwrite)
