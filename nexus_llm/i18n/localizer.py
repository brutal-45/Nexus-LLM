"""
Localization Manager for Nexus-LLM

Provides a Localizer class that loads language-specific YAML files
and resolves translated strings with optional format parameters.

Supports:
- Multiple locales loaded from YAML files
- Fallback to English when a key is missing in the active locale
- String interpolation with named placeholders: t("chat.saved", path="/tmp")
- Lazy loading and caching of locale files
- Runtime locale switching
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

class LocalizationError(Exception):
    """Base exception for localization errors."""


class LocaleNotFoundError(LocalizationError):
    """Raised when a requested locale file cannot be found."""

    def __init__(self, locale: str, path: Optional[Path] = None) -> None:
        self.locale = locale
        self.path = path
        msg = f"Locale '{locale}' not found"
        if path:
            msg += f" at {path}"
        super().__init__(msg)


class KeyNotFoundError(LocalizationError):
    """Raised when a translation key cannot be resolved."""

    def __init__(self, key: str, locale: str) -> None:
        self.key = key
        self.locale = locale
        super().__init__(f"Key '{key}' not found in locale '{locale}'")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

I18N_DIR = Path(__file__).parent

DEFAULT_LOCALE = "en"

SUPPORTED_LOCALES = ["en", "zh", "ja", "es"]


# ---------------------------------------------------------------------------
# Localizer
# ---------------------------------------------------------------------------

class Localizer:
    """Manages localization strings for Nexus-LLM.

    Loads YAML locale files from disk, caches them, and resolves
    dot-separated keys like ``"chat.welcome"`` to the translated string.

    Example::

        loc = Localizer()
        loc.set_locale("es")
        msg = loc.t("chat.welcome")       # Spanish welcome message
        msg = loc.t("chat.saved", path="/tmp/chat.json")
    """

    def __init__(
        self,
        locale_dir: Optional[Union[str, Path]] = None,
        default_locale: str = DEFAULT_LOCALE,
    ) -> None:
        """Initialize the Localizer.

        Args:
            locale_dir: Directory containing YAML locale files.
                        Defaults to the bundled i18n directory.
            default_locale: Fallback locale when keys are missing.
        """
        self._locale_dir = Path(locale_dir) if locale_dir else I18N_DIR
        self._default_locale = default_locale
        self._current_locale = default_locale
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._flat_cache: Dict[str, Dict[str, str]] = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_locale(self) -> str:
        """Return the currently active locale code."""
        return self._current_locale

    @property
    def default_locale(self) -> str:
        """Return the default (fallback) locale code."""
        return self._default_locale

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_locale(self, locale: str) -> None:
        """Switch the active locale.

        Args:
            locale: A locale code such as 'en', 'zh', 'ja', or 'es'.

        Raises:
            LocaleNotFoundError: If the locale file does not exist.
        """
        self._ensure_loaded(locale)
        self._current_locale = locale

    def t(self, key: str, **kwargs: Any) -> str:
        """Translate a key using the current locale.

        Key format is dot-separated, e.g. ``"chat.welcome"`` resolves to
        the ``chat.welcome`` value in the locale YAML.

        Keyword arguments are used for string interpolation:
        ``t("chat.saved", path="/tmp")`` → ``"Chat saved to /tmp"``

        If the key is missing in the current locale, falls back to the
        default locale. If still missing, returns the key itself.

        Args:
            key: Dot-separated translation key.
            **kwargs: Format parameters for interpolation.

        Returns:
            The translated (and interpolated) string.
        """
        template = self._resolve(key, self._current_locale)
        if template is None:
            template = self._resolve(key, self._default_locale)
        if template is None:
            return key

        try:
            return template.format(**kwargs)
        except KeyError:
            # Return the template as-is if some format params are missing
            return template

    def get(self, key: str, locale: Optional[str] = None) -> Optional[str]:
        """Get a raw translation string without interpolation.

        Args:
            key: Dot-separated translation key.
            locale: Locale to look up. Defaults to current locale.

        Returns:
            The raw string, or None if not found.
        """
        loc = locale or self._current_locale
        return self._resolve(key, loc)

    def list_locales(self) -> List[str]:
        """Return a list of available locale codes.

        Scans the locale directory for .yaml files.
        """
        locales: List[str] = []
        if self._locale_dir.exists():
            for f in sorted(self._locale_dir.glob("*.yaml")):
                locales.append(f.stem)
        return locales

    def list_keys(self, locale: Optional[str] = None) -> List[str]:
        """Return all available translation keys for a locale.

        Args:
            locale: Locale code. Defaults to current locale.

        Returns:
            Sorted list of dot-separated keys.
        """
        loc = locale or self._current_locale
        flat = self._get_flat(loc)
        return sorted(flat.keys())

    def reload(self, locale: Optional[str] = None) -> None:
        """Clear cached locale data.

        Args:
            locale: If specified, only clear cache for that locale.
                    Otherwise, clear the entire cache.
        """
        if locale is not None:
            self._cache.pop(locale, None)
            self._flat_cache.pop(locale, None)
        else:
            self._cache.clear()
            self._flat_cache.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_loaded(self, locale: str) -> None:
        """Ensure a locale is loaded into cache.

        Args:
            locale: Locale code to load.

        Raises:
            LocaleNotFoundError: If the locale file does not exist.
        """
        if locale in self._cache:
            return

        filepath = self._locale_dir / f"{locale}.yaml"
        if not filepath.exists():
            raise LocaleNotFoundError(locale, filepath)

        with open(filepath, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if data is None:
            data = {}

        self._cache[locale] = data

    def _get_flat(self, locale: str) -> Dict[str, str]:
        """Return a flattened key→value map for a locale.

        Nested dictionaries are joined with dots.
        """
        if locale in self._flat_cache:
            return self._flat_cache[locale]

        self._ensure_loaded(locale)
        data = self._cache[locale]
        flat: Dict[str, str] = {}
        self._flatten(data, prefix="", result=flat)
        self._flat_cache[locale] = flat
        return flat

    def _resolve(self, key: str, locale: str) -> Optional[str]:
        """Resolve a dot-separated key in a given locale.

        Args:
            key: Dot-separated key, e.g. "chat.welcome".
            locale: Locale code.

        Returns:
            The translated string, or None if not found.
        """
        flat = self._get_flat(locale)
        return flat.get(key)

    @staticmethod
    def _flatten(
        obj: Dict[str, Any],
        prefix: str,
        result: Dict[str, str],
    ) -> None:
        """Recursively flatten a nested dict into dot-separated keys."""
        for key, value in obj.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                Localizer._flatten(value, full_key, result)
            elif isinstance(value, str):
                result[full_key] = value
            else:
                result[full_key] = str(value)


# ---------------------------------------------------------------------------
# Module-level convenience API
# ---------------------------------------------------------------------------

_default_localizer: Optional[Localizer] = None


def _get_localizer() -> Localizer:
    """Return (and lazily create) the default Localizer singleton."""
    global _default_localizer
    if _default_localizer is None:
        _default_localizer = Localizer()
    return _default_localizer


def get_localizer() -> Localizer:
    """Return the default Localizer instance."""
    return _get_localizer()


def set_locale(locale: str) -> None:
    """Set the active locale using the default Localizer."""
    _get_localizer().set_locale(locale)


def t(key: str, **kwargs: Any) -> str:
    """Translate a key using the default Localizer.

    This is the primary translation function used throughout the
    application::

        from nexus_llm.i18n import t
        msg = t("chat.welcome")
        msg = t("chat.saved", path="/tmp/chat.json")
    """
    return _get_localizer().t(key, **kwargs)
